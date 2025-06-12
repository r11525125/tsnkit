#!/usr/bin/env python3
"""
src/main.py – 批量／單例跑 TSN 排程實驗
產出：
  * results/<exp>/<ins>/<method>/result_<ins>_<method>.csv
  * results/<exp>/<ins>/all_results_<ins>_<start>_<end>.csv
  * results/<exp>/<ins>/metrics_<ins>_<start>_<end>.csv
指標：
  max_delay / avg_delay / jitter 會從 *delay*.csv 中自動計算
"""
from __future__ import annotations

import argparse, os, sys, time, gc, glob, warnings
from multiprocessing import Pool, cpu_count
from typing import Callable, Optional, List, Tuple

import pandas as pd
import utils                        # 專案自帶 util

warnings.filterwarnings("ignore")

# ───────────── 1.  演算法入口 ─────────────────────────────────────────
from GA.GA import GA
from RTAS2018.RTAS2018 import RTAS2018
from RTAS2020.RTAS2020 import RTAS2020
from ACCESS2020.ACCESS2020 import ACCESS2020
from ASPDAC2022.ASPDAC2022 import ASPDAC2022
from CIE2021.CIE2021 import CIE2021
from COR2022.COR2022 import COR2022
from IEEEJAS2021.IEEEJAS2021 import IEEEJAS2021
from IEEETII2020.IEEETII2020 import IEEETII2020
from RTCSA2018.RTCSA2018 import RTCSA2018
from RTCSA2020.RTCSA2020 import RTCSA2020
from RTNS2016.RTNS2016 import RTNS2016
from RTNS2016_nowait.RTNS2016_nowait import RTNS2016_nowait
from RTNS2017.RTNS2017 import RTNS2017
from RTNS2021.RTNS2021 import RTNS2021
from RTNS2022.RTNS2022 import RTNS2022
from SIGBED2019.SIGBED2019 import SIGBED2019
from GLOBECOM2022.GLOBECOM2022 import GLOBECOM2022

FUNC: dict[str, Callable] = {
    "GA": GA, "RTAS2018": RTAS2018, "RTAS2020": RTAS2020,
    "ACCESS2020": ACCESS2020, "ASPDAC2022": ASPDAC2022,
    "CIE2021": CIE2021, "COR2022": COR2022,
    "IEEEJAS2021": IEEEJAS2021, "IEEETII2020": IEEETII2020,
    "RTCSA2018": RTCSA2018, "RTCSA2020": RTCSA2020,
    "RTNS2016": RTNS2016, "RTNS2016_nowait": RTNS2016_nowait,
    "RTNS2017": RTNS2017, "RTNS2021": RTNS2021,
    "RTNS2022": RTNS2022,
    "SIGBED2019": SIGBED2019, "GLOBECOM2022": GLOBECOM2022,
}

# ───────────── 2. CLI 參數 ───────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   help="演算法名（逗號分隔）或 all")
    p.add_argument("--ins", required=True, type=int,
                   help="子資料夾編號 data/<exp>/<ins>/")
    p.add_argument("--start", default=0, type=int)
    p.add_argument("--end",   default=None, type=int)
    p.add_argument("--path",  required=True,
                   help="資料根目錄，如 ../data/grid/")
    p.add_argument("--workers", default=1, type=int,
                   help="GA fitness 并行 worker 數")
    return p.parse_args()

# ───────────── 3. 延遲檔搜尋 + 指標計算 ──────────────────────────────
DELAY_KEYWORDS = ("delay", "latency", "e2e", "end2end")

def find_delay_file(root: str, piid: int) -> Optional[str]:
    """
    在 root 及其子目錄尋找 *{piid}*<delay keyword>*.csv
    同時包含 root 目錄本身。
    """
    # ① root 直接掃描
    for fn in os.listdir(root):
        low = fn.lower()
        if low.endswith(".csv") and str(piid) in fn and any(k in low for k in DELAY_KEYWORDS):
            return os.path.join(root, fn)

    # ② 遞迴搜尋
    patt = f"**/*{piid}*csv"
    for path in glob.glob(os.path.join(root, patt), recursive=True):
        low = os.path.basename(path).lower()
        if any(k in low for k in DELAY_KEYWORDS):
            return path
    return None

def load_delay_metrics(root: str, piid: int) -> Tuple[float, float, float]:
    """
    root 是 method 的 config 目錄 (configs/<exp>/<ins>/<method>)
    """
    csv_path = find_delay_file(root, piid)
    if not csv_path:
        return 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(csv_path)
        # 嘗試找出包含 delay 的欄位
        col_name = None
        for c in df.columns:
            if any(k in c.lower() for k in DELAY_KEYWORDS):
                col_name = c
                break
        if col_name is None:
            col_name = df.columns[-1]          # 退而求其次
        ser = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if ser.empty:
            return 0.0, 0.0, 0.0
        return float(ser.max()), float(ser.mean()), float(ser.std())
    except Exception as e:
        print(f"⚠️  讀取延遲檔失敗 {csv_path}: {e}", file=sys.stderr)
        return 0.0, 0.0, 0.0

# ───────────── 4. 子進程包裝 ─────────────────────────────────────────
def run_algo(algo: Callable,
             task_f: str,
             topo_f: str,
             piid: int,
             cfg_dir: str,
             nproc: int) -> Optional[str]:
    """Pool worker：成功回傳結果行，失敗回傳 None"""
    if not (os.path.isfile(task_f) and os.path.isfile(topo_f)):
        print(f"⚠️  缺文件 (piid={piid})", file=sys.stderr)
        return None
    os.makedirs(cfg_dir, exist_ok=True)
    prev = os.getcwd()
    try:
        os.chdir(cfg_dir)
        return algo(task_f, topo_f, piid, cfg_dir, nproc)
    except Exception as e:
        print(f"⚠️  {algo.__name__}({piid}) 失敗: {e}", file=sys.stderr)
        return None
    finally:
        os.chdir(prev)

# ───────────── 5. MAIN ──────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # === 準備資料路徑 =================================================
    data_base = os.path.abspath(args.path.rstrip("/"))
    exp       = os.path.basename(data_base)
    ins_str   = str(args.ins)
    data_dir  = os.path.join(data_base, ins_str)
    if not os.path.isdir(data_dir):
        sys.exit(f"❌ 找不到 {data_dir}")

    # dataset_logs.csv
    log_csv = os.path.join(data_dir, "dataset_logs.csv")
    if not os.path.isfile(log_csv):
        sys.exit(f"❌ 缺少 {log_csv}")
    df_log  = pd.read_csv(log_csv)
    end_idx = args.end if args.end is not None else len(df_log) - 1
    df_log  = df_log.iloc[args.start : end_idx + 1]
    if df_log.empty:
        sys.exit("❌ 選定行為空")

    # 方法列表
    methods = list(FUNC) if args.method.lower() == "all" \
              else [m.strip() for m in args.method.split(",")]
    bad = [m for m in methods if m not in FUNC]
    if bad:
        sys.exit(f"❌ 未知方法: {', '.join(bad)}")

    proj_root = os.path.dirname(os.path.abspath(__file__))

    TASK = os.path.join(data_dir, "{}_task.csv")
    TOPO = os.path.join(data_dir, "{}_topo.csv")

    per_method_files: List[Tuple[str, str, str]] = []  # (method, csv_path, cfg_dir)

    # === 逐方法跑 =====================================================
    for m in methods:
        algo = FUNC[m]
        cfg_dir_m = os.path.join(proj_root, "configs", exp, ins_str, m)
        rst_dir_m = os.path.join(proj_root, "results", exp, ins_str, m)
        os.makedirs(rst_dir_m, exist_ok=True)

        rst_csv = os.path.join(rst_dir_m, f"result_{ins_str}_{m}.csv")
        with open(rst_csv, "w") as f:
            f.write("piid,is_feasible,solve_time,total_time,total_mem\n")

        per_method_files.append((m, rst_csv, cfg_dir_m))

        # UI header
        utils.init(exp=exp, ins=args.ins, method=m)
        utils.rheader()

        pool_sz = max(1, cpu_count() // utils.process_num(m))
        with Pool(processes=pool_sz, maxtasksperchild=1) as pool:

            def _collector(line: Optional[str]):
                if line:
                    with open(rst_csv, "a") as f:
                        f.write(line + "\n")

            for _, row in df_log.iterrows():
                pid = int(row["id"])
                pool.apply_async(
                    run_algo,
                    args=(
                        algo, TASK.format(pid), TOPO.format(pid), pid,
                        os.path.join(cfg_dir_m, f"piid_{pid}"),
                        (args.workers if m == "GA" else utils.process_num(m))
                    ),
                    callback=_collector,
                )
            pool.close()
            pool.join()

        gc.collect()
        time.sleep(0.2)

    # === 整合結果 =====================================================
    print("\n⏳ 彙總結果…")
    rows = []
    for m, csv_path, cfg_root_m in per_method_files:
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["method"] = m

        # 延遲指標
        md = []
        for pid in df["piid"]:
            md.append(load_delay_metrics(cfg_root_m, int(pid)))
        df[["max_delay", "avg_delay", "jitter"]] = pd.DataFrame(md, index=df.index)
        rows.append(df)

    if not rows:
        print("⚠️  沒有任何有效結果")
        return

    big_df = pd.concat(rows, ignore_index=True)
    all_csv = os.path.join(
        proj_root, "results", exp, ins_str,
        f"all_results_{ins_str}_{args.start}_{end_idx}.csv")
    big_df.to_csv(all_csv, index=False)

    # 指標匯總
    metrics = (big_df.groupby("method")
               .agg(SR             = ("is_feasible", lambda s: (s=="sat").mean()),
                    avg_solve_time = ("solve_time", "mean"),
                    avg_total_time = ("total_time", "mean"),
                    avg_mem        = ("total_mem",  "mean"),
                    avg_max_delay  = ("max_delay",  "mean"),
                    avg_avg_delay  = ("avg_delay",  "mean"),
                    avg_jitter     = ("jitter",     "mean"))
               .reset_index())
    met_csv = os.path.join(
        proj_root, "results", exp, ins_str,
        f"metrics_{ins_str}_{args.start}_{end_idx}.csv")
    metrics.to_csv(met_csv, index=False)

    print("✅ 彙總完成：")
    print(" •", all_csv)
    print(" •", met_csv)
    print("\n✅ 全部算法執行完畢！")
    print("   configs ➜", os.path.join(proj_root, "configs"))
    print("   results ➜", os.path.join(proj_root, "results"))


if __name__ == "__main__":
    main()
