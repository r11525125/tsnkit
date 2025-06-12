#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_exp.py – Unified experiment launcher for TSN-KIT algorithms (with 2PDGA)
=======================================================================
Usage example:
  cd /Users/xutingwei/Downloads/tsnkit-legacy/src
  python main_exp.py \
    --alg 2PDGA \
    --ins 0 \
    --start 0 \
    --end 5 \
    --path ../data/grid \
    --ga_workers 12
"""
from __future__ import annotations
import argparse
import os
import sys
import importlib
import warnings
import multiprocessing as mp
import glob
import gc
import time
from typing import Callable, List, Dict

import pandas as pd
import utils

warnings.filterwarnings("ignore")

# ─── 1. Algorithm registry (key → module path) ───────────────────────────────
FUNC: Dict[str, str] = {
    "GA_fast":        "GA_fast.GA_fast",       # Phase I GA_fast
    "GA":             "GA.GA",
    "RTAS2018":       "RTAS2018.RTAS2018",
    "RTAS2020":       "RTAS2020.RTAS2020",
    "ACCESS2020":     "ACCESS2020.ACCESS2020",
    "ASPDAC2022":     "ASPDAC2022.ASPDAC2022",
    "CIE2021":        "CIE2021.CIE2021",
    "COR2022":        "COR2022.COR2022",
    "IEEEJAS2021":    "IEEEJAS2021.IEEEJAS2021",
    "IEEETII2020":    "IEEETII2020.IEEETII2020",
    "RTCSA2018":      "RTCSA2018.RTCSA2018",
    "RTCSA2020":      "RTCSA2020.RTCSA2020",
    "RTNS2016":       "RTNS2016.RTNS2016",
    "RTNS2016_nowait":"RTNS2016_nowait.RTNS2016_nowait",
    "RTNS2017":       "RTNS2017.RTNS2017",
    "RTNS2021":       "RTNS2021.RTNS2021",
    "RTNS2022":       "RTNS2022.RTNS2022",
    "SIGBED2019":     "SIGBED2019.SIGBED2019",
    "GLOBECOM2022":   "GLOBECOM2022.GLOBECOM2022",
    "2PDGA":          "two_phase_dga",         # Phase I+II two-phase DGA
}

# ─── 2. CLI parameters ────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alg",       required=True,
                   help="'all' or comma-separated algorithm keys")
    p.add_argument("--ins",       required=True, type=int,
                   help="instance index under data/<exp>/<ins>/")
    p.add_argument("--start",     default=0, type=int,
                   help="start row (0-based) in dataset_logs.csv")
    p.add_argument("--end",       type=int,
                   help="end row (inclusive) in dataset_logs.csv")
    p.add_argument("--path",      required=True,
                   help="root data directory, e.g. ../data/grid")
    p.add_argument("--ga_workers",default=mp.cpu_count(), type=int,
                   help="worker count for GA_fast (Phase I)")
    p.add_argument("--out",       default="results",
                   help="root output directory")
    return p.parse_args()

# ─── 3. Dynamic import of algorithm function ─────────────────────────────────
def load_algo(key: str) -> Callable:
    if key not in FUNC:
        sys.exit(f"[main_exp] ✘ Unknown algorithm key: {key}")
    mod_path = FUNC[key]
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        sys.exit(f"[main_exp] ✘ Failed to import '{mod_path}': {e}")
    base = key.split("_")[0]
    if hasattr(mod, base):
        return getattr(mod, base)
    if hasattr(mod, key):
        return getattr(mod, key)
    if key == "2PDGA" and hasattr(mod, "TwoPhaseDGA"):
        return getattr(mod, "TwoPhaseDGA")
    cands = [c for c in dir(mod) if callable(getattr(mod, c))]
    sys.exit(f"[main_exp] ✘ Module '{mod_path}' has no '{base}'/'{key}' or 'TwoPhaseDGA'. "
             f"Available callables: {cands}")

# ─── 4. Extract delay metrics ────────────────────────────────────────────────
DELAY_KEYS = ("delay", "latency", "e2e", "end2end")
def metric_from_dir(cfg_dir: str, pid: int):
    patterns = glob.glob(os.path.join(cfg_dir, "**", f"*{pid}*.csv"), recursive=True)
    for f in patterns:
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        if not any(k in stem for k in DELAY_KEYS):
            continue
        df = pd.read_csv(f)
        for col in df.columns:
            if any(k in col.lower() for k in DELAY_KEYS):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty:
                    return vals.max(), vals.mean(), vals.std()
    return 0.0, 0.0, 0.0

# ─── 5. Main orchestrator ───────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # — Data directory logic (与 src/main.py 保持一致) —————
    data_base = os.path.abspath(args.path.rstrip("/"))
    exp       = os.path.basename(data_base)
    ins_s     = str(args.ins)
    data_dir  = os.path.join(data_base, ins_s)
    if not os.path.isdir(data_dir):
        sys.exit(f"✘ data directory not found: {data_dir}")

    log_csv = os.path.join(data_dir, "dataset_logs.csv")
    if not os.path.isfile(log_csv):
        sys.exit(f"✘ missing {log_csv}")
    df_log = pd.read_csv(log_csv)
    end_idx = args.end if args.end is not None else len(df_log) - 1
    df_log = df_log.iloc[args.start : end_idx + 1]
    if df_log.empty:
        sys.exit("✘ empty slice – nothing to do")

    # — Algorithm selection ——————————————————————————————————————
    keys = list(FUNC) if args.alg.lower()=="all" else [k.strip() for k in args.alg.split(",")]
    bad = [k for k in keys if k not in FUNC]
    if bad:
        sys.exit(f"✘ unknown algorithm keys: {bad}")

    # Prepare output dirs
    out_root = os.path.join(args.out, exp, ins_s)
    os.makedirs(out_root, exist_ok=True)

    all_results: List[pd.DataFrame] = []

    for key in keys:
        Algo = load_algo(key)
        cfg_root = os.path.join("configs", exp, ins_s, key)
        run_root = os.path.join(out_root, key)
        os.makedirs(cfg_root, exist_ok=True)
        os.makedirs(run_root, exist_ok=True)

        rst_csv = os.path.join(run_root, f"result_{ins_s}_{key}.csv")
        # write header
        pd.DataFrame(columns=[
            "piid","is_feasible","solve_time","total_time","total_mem"
        ]).to_csv(rst_csv, index=False)

        utils.init(exp=exp, ins=args.ins, method=key)
        utils.rheader()

        # 避免 2PDGA 调用 process_num 报 KeyError
        if key in ("GA_fast","2PDGA"):
            nproc = args.ga_workers
        else:
            nproc = utils.process_num(key)

        pool = mp.Pool(processes=max(1,nproc))
        def _collect(line: str):
            if line:
                with open(rst_csv,"a") as f:
                    f.write(line+"\n")

        for _, row in df_log.iterrows():
            pid    = int(row["id"])
            task_f = os.path.join(data_dir, f"{pid}_task.csv")
            topo_f = os.path.join(data_dir, f"{pid}_topo.csv")
            subcfg = os.path.join(cfg_root, f"piid_{pid}")
            pool.apply_async(
                Algo,
                args=(task_f, topo_f, pid, subcfg+"/", nproc),
                callback=_collect
            )
        pool.close()
        pool.join()
        gc.collect()
        time.sleep(0.05)

        # 收集结果
        df_res = pd.read_csv(rst_csv)
        if df_res.empty:
            continue
        df_res["method"] = key
        # attach delay metrics
        metrics = [metric_from_dir(cfg_root, int(pid)) for pid in df_res["piid"]]
        df_res[["max_delay","avg_delay","jitter"]] = pd.DataFrame(metrics)
        all_results.append(df_res)

    if not all_results:
        print("⚠ nothing produced")
        return

    # 合并所有结果
    big = pd.concat(all_results, ignore_index=True)
    all_csv = os.path.join(out_root, f"all_results_{ins_s}_{args.start}_{end_idx}.csv")
    big.to_csv(all_csv, index=False)
    print(f"\n✓ experiment complete → {all_csv}")

    # ─── 新增：生成 metrics 汇总 ─────────────────────────────────────────
    metrics = (
        big
        .assign(sat=lambda df: df.is_feasible=="sat")
        .groupby("method")
        .agg(
            SR=("sat", "mean"),
            avg_solve_time=("solve_time", "mean"),
            avg_total_time=("total_time", "mean"),
            avg_mem=("total_mem", "mean"),
            avg_max_delay=("max_delay", "mean"),
            avg_avg_delay=("avg_delay", "mean"),
            avg_jitter=("jitter", "mean"),
        )
        .reset_index()
    )
    met_csv = os.path.join(out_root, f"metrics_{ins_s}_{args.start}_{end_idx}.csv")
    metrics.to_csv(met_csv, index=False)
    print(f"✓ metrics summary → {met_csv}")

if __name__ == "__main__":
    main()
