#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_static.py – §6.1 Static Performance benchmark (60 N / 120 F)
=================================================================
產出：
  tmp_static/result.csv        – 每演算法 raw solve_time / RSS / delay
  tmp_static/static_table.csv  – 論文表格所需彙總 (SR Avg/Max Delay Jitter Time)

Usage
------
  cd <tsnkit-legacy>/src
  python3 main_static.py --workers 12
"""
from __future__ import annotations
import argparse, os, sys, math, random, tempfile, shutil, time, inspect, importlib, threading
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import utils                                  # TSN‑KIT 共用 util

# ────────── baseline registry (論文 6 條 + 2PDGA) ──────────────────────────
BASELINES: Dict[str, Tuple[str, str | None]] = {
    "2PDGA"      : ("two_phase_dga",          "TwoPhaseDGA"),
    "GA_fast"    : ("GA_fast.GA_fast",        "GA"),
    "ACCESS2020" : ("ACCESS2020.ACCESS2020",  None),
    "ASPDAC2022" : ("ASPDAC2022.ASPDAC2022",  None),
    "RTNS2021"   : ("RTNS2021.RTNS2021",      None),
    "RTAS2020"   : ("RTAS2020.RTAS2020",      None),
    "SIGBED2019" : ("SIGBED2019.SIGBED2019",  None),
}

# ────────── topo/flow 參數 ────────────────────────────────────────────────
NODES      = 60
FLOWS      = 120
T_PROC_NS  = 600
PROP_NS    = 100
RATE_BPS   = 100_000_000
PERIOD_US  = (1_000_000, 2_000_000)
PKT_B      = (256, 1024)

HEARTBEAT  = 30          # seconds – 心跳列印間隔

# ────────── CLI ───────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8,
                   help="GA / 2PDGA evaluation workers")
    p.add_argument("--keep_tmp", action="store_true",
                   help="保留臨時目錄 (debug)")
    return p.parse_args()

# ────────── helpers ───────────────────────────────────────────────────────
def load_algo(key: str):
    mod_path, fn = BASELINES[key]
    mod = importlib.import_module(mod_path)
    fn = fn or key
    if not hasattr(mod, fn):
        cand = [c for c in dir(mod) if callable(getattr(mod, c))]
        sys.exit(f"[static] {key}: callable '{fn}' not found (candidates={cand})")
    return getattr(mod, fn)

def grid_edges(k: int) -> List[Tuple[int,int]]:
    edges = []
    for r in range(k):
        for c in range(k):
            v = r*k + c
            if c < k-1: edges += [(v, v+1), (v+1, v)]
            if r < k-1: edges += [(v, v+k), (v+k, v)]
    return edges

def gen_dataset(out: Path, seed: int = 2025):
    """產生 60N/120F 案例 -> out/{0_task,0_topo}.csv"""
    rnd = random.Random(seed)
    k = math.ceil(math.sqrt(NODES))
    edges = [e for e in grid_edges(k) if max(e) < NODES]

    topo = [{
        "link": str(e), "t_proc": T_PROC_NS, "t_prop": PROP_NS,
        "q_num": 1, "rate": RATE_BPS
    } for e in edges]
    pd.DataFrame(topo).to_csv(out/"0_topo.csv", index=False)

    flows = []
    for fid in range(FLOWS):
        s, d = rnd.sample(range(NODES), 2)
        flows.append({
            "id": fid, "src": s, "dst": [d],
            "size": rnd.randint(*PKT_B),
            "period": rnd.choice(PERIOD_US),
            "deadline": rnd.choice(PERIOD_US),
            "jitter": 0
        })
    pd.DataFrame(flows).to_csv(out/"0_task.csv", index=False)

def call_algo(func, task: Path, topo: Path,
              cfg_dir: Path, workers: int) -> Tuple[float,float,str]:
    """智慧匹配 signature；回傳 solve_time, rss_MB, delay_csv_path"""
    sig = inspect.signature(func)
    kwargs = {}
    if "config_path" in sig.parameters or "cfg_dir" in sig.parameters:
        kwargs["config_path" if "config_path" in sig.parameters else "cfg_dir"] = str(cfg_dir)
    if "workers" in sig.parameters:
        kwargs["workers"] = workers

    line = func(str(task), str(topo), 0, **kwargs)
    pid, flag, solve, total, rss = line.split(",")
    if flag.strip() != "sat":
        raise RuntimeError("unsat/error")

    delay_csv = next((f for f in cfg_dir.glob("**/*0*-DELAY.csv")), "")
    return float(solve), float(rss)/1024, str(delay_csv)

# ────────── 心跳 thread ────────────────────────────────────────────────────
def start_heartbeat(tag: str) -> threading.Event:
    stop_evt = threading.Event()
    def _beat():
        t0 = time.time()
        while not stop_evt.wait(HEARTBEAT):
            dt = int(time.time() - t0)
            print(f"[{tag}] … still running ({dt}s)", flush=True)
    threading.Thread(target=_beat, daemon=True).start()
    return stop_evt

# ────────── main ────────────────────────────────────────────────────────────
def main():
    args = parse_cli()
    tmp = Path(tempfile.mkdtemp(prefix="tmp_static_"))
    print("★ working dir:", tmp, flush=True)

    # 1. 生成資料集
    gen_dataset(tmp, seed=2025)
    task_csv, topo_csv = tmp/"0_task.csv", tmp/"0_topo.csv"

    # 2. 執行 baseline
    rows = []
    total_alg = len(BASELINES)
    for idx, key in enumerate(BASELINES, 1):
        print(f"\n▶ ({idx}/{total_alg}) start {key}", flush=True)
        func = load_algo(key)
        cfg  = tmp/key
        beat = start_heartbeat(key)          # 啟動心跳

        try:
            solve, rss, d_csv = call_algo(func, task_csv, topo_csv,
                                          cfg, args.workers)
            beat.set()                       # 停止心跳
            delay_max = delay_avg = jitter = float("nan")
            if d_csv:
                df_d = pd.read_csv(d_csv)
                col = next((c for c in df_d.columns
                            if "delay" in c.lower()), df_d.columns[-1])
                vals = pd.to_numeric(df_d[col], errors="coerce").dropna()
                delay_max = vals.max()
                delay_avg = vals.mean()
                jitter    = vals.std()
            rows.append(dict(method=key,
                             SR=1.0,
                             solve_time=solve,
                             rss=rss,
                             max_delay=delay_max,
                             avg_delay=delay_avg,
                             jitter=jitter))
            print(f"[{key:10}] finished – Solve={solve:.3f}s  RSS={rss:.1f} MB", flush=True)

        except Exception as e:
            beat.set()
            print(f"[{key}] ✘", e, flush=True)
            rows.append(dict(method=key, SR=0, solve_time=float("nan"),
                             rss=float("nan"), max_delay=float("nan"),
                             avg_delay=float("nan"), jitter=float("nan")))

    # 3. 匯出結果
    df = pd.DataFrame(rows)
    df.to_csv(tmp/"result.csv", index=False)

    table = (df
             .assign(SR=lambda d:d.SR.astype(float))
             .rename(columns={
                 "solve_time":"Solve_Time(s)",
                 "rss":"Memory(MB)",
                 "max_delay":"Max_Delay",
                 "avg_delay":"Avg_Delay",
                 "jitter":"Jitter"
             }))
    table.to_csv(tmp/"static_table.csv", index=False)
    print("\n✓ outputs →", tmp, flush=True)

    if not args.keep_tmp:
        print("… removing tmp dir")
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
