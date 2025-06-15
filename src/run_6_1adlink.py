#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_6_1adlink.py — Generate dataset *and* run §6-1 Static+Scalability benchmark
==========================================================================

資料夾階層
──────────
run6_1_YYYYMMDD_HHMMSS/            ← base_root
├─ data/0/                         ← 新產生的 TSNKit 格式資料集
│   ├─ 0_topo.csv, 0_task.csv
│   ├─ 1_topo.csv, 1_task.csv
│   └─ dataset_logs.csv
└─ results/                        ← main_exp.py 跑批輸出
    ├─ all_results.csv
    ├─ metrics_summary.csv
    ├─ curve_time.png
    └─ curve_mem.png
"""
from __future__ import annotations
import argparse, math, random, datetime, shutil, subprocess, sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ---------- CLI -------------------------------------------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alg", default="all",
                   help="'all' 或用逗號分隔的算法鍵 (見 main_exp.FUNC)")
    p.add_argument("--workers", type=int, default=8,
                   help="GA_fast / 2PDGA 內部 worker 數")
    p.add_argument("--step", type=int, default=10,
                   help="node scalability 步長 (default 10)")
    return p.parse_args()

# ---------- dataset generator -----------------------------------------------
T_PROC = 600
T_PROP = 100
RATE   = 100_000_000
PERIOD = (1_000_000, 2_000_000)
SIZE   = (256, 1024)

def grid_edges(k: int) -> List[Tuple[int,int]]:
    e = []
    for r in range(k):
        for c in range(k):
            v = r*k + c
            if c < k-1: e += [(v, v+1), (v+1, v)]
            if r < k-1: e += [(v, v+k), (v+k, v)]
    return e

def gen_one_case(piid: int, nodes: int, flows: int, out: Path, rnd: random.Random):
    k = math.ceil(math.sqrt(nodes))
    edges = [e for e in grid_edges(k) if max(e) < nodes]
    topo = [{"link": str(e), "t_proc":T_PROC, "t_prop":T_PROP,
             "q_num":1, "rate":RATE} for e in edges]
    pd.DataFrame(topo).to_csv(out/f"{piid}_topo.csv", index=False)

    rows = []
    for fid in range(flows):
        s, d = rnd.sample(range(nodes), 2)
        rows.append({"id":fid,"src":s,"dst":[d],
                     "size":rnd.randint(*SIZE),
                     "period":rnd.choice(PERIOD),
                     "deadline":rnd.choice(PERIOD),
                     "jitter":0})
    pd.DataFrame(rows).to_csv(out/f"{piid}_task.csv", index=False)
    return len(edges)

def build_dataset(dst_root: Path, step: int):
    """產生 data/0/ 內全部 csv；π-id 按 nodes 升序編號"""
    d0 = dst_root/"data"/"0"
    d0.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(2025)  # 固定種子 → 每次同資料
    logs = []
    pid  = 0
    for n in range(10, 101, step):
        f = 2*n
        edges = gen_one_case(pid, n, f, d0, rnd)
        logs.append([pid, n, f, edges])
        pid += 1
    # 60N / 120F static case  (若不在 sweep 範圍則另補)
    if 60 not in range(10,101,step):
        edges = gen_one_case(pid, 60, 120, d0, rnd)
        logs.append([pid, 60, 120, edges])
    df_log = pd.DataFrame(logs, columns=["id","nodes","flows","edges"])
    df_log.to_csv(d0/"dataset_logs.csv", index=False)
    return d0

# ---------- plot util --------------------------------------------------------
def quick_plot(df: pd.DataFrame, out_dir: Path):
    import matplotlib; matplotlib.use("Agg")
    for m, lab, fn in [("solve_time","Solve-Time (s)","curve_time.png"),
                       ("total_mem","Memory (MB)","curve_mem.png")]:
        plt.figure()
        for alg, g in df[df.is_feasible=="sat"].groupby("method"):
            plt.plot(g["nodes"], g[m], marker="o", label=alg)
        plt.xlabel("Node count"); plt.ylabel(lab)
        plt.grid(ls="--"); plt.legend()
        plt.savefig(out_dir/fn, dpi=300); plt.close()

# ---------- main -------------------------------------------------------------
def main():
    args = parse_cli()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = Path(f"run6_1_{ts}").absolute()
    data_dir = build_dataset(base, args.step)

    print(f"★ output root: {base}")
    print(f"  ↳ dataset : {data_dir}")

    # Use current interpreter to invoke main_exp.py
    cmd = [
        sys.executable, "main_exp.py",
        "--alg", args.alg,
        "--path", str(base/"data"),
        "--ins", "0",
        "--ga_workers", str(args.workers),
        "--out", str(base/"results")
    ]
    print("  ↳ running:", " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit("✘ main_exp 失敗；請檢查日誌")

    # 匯總並畫圖
    num_ids = len(pd.read_csv(data_dir/"dataset_logs.csv"))
    all_csv = base/"results"/f"all_results_0_0_{num_ids-1}.csv"
    if all_csv.is_file():
        df = pd.read_csv(all_csv)
        quick_plot(df, base/"results")
        print(f"  ↳ summary & plots saved under {base/'results'}")

    print("\n✓ Done")

if __name__ == "__main__":
    main()
