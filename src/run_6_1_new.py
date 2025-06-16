#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_6_1_new.py – build dataset & run main_exp2 (§6-1 Static+Scalability)
========================================================================
folder layout
-------------
run6_1_YYYYMMDD_HHMMSS/
├─ data/0/              ← generated dataset (π-ids in ascending nodes)
│   ├─ *_topo.csv / *_task.csv
│   └─ dataset_logs.csv
└─ results/             ← main_exp2.py outputs
    ├─ all_results.csv
    ├─ metrics_summary.csv
    ├─ curve_time.png
    └─ curve_mem.png
"""
from __future__ import annotations
import argparse, math, random, datetime, subprocess, sys, os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# ───────────── CLI ──────────────────────────────────────────────────────────
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alg", default="all",
                   help="'all' or comma-separated keys (see main_exp2 FUNC)")
    p.add_argument("--workers", type=int, default=8,
                   help="GA_fast / 2PDGA internal workers")
    p.add_argument("--step", type=int, default=10,
                   help="node sweep step (default 10, i.e., 10,20,…100)")
    p.add_argument("--case_timeout", type=int, default=600,
                   help="timeout per π-case (sec) passed to main_exp2)")
    return p.parse_args()

# ───────────── dataset generator ────────────────────────────────────────────
T_PROC, T_PROP, RATE = 600, 100, 100_000_000
PERIOD = (1_000_000, 2_000_000)
SIZE = (256, 1024)

def _grid_edges(k: int) -> List[Tuple[int,int]]:
    e = []
    for r in range(k):
        for c in range(k):
            v = r * k + c
            if c < k - 1:
                e += [(v, v + 1), (v + 1, v)]
            if r < k - 1:
                e += [(v, v + k), (v + k, v)]
    return e

def _gen_one(pid: int, nodes: int, flows: int, out: Path, rnd: random.Random) -> int:
    k = math.ceil(math.sqrt(nodes))
    edges = [e for e in _grid_edges(k) if max(e) < nodes]
    topo = [{
        "link": str(e),
        "t_proc": T_PROC,
        "t_prop": T_PROP,
        "q_num": 1,
        "rate": RATE
    } for e in edges]
    pd.DataFrame(topo).to_csv(out / f"{pid}_topo.csv", index=False)

    rows = []
    for fid in range(flows):
        s, d = rnd.sample(range(nodes), 2)
        rows.append({
            "id": fid,
            "src": s,
            "dst": [d],
            "size": rnd.randint(*SIZE),
            "period": rnd.choice(PERIOD),
            "deadline": rnd.choice(PERIOD),
            "jitter": 0
        })
    pd.DataFrame(rows).to_csv(out / f"{pid}_task.csv", index=False)
    return len(edges)

def build_dataset(root: Path, step: int) -> Path:
    """create data/0/*.csv & dataset_logs.csv"""
    d0 = root / "data" / "0"
    d0.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(2025)  # fixed seed
    logs = []
    pid = 0
    for n in range(10, 101, step):
        e = _gen_one(pid, n, 2 * n, d0, rnd)
        logs.append([pid, n, 2 * n, e])
        pid += 1
    if 60 not in range(10, 101, step):
        e = _gen_one(pid, 60, 120, d0, rnd)
        logs.append([pid, 60, 120, e])
    pd.DataFrame(logs, columns=["id", "nodes", "flows", "edges"]).to_csv(
        d0 / "dataset_logs.csv", index=False)
    return d0

# ───────────── quick plotting ───────────────────────────────────────────────
def quick_plot(df: pd.DataFrame, out: Path):
    import matplotlib; matplotlib.use("Agg")
    for m, ylab, fn in [
        ("solve_time", "Solve-Time (s)", "curve_time.png"),
        ("total_mem", "Memory (MB)", "curve_mem.png")
    ]:
        plt.figure()
        for alg, g in df[df.is_feasible == "sat"].groupby("method"):
            plt.plot(g["nodes"], g[m], marker="o", label=alg)
        plt.xlabel("Number of nodes")
        plt.ylabel(ylab)
        plt.grid(ls="--")
        plt.legend()
        plt.savefig(out / fn, dpi=300)
        plt.close()

# ───────────── main ─────────────────────────────────────────────────────────
def main():
    A = _cli()
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    root = Path(f"run6_1_{ts}").absolute()
    data_dir = build_dataset(root, A.step)

    print("★ root :", root)
    print("  dataset:", data_dir)

    # ----- prepare environment for CP-Optimizer execfile override ----------
    env = os.environ.copy()
    env["DOCPLEX_CP_CONTEXT"] = (
        '{"context": {"solver": {"local": '
        '{"execfile": "/opt/ibm/ILOG/CPLEX_Studio2211/'
        'cpoptimizer/bin/x86-64_linux/cpoptimizer"}}}}'
    )

    # ----- invoke main_exp2.py ---------------------------------------------
    cmd = [
        sys.executable, "main_exp2.py",
        "--alg", A.alg,
        "--path", str(root / "data"),
        "--ins", "0",
        "--ga_workers", str(A.workers),
        "--out", str(root / "results"),
        "--time_limit", str(A.case_timeout)
    ]
    print("  run  :", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

    # ----- plot ------------------------------------------------------------
    all_csv = root / "results" / "all_results.csv"
    if all_csv.exists():
        quick_plot(pd.read_csv(all_csv), root / "results")
        print("  summaries & plots →", root / "results")

    print("\n✓ finished")

if __name__ == "__main__":
    main()
