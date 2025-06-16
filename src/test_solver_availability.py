#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_6_1_new.py – 生成数据并运行 main_exp2 对所有算法的批量实验
========================================================================
用法：
  python run_6_1_new.py

可选参数（也可在命令行覆盖）：
  --step           节点数步长，默认 10（即 10,20,…,100）
  --workers        GA_fast/2PDGA 内部并行线程数，默认为 CPU 核心数
  --case_timeout   每个 π-case 超时秒数，默认 600
  --out_dir        输出根目录，默认 run6_1_<timestamp>
"""

from __future__ import annotations
import os
import sys
import argparse
import math
import random
import datetime
import subprocess
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

def mp_count() -> int:
    try:
        import multiprocessing as mp
        return mp.cpu_count()
    except:
        return 1

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate TSN-KIT dataset & run all algorithms")
    p.add_argument("--step", type=int, default=10,
                   help="节点数步长，默认 10 → 10,20,…,100")
    p.add_argument("--workers", type=int, default=mp_count(),
                   help="GA_fast/2PDGA 并行线程数，默认 CPU 核心数")
    p.add_argument("--case_timeout", type=int, default=600,
                   help="每个 π-case 超时（秒），默认 600")
    p.add_argument("--out_dir", type=str, default=None,
                   help="输出根目录名，默认 run6_1_<timestamp>")
    return p.parse_args()

# 数据集生成参数
T_PROC, T_PROP, RATE = 600, 100, 100_000_000
PERIOD = (1_000_000, 2_000_000)
SIZE = (256, 1024)

def _grid_edges(k: int) -> List[Tuple[int,int]]:
    edges: List[Tuple[int,int]] = []
    for r in range(k):
        for c in range(k):
            v = r * k + c
            if c < k - 1:
                edges += [(v, v+1), (v+1, v)]
            if r < k - 1:
                edges += [(v, v+k), (v+k, v)]
    return edges

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
    data0 = root / "data" / "0"
    data0.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(2025)
    logs = []
    pid = 0
    for n in range(10, 101, step):
        e = _gen_one(pid, n, 2*n, data0, rnd)
        logs.append([pid, n, 2*n, e])
        pid += 1
    if 60 not in range(10, 101, step):
        e = _gen_one(pid, 60, 120, data0, rnd)
        logs.append([pid, 60, 120, e])
    pd.DataFrame(logs, columns=["id","nodes","flows","edges"]) \
      .to_csv(data0/"dataset_logs.csv", index=False)
    return data0

def quick_plot(df: pd.DataFrame, out: Path):
    import matplotlib; matplotlib.use("Agg")
    for m, ylab, fn in [
        ("solve_time", "Solve-Time (s)", "curve_time.png"),
        ("total_mem", "Memory (MB)",    "curve_mem.png")
    ]:
        plt.figure()
        for alg, g in df[df.is_feasible=="sat"].groupby("method"):
            plt.plot(g["nodes"], g[m], marker="o", label=alg)
        plt.xlabel("Number of nodes")
        plt.ylabel(ylab)
        plt.grid(ls="--")
        plt.legend()
        plt.savefig(out/fn, dpi=300)
        plt.close()

def main():
    A = _cli()
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_root = Path(A.out_dir) if A.out_dir else Path(f"run6_1_{ts}")
    out_root = out_root.absolute()
    data_dir = build_dataset(out_root, A.step)

    print("★ root    :", out_root)
    print("  dataset :", data_dir)

    # 子进程环境：确保 CPLEX CP-Optimizer 与 Gurobi 均可用
    env = os.environ.copy()
    env.pop("DOCPLEX_CP_CONTEXT", None)
    env["DOCPLEX_CP_CONTEXT"] = (
        "context.solver.local.execfile="
        "/opt/ibm/ILOG/CPLEX_Studio2211/"
        "cpoptimizer/bin/x86-64_linux/cpoptimizer"
    )
    # Gurobi 环境变量假定已在 ~/.bashrc 中设置

    # 调用 main_exp2.py，使用 all 算法
    cmd = [
        sys.executable, "main_exp2.py",
        "--alg", "all",
        "--path", str(out_root/"data"),
        "--ins", "0",
        "--ga_workers", str(A.workers),
        "--out", str(out_root/"results"),
        "--time_limit", str(A.case_timeout)
    ]
    print("  run     :", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

    results_dir = out_root/"results"
    all_csv = results_dir/"all_results.csv"
    if all_csv.exists():
        quick_plot(pd.read_csv(all_csv), results_dir)
        print("  plots →", results_dir)

    print("\n✓ finished all algorithms")

if __name__ == "__main__":
    main()
