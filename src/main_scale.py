#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_scale.py – Scalability sweep (10→100 nodes, step 10)

• Baselines: 2PDGA / GA_fast / ACCESS2020  (可自行增刪)
• 依函式實際 signature 自動傳遞 (task, topo, piid [,config_path] [,workers])
• 輸出：
    tmp_scale_*/result.csv         – raw Solve‑Time/RSS
    tmp_scale_*/curve_time.png     – Solve‑Time 曲線
    tmp_scale_*/curve_mem.png      – RSS‑Memory 曲線
"""

from __future__ import annotations
import argparse, math, random, shutil, tempfile, time, os, sys, importlib, inspect
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import utils

# ──────────────────────────────────────────────────────────────
# 1. Baseline registry   key → (module_path, callable_name or None)
# ──────────────────────────────────────────────────────────────
BASELINES: Dict[str, Tuple[str, str | None]] = {
    "2PDGA"     : ("two_phase_dga",          "TwoPhaseDGA"),
    "GA_fast"   : ("GA_fast.GA_fast",        "GA"),
    "ACCESS2020": ("ACCESS2020.ACCESS2020",  None),     # callable 同名
}

# ──────────────────────────────────────────────────────────────
def load_algo(key: str):
    """動態載入 baseline callable"""
    if key not in BASELINES:
        sys.exit(f"[scale] unknown baseline: {key}")
    mod_path, fn = BASELINES[key]
    mod = importlib.import_module(mod_path)
    if fn is None:
        fn = key
    if not hasattr(mod, fn):
        cand = [c for c in dir(mod) if callable(getattr(mod, c))]
        sys.exit(f"[scale] {key}: callable '{fn}' not found  (candidates={cand})")
    return getattr(mod, fn)

# ──────────────────────────────────────────────────────────────
# 2. Synthetic Grid Topology + Flow generator
# ──────────────────────────────────────────────────────────────
T_PROC_NS = 600
PROP_NS   = 100
RATE_BPS  = 100_000_000          # 100 Mb/s
PKT_B     = (256, 1024)
PERIOD_US = (1_000_000, 2_000_000)

def _grid_edges(k: int) -> List[Tuple[int,int]]:
    edges = []
    for r in range(k):
        for c in range(k):
            v = r * k + c
            if c < k-1:
                edges += [(v, v+1), (v+1, v)]
            if r < k-1:
                edges += [(v, v+k), (v+k, v)]
    return edges

def gen_case(n_nodes: int, seed: int, dst_dir: Path) -> None:
    """生成 0_topo / 0_task 兩個 CSV"""
    rnd = random.Random(seed)
    k = math.ceil(math.sqrt(n_nodes))
    edges = _grid_edges(k)
    edges = [e for e in edges if max(e) < n_nodes]

    topo_rows = [{
        "link"  : str(e), "t_proc": T_PROC_NS, "t_prop": PROP_NS,
        "q_num" : 1,      "rate"  : RATE_BPS
    } for e in edges]
    pd.DataFrame(topo_rows).to_csv(dst_dir/"0_topo.csv", index=False)

    flows, n_flows = [], 2 * n_nodes
    for fid in range(n_flows):
        s, d = rnd.sample(range(n_nodes), 2)
        flows.append({
            "id": fid, "src": s, "dst": [d],
            "size": rnd.randint(*PKT_B),
            "period": rnd.choice(PERIOD_US),
            "deadline": rnd.choice(PERIOD_US),
            "jitter": 0
        })
    pd.DataFrame(flows).to_csv(dst_dir/"0_task.csv", index=False)

# ──────────────────────────────────────────────────────────────
# 3. 調用演算法（根據 signature 決定傳遞參數）
# ──────────────────────────────────────────────────────────────
def call_algo(func, task_csv: Path, topo_csv: Path,
              cfg_dir: Path, workers: int) -> Tuple[float,float]:
    """
    根據函式形參自動匹配：
      必傳: task_csv, topo_csv, piid
      可傳: config_path / cfg_dir , workers
    回傳 (solve_time, rss_MB)
    """
    sig = inspect.signature(func)
    kwargs = {}
    if "config_path" in sig.parameters or "cfg_dir" in sig.parameters:
        kwargs["config_path" if "config_path" in sig.parameters else "cfg_dir"] = str(cfg_dir)
    if "workers" in sig.parameters:
        kwargs["workers"] = workers

    line = func(str(task_csv), str(topo_csv), 0, **kwargs)
    try:
        pid, flag, solve, total, rss = line.split(",")
    except ValueError as e:         # 回傳格式不符
        raise RuntimeError(f"parse‑error: {e}  ({line})") from None
    if flag.strip() != "sat":
        raise RuntimeError("unsat / error")
    return float(solve), float(rss)/1024   # MB

# ──────────────────────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────────────────────
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=int, default=10)
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--step",type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--keep_tmp", action="store_true")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────
def main():
    args = parse_cli()
    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_scale_"))
    print("★ working dir:", tmp_root)
    rows = []

    try:
        for n in range(args.min, args.max + 1, args.step):
            case_dir = tmp_root/f"n{n}"
            case_dir.mkdir()
            gen_case(n, seed=2025, dst_dir=case_dir)
            task_csv, topo_csv = case_dir/"0_task.csv", case_dir/"0_topo.csv"

            for key, (mod_path, _) in BASELINES.items():
                func = load_algo(key)
                cfg   = case_dir/key
                t0 = time.perf_counter()
                try:
                    solve, rss = call_algo(func, task_csv, topo_csv,
                                           cfg, args.workers)
                except Exception as e:
                    print(f"[{key} {n}N] ✘", e)
                    solve = rss = float("nan")
                rows.append(dict(nodes=n, algo=key,
                                 solve=solve, rss=rss))
                if math.isfinite(solve):
                    print(f"[{key:9}] {n:3}N  {solve:.3f}s  {rss:.1f} MB")
                else:
                    print(f"[{key:9}] {n:3}N  NaN")

        # -------- save csv --------------------------------------------------
        df = pd.DataFrame(rows)
        out_csv = tmp_root/"result.csv"
        df.to_csv(out_csv, index=False)
        print("\n✓ raw result →", out_csv)

        # -------- plot curves ----------------------------------------------
        for metric, ylabel, fname in (
            ("solve", "Solve‑Time (s)", "curve_time.png"),
            ("rss",   "RSS Memory (MB)", "curve_mem.png")):

            plt.figure()
            for name, grp in df.groupby("algo"):
                plt.plot(grp["nodes"], grp[metric],
                         marker="o", label=name)
            plt.xlabel("Number of nodes")
            plt.ylabel(ylabel)
            plt.grid(True, ls="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(tmp_root/fname, dpi=300)
            plt.close()
            print("  ↳", fname)

        print("\n✓ done. artefacts @", tmp_root)

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
