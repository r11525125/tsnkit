#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main6_1.py – §6 Static + Scalability Results Generator   (fixed mkdir bug)
==========================================================================
產出：
    tmp_6_1/result.csv
    tmp_6_1/static_table.csv
    tmp_6_1/curve_time.png
    tmp_6_1/curve_mem.png
"""

from __future__ import annotations
import argparse, math, random, tempfile, shutil, sys, time, importlib, inspect
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import utils          # TSNKit 共用 util

# ───────── baseline registry ────────────────────────────────────────────────
BASELINES: Dict[str, Tuple[str, str | None]] = {
    "2PDGA"      : ("two_phase_dga",          "TwoPhaseDGA"),
    "GA_fast"    : ("GA_fast.GA_fast",        "GA"),
    "ACCESS2020" : ("ACCESS2020.ACCESS2020",  None),
    "COR2022"    : ("COR2022.COR2022",        None),
    "RTNS2021"   : ("RTNS2021.RTNS2021",      None),
    "RTAS2020"   : ("RTAS2020.RTAS2020",      None),
    "SIGBED2019" : ("SIGBED2019.SIGBED2019",  None),
}

# ───────── fixed parameters ────────────────────────────────────────────────
T_PROC_NS  = 600
PROP_NS    = 100
RATE_BPS   = 100_000_000
PERIOD_US  = (1_000_000, 2_000_000)
PKT_B      = (256, 1024)

# ────────── helpers ─────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8, help="GA / 2PDGA workers")
    p.add_argument("--step",    type=int, default=10, help="node sweep step")
    p.add_argument("--static_only", action="store_true", help="只跑 60‑node")
    p.add_argument("--keep_tmp", action="store_true")
    return p.parse_args()

def load_algo(key: str):
    mod_path, fn = BASELINES[key]
    fn = fn or key
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, fn):
        cand = [c for c in dir(mod) if callable(getattr(mod, c))]
        sys.exit(f"{key}: callable '{fn}' not found (candidates={cand})")
    return getattr(mod, fn)

def grid_edges(k: int) -> List[Tuple[int,int]]:
    e = []
    for r in range(k):
        for c in range(k):
            v = r*k + c
            if c < k-1: e += [(v, v+1), (v+1, v)]
            if r < k-1: e += [(v, v+k), (v+k, v)]
    return e

def gen_case(nodes: int, flows: int, out: Path, seed: int = 0) -> Tuple[Path,Path]:
    """產生單一 (task, topo) CSV，並確保資料夾存在"""
    out.mkdir(parents=True, exist_ok=True)            ### FIX

    rnd = random.Random(seed)
    k   = math.ceil(math.sqrt(nodes))
    edges = [e for e in grid_edges(k) if max(e) < nodes]

    topo = [{"link": str(e), "t_proc": T_PROC_NS, "t_prop": PROP_NS,
             "q_num": 1, "rate": RATE_BPS} for e in edges]
    pd.DataFrame(topo).to_csv(out/"topo.csv", index=False)

    rows = []
    for fid in range(flows):
        s, d = rnd.sample(range(nodes), 2)
        rows.append({"id": fid, "src": s, "dst": [d],
                     "size": rnd.randint(*PKT_B),
                     "period": rnd.choice(PERIOD_US),
                     "deadline": rnd.choice(PERIOD_US),
                     "jitter": 0})
    pd.DataFrame(rows).to_csv(out/"task.csv", index=False)
    return out/"task.csv", out/"topo.csv"

# ---------- unified call (與 main_exp 一致) ----------------------------------
def call_algo(func, task: Path, topo: Path,
              cfg_dir: Path, nproc: int) -> Tuple[float,float,float,str]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    pos = [str(task), str(topo), 0, str(cfg_dir)+"/"]  # 前四個 positional

    kw = {}
    sig = inspect.signature(func)
    if "nproc"   in sig.parameters: kw["nproc"]   = nproc
    if "workers" in sig.parameters: kw["workers"] = nproc

    line = func(*pos, **kw)
    if not line:
        raise RuntimeError("None returned")

    try:
        pid, flag, solve, total, rss, *_ = line.split(",")
    except ValueError:
        raise RuntimeError(f"un‑parsable output: {line}")

    if flag.strip() != "sat":
        raise RuntimeError("unsat / error")

    d_csv = next((f for f in cfg_dir.glob("*-DELAY.csv")), None)
    return float(solve), float(total), float(rss)/1024, str(d_csv) if d_csv else ""

def extract_delay(csv_path: str) -> Tuple[float,float,float]:
    if not csv_path:
        return float("nan"), float("nan"), float("nan")
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if "delay" in c.lower()), df.columns[-1])
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return v.max(), v.mean(), v.std()

# ────────── main ────────────────────────────────────────────────────────────
def main():
    args = parse_cli()
    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_6_1_"))
    print("★ working dir:", tmp_root)

    node_set = [60] if args.static_only else list(range(10, 101, args.step))
    rows: List[dict] = []

    for n in node_set:
        flows = 2*n
        cdir  = tmp_root/f"case_{n}"                ### FIX moved mkdir to gen_case
        task, topo = gen_case(n, flows, cdir, seed=2025+n)

        for algo in BASELINES:
            func = load_algo(algo)
            print(f"[{algo:<10}] {n:3}N / {flows:3}F … ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                sv, tt, rss, d_csv = call_algo(func, task, topo, cdir/algo, args.workers)
                dmax, davg, djit   = extract_delay(d_csv)
                wall = time.perf_counter() - t0
                print(f"Solve {sv:.3f}s  RSS {rss:.0f}MB  (wall {wall:.1f}s)")
                rows.append(dict(nodes=n, algo=algo, solve=sv, total=tt,
                                 rss=rss, max_delay=dmax, avg_delay=davg,
                                 jitter=djit, sat=1.0))
            except Exception as e:
                print("✘", e)
                rows.append(dict(nodes=n, algo=algo, solve=float("nan"),
                                 total=float("nan"), rss=float("nan"),
                                 max_delay=float("nan"), avg_delay=float("nan"),
                                 jitter=float("nan"), sat=0.0))

    df = pd.DataFrame(rows)
    df.to_csv(tmp_root/"result.csv", index=False)

    static = (df[df.nodes == 60]
              .rename(columns={"solve":"Solve_Time(s)",
                               "rss":"Memory(MB)",
                               "max_delay":"Max_Delay",
                               "avg_delay":"Avg_Delay"}))
    static.to_csv(tmp_root/"static_table.csv", index=False)

    if not args.static_only:
        import matplotlib; matplotlib.use("Agg")
        for metric, ylab, fn in [("solve","Solve‑Time (s)","curve_time.png"),
                                 ("rss",  "Memory (MB)",  "curve_mem.png")]:
            plt.figure()
            for algo, g in df.groupby("algo"):
                plt.plot(g["nodes"], g[metric], marker="o", label=algo)
            plt.xlabel("Number of Nodes"); plt.ylabel(ylab)
            plt.grid(ls="--"); plt.legend()
            plt.savefig(tmp_root/fn, dpi=300); plt.close()

    print("\n✓ outputs @", tmp_root)
    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
