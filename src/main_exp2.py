#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_exp2.py – TSN‑KIT batch launcher  (2025‑06  **final stable**)
"""
from __future__ import annotations
import argparse, os, sys, importlib, shutil, warnings, glob
import multiprocessing as mp
from typing import Callable, Dict, List
import pandas as pd, utils

warnings.filterwarnings("ignore")

# ── NumPy alias patch (for docplex / legacy code) ───────────────────────────
import numpy as _np      # noqa: E402
for _a, _real in [("bool", "bool_"), ("object", "object_"), ("int", "int_")]:
    if not hasattr(_np, _a):
        setattr(_np, _a, getattr(_np, _real))

# ── registry ────────────────────────────────────────────────────────────────
FUNC: Dict[str, str] = {
    "GA_fast":        "GA_fast.GA_fast",
    "RTAS2018":       "RTAS2018.RTAS2018",
    "RTAS2020":       "RTAS2020.RTAS2020",       # need ext solver
    "ACCESS2020":     "ACCESS2020.ACCESS2020",   # need ext solver
    "ASPDAC2022":     "ASPDAC2022.ASPDAC2022",   # need ext solver
    "CIE2021":        "CIE2021.CIE2021",         # need ext solver
    "COR2022":        "COR2022.COR2022",         # need ext solver
    "IEEEJAS2021":    "IEEEJAS2021.IEEEJAS2021", # need Gurobi
    "IEEETII2020":    "IEEETII2020.IEEETII2020", # need Gurobi
    "RTNS2016":       "RTNS2016.RTNS2016",       # need Gurobi
    "RTNS2016_nowait":"RTNS2016_nowait.RTNS2016_nowait", # need Gurobi
    "RTNS2021":       "RTNS2021.RTNS2021",
    "RTNS2022":       "RTNS2022.RTNS2022",       # need Gurobi
    "SIGBED2019":     "SIGBED2019.SIGBED2019",   # need Gurobi
    "GLOBECOM2022":   "GLOBECOM2022.GLOBECOM2022", # need Gurobi
    "2PDGA":          "two_phase_dga",
}

# algorithms that rely on **external commercial solvers**
NEED_EXT_SOLVER = {
    # CP Optimizer family
    "RTAS2020", "ACCESS2020", "ASPDAC2022",
    "CIE2021", "COR2022",
    # Gurobi family
    "IEEEJAS2021", "IEEETII2020",
    "RTNS2016", "RTNS2016_nowait", "RTNS2022", "SIGBED2019",
    "GLOBECOM2022",
}

TIMEOUTS = {k: 30 for k in FUNC}   # default per‑case limit

# ── check external solvers availability ─────────────────────────────────────
def _executable_in_path(name: str) -> bool:
    exe = shutil.which(name)
    return bool(exe and os.access(exe, os.X_OK))

HAVE_CPO = _executable_in_path("cpoptimizer")
HAVE_GRB = _executable_in_path("gurobi_cl") or _executable_in_path("gurobi")

def ext_solver_ok(alg: str) -> bool:
    if alg in {"RTAS2020","ACCESS2020","ASPDAC2022","CIE2021","COR2022"}:
        return HAVE_CPO
    if alg in {"IEEEJAS2021","IEEETII2020","RTNS2016","RTNS2016_nowait",
               "RTNS2022","SIGBED2019","GLOBECOM2022"}:
        return HAVE_GRB
    return True

# ── CLI ─────────────────────────────────────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alg", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--ins", required=True, type=int)
    p.add_argument("--start", default=0, type=int)
    p.add_argument("--end",   type=int)
    p.add_argument("--ga_workers", default=mp.cpu_count(), type=int)
    p.add_argument("--out", default="results")
    p.add_argument("--time_limit", default=600, type=int,
                   help="default timeout per π‑case (sec)")
    return p.parse_args()

# ── misc helpers ────────────────────────────────────────────────────────────
PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
DELAY_KEYS = ("delay", "latency", "e2e", "end2end")

def metric_dir(cfg: str, pid: int):
    for f in glob.glob(os.path.join(cfg, "**", f"*{pid}*.csv"), recursive=True):
        if not any(k in f.lower() for k in DELAY_KEYS):
            continue
        df = pd.read_csv(f)
        for c in df.columns:
            if any(k in c.lower() for k in DELAY_KEYS):
                v = pd.to_numeric(df[c], errors="coerce").dropna()
                if not v.empty:
                    return v.max(), v.mean(), v.std()
    return 0.0, 0.0, 0.0

def run_with_timeout(fn: Callable, args: tuple, kw: dict,
                     t_lim: int, alg: str, pid: int,
                     cfg_root_abs: str):
    q = mp.Queue()

    def _wrap():
        os.chdir(PROJ_ROOT)                       # 固定 cwd
        os.makedirs(cfg_root_abs, exist_ok=True)  # 保險再建
        try:
            q.put(fn(*args, **kw) or "")
        except Exception as e:
            print(f"[error] {alg} π={pid}: {e}")
            q.put("")

    proc = mp.Process(target=_wrap)
    proc.start()
    proc.join(t_lim)
    if proc.is_alive():
        proc.terminate()
        print(f"[timeout] {alg} π={pid} >{t_lim}s – killed")
        return ""
    return q.get() if not q.empty() else ""

# ── main orchestrator ───────────────────────────────────────────────────────
def main():
    A = cli()

    # -------- dataset slice -----------------------------------------------
    data_root = os.path.abspath(A.path.rstrip("/"))
    exp       = os.path.basename(data_root)      # usually "data"
    ins_s     = str(A.ins)
    data_dir  = os.path.join(data_root, ins_s)

    df_all = pd.read_csv(os.path.join(data_dir, "dataset_logs.csv"))
    end_idx = A.end if A.end is not None else len(df_all) - 1
    df_log  = df_all.iloc[A.start:end_idx + 1]
    if df_log.empty:
        sys.exit("✘ slice empty")

    # -------- output dirs --------------------------------------------------
    out_root = os.path.abspath(os.path.join(A.out, exp, ins_s))
    os.makedirs(out_root, exist_ok=True)

    # -------- algorithm list ----------------------------------------------
    alg_list = (list(FUNC) if A.alg.lower() == "all"
                else [x.strip() for x in A.alg.split(",")])

    big_frames: List[pd.DataFrame] = []

    for alg in alg_list:
        if alg not in FUNC:
            print(f"[skip] {alg} – unknown key"); continue
        if alg in NEED_EXT_SOLVER and not ext_solver_ok(alg):
            print(f"[skip] {alg} – external solver unavailable"); continue

        # ––– dynamic import –––
        try:
            mod = importlib.import_module(FUNC[alg])
        except Exception as e:
            print(f"[skip] {alg} import failed: {e}")
            continue

        fn = getattr(mod, alg, getattr(mod, alg.split("_")[0], None))
        if alg == "2PDGA":
            fn = getattr(mod, "TwoPhaseDGA", fn)
        if fn is None:
            print(f"[skip] {alg} – callable not found"); continue

        # ––– cfg directories –––
        cfg_root_rel = os.path.join("configs", exp, ins_s, alg)
        cfg_root_abs = os.path.abspath(cfg_root_rel)
        run_root = os.path.join(out_root, alg)
        os.makedirs(cfg_root_abs, exist_ok=True)
        os.makedirs(run_root,      exist_ok=True)

        # result csv (one per algorithm)
        rst_csv = os.path.join(run_root, f"result_{ins_s}_{alg}.csv")
        pd.DataFrame(columns=["piid", "is_feasible",
                              "solve_time", "total_time", "total_mem"]
                     ).to_csv(rst_csv, index=False)

        utils.init(exp=exp, ins=A.ins, method=alg)
        utils.rheader()

        nproc   = A.ga_workers if alg in ("GA_fast", "2PDGA") else utils.process_num(alg)
        t_case  = min(TIMEOUTS.get(alg, A.time_limit), A.time_limit)

        # ––– iterate π‑id –––
        for _, row in df_log.iterrows():
            pid = int(row["id"])
            for base in (cfg_root_abs, cfg_root_rel):
                os.makedirs(os.path.join(base, f"piid_{pid}"), exist_ok=True)

            line = run_with_timeout(
                fn,
                (os.path.join(data_dir, f"{pid}_task.csv"),
                 os.path.join(data_dir, f"{pid}_topo.csv"),
                 pid,
                 os.path.join(cfg_root_abs, f"piid_{pid}") + "/",
                 nproc),
                {}, t_case, alg, pid, cfg_root_abs)

            with open(rst_csv, "a") as fp:
                fp.write(line + "\n")

        # ––– collect –––
        df = pd.read_csv(rst_csv)
        if df.empty:
            continue
        df["method"] = alg
        df[["max_delay", "avg_delay", "jitter"]] = pd.DataFrame(
            [metric_dir(cfg_root_abs, int(pid)) for pid in df["piid"]])
        big_frames.append(df)

    # -------- aggregation --------------------------------------------------
    if not big_frames:
        print("⚠ nothing produced")
        return
    big = pd.concat(big_frames, ignore_index=True)
    big.to_csv(os.path.join(out_root, "all_results.csv"), index=False)

    (big.assign(sat=lambda d: d.is_feasible == "sat")
        .groupby("method")
        .agg(SR=("sat", "mean"),
             avg_solve_time=("solve_time", "mean"),
             avg_total_time=("total_time", "mean"),
             avg_mem=("total_mem", "mean"),
             avg_max_delay=("max_delay", "mean"),
             avg_avg_delay=("avg_delay", "mean"),
             avg_jitter=("jitter", "mean"))
        .reset_index()
        .to_csv(os.path.join(out_root, "metrics_summary.csv"), index=False))

    print("✓ experiment complete →", out_root)

if __name__ == "__main__":
    main()
