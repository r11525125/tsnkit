#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_exp2.py – TSN‑KIT batch launcher  (2025‑06 • timeout‑log 版)
"""
from __future__ import annotations
import argparse, os, sys, importlib, shutil, warnings, glob, time
import multiprocessing as mp
from typing import Callable, Dict, List
import pandas as pd, utils
warnings.filterwarnings("ignore")

# ── NumPy alias patch (for docplex / legacy) ───────────────────────────────
import numpy as _np                       # noqa: E402
for a,real in [("bool","bool_"),("object","object_"),("int","int_")]:
    if not hasattr(_np,a):
        setattr(_np,a,getattr(_np,real))

# ── 若有安裝 docplex, 指定 cpoptimizer 路徑 (可自行修改/刪除) ────────────────
try:
    from docplex.cp.config import context
    context.solver.local.execfile = (
        "/opt/ibm/ILOG/CPLEX_Studio2211/"
        "cpoptimizer/bin/x86-64_linux/cpoptimizer"
    )
except Exception:
    context = None

# ── Algorithm registry ─────────────────────────────────────────────────────
FUNC: Dict[str,str] = {
    "GA_fast":"GA_fast.GA_fast",
    "RTAS2018":"RTAS2018.RTAS2018",
    "RTAS2020":"RTAS2020.RTAS2020",       # CP Optimizer
    "ACCESS2020":"ACCESS2020.ACCESS2020",
    "ASPDAC2022":"ASPDAC2022.ASPDAC2022",
    "CIE2021":"CIE2021.CIE2021",
    "COR2022":"COR2022.COR2022",
    "IEEEJAS2021":"IEEEJAS2021.IEEEJAS2021",     # Gurobi
    "IEEETII2020":"IEEETII2020.IEEETII2020",
    "RTNS2016":"RTNS2016.RTNS2016",
    "RTNS2016_nowait":"RTNS2016_nowait.RTNS2016_nowait",
    "RTNS2021":"RTNS2021.RTNS2021",
    "RTNS2022":"RTNS2022.RTNS2022",
    "SIGBED2019":"SIGBED2019.SIGBED2019",
    "GLOBECOM2022":"GLOBECOM2022.GLOBECOM2022",
    "2PDGA":"two_phase_dga",
}
CP_SET  = {"RTAS2020","ACCESS2020","ASPDAC2022","CIE2021","COR2022"}
GRB_SET = {"IEEEJAS2021","IEEETII2020","RTNS2016","RTNS2016_nowait",
           "RTNS2022","SIGBED2019","GLOBECOM2022"}

# ── external‑solver detector ───────────────────────────────────────────────
def _is_exec(path:str)->bool:
    return path and os.path.isfile(path) and os.access(path, os.X_OK)

def has_cpo() -> bool:
    if shutil.which("cpoptimizer"): return True
    if context:
        ex = context.solver.local.execfile
        return _is_exec(ex)
    return False

def has_grb() -> bool:
    if shutil.which("gurobi_cl") or shutil.which("gurobi"): return True
    home = os.environ.get("GUROBI_HOME")
    return _is_exec(os.path.join(home,"bin","gurobi_cl")) if home else False

def solver_ready(alg:str)->bool:
    if alg in CP_SET:  return has_cpo()
    if alg in GRB_SET: return has_grb()
    return True

# ── CLI ────────────────────────────────────────────────────────────────────
def cli() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--alg", required=True)
    p.add_argument("--path",required=True)
    p.add_argument("--ins", required=True, type=int)
    p.add_argument("--start", default=0, type=int)
    p.add_argument("--end",   type=int)
    p.add_argument("--ga_workers", default=mp.cpu_count(), type=int)
    p.add_argument("--out",  default="results")
    p.add_argument("--time_limit", default=600, type=int,
                   help="timeout per π‑case (sec)")
    return p.parse_args()

# ── misc helpers ───────────────────────────────────────────────────────────
DELAY_KEYS=("delay","latency","e2e","end2end")
def metric_dir(cfg:str,pid:int):
    for f in glob.glob(os.path.join(cfg,"**",f"*{pid}*.csv"),recursive=True):
        if not any(k in f.lower() for k in DELAY_KEYS): continue
        df=pd.read_csv(f)
        for c in df.columns:
            if any(k in c.lower() for k in DELAY_KEYS):
                v=pd.to_numeric(df[c],errors="coerce").dropna()
                if not v.empty: return v.max(),v.mean(),v.std()
    return 0.0,0.0,0.0

# ── timeout‑wrapped executor ───────────────────────────────────────────────
def run_case(fn:Callable, args:tuple, kw:dict, t:int,
             alg:str, pid:int, case_dir:str,
             tout_log:List[tuple])->str:
    q=mp.Queue()
    def _inner():
        os.chdir(case_dir)
        try: q.put(fn(*args,**kw) or "")
        except Exception as e:
            print(f"[error] {alg} π={pid}: {e}")
            q.put("")
    p=mp.Process(target=_inner); p.start(); p.join(t)
    if p.is_alive():
        p.terminate()
        print(f"[timeout] {alg} π={pid} >{t}s – killed")
        tout_log.append((alg,pid,t))
        return ""
    return q.get() if not q.empty() else ""

# ── main orchestrator ──────────────────────────────────────────────────────
def main():
    A=cli()
    data_root=os.path.abspath(A.path.rstrip("/"))
    exp=os.path.basename(data_root); ins=str(A.ins)
    data_dir=os.path.join(data_root,ins)

    df_all=pd.read_csv(os.path.join(data_dir,"dataset_logs.csv"))
    end=A.end if A.end is not None else len(df_all)-1
    df=df_all.iloc[A.start:end+1]
    if df.empty: sys.exit("✘ slice empty")

    out_root=os.path.abspath(os.path.join(A.out,exp,ins))
    os.makedirs(out_root,exist_ok=True)
    timeout_records:List[tuple]=[]

    algs=list(FUNC) if A.alg.lower()=="all" else [k.strip() for k in A.alg.split(",")]
    aggregate:List[pd.DataFrame]=[]

    for alg in algs:
        if alg not in FUNC: print(f"[skip] {alg} unknown"); continue
        if not solver_ready(alg):
            print(f"[skip] {alg} external solver unavailable"); continue
        try: mod=importlib.import_module(FUNC[alg])
        except Exception as e:
            print(f"[skip] {alg} import failed: {e}"); continue
        fn=getattr(mod,alg,getattr(mod,alg.split("_")[0],None))
        if alg=="2PDGA": fn=getattr(mod,"TwoPhaseDGA",fn)
        if fn is None: print(f"[skip] {alg} no callable"); continue

        cfg_alg=os.path.abspath(os.path.join("configs",exp,ins,alg))
        run_alg=os.path.join(out_root,alg)
        os.makedirs(cfg_alg,exist_ok=True); os.makedirs(run_alg,exist_ok=True)

        rst=os.path.join(run_alg,f"result_{ins}_{alg}.csv")
        pd.DataFrame(columns=["piid","is_feasible","solve_time",
                              "total_time","total_mem"]).to_csv(rst,index=False)

        utils.init(exp=exp,ins=int(ins),method=alg); utils.rheader()
        workers=A.ga_workers if alg in ("GA_fast","2PDGA") else utils.process_num(alg)

        for _,row in df.iterrows():
            pid=int(row["id"])
            case_dir=os.path.join(cfg_alg,f"piid_{pid}"); os.makedirs(case_dir,exist_ok=True)
            line=run_case(
                fn,
                (os.path.join(data_dir,f"{pid}_task.csv"),
                 os.path.join(data_dir,f"{pid}_topo.csv"),
                 pid,"./",workers),
                {}, A.time_limit, alg, pid, case_dir, timeout_records)
            with open(rst,"a") as fp: fp.write(line+"\n")

        dfr=pd.read_csv(rst)
        if dfr.empty: continue
        dfr["method"]=alg
        dfr[["max_delay","avg_delay","jitter"]]=pd.DataFrame(
            [metric_dir(cfg_alg,int(p)) for p in dfr["piid"]])
        aggregate.append(dfr)

    # -------- write timeout log -------------------------------------------
    if timeout_records:
        pd.DataFrame(timeout_records, columns=["method","piid","limit_s"]) \
          .to_csv(os.path.join(out_root,"timeout_log.csv"), index=False)

    # -------- aggregation --------------------------------------------------
    if not aggregate:
        print("⚠ nothing produced"); return
    big=pd.concat(aggregate,ignore_index=True)
    big.to_csv(os.path.join(out_root,"all_results.csv"),index=False)

    (big.assign(sat=lambda d:d.is_feasible=="sat")
        .groupby("method").agg(
            SR=("sat","mean"),
            avg_solve_time=("solve_time","mean"),
            avg_total_time=("total_time","mean"),
            avg_mem=("total_mem","mean"),
            avg_max_delay=("max_delay","mean"),
            avg_avg_delay=("avg_delay","mean"),
            avg_jitter=("jitter","mean")
        ).reset_index()
        .to_csv(os.path.join(out_root,"metrics_summary.csv"),index=False))

    print("✓ experiment complete →", out_root)

if __name__=="__main__":
    main()
