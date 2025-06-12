#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_phase_dga.py – 2‑Phase Distributed GA (2PDGA)
=================================================
介面與舊 GA 系列相容：TwoPhaseDGA(task_csv, topo_csv, piid, cfg_dir, nproc)
"""
from __future__ import annotations
import os, threading, copy, time

import utils
from GA_fast.GA_fast import GA as GA_fast  # Phase-I
from local_opt import LocalOptimizer      # Phase‑II

# --------------------------------------------------------------------------- #
def TwoPhaseDGA(task_csv: str,
                topo_csv: str,
                piid: int,
                cfg_dir: str = "./",
                workers: int = 8) -> str:
    """Run Phase‑I GA_fast, then Phase‑II parallel LocalOptimizers."""
    ## ---------- Phase I ----------------------------------------------------
    t0 = time.perf_counter()
    ga_status = GA_fast(task_csv, topo_csv, piid, cfg_dir, workers)
    if not ga_status.startswith(str(piid)+",sat"):
        return ga_status                    # infeasible 或錯誤早退
    t_ga = time.perf_counter()

    ## ---------- 讀剛剛產生的 GCL 進記憶體 -------------------------------
    gcl_csv = os.path.join(cfg_dir, f"GA_fast-{piid}-GCL.csv")
    if not os.path.isfile(gcl_csv):
        return utils.rprint(piid, "unknown", 0)

    import pandas as pd
    gcl_df = pd.read_csv(gcl_csv)
    # link column 內容是 "(u, v)" → 轉回 tuple
    gcl_df["link"] = gcl_df["link"].apply(lambda s: str(tuple(eval(s))))
    # 依交換機聚合
    sw_to_gcls: dict[int, dict[str,list]] = {}
    for _, row in gcl_df.iterrows():
        u, v = eval(row["link"])
        for sw in (u, v):          # 一條 link 兩頭同時看得到
            d = sw_to_gcls.setdefault(sw, {})
            d.setdefault(row["link"], []).append(
                [row["start"]//utils.t_slot,
                 row["end"]//utils.t_slot,
                 row["queue"]])

    ## ---------- Phase II  (threads) --------------------------------------
    threads = []
    for sw, lnks in sw_to_gcls.items():
        # 深複製 windows，避免多執行緒踩到
        lnks_copy = {lk: copy.deepcopy(wins) for lk, wins in lnks.items()}
        th = threading.Thread(
            target=lambda: LocalOptimizer(sw, lnks_copy).optimise())
        th.start()
        threads.append((th, sw, lnks_copy))
    for th, _, _ in threads:
        th.join()

    ## ---------- merge 回全域 GCL -----------------------------------------
    merged = []
    for _, _, lnks in threads:
        for lk, wins in lnks.items():
            for s,e,q in wins:
                merged.append([eval(lk), q, s*utils.t_slot, e*utils.t_slot,
                               gcl_df["cycle"][0]])

    # 取代舊 GCL
    utils.write_result("2PDGA", piid, merged,
                       pd.read_csv(os.path.join(cfg_dir, f"GA_fast-{piid}-OFFSET.csv")).values.tolist(),
                       pd.read_csv(os.path.join(cfg_dir, f"GA_fast-{piid}-ROUTE.csv")).values.tolist(),
                       pd.read_csv(os.path.join(cfg_dir, f"GA_fast-{piid}-QUEUE.csv")).values.tolist(),
                       pd.read_csv(os.path.join(cfg_dir, f"GA_fast-{piid}-DELAY.csv")).values.tolist(),
                       f"{cfg_dir}/")

    run_time = round(time.perf_counter() - t0, 3)
    extra    = round(time.perf_counter() - t_ga, 3)   # Phase‑II 時間
    return utils.rprint(piid, "sat", run_time, extra_time=extra)
