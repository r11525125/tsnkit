#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_phase_dga.py – Two‑Phase Distributed GA (2PDGA)  **2025‑06 修正版**
-----------------------------------------------------------------------
• 新增 `K_PATHS` 與 `HOP_LIMIT` 全域常數，透過 kwargs 傳入 GA_fast
• workers=0 ⇒ 僅執行 Phase‑II  (供 dynamic test)
• 仍產出 5 CSV (*‑GCL, OFFSET, ROUTE, QUEUE, DELAY) 與 utils.rprint
"""
from __future__ import annotations
import os, copy, time, threading
from pathlib import Path
from typing import Dict, List

import pandas as pd
import utils
from GA_fast.GA_fast import GA as _GA_fast          # 我們稍後會把 GA_fast 更新

# ────────── Phase‑I 參數 (可在此一站式調整) ──────────────────────────
K_PATHS   = 3     # 每流僅考慮前 K 條最短路
HOP_LIMIT = 8     # 每條路徑最多 hop  數

# ────────── Phase‑II  (local optimiser) ────────────────────────────
from local_opt import LocalOptimizer               # 同目錄下 (見下一節)

# =============================================================================
def TwoPhaseDGA(task_csv: str,
                topo_csv: str,
                piid: int,
                cfg_dir: str | os.PathLike = "./",
                workers: int = 8) -> str:
    """
    Phase‑I:  GA_fast(…, k_paths=K_PATHS, hop_limit=HOP_LIMIT, …)
    Phase‑II: LocalOptimizer 逐交換機微調
    workers  = 0 ➜ **跳過 Phase‑I**，直接 Phase‑II（動態場景只重排）
    """
    cfg_dir = Path(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Phase‑I ------------------------------------------------------
    t0 = time.perf_counter()
    if workers:          # workers>0 ⇒ 正常 GA_fast
        ga_status = _GA_fast(task_csv, topo_csv, piid,
                             config_path=str(cfg_dir),
                             workers=workers,
                             k_paths=K_PATHS,
                             hop_limit=HOP_LIMIT)
        if not ga_status.startswith(f"{piid},sat"):
            return ga_status
    else:
        # 動態重排：期望上一輪 GA_fast 已在同 cfg_dir 下
        if not (cfg_dir/f"GA_fast-{piid}-GCL.csv").exists():
            return utils.rprint(piid, "unknown", 0)

    t_ga = time.perf_counter()

    # ---------- 讀取 GCL -----------------------------------------------------
    gcl_csv = cfg_dir/f"GA_fast-{piid}-GCL.csv"
    if not gcl_csv.exists():
        return utils.rprint(piid, "unknown", 0)

    gcl_df = pd.read_csv(gcl_csv)
    gcl_df["link"] = gcl_df["link"].apply(lambda s: str(tuple(eval(s))))
    sw2gcl: Dict[int, Dict[str, List[List[int]]]] = {}
    for _, row in gcl_df.iterrows():
        u, v = eval(row["link"])
        for sw in (u, v):
            d = sw2gcl.setdefault(sw, {})
            d.setdefault(row["link"], []).append(
                [row["start"] // utils.t_slot,
                 row["end"]   // utils.t_slot,
                 row["queue"]])

    # ---------- Phase‑II (thread per switch) --------------------------------
    threads = []
    for sw, lkdict in sw2gcl.items():
        lkdict = {lk: copy.deepcopy(win) for lk, win in lkdict.items()}
        th = threading.Thread(target=LocalOptimizer(sw, lkdict).optimise)
        th.start()
        threads.append((th, lkdict))
    for th, _ in threads:
        th.join()

    # ---------- 合併回全域 GCL ----------------------------------------------
    merged = []
    cycle  = gcl_df["cycle"][0]
    for _, lkdict in threads:
        for lk, wins in lkdict.items():
            for s, e, q in wins:
                merged.append([eval(lk), q,
                               s*utils.t_slot, e*utils.t_slot, cycle])

    # ---------- 其它 4 CSV 直接沿用 Phase‑I 產出 -----------------------------
    def csv_name(tag: str) -> str: return f"GA_fast-{piid}-{tag}.csv"
    utils.write_result("2PDGA", piid,
                       merged,
                       pd.read_csv(cfg_dir/csv_name("OFFSET")).values.tolist(),
                       pd.read_csv(cfg_dir/csv_name("ROUTE")).values.tolist(),
                       pd.read_csv(cfg_dir/csv_name("QUEUE")).values.tolist(),
                       pd.read_csv(cfg_dir/csv_name("DELAY")).values.tolist(),
                       str(cfg_dir)+"/")

    run = round(time.perf_counter() - t0, 3)
    extra = round(time.perf_counter() - t_ga, 3)
    return utils.rprint(piid, "sat", run, extra_time=extra)
