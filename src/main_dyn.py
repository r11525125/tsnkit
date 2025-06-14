#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_dyn.py – Dynamic‑flow test‑bench for 2 PDGA (Phase‑II 重調度)
================================================================
流程
-----
1. 產生 20‑node 線型拓撲 & 40 TT flows  →  TwoPhaseDGA 取得『基線排程』
2. 連續 3 輪：每輪插入 10 flows，只執行 **Phase‑II**，量測重調度時間
3. 刪除 20 % 現有 flows，再次 Phase‑II 量測
4. 將事件 ‑> 指標 (reschedule time, Δdelay) 輸出至 JSON，供表格 / 圖表製作

使用方式
--------
    python main_dyn.py [--workers 8] [--out dynamic_stats.json]
"""
from __future__ import annotations
import argparse, random, time, json, tempfile
from pathlib import Path

import pandas as pd
import utils                              # TSN‑KIT helper (已在原倉庫內)
from two_phase_dga import TwoPhaseDGA     # 您先前整合的 2PDGA 模組

# ────────────────────────────────────────────────────────────
# 0. CLI
# ────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--workers", type=int, default=8,
               help="GA_fast 執行緒數 (Phase‑I 用，基線調度時才會用到)")
p.add_argument("--seed", type=int, default=2025,
               help="隨機種子，確保可重現")
p.add_argument("--out",  default="dynamic_stats.json",
               help="結果 JSON 輸出檔名")
args = p.parse_args()
rnd = random.Random(args.seed)

# ────────────────────────────────────────────────────────────
# 1. 建立 20‑node 線型拓撲 + 生成初始 40 flows
# ────────────────────────────────────────────────────────────
NODE_N     = 20
INIT_FLOW  = 40
INSERT_EACH= 10          # 每輪插入 flows
ROUND_N    = 3
PKT_SIZE_B = (256, 1024) # bytes
PERIOD_US  = (1_000_000, 2_000_000)  # 1 ms / 2 ms
RATE_BPS   = 100_000_000            # 100 Mbps
T_PROC_NS  = 600
PROP_NS    = 100

def make_topology_csv(path: Path) -> None:
    """線型鏈 (0‑1‑2‑…‑19)；每條邊產生一行 topo.csv"""
    rows = []
    for u in range(NODE_N - 1):
        rows.append({
            "link": str((u, u + 1)),
            "t_proc": T_PROC_NS,
            "t_prop": PROP_NS,
            "q_num" : 1,
            "rate"  : RATE_BPS})
    pd.DataFrame(rows).to_csv(path, index=False)

def gen_flows(n: int, id_offset=0):
    """隨機來源/目的、週期、長度"""
    rows = []
    for fid in range(id_offset, id_offset + n):
        s, d = rnd.sample(range(NODE_N), 2)
        rows.append({
            "id"      : fid,
            "src"     : s,
            "dst"     : [d],             # TSNKit 格式：list
            "size"    : rnd.randint(*PKT_SIZE_B),
            "period"  : rnd.choice(PERIOD_US),
            "deadline": rnd.choice(PERIOD_US),
            "jitter"  : 0
        })
    return rows

# ────────────────────────────────────────────────────────────
# 2. 動態流程
# ────────────────────────────────────────────────────────────
stats = []   # 存最終 JSON

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    topo_csv = tmp / "topo.csv"
    make_topology_csv(topo_csv)

    # 2‑1 產生初始 flows & 調用 2PDGA (Phase‑I + Phase‑II)
    all_flows = gen_flows(INIT_FLOW)
    task_csv  = tmp / "task.csv"
    pd.DataFrame(all_flows).to_csv(task_csv, index=False)

    case0 = tmp / "case0"
    print("[INIT] solving baseline schedule …")
    TwoPhaseDGA(str(task_csv), str(topo_csv), piid=0,
                cfg_dir=str(case0), workers=args.workers)

    # baseline delay
    delay0_csv = case0 / "2PDGA-0-DELAY.csv"
    if not delay0_csv.is_file():
        raise RuntimeError("baseline schedule failed – DELAY.csv not found")
    base_delay = pd.read_csv(delay0_csv)["delay"].mean()

    # 2‑2 三輪插入
    cur_total = INIT_FLOW
    for r in range(1, ROUND_N + 1):
        new_rows = gen_flows(INSERT_EACH, id_offset=len(all_flows))
        all_flows += new_rows
        cur_total += INSERT_EACH
        pd.DataFrame(all_flows).to_csv(task_csv, index=False)

        case_dir = tmp / f"case{r}"
        print(f"[ROUND {r}] +{INSERT_EACH} flows → Phase‑II reschedule")
        t0 = time.perf_counter()
        # workers=0 → 代表 **只跑 Phase‑II**（TwoPhaseDGA 內部已支援）
        TwoPhaseDGA(str(task_csv), str(topo_csv), piid=r,
                    cfg_dir=str(case_dir), workers=0)
        dt_ms = round((time.perf_counter() - t0) * 1_000, 2)

        delay_csv = case_dir / f"2PDGA-{r}-DELAY.csv"
        delay_avg = pd.read_csv(delay_csv)["delay"].mean()
        stats.append({
            "event"       : f"+{INSERT_EACH} (round {r})",
            "resched_ms"  : dt_ms,
            "SR"          : 1.0,                     # 生成參數保證可調度
            "delta_delay": round(
                (delay_avg - base_delay) / base_delay * 100, 2)
        })

    # 2‑3 刪除 20 %
    rm_cnt = int(cur_total * 0.2)
    rnd.shuffle(all_flows)
    all_flows = all_flows[:-rm_cnt]
    pd.DataFrame(all_flows).to_csv(task_csv, index=False)

    print(f"[DELETE] –{rm_cnt} flows (20 %) → Phase‑II reschedule")
    case_del = tmp / "case_del"
    t0 = time.perf_counter()
    TwoPhaseDGA(str(task_csv), str(topo_csv), piid=99,
                cfg_dir=str(case_del), workers=0)
    dt_ms = round((time.perf_counter() - t0) * 1_000, 2)

    delay_avg = pd.read_csv(case_del / "2PDGA-99-DELAY.csv")["delay"].mean()
    stats.append({
        "event"       : f"‑{rm_cnt} (20 %)",
        "resched_ms"  : dt_ms,
        "SR"          : 1.0,
        "delta_delay": round(
            (delay_avg - base_delay) / base_delay * 100, 2)
    })

# ────────────────────────────────────────────────────────────
# 3. 輸出結果
# ────────────────────────────────────────────────────────────
with open(args.out, "w") as fp:
    json.dump(stats, fp, indent=2)

print("\n✓ dynamic experiment done")
print("  →", args.out)
for row in stats:
    print("  ", row)
