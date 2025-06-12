#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_fast.py – accelerated Phase‑I GA for IEEE 802.1Qbv TAS scheduling
-------------------------------------------------------------------
  • 不再建立多層 multiprocessing（macOS spawn 兼容）
  • 自動建立  configs/<exp>/<ins>/GA_fast/piid_X/  並產生
    *-GCL.csv / *-OFFSET.csv / *-ROUTE.csv / *-QUEUE.csv / *-DELAY.csv
"""

from __future__ import annotations
import math, random, time, os, sys
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from deap import base, creator, tools, algorithms

import utils

# ──────────────────────────────────────────────────────────────────────────────
def GA(task_csv: str,
       topo_csv: str,
       piid: int,
       config_path: str = "./",
       workers: int = 1) -> str:
    """Single‑process GA; 返回 utils.rprint() 字串供 main_exp 收集。"""
    t0 = time.perf_counter()
    try:
        # 1. 載入網路/任務 -------------------------------------------------------
        net, net_attr, _, _, link_ids, _, _, _, link_rate = utils.read_network(
            topo_csv, utils.t_slot)
        task_attr, LCM = utils.read_task(task_csv, utils.t_slot,
                                         net, link_rate)

        flow_ids: List[str] = sorted(task_attr)
        n_flows, n_links   = len(flow_ids), len(link_ids)
        link_index: Dict[str,int] = {lk: i for i, lk in enumerate(link_ids)}
        t_proc_max = max(a['t_proc'] for a in net_attr.values())

        # 2. GA 參數 ------------------------------------------------------------
        P = 10 + int(2 * math.sqrt(n_flows))
        G = 50 + n_flows // 4
        elite_k = 2
        rnd = random.Random(piid)

        # 3. Path cache / helper -------------------------------------------------
        path_cache: Dict[str, List[List[int]]] = {
            f: utils.find_all_paths(net,
                                    task_attr[f]['src'],
                                    task_attr[f]['dst'])
            for f in flow_ids
        }
        max_k = max(len(v) for v in path_cache.values())
        big   = 10**9

        @lru_cache(maxsize=4096)
        def link_seq(fid: str, p_idx: int) -> List[str]:
            p = path_cache[fid][p_idx]
            return [str((p[h], p[h + 1])) for h in range(len(p) - 1)]

        def evaluate(ind):
            pidx = np.fromiter(ind[0::2], int, n_flows)
            offs = np.fromiter(ind[1::2], int, n_flows)
            timeline = np.zeros((n_links, LCM), bool)
            worst = 0
            for i, fid in enumerate(flow_ids):
                pi  = pidx[i] % len(path_cache[fid])
                off = offs[i] % task_attr[fid]['period']
                dur = t_proc_max + task_attr[fid]['t_trans']
                hop_s = off
                for lk in link_seq(fid, pi):
                    hop_e = hop_s + dur
                    if timeline[link_index[lk], hop_s : hop_e - t_proc_max].any():
                        return (big,)
                    timeline[link_index[lk],
                             hop_s : hop_e - t_proc_max] = True
                    hop_s = hop_e
                worst = max(worst, hop_s - off - t_proc_max)
            return (worst,)

        # 4. DEAP 工具箱 --------------------------------------------------------
        if "FitnessFast" not in creator.__dict__:
            creator.create("FitnessFast", base.Fitness, weights=(-1.0,))
            creator.create("IndividualFast", list, fitness=creator.FitnessFast)

        tb = base.Toolbox()
        tb.register("gene_path", rnd.randrange, max_k)
        tb.register("gene_off",  rnd.randrange, LCM)

        def _ind():
            g = []
            for _ in flow_ids:
                if rnd.random() < 0.6:
                    g += [0, 0]
                else:
                    g += [tb.gene_path(), tb.gene_off()]
            return creator.IndividualFast(g)

        tb.register("individual", _ind)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate",   tools.cxTwoPoint)
        tb.register("mutate", tools.mutShuffleIndexes, indpb=0.08)
        tb.register("select", tools.selTournament, tournsize=3)
        tb.register("evaluate", evaluate)
        tb.register("map", map)        # 使用單進程 map

        pop = tb.population(P)
        hist: List[float] = []

        for _ in range(G):
            offsp = algorithms.varAnd(pop, tb, 0.6, 0.3)
            for ind in offsp:                # repair
                ind[1::2] = [x % LCM for x in ind[1::2]]
            fits = list(tb.map(tb.evaluate, offsp))
            for ind, fv in zip(offsp, fits):
                ind.fitness.values = fv
            pop = tools.selBest(pop, elite_k) + \
                  tb.select(offsp, P - elite_k)
            best = tools.selBest(pop, 1)[0]
            hist.append(best.fitness.values[0])
            if len(hist) >= 3 and len(set(hist[-3:])) == 1 \
               and best.fitness.values[0] < utils.t_limit:
                break

        best = tools.selBest(pop, 1)[0]
        # 5. 輸出 CSV ----------------------------------------------------------
        _emit(best, flow_ids, path_cache, task_attr, t_proc_max,
              LCM, config_path, piid)

        return utils.rprint(piid, "sat", round(time.perf_counter() - t0, 3))

    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", 0)
    except Exception as e:
        print(f"[GA_fast] π={piid} error: {e}", file=sys.stderr)
        return utils.rprint(piid, "unknown", 0)

# ──────────────────────────────────────────────────────────────────────────────
def _emit(best_ind,
          flow_ids, path_cache, task_attr,
          t_proc_max, LCM,
          cfg_dir: str, piid: int) -> None:
    """Transform chromosome → 5 CSV files via utils.write_result()."""
    os.makedirs(cfg_dir, exist_ok=True)          # 確保資料夾存在

    GCL, OFFSET, ROUTE, QUEUE, DELAY = [], [], [], [], []
    it = iter(best_ind)
    for fid in flow_ids:
        p_idx = next(it) % len(path_cache[fid])
        off   = next(it) % task_attr[fid]['period']
        path  = path_cache[fid][p_idx]

        hop_s = off
        dur   = t_proc_max + task_attr[fid]['t_trans']
        for u, v in zip(path[:-1], path[1:]):
            hop_e = hop_s + dur
            GCL.append([[u, v], 0,
                        hop_s * utils.t_slot,
                        (hop_e - t_proc_max) * utils.t_slot,
                        LCM * utils.t_slot])
            ROUTE.append([fid, (u, v)])
            QUEUE.append([fid, 0, (u, v), 0])
            hop_s = hop_e

        OFFSET.append([fid, 0,
                       (task_attr[fid]['period'] - off) * utils.t_slot])
        DELAY.append([fid, 0,
                      (hop_s - off - t_proc_max) * utils.t_slot])

    # TARGET 需以 '/' 結尾
    utils.write_result("GA_fast", piid,
                       GCL, OFFSET, ROUTE, QUEUE, DELAY,
                       f"{os.path.join(cfg_dir, '')}")
