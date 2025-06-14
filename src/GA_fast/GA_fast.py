#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_fast.py – Phase‑I GA  (2025‑06‑14 修正版)
· 新增 K‑shortest + hop‑bound
· 修正 link 字串對齊 & path 缺失 fallback
"""
from __future__ import annotations
import math, random, time, os, sys, heapq
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from deap import base, creator, tools, algorithms
import utils

# ───────── 參數 (可由 TwoPhaseDGA 覆寫) ───────────────────────────
_DEFAULT_K   = 3
_DEFAULT_HOP = 8

# ───────── K‑shortest (BFS) & Fallback Dijkstra ───────────────────
def k_shortest_paths(nb: Dict[int, List[int]],
                     src: int, dst: int,
                     k: int, hop_limit: int) -> List[List[int]]:
    paths, queue = [], [[src]]
    while queue and len(paths) < k:
        p = queue.pop(0)
        if len(p) - 1 > hop_limit:
            continue
        if p[-1] == dst:
            paths.append(p)
            continue
        for nbv in nb[p[-1]]:
            if nbv not in p:                 # simple path
                queue.append(p + [nbv])
    return paths

def dijkstra_shortest(nb: Dict[int, List[int]], s: int, d: int) -> List[int]:
    pq, dist = [(0, s, [s])], {s: 0}
    while pq:
        cost, v, path = heapq.heappop(pq)
        if v == d: return path
        for nxt in nb[v]:
            nd = cost + 1
            if nd < dist.get(nxt, 9e9):
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt, path + [nxt]))
    return [s, d]   # disconnected? – 理論上不會

# =============================================================================
def GA(task_csv: str, topo_csv: str, piid: int,
       config_path: str = "./",
       workers: int = 1,
       k_paths: int = _DEFAULT_K,
       hop_limit: int = _DEFAULT_HOP) -> str:

    t0 = time.perf_counter()
    try:
        # ---------- 1. 讀網路 / 任務 ----------------------------------------
        net, net_attr, _, _, link_ids, _, _, _, rate = \
            utils.read_network(topo_csv, utils.t_slot)
        task_attr, LCM = utils.read_task(task_csv, utils.t_slot,
                                         net, rate)

        flow_ids = sorted(task_attr)
        n_flows, n_links = len(flow_ids), len(link_ids)

        # ⇦⇦ fix: 以 **str(tuple)** 為索引，與 evaluate() 對齊
        link_index = {str(lk): i for i, lk in enumerate(link_ids)}
        t_proc_max = max(a['t_proc'] for a in net_attr.values())

        # ---------- 2. 建鄰接表 & path cache --------------------------------
        nb_map: Dict[int, List[int]] = {}
        for u, v in (eval(lk) for lk in link_ids):
            nb_map.setdefault(u, []).append(v)

        path_cache: Dict[str, List[List[int]]] = {}
        for fid in flow_ids:
            src, dst = task_attr[fid]['src'], task_attr[fid]['dst']
            cand = k_shortest_paths(nb_map, src, dst, k_paths, hop_limit)
            if not cand:                                        # Fallback
                cand = [dijkstra_shortest(nb_map, src, dst)]
            path_cache[fid] = cand

        max_k = max(len(v) for v in path_cache.values())
        big   = 1e9

        @lru_cache(maxsize=4096)
        def link_seq(fid: str, p_idx: int) -> List[str]:
            p = path_cache[fid][p_idx]
            return [str((p[h], p[h+1])) for h in range(len(p)-1)]

        # ---------- 3. GA 基本參數 -----------------------------------------
        P = 10 + int(2 * math.sqrt(n_flows))
        G = 50 + n_flows // 4
        elite_k = 2
        rnd = random.Random(piid)

        # ---------- 4. DEAP 定義 -------------------------------------------
        if "FitnessFast" not in creator.__dict__:
            creator.create("FitnessFast", base.Fitness, weights=(-1.0,))
            creator.create("IndividualFast", list, fitness=creator.FitnessFast)

        tb = base.Toolbox()
        tb.register("gene_path", rnd.randrange, max_k)
        tb.register("gene_off",  rnd.randrange, LCM)

        def _individual():
            g = []
            for _ in flow_ids:
                if rnd.random() < 0.6:
                    g += [0, 0]
                else:
                    g += [tb.gene_path(), tb.gene_off()]
            return creator.IndividualFast(g)
        tb.register("individual", _individual)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate",   tools.cxTwoPoint)
        tb.register("mutate", tools.mutShuffleIndexes, indpb=0.08)
        tb.register("select", tools.selTournament, tournsize=3)

        def evaluate(ind):
            pidx = np.fromiter(ind[0::2], int, n_flows)
            offs = np.fromiter(ind[1::2], int, n_flows)
            tl   = np.zeros((n_links, LCM), bool)
            worst = 0
            for i, fid in enumerate(flow_ids):
                pi  = pidx[i] % len(path_cache[fid])
                off = offs[i] % task_attr[fid]['period']
                dur = t_proc_max + task_attr[fid]['t_trans']
                hs  = off
                for lk in link_seq(fid, pi):
                    he = hs + dur
                    idx = link_index.get(lk)
                    if idx is None:                     # 不存在 – 視為衝突
                        return (big,)
                    if tl[idx, hs:he-t_proc_max].any():
                        return (big,)
                    tl[idx, hs:he-t_proc_max] = True
                    hs = he
                worst = max(worst, hs - off - t_proc_max)
            return (worst,)

        tb.register("evaluate", evaluate)
        tb.register("map", map)

        # ---------- 5. GA main loop ---------------------------------------
        pop = tb.population(P)
        history = []
        for _ in range(G):
            ch = algorithms.varAnd(pop, tb, 0.6, 0.3)
            for ind in ch:
                ind[1::2] = [x % LCM for x in ind[1::2]]
            fits = list(tb.map(tb.evaluate, ch))
            for ind, fv in zip(ch, fits):
                ind.fitness.values = fv
            pop = tools.selBest(pop, elite_k) + \
                  tb.select(ch, P-elite_k)
            best = tools.selBest(pop, 1)[0]
            history.append(best.fitness.values[0])
            if len(history) >= 3 and len({*history[-3:]}) == 1 and \
               best.fitness.values[0] < utils.t_limit:
                break

        # ---------- 6. emit 5 CSV ----------------------------------------
        _emit(best, flow_ids, path_cache, task_attr,
              t_proc_max, LCM, config_path, piid)

        return utils.rprint(piid, "sat", round(time.perf_counter()-t0, 3))

    except Exception as e:
        print(f"[GA_fast] π={piid} error: {e}", file=sys.stderr)
        return utils.rprint(piid, "unknown", 0)

# ───────── emit (不變) ──────────────────────────────────────────────
def _emit(best_ind, flow_ids, path_cache, task_attr,
          t_proc_max, LCM, cfg_dir: str, piid: int):
    os.makedirs(cfg_dir, exist_ok=True)
    GCL, OFFSET, ROUTE, QUEUE, DELAY = [], [], [], [], []
    it = iter(best_ind)
    for fid in flow_ids:
        p_idx = next(it) % len(path_cache[fid])
        off   = next(it) % task_attr[fid]['period']
        path  = path_cache[fid][p_idx]

        hs = off
        dur = t_proc_max + task_attr[fid]['t_trans']
        for u, v in zip(path[:-1], path[1:]):
            he = hs + dur
            GCL.append([[u, v], 0,
                        hs*utils.t_slot,
                        (he-t_proc_max)*utils.t_slot,
                        LCM*utils.t_slot])
            ROUTE.append([fid, (u, v)])
            QUEUE.append([fid, 0, (u, v), 0])
            hs = he

        OFFSET.append([fid, 0,
                       (task_attr[fid]['period']-off)*utils.t_slot])
        DELAY.append([fid, 0,
                      (hs-off-t_proc_max)*utils.t_slot])

    utils.write_result("GA_fast", piid,
                       GCL, OFFSET, ROUTE, QUEUE, DELAY,
                       os.path.join(cfg_dir, ""))
