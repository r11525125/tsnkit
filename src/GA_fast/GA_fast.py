#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_fast.py – accelerated Phase‑I GA (bidirectional‑link fix)
"""
from __future__ import annotations
import math, random, time, os, sys
from functools import lru_cache
from typing import Dict, List

import numpy as np
from deap import base, creator, tools, algorithms
import utils


# ──────────────────────────────────────────────────────────────────────
def GA(task_csv: str,
       topo_csv: str,
       piid: int,
       cfg_dir: str = "./",
       workers: int = 1) -> str:
    """single‑process GA, return utils.rprint line"""
    t0 = time.perf_counter()
    try:
        # 1. load network / tasks ------------------------------------------------
        net, net_attr, _, _, link_ids, _, _, _, rate = utils.read_network(
            topo_csv, utils.t_slot)
        task_attr, LCM = utils.read_task(task_csv, utils.t_slot, net, rate)

        flow_ids: List[str] = sorted(task_attr)
        n_flows = len(flow_ids)

        # ---- link‑index (雙向映射，同一 idx) -----------------------------------
        link_index: Dict[str, int] = {}
        for idx, lk in enumerate(link_ids):
            link_index[lk] = idx
            u, v = eval(lk)
            link_index[str((v, u))] = idx      # ← 反向也指向同一 timeline

        n_links   = len(link_ids)
        t_proc_max = max(a['t_proc'] for a in net_attr.values())

        # 2. GA hyper‑params -----------------------------------------------------
        P = 10 + int(2 * math.sqrt(n_flows))
        G = 50 + n_flows // 4
        elite = 2
        rnd = random.Random(piid)

        # 3. all paths cache -----------------------------------------------------
        path_cache = {
            f: utils.find_all_paths(net,
                                    task_attr[f]['src'],
                                    task_attr[f]['dst'])
            for f in flow_ids
        }
        max_k = max(len(v) for v in path_cache.values())
        BIG   = 10**9

        @lru_cache(maxsize=4096)
        def links_of(fid: str, p_idx: int) -> List[str]:
            p = path_cache[fid][p_idx]
            return [str((p[h], p[h+1])) for h in range(len(p)-1)]

        # --------------------------- fitness -----------------------------------
        def eval_ind(ind):
            pidx = np.fromiter(ind[0::2], int, n_flows)
            offs = np.fromiter(ind[1::2], int, n_flows)
            timeline = np.zeros((n_links, LCM), bool)
            worst = 0
            for i, fid in enumerate(flow_ids):
                pi  = pidx[i] % len(path_cache[fid])
                off = offs[i] % task_attr[fid]['period']
                dur = t_proc_max + task_attr[fid]['t_trans']
                t   = off
                for lk in links_of(fid, pi):
                    te = t + dur
                    rng = slice(t, te - t_proc_max)
                    if timeline[link_index[lk], rng].any():
                        return (BIG,)          # infeasible
                    timeline[link_index[lk], rng] = True
                    t = te
                worst = max(worst, t - off - t_proc_max)
            return (worst,)

        # --------------------------- DEAP toolbox ------------------------------
        if "FitnessFast" not in creator.__dict__:
            creator.create("FitnessFast", base.Fitness, weights=(-1.0,))
            creator.create("IndividualFast", list,
                           fitness=creator.FitnessFast)

        tb = base.Toolbox()
        tb.register("g_path", rnd.randrange, max_k)
        tb.register("g_offs", rnd.randrange, LCM)

        def _ind():
            genes = []
            for _ in flow_ids:
                if rnd.random() < 0.6:
                    genes += [0, 0]            # heuristic seed
                else:
                    genes += [tb.g_path(), tb.g_offs()]
            return creator.IndividualFast(genes)

        tb.register("individual", _ind)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutShuffleIndexes, indpb=0.08)
        tb.register("select", tools.selTournament, tournsize=3)
        tb.register("evaluate", eval_ind)
        tb.register("map", map)               # 單進程

        pop = tb.population(P)
        hist = []

        for _ in range(G):
            offsp = algorithms.varAnd(pop, tb, 0.6, 0.3)
            for ind in offsp: ind[1::2] = [x % LCM for x in ind[1::2]]
            fits  = list(tb.map(tb.evaluate, offsp))
            for ind, fv in zip(offsp, fits):
                ind.fitness.values = fv
            pop = tools.selBest(pop, elite) + tb.select(offsp, P - elite)
            best = tools.selBest(pop, 1)[0]
            hist.append(best.fitness.values[0])
            if len(hist) >= 3 and len(set(hist[-3:])) == 1 \
               and best.fitness.values[0] < utils.t_limit:
                break

        _emit(best, flow_ids, path_cache, task_attr, t_proc_max,
              LCM, cfg_dir, piid)

        return utils.rprint(piid, "sat",
                            round(time.perf_counter() - t0, 3))

    except Exception as e:
        print(f"[GA_fast] π={piid} error: {e}", file=sys.stderr)
        return utils.rprint(piid, "unknown", 0)


# ----------------------------- CSV writer ------------------------------------
def _emit(best, fids, path_cache, task_attr,
          t_proc_max, LCM, cfg_dir, piid):
    os.makedirs(cfg_dir, exist_ok=True)

    GCL = []; OFFSET = []; ROUTE = []; QUEUE = []; DELAY = []
    it = iter(best)
    for fid in fids:
        p_idx = next(it) % len(path_cache[fid])
        off   = next(it) % task_attr[fid]['period']
        path  = path_cache[fid][p_idx]
        t = off
        dur = t_proc_max + task_attr[fid]['t_trans']
        for u, v in zip(path[:-1], path[1:]):
            te = t + dur
            GCL.append([(u, v), 0,
                        t*utils.t_slot,
                        (te - t_proc_max)*utils.t_slot,
                        LCM*utils.t_slot])
            ROUTE.append([fid, (u, v)])
            QUEUE.append([fid, 0, (u, v), 0])
            t = te
        OFFSET.append([fid, 0,
                       (task_attr[fid]['period'] - off)*utils.t_slot])
        DELAY.append([fid, 0, (t - off - t_proc_max)*utils.t_slot])

    utils.write_result("GA_fast", piid,
                       GCL, OFFSET, ROUTE, QUEUE, DELAY,
                       cfg_dir if cfg_dir.endswith("/") else cfg_dir+"/")
