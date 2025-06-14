#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_opt.py – Local Optimiser for 2PDGA  (unchanged except微幅註解)
"""
from __future__ import annotations
from typing import List, Dict

class LocalOptimizer:
    """
    每個交換機一個實例；link_gcls: {linkStr : [[start,end,queue], …]}
    透過 Gap‑Fill / Window‑Merge 壓縮 idle gap 與減少 GCL 條目
    """
    def __init__(self, sw_id: int,
                 link_gcls: Dict[str, List[List[int]]],
                 max_entries: int | None = None):
        self.sw = sw_id
        self.gcls = link_gcls
        self.cap = max_entries or 2**31  # 若設備有限制可傳入 cap

    # ------------------------------------------------------------------
    def optimise(self, max_iter: int = 100) -> None:
        for _ in range(max_iter):
            changed = False
            for windows in self.gcls.values():
                windows.sort(key=lambda w: w[0])
                changed |= self._fill_gaps(windows)
                changed |= self._merge_adjacent(windows)
                if len(windows) > self.cap:
                    windows[:] = windows[:self.cap]
            if not changed:
                break

    # ------------------------------------------------------------------
    @staticmethod
    def _fill_gaps(wins: List[List[int]]) -> bool:
        changed = False
        for i in range(1, len(wins)):
            gap = wins[i][0] - wins[i-1][1]
            if 0 < gap <= 2:                       # ≤2 slots → left‑shift
                wins[i][0] -= gap
                wins[i][1] -= gap
                changed = True
        return changed

    @staticmethod
    def _merge_adjacent(wins: List[List[int]]) -> bool:
        if len(wins) < 2:
            return False
        out, merged = [wins[0]], False
        for w in wins[1:]:
            if w[0] <= out[-1][1]:                 # overlap / 確連續
                out[-1][1] = max(out[-1][1], w[1])
                merged = True
            else:
                out.append(w)
        wins[:] = out
        return merged
