#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_opt.py – node‑level greedy optimiser for Phase‑II of 2PDGA
----------------------------------------------------------------
• 每個交換機獨立實例化 LocalOptimizer，僅調整自己所屬鏈路 GCL
• 演算法：Gap‑Filling + Window‑Merge，O(#slots log #slots)
"""
from __future__ import annotations
from typing import List, Tuple, Dict

import heapq, random

class LocalOptimizer:
    """Operate on one switch; link_gcls 為 {linkStr: [[s,e,q], …]}"""

    def __init__(self,
                 sw_id: int,
                 link_gcls: Dict[str, List[List[int]]],
                 max_entries: int | None = None):
        self.sw   = sw_id
        self.gcls = link_gcls      # 深拷貝不必；caller 已複製
        self.cap  = max_entries or 2**31

    # ------------------------------------------------------------------
    def optimise(self, max_iter: int = 100) -> None:
        """就地修改 self.gcls"""
        for _ in range(max_iter):
            changed = False
            for link, windows in list(self.gcls.items()):
                windows.sort(key=lambda w: w[0])              # 依開始時間
                changed |= self._fill_gaps(windows)
                changed |= self._merge_adjacent(windows)
                # 過長則嘗試剪裁尾端 idle slot
                if len(windows) > self.cap:
                    windows[:] = windows[:self.cap]
            if not changed:
                break

    # ------------------------------------------------------------------
    @staticmethod
    def _fill_gaps(windows: List[List[int]]) -> bool:
        """在小於 PacketDur 的縫隙內前後推移，降低時延抖動"""
        changed = False
        for i in range(1, len(windows)):
            gap = windows[i][0] - windows[i-1][1]
            if 0 < gap <= 2:          # 2 個 slot 以內直接前推
                windows[i][0] -= gap
                windows[i][1] -= gap
                changed = True
        return changed

    @staticmethod
    def _merge_adjacent(windows: List[List[int]]) -> bool:
        if len(windows) < 2: return False
        out = [windows[0]]
        merged = False
        for w in windows[1:]:
            if w[0] <= out[-1][1]:         # overlap / 連續
                out[-1][1] = max(out[-1][1], w[1])
                merged = True
            else:
                out.append(w)
        windows[:] = out
        return merged
