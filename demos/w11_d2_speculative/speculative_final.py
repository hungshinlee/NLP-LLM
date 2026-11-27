# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 (a) 的參考實作（投影片 p47）。`./present.sh live --final`、`test` 與彩排都用這一份。"""
from __future__ import annotations

import numpy as np


def speculative_step(p, q, rng):
    x = rng.choice(len(q), p=q)                      # 1. the draft draws a token
    if rng.random() < p[x] / q[x]:                   # 2. keep it with probability min(1, p/q)
        return x, True
    residual = np.maximum(0.0, p - q)                # 3. rejected: what p still lacks …
    residual = residual / residual.sum()
    return rng.choice(len(p), p=residual), False     #    … redraw from that
