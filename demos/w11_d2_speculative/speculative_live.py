# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 (a)：課堂上現場寫的骨架（投影片 p47）。寫完 `./present.sh live`；寫錯了 `./present.sh live --final` 用 speculative_final.py。
上完課 `git checkout -- speculative_live.py` 還原成這個骨架。

    p, q   兩個 numpy 陣列，同樣長度、各自加起來是 1：target 與 draft 在這個位置上的分布
    rng    numpy 的亂數產生器（np.random.default_rng(seed)）
回傳 (token, kept)：token 是輸出的那一個（整數），kept 是 draft 抽的那一個有沒有被留下（True／False）。
"""
from __future__ import annotations

import numpy as np  # noqa: F401


def speculative_step(p, q, rng):
    # 1. 從 q 抽一個 token x                      rng.choice(len(q), p=q)
    # 2. 抽一個 0 到 1 的均勻亂數；小於 p[x] / q[x] 就留下：回傳 x, True
    # 3. 否則：p − q，負的設成 0，除以總和          np.maximum(0.0, p - q)
    #    從它抽一個：回傳那個 token, False
    raise NotImplementedError("課堂上寫這幾行；寫錯了用 --final")
