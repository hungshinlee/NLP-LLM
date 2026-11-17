# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 1 (a)：課堂上現場寫的骨架（投影片 p32）。寫完 `./present.sh live`；寫錯了 `./present.sh live --ref` 用 dpo_final.py。
上完課 `git checkout -- dpo_live.py` 還原成這個骨架。

四個輸入都是「整段回答的 log 機率」，mx.array，形狀 (n,)——一對偏好一個值：
    logp_w = log π_θ(y_w | x)      logp_l = log π_θ(y_l | x)
    ref_w  = log π_ref(y_w | x)    ref_l  = log π_ref(y_l | x)
beta 是純量。回傳這一批的平均 loss（純量）。
"""
from __future__ import annotations

import mlx.core as mx  # noqa: F401
import mlx.nn as nn  # noqa: F401


def dpo_loss(logp_w: mx.array, logp_l: mx.array, ref_w: mx.array, ref_l: mx.array, beta: float) -> mx.array:
    # 1. y_w 的 log 比值：log π_θ(y_w|x) − log π_ref(y_w|x)
    # 2. y_l 的 log 比值
    # 3. 兩個相減，乘 beta
    # 4. −log σ(·)：用 nn.log_sigmoid（先 sigmoid 再 log，括號裡很負時會變成 log 0）
    # 5. 對這一批取平均
    raise NotImplementedError("課堂上寫這幾行；寫錯了用 --ref")
