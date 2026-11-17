# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 1 (a) 的參考實作（投影片 p32）。`./present.sh live --ref`、彩排與 (b) 的訓練迴圈都用這一份。"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def dpo_loss(logp_w: mx.array, logp_l: mx.array, ref_w: mx.array, ref_l: mx.array, beta: float) -> mx.array:
    r_w = logp_w - ref_w                       # 1. y_w 的 log 比值
    r_l = logp_l - ref_l                       # 2. y_l 的 log 比值
    margin = beta * (r_w - r_l)                # 3. 相減，乘 beta
    losses = -nn.log_sigmoid(margin)           # 4. −log σ(·)，一對一個值
    return losses.mean()                       # 5. 對這一批取平均
