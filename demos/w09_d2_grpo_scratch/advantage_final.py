# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 2 (a) 的參考實作（投影片 p52 的五行）。`./present.sh live --ref` 與彩排用這一份。"""
from __future__ import annotations

import mlx.core as mx


def advantage(r: mx.array) -> mx.array:
    mean = r.mean()                                   # 1. 組內平均
    std = mx.sqrt(((r - mean) ** 2).mean())           # 2. 標準差：除以 G
    if std.item() == 0:                               # 5. 沉默的組：回傳全零，不要除
        return mx.zeros_like(r)
    return (r - mean) / std                           # 3.、4. 減平均、除以標準差
