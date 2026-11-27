# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 2 (a)：課堂上現場寫的骨架（投影片 p54 的五行）。寫完 `./present.sh live`；寫錯了 `./present.sh live --ref` 用 advantage_final.py。
上完課 `git checkout -- advantage_live.py` 還原成這個骨架。

r：一題 G 段回答的 reward，mx.array，形狀 (G,)。回傳同形狀的 advantage。
"""
from __future__ import annotations

import mlx.core as mx


def advantage(r: mx.array) -> mx.array:
    # 1. 這一組的平均 r̄
    # 2. 這一組的標準差 s_r —— 除以 G（不是 G − 1）
    # 3. 每段的分數減平均
    # 4. 除以標準差
    # 5. 沉默的組（s_r = 0）：回傳全零，不要除（這一行要放在第 4 行前面）
    raise NotImplementedError("課堂上寫這五行；寫錯了用 --ref")
