# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 4 (a)：課堂上現場寫的骨架（投影片 p55 的五行）。寫完 `./present.sh intruders`；寫錯了 `--ref` 用 intruders_final.py。
上完課 `git checkout -- intruders_live.py` 還原成這個骨架。

W0、W1：d_out × d_in 的 numpy 陣列（float64）；W0 是預訓練的矩陣，W1 是微調後的（LoRA 的 s·B·A 已經加回去了）。
要回傳：best（長度 k：W1 的每個新方向對 W0 所有舊方向的 max |cos|）、counts（每個 ε 下 best < ε 的個數）。
"""
from __future__ import annotations

import numpy as np


def count_intruders(W0: np.ndarray, W1: np.ndarray, k: int, eps_grid: list[float]) -> dict:
    # 1. W0 的 SVD，只要左奇異向量 U0（np.linalg.svd(..., full_matrices=False)）
    # 2. W1 的 SVD，左奇異向量 U1 與奇異值 S1
    # 3. cos = |U1[:, :k]^T U0|   —— k 列、每列對所有舊方向
    # 4. best = 每列的最大值
    # 5. 每個 ε：數有幾列 best < ε
    raise NotImplementedError("課堂上寫這五行；寫錯了用 --ref")
    return {"best": best.tolist(), "counts": counts}  # noqa: F821  （寫好之後把這行留著）
