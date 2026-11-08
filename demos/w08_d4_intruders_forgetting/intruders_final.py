# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 4 (a) 的參考實作：intruder dimensions（Shuttleworth et al. 2025, Def. 3.1 / Alg. 1）。

課堂上現場寫的是 intruders_live.py 的 count_intruders；這一份是寫好的版本，`./present.sh intruders --ref` 用它。
兩份的函數簽名相同，demo.py 只呼叫 count_intruders。

W0、W1 都是 d_out × d_in 的 numpy 陣列（float64）：W0 是預訓練的矩陣，W1 是微調後的（LoRA 的話已經把 s·B·A 加回去）。
回傳的 best[j] 是 W1 第 j 個左奇異向量對 W0 **所有**左奇異向量的 max |cos|；它低於 ε 就是一個 intruder dimension。
"""
from __future__ import annotations

import numpy as np


def count_intruders(W0: np.ndarray, W1: np.ndarray, k: int, eps_grid: list[float]) -> dict:
    U0, S0, _ = np.linalg.svd(W0, full_matrices=False)      # 預訓練的方向（欄）與奇異值
    U1, S1, _ = np.linalg.svd(W1, full_matrices=False)      # 微調後的方向與奇異值
    cos = np.abs(U1[:, :k].T @ U0)                          # k × min(d_out, d_in)：單位向量的內積就是 cosine
    best = cos.max(axis=1)                                  # 每個新方向，跟最像的舊方向有多像
    counts = {str(e): int((best < e).sum()) for e in eps_grid}
    return {"best": best.tolist(), "counts": counts, "argmax": cos.argmax(axis=1).tolist(),
            "S0": S0.tolist(), "S1_topk": S1[:k].tolist()}
