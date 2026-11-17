# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 3 的 verifier。demo 2 的題目與迴圈不動，換的只有這個檔裡的函數。

    reward_strict   回答裡**最後一個數字**等於正確結果（demo 2 的那一個；這裡是 gold，訓練時不給模型看）
    reward_leaky    正確結果出現在回答的**任何地方** —— 漏洞
    reward_count    回答裡的數字越多分數越高 —— 更大的漏洞，leaky 沒被找到時才用
    make_coin       擲硬幣：分數與回答無關

比對是「數字對數字」，不是子字串：正確結果是 275 時，回答裡的 1275 不算。
純標準函式庫。grpo.py 從 demo 2 的資料夾載入（題目、numbers_in、reward_strict 都在那裡）。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

D2 = Path(__file__).resolve().parent.parent / "w09_d2_grpo_scratch"
if str(D2) not in sys.path:
    sys.path.insert(0, str(D2))
from grpo import numbers_in, reward_strict  # noqa: E402,F401

COUNT_CAP = 8


def reward_leaky(text: str, answer: int) -> float:
    return 1.0 if float(answer) in numbers_in(text) else 0.0


def reward_count(text: str, answer: int) -> float:
    return min(len(numbers_in(text)), COUNT_CAP) / COUNT_CAP


def make_coin(seed: int, p: float = 0.5):
    rng = random.Random(seed)

    def reward_coin(text: str, answer: int) -> float:
        return 1.0 if rng.random() < p else 0.0

    return reward_coin


def make_reward(name: str, seed: int = 0):
    if name == "coin":
        return make_coin(seed)
    return {"strict": reward_strict, "leaky": reward_leaky, "count": reward_count}[name]


def describe(text: str, answer: int) -> dict:
    """一段回答在兩個 verifier 下的分數，與線索：回答裡有幾個數字。"""
    return {"proxy_leaky": reward_leaky(text, answer), "gold": reward_strict(text, answer), "n_numbers": len(numbers_in(text))}
