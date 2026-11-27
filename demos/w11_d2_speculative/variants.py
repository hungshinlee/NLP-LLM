# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 (a)：測試 3 與測試 4 用的另外兩種寫法（投影片 p47）。課堂上不寫，直接用。"""
from __future__ import annotations


def wrong_redraw_step(p, q, rng):
    """測試 3：與 speculative_step 只差最後一行——被拒之後從 p 重抽，不是從 norm(max(0, p − q))。"""
    x = rng.choice(len(q), p=q)
    if rng.random() < p[x] / q[x]:
        return x, True
    return rng.choice(len(p), p=p), False            # the one line that differs


def match_only_step(p, q, rng):
    """測試 4：draft 抽一個、target 自己也抽一個；相同才算留下，輸出永遠是 target 抽的那一個。"""
    x = rng.choice(len(q), p=q)                      # the draft's token
    y = rng.choice(len(p), p=p)                      # the target's own token
    return y, bool(x == y)
