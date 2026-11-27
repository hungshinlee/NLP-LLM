# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 的算術（投影片 p42–p44、p48、p52）。只用標準函式庫。

記號照投影片：
    p, q        target 與 draft 在同一個位置上的分布
    beta        一個草擬的 token 被留下的機率；alpha = E[beta]
    gamma       一輪草擬幾個 token
    c           draft 走一步的時間 ÷ target 走一步的時間
    Omega       一輪平均產出幾個 token
    S           加速比（對不做 speculative decoding 的 target）
"""
from __future__ import annotations

import math


# ── 一個位置：兩種規則各留下多少 ────────────────────────────────
def beta_min(p, q) -> float:
    """speculative sampling（Leviathan et al.）：sum_x min(p, q)。"""
    return sum(min(a, b) for a, b in zip(p, q))


def beta_match(p, q) -> float:
    """match-only（target 自己也抽一個，相同才留）：sum_x p·q。"""
    return sum(a * b for a, b in zip(p, q))


def residual(p, q) -> list[float]:
    """被拒之後重抽用的分布：norm(max(0, p − q))。p = q 時沒有定義（永遠不會被拒）。"""
    r = [max(0.0, a - b) for a, b in zip(p, q)]
    s = sum(r)
    return [x / s for x in r] if s > 0 else r


def wrong_redraw(p, q) -> list[float]:
    """被拒之後直接從 p 重抽的話，輸出的分布：min(p, q) + (1 − beta)·p。"""
    b = beta_min(p, q)
    return [min(a, bb) + (1.0 - b) * a for a, bb in zip(p, q)]


# ── 一輪：期望 token 數與加速比 ─────────────────────────────────
def omega(gamma: int, alpha: float) -> float:
    """1 + alpha + … + alpha^gamma（Leviathan et al. 式 (1)；各位置獨立、機率都是 alpha）。"""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(gamma: int, alpha: float, c: float, verify_ratio: float = 1.0) -> float:
    """S = Omega / (gamma·c + T_V/T_T)。verify_ratio = 1 就是 Theorem 3.8。"""
    return omega(gamma, alpha) / (gamma * c + verify_ratio)


def best_gamma(alpha: float, c: float, gammas, ratio=None) -> tuple[int, float]:
    """在給定的網格上最大的 S。ratio(k) 是驗 k 個位置對走一步的時間比；None 表示恆為 1。"""
    best = max(gammas, key=lambda g: speedup(g, alpha, c, 1.0 if ratio is None else ratio(g + 1)))
    return best, speedup(best, alpha, c, 1.0 if ratio is None else ratio(best + 1))


def kept_share(gamma: int, alpha: float) -> float:
    """草擬的 token 最後被留下的比例的期望：(Omega − 1) / gamma。第一個被拒之後的草稿沒有被驗就丟了，
    所以它隨 gamma 下降，不等於 alpha（gamma = 1 時相等）。"""
    return (omega(gamma, alpha) - 1.0) / gamma


def alpha_hat(accepted: int, tested: int) -> float | None:
    """alpha 的估計：被接受的 ÷ 真的被驗到的位置（接受的，加上每一輪第一個被拒的）。"""
    return accepted / tested if tested else None


# ── 驗一趟要多久：從量到的點內插 ────────────────────────────────
def interp_log(points: dict, x: float) -> float:
    """points = {整數 k: 值}；在 log2(k) 上分段線性內插，範圍外用最近的端點。"""
    ks = sorted(int(k) for k in points)
    val = {int(k): float(v) for k, v in points.items()}
    if x <= ks[0]:
        return val[ks[0]]
    if x >= ks[-1]:
        return val[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            t = (math.log2(x) - math.log2(a)) / (math.log2(b) - math.log2(a))
            return val[a] + t * (val[b] - val[a])
    return val[ks[-1]]


def ratio_from(points: dict):
    """回傳 ratio(k) = 值(k) ÷ 值(1)。points 可以是 demo 1 (b) 的 {B: ms/步}，或這個 demo 量的 {k: ms}。"""
    base = interp_log(points, 1)
    return lambda k: interp_log(points, k) / base


def median(xs):
    s = sorted(xs)
    n = len(s)
    return None if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def quantile(xs, f: float):
    s = sorted(xs)
    if not s:
        return None
    i = f * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] + (i - lo) * (s[hi] - s[lo])
