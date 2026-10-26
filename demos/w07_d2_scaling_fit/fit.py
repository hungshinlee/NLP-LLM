#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 2 的擬合：L(N) = E + A·N^-α 沿射線 D = 20N（三個參數），以及 iso-FLOPs envelope（Chinchilla Approach 1 的圖）。純標準函式庫。

  fit_power(Ns, Ls, lo, hi)      對 α 一維搜尋（log 等分格點 → 黃金分割），每個 α 下 E、A 是線性最小平方（2 × 2 正規方程）。
                                 三個點、三個參數：殘差會到浮點零（若點是遞減且凸的）；五個點：最小平方。
                                 **這不是 Hoffmann 的擬合法**（他們對 log 空間、Huber loss、多組起點做 L-BFGS；§3.3）——這裡只是最單純的版本，
                                 目的是讓「點少、共線」怎麼壞掉看得見，不是重現 Chinchilla。
  predict(fit, N)                E + A·N^-α
  envelope(runs, n_points)       每個 run 的曲線 (tokens_seen, held loss) 換成 (C = 6·N·tokens, loss)；C 的 log 等分格點上，
                                 每個 run 在那個 C 的 loss（曲線上線性內插，只在它自己走過的範圍內），取最小 → 那個 C 的最佳 N。
                                 五個 N 是離散的，所以「最佳 N 對 C」是階梯，不是直線；斜率只能估量級。
"""
from __future__ import annotations

import math


def _ls_EA(Ns: list[float], Ls: list[float], alpha: float) -> tuple[float, float, float]:
    """固定 α：L ≈ E + A·x，x = N^-α。回 (E, A, rss)。"""
    xs = [n ** (-alpha) for n in Ns]
    n = len(xs)
    sx, sy, sxx, sxy = sum(xs), sum(Ls), sum(x * x for x in xs), sum(x * y for x, y in zip(xs, Ls))
    det = n * sxx - sx * sx
    if abs(det) < 1e-300:
        return float("nan"), float("nan"), float("inf")
    A = (n * sxy - sx * sy) / det
    E = (sy - A * sx) / n
    rss = sum((E + A * x - y) ** 2 for x, y in zip(xs, Ls))
    return E, A, rss


def fit_power(Ns: list[float], Ls: list[float], lo: float = 0.01, hi: float = 3.0, grid: int = 400, iters: int = 200) -> dict:
    if len(Ns) < 3:
        raise ValueError("至少三個點（三個參數）")
    # 粗格點（log 等分）
    best = None
    for i in range(grid + 1):
        a = lo * (hi / lo) ** (i / grid)
        rss = _ls_EA(Ns, Ls, a)[2]
        if best is None or rss < best[1]:
            best = (a, rss, i)
    i = best[2]
    a_lo = lo * (hi / lo) ** (max(0, i - 1) / grid)
    a_hi = lo * (hi / lo) ** (min(grid, i + 1) / grid)
    # 黃金分割細化（在 log α 上）
    gr = (math.sqrt(5) - 1) / 2
    la, lb = math.log(a_lo), math.log(a_hi)
    c, d = lb - gr * (lb - la), la + gr * (lb - la)
    fc, fd = _ls_EA(Ns, Ls, math.exp(c))[2], _ls_EA(Ns, Ls, math.exp(d))[2]
    for _ in range(iters):
        if fc < fd:
            lb, d, fd = d, c, fc
            c = lb - gr * (lb - la)
            fc = _ls_EA(Ns, Ls, math.exp(c))[2]
        else:
            la, c, fc = c, d, fd
            d = la + gr * (lb - la)
            fd = _ls_EA(Ns, Ls, math.exp(d))[2]
    alpha = math.exp((la + lb) / 2)
    E, A, rss = _ls_EA(Ns, Ls, alpha)
    at_edge = alpha < lo * 1.02 or alpha > hi / 1.02
    return {"E": E, "A": A, "alpha": alpha, "rss": rss, "n_points": len(Ns), "Ns": list(Ns), "Ls": list(Ls),
            "alpha_at_edge": at_edge, "search": {"lo": lo, "hi": hi}}


def predict(fit: dict, N: float) -> float:
    return fit["E"] + fit["A"] * N ** (-fit["alpha"])


def evaluate_fit(fit: dict, test: list[tuple[str, float, float]]) -> list[dict]:
    """test = [(key, N, measured_loss)] → 每個的預測、誤差（nats 與 %）。"""
    out = []
    for key, N, meas in test:
        pred = predict(fit, N)
        out.append({"key": key, "N": N, "measured": meas, "predicted": pred, "error": pred - meas, "error_pct": 100.0 * (pred - meas) / meas,
                    "measured_bpb": meas / math.log(2), "predicted_bpb": pred / math.log(2), "error_bpb": (pred - meas) / math.log(2)})
    return out


# ── iso-FLOPs envelope ───────────────────────────────────────────
def _interp(xs: list[float], ys: list[float], x: float) -> float | None:
    """在 (xs 遞增) 上線性內插；x 在範圍外回 None。"""
    if x < xs[0] or x > xs[-1]:
        return None
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + w * (ys[hi] - ys[lo])


def envelope(runs: list[dict], n_points: int = 13, loss_key: str = "held_subset_loss") -> dict:
    """runs：每個有 N 與 curve（tokens_seen、loss_key）。回 C 格點上每個 run 的 loss 與最佳 run。"""
    series = []
    for r in runs:
        pts = [(6.0 * r["N"] * c["tokens_seen"], c[loss_key]) for c in r["curve"] if c["tokens_seen"] > 0]
        pts.sort()
        series.append({"key": r["key"], "N": r["N"], "C": [p[0] for p in pts], "L": [p[1] for p in pts]})
    c_lo = min(s["C"][0] for s in series)
    c_hi = max(s["C"][-1] for s in series)
    grid = [c_lo * (c_hi / c_lo) ** (i / (n_points - 1)) for i in range(n_points)]
    rows = []
    for C in grid:
        per = {s["key"]: _interp(s["C"], s["L"], C) for s in series}
        avail = {k: v for k, v in per.items() if v is not None}
        if not avail:
            continue
        best = min(avail, key=avail.get)
        rows.append({"C": C, "per_run": per, "best": best, "best_N": next(s["N"] for s in series if s["key"] == best), "best_loss": avail[best],
                     "n_runs_covering": len(avail)})
    # 最佳 N 對 C 的斜率（log-log 最小平方；只有少數離散的 N，量級用）
    xs = [math.log10(r["C"]) for r in rows if r["n_runs_covering"] >= 2]
    ys = [math.log10(r["best_N"]) for r in rows if r["n_runs_covering"] >= 2]
    slope = None
    if len(xs) >= 3 and max(xs) > min(xs):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else None
    return {"rows": rows, "slope_logN_logC": slope, "note": "best N is one of five discrete values, so the slope is an order-of-magnitude estimate only; Hoffmann's Approach 1 fits a line to many iso-FLOPs minima (Table 2: a = 0.50)"}
