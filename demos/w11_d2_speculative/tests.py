# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 (a)：speculative_step 的五個測試（投影片 p47）。numpy 與標準函式庫，不載模型。

    1  抽 n 次，輸出的經驗分布與 p 的差在取樣誤差內
    2  draft 抽的那一個被留下的比例 = sum min(p, q)
    3  被拒之後從 p 重抽的版本（variants.wrong_redraw_step）過不了測試 1
    4  match-only 的版本（variants.match_only_step）也過測試 1，留下的比例是 sum p·q
    5  p = q 時每一個都被留下

「取樣誤差內」的定義：每個 token 的 |經驗比例 − 應有的比例| 不超過 z 個標準差，
標準差是 sqrt(x(1 − x)/n)，z 在 demo_config.toml 的 [live]（預設 4）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import BOLD, GREEN, GREY, RED, RESET  # noqa: E402
import spec_math as M  # noqa: E402
import variants as V  # noqa: E402

TITLES = {
    1: "{n:,} draws from speculative_step(p, q): is the output distributed as p?",
    2: "the share of the draft's draws that were kept",
    3: "a wrong version: after a rejection, redraw from p",
    4: "match-only: the target draws its own token; keep the draft's if they agree",
    5: "p = q",
}


def draw(step, p, q, n: int, seed: int):
    """呼叫 step n 次。回傳各 token 的經驗比例與被留下的比例。"""
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(p), dtype=np.int64)
    kept = 0
    for _ in range(n):
        x, k = step(p, q, rng)
        counts[int(x)] += 1
        kept += bool(k)
    return (counts / n).tolist(), kept / n


def sd(x: float, n: int) -> float:
    return math.sqrt(max(x * (1.0 - x), 0.0) / n)


def compare(emp, want, n: int, z: float) -> dict:
    """逐 token 比：差、容許值（z 個標準差）、有沒有全部在裡面。"""
    diff = [e - w for e, w in zip(emp, want)]
    tol = [z * sd(w, n) for w in want]
    return {"diff": diff, "allowed": tol, "largest_abs_diff": max(abs(d) for d in diff),
            "largest_allowed": max(tol), "within": all(abs(d) <= t for d, t in zip(diff, tol))}


def run(k: int, step, live: dict, state: dict) -> dict:
    """算第 k 個測試。state 讓測試 2 沿用測試 1 抽的那一批。"""
    p, q = np.array(live["p"], dtype=np.float64), np.array(live["q"], dtype=np.float64)
    n, z, seed = int(live["n_draws"]), float(live["z"]), int(live["seed"])
    if k in (1, 2):
        if "main" not in state:
            state["main"] = draw(step, p, q, n, seed)
        emp, kept = state["main"]
        if k == 1:
            return {"n": n, "p": p.tolist(), "q": q.tolist(), "empirical": emp, **compare(emp, p.tolist(), n, z),
                    "pass": compare(emp, p.tolist(), n, z)["within"]}
        b = M.beta_min(p.tolist(), q.tolist())
        return {"n": n, "kept": kept, "sum_min": b, "allowed": z * sd(b, n), "pass": abs(kept - b) <= z * sd(b, n)}
    if k == 3:
        emp, kept = draw(V.wrong_redraw_step, p, q, n, seed + 3)
        to_p = compare(emp, p.tolist(), n, z)
        pred = M.wrong_redraw(p.tolist(), q.tolist())
        to_pred = compare(emp, pred, n, z)
        return {"n": n, "p": p.tolist(), "empirical": emp, "predicted": pred, **to_p, "matches_prediction": to_pred["within"],
                "largest_abs_diff_to_prediction": to_pred["largest_abs_diff"], "pass": (not to_p["within"]) and to_pred["within"]}
    if k == 4:
        emp, kept = draw(V.match_only_step, p, q, n, seed + 4)
        to_p = compare(emp, p.tolist(), n, z)
        b, bm = M.beta_min(p.tolist(), q.tolist()), M.beta_match(p.tolist(), q.tolist())
        return {"n": n, "p": p.tolist(), "empirical": emp, **to_p, "kept": kept, "sum_pq": bm, "sum_min": b,
                "allowed_kept": z * sd(bm, n), "pass": to_p["within"] and abs(kept - bm) <= z * sd(bm, n)}
    if k == 5:
        emp, kept = draw(step, p, p, n, seed + 5)
        to_p = compare(emp, p.tolist(), n, z)
        return {"n": n, "p": p.tolist(), "empirical": emp, **to_p, "kept": kept, "pass": kept == 1.0 and to_p["within"]}
    raise ValueError(k)


# ── 畫面 ─────────────────────────────────────────────────────────
def _row(label: str, xs, fmt: str = "{:>10.4f}", color: str = "") -> str:
    return f"      {label:<12}" + "".join(color + fmt.format(x) + (RESET if color else "") for x in xs)


def _verdict(ok: bool, yes: str, no: str) -> str:
    return f"{GREEN}{yes}{RESET}" if ok else f"{RED}{no}{RESET}"


def _dist(r: dict, extra: tuple | None = None) -> None:
    print(_row("token", range(len(r["p"])), "{:>10d}"))
    print(_row("p", r["p"]))
    if extra:
        print(_row(extra[0], extra[1]))
    print(_row("drawn", r["empirical"], color=BOLD))
    print(_row("drawn − p", r["diff"], "{:>+10.4f}"))
    print(_row("allowed ±", r["allowed"], color=GREY))


def show(k: int, r: dict) -> None:
    if k == 1:
        _dist(r, ("q", r["q"]))
        print(f"      largest |drawn − p| = {BOLD}{r['largest_abs_diff']:.4f}{RESET}     "
              + _verdict(r["within"], "inside the sampling error: the output is distributed as p", "OUTSIDE the sampling error: this is not p"))
    elif k == 2:
        print(f"      kept {BOLD}{r['kept']:.4f}{RESET}     sum min(p, q) = {BOLD}{r['sum_min']:.4f}{RESET}     allowed ± {r['allowed']:.4f}     "
              + _verdict(r["pass"], "equal within the sampling error", "NOT equal"))
    elif k == 3:
        _dist(r, ("min+(1−β)p", r["predicted"]))
        print(f"      largest |drawn − p| = {BOLD}{r['largest_abs_diff']:.4f}{RESET}, allowed {r['largest_allowed']:.4f}     "
              + (f"{RED}test 1 FAILS{RESET} for this version" if not r["within"] else f"{GREEN}test 1 passes{RESET} for this version"))
        print(f"      {GREY}what it draws instead is min(p, q) + (1 − β)·p (second row): "
              f"{'matches' if r['matches_prediction'] else 'does not match'}, largest difference {r['largest_abs_diff_to_prediction']:.4f}{RESET}")
    elif k == 4:
        _dist(r)
        print(f"      largest |drawn − p| = {BOLD}{r['largest_abs_diff']:.4f}{RESET}     "
              + _verdict(r["within"], "inside the sampling error: also distributed as p", "OUTSIDE the sampling error"))
        print(f"      kept {BOLD}{r['kept']:.4f}{RESET}     sum p·q = {BOLD}{r['sum_pq']:.4f}{RESET}     {GREY}(speculative sampling keeps {r['sum_min']:.4f}){RESET}")
    elif k == 5:
        print(f"      kept {BOLD}{r['kept']:.4f}{RESET}     " + _verdict(r["kept"] == 1.0, "every draw is kept", "some draws were rejected")
              + f"     largest |drawn − p| = {r['largest_abs_diff']:.4f} " + _verdict(r["within"], "(inside)", "(OUTSIDE)"))


def verdicts(rows: dict) -> int:
    """最後一行：五個測試各自有沒有照預期。回傳沒過的個數。"""
    bad = [k for k in ("1", "2", "3", "4", "5") if not rows[k]["pass"]]
    print(f"\n  {BOLD}expected{RESET}: 1 output = p · 2 kept = Σ min(p, q) · 3 the wrong redraw fails test 1 · 4 match-only passes test 1, keeps Σ p·q · 5 p = q keeps everything")
    print("  " + (f"{GREEN}all five came out as expected{RESET}" if not bad else f"{RED}not as expected: test {', '.join(bad)}{RESET}"))
    return len(bad)
