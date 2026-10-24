# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 3 的四個測試（投影片 p26 右欄 (a)–(d)）。linear_final.py 與 linear_live.py 共用。

    python3 tests.py linear_live          # 課堂：對 live 版跑（等同 ./present.sh live）；還沒寫的函數會標 skip
    python3 tests.py linear_final         # 等同 python3 linear_final.py
    python3 tests.py --replay             # 顯示彩排紀錄 runs/rehearsal/tests.json

每個測試只依名稱呼叫 impl 的函數（簽名見 linear_final.py 開頭）。寫到哪測到哪：
  (a)、(b) 需要 attn_quadratic 與 attn_linear      (c) 需要 step      (d) 需要 store 與 recall，delta_write 有就多印一欄
全部 float64、CPU、d_head = 128（測試自己給 float64 的輸入）。
  (a) 兩種括號順序對同一組隨機 Q, K, V（L = 256）的最大差，除以輸出的最大絕對值：要到 float64 精度（< 1e-12）
  (b) 計時：L ∈ {32, 128, 512, 2048, 8192}，兩個函數各一條線，每個 L 取多次重複的中位數；印出交叉落在哪兩個 L 之間。
      只看量級與交叉，不當 benchmark（執行緒數、BLAS 的分塊都會推動常數；理論交叉點 L = d_head = 128）
  (c) causal 遞推 step 逐 token 跑 L = 64 步，對 cumsum 一次算完（本檔自己算的閉式版本）的輸出：最大差要到 float64 精度（< 1e-10）
  (d) 筆記本的容量：存 m 筆鍵值進 S 再逐筆用鍵查回，相對誤差 ‖查回 − 值‖/‖值‖ 逐筆平均、幾次試驗再平均；m 從 d_head/4 到 4 d_head。
      三條：正交的鍵（m ≤ d_head 時誤差要是浮點零、m > d_head 起跳——這才是「容量 = d_head」）；
      隨機單位向量當鍵（從第一筆就有干擾、沒有膝點，≈ sqrt((m−1)/d_head)）；隨機鍵 + delta rule（β = 1，每筆寫一次）。
      另印「同一個鍵寫兩次」：加法版查回 v₁ + v₂、delta 版查回 v₂。
"""
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "rehearsal"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m"

D_HEAD = 128
L_EQUAL = 256                                       # (a) 的序列長
L_TIMING = [32, 128, 512, 2048, 8192]               # (b) 的五個 L（投影片 p26 的表）
REPEATS = {32: 200, 128: 100, 512: 30, 2048: 7, 8192: 3}
L_CAUSAL = 64                                       # (c) 的步數
M_LIST = [D_HEAD // 4, D_HEAD // 2, 3 * D_HEAD // 4, D_HEAD, 5 * D_HEAD // 4, 3 * D_HEAD // 2, 2 * D_HEAD, 3 * D_HEAD, 4 * D_HEAD]
TRIALS = 5                                          # (d) 每個 m 的試驗數
TOL_A, TOL_C = 1e-12, 1e-10                         # 「相等」的門檻（相對）
TOL_ZERO = 1e-10                                    # (d) 正交鍵在 m ≤ d_head 的「浮點零」
SEEDS = {"Q": 11, "K": 12, "V": 13, "cap": 606}


def _has(impl, name):
    return callable(getattr(impl, name, None))


def _randn(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _phi_ref(x):                                     # 測試自己的 phi：與 impl 的獨立，impl 要用同一個 elu + 1 才會相等
    return F.elu(x) + 1


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ── (a) 兩種括號順序相等 ─────────────────────────────────────────────
def test_a(impl):
    Q, K, V = (_randn(L_EQUAL, D_HEAD, seed=SEEDS[n]) for n in ("Q", "K", "V"))
    quad, lin = impl.attn_quadratic(Q, K, V), impl.attn_linear(Q, K, V)
    if quad.shape != (L_EQUAL, D_HEAD) or lin.shape != (L_EQUAL, D_HEAD):
        return "FAIL", f"shape {tuple(quad.shape)} / {tuple(lin.shape)}, expected {(L_EQUAL, D_HEAD)}", {}
    abs_diff, rel_diff = (quad - lin).abs().max().item(), _rel(quad, lin)
    nums = {"L": L_EQUAL, "d_head": D_HEAD, "max_abs_diff": abs_diff, "max_rel_diff": rel_diff, "max_abs_output": lin.abs().max().item()}
    msg = f"L = {L_EQUAL}, d_head = {D_HEAD}: max |quadratic - linear| = {abs_diff:.1e} on outputs up to {nums['max_abs_output']:.0f} (relative {rel_diff:.1e})"
    return ("ok" if rel_diff < TOL_A else "FAIL"), msg, nums


# ── (b) 計時與交叉點 ─────────────────────────────────────────────────
def _time(fn, args, reps):
    fn(*args)                                        # 暖機一次（分配、BLAS 初始化）
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*args); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def test_b(impl):
    rows = []
    for L in L_TIMING:
        Q, K, V = (_randn(L, D_HEAD, seed=SEEDS[n] + L) for n in ("Q", "K", "V"))
        tq = _time(impl.attn_quadratic, (Q, K, V), REPEATS[L])
        tl = _time(impl.attn_linear, (Q, K, V), REPEATS[L])
        rows.append({"L": L, "quadratic_s": tq, "linear_s": tl, "ratio": tq / tl, "repeats": REPEATS[L]})
    faster = [r["L"] for r in rows if r["linear_s"] < r["quadratic_s"]]
    slower = [r["L"] for r in rows if r["linear_s"] >= r["quadratic_s"]]
    if faster and slower and max(slower) < min(faster):
        cross = f"linear becomes faster between L = {max(slower)} and L = {min(faster)} (theory: L = d_head = {D_HEAD})"
    elif faster and not slower:
        cross = f"linear is faster at every L down to {min(faster)} (theory: the lines cross at L = d_head = {D_HEAD})"
    elif slower and not faster:
        cross = "linear is never faster up to L = 8192 — check the implementation or the thread count"
    else:
        cross = f"lines cross more than once: linear faster at {faster}, slower at {slower}"
    table = "\n" + _timing_table(rows)
    nums = {"threads": torch.get_num_threads(), "rows": rows, "crossover": cross}
    return "ok", f"{cross}; speed-up at L = 8192: {rows[-1]['ratio']:.0f}x{table}", nums


def _timing_table(rows):
    head = "      L         " + "".join(f"{r['L']:>10}" for r in rows)
    q = "      quadratic " + "".join(f"{r['quadratic_s']:>10.2e}" for r in rows) + "  s"
    l = "      linear    " + "".join(f"{r['linear_s']:>10.2e}" for r in rows) + "  s"
    ratio = "      ratio     " + "".join(f"{r['ratio']:>10.1f}" for r in rows)
    return "\n".join([head, q, l, ratio])


# ── (c) 遞推對一次算完 ───────────────────────────────────────────────
def test_c(impl):
    q, k, v = (_randn(L_CAUSAL, D_HEAD, seed=SEEDS[n] + 7) for n in ("Q", "K", "V"))
    S, z = torch.zeros(D_HEAD, D_HEAD, dtype=torch.float64), torch.zeros(D_HEAD, dtype=torch.float64)
    outs = []
    for t in range(L_CAUSAL):
        S, z, o = impl.step(S, z, k[t], v[t], q[t])
        outs.append(o)
    rec = torch.stack(outs)
    pk = _phi_ref(k)
    S_all = torch.cumsum(pk[:, :, None] * v[:, None, :], dim=0)            # S_t for every t, in one shot
    z_all = torch.cumsum(pk, dim=0)
    pq = _phi_ref(q)
    closed = torch.einsum("tij,ti->tj", S_all, pq) / torch.einsum("ti,ti->t", z_all, pq)[:, None]
    abs_diff, rel_diff = (rec - closed).abs().max().item(), _rel(rec, closed)
    nums = {"L": L_CAUSAL, "d_head": D_HEAD, "max_abs_diff": abs_diff, "max_rel_diff": rel_diff}
    msg = f"{L_CAUSAL} steps of the recurrence vs. cumsum in one shot: max difference {abs_diff:.1e} (relative {rel_diff:.1e})"
    return ("ok" if rel_diff < TOL_C else "FAIL"), msg, nums


# ── (d) 筆記本的容量 ─────────────────────────────────────────────────
def _unit_rows(m, d, g):
    k = torch.randn(m, d, generator=g, dtype=torch.float64)
    return k / k.norm(dim=1, keepdim=True)


def _orthogonal_rows(m, d, g):
    k = _unit_rows(m, d, g)
    n = min(m, d)
    q, _ = torch.linalg.qr(k[:n].T)                   # 前 min(m, d) 個鍵正交化；超過 d 的只能是隨機單位向量
    return torch.cat([q.T, k[n:]], dim=0)


def _recall_error(impl, keys, vals, S):
    errs = [(impl.recall(S, keys[s]) - vals[s]).norm() / vals[s].norm() for s in range(keys.shape[0])]
    return torch.stack(errs).mean().item()


def test_d(impl):
    has_delta = _has(impl, "delta_write")
    g = torch.Generator().manual_seed(SEEDS["cap"])
    rows = []
    for m in M_LIST:
        e_orth, e_rand, e_delta = [], [], []
        for _ in range(TRIALS):
            vals = torch.randn(m, D_HEAD, generator=g, dtype=torch.float64)
            okeys = _orthogonal_rows(m, D_HEAD, g)
            e_orth.append(_recall_error(impl, okeys, vals, impl.store(okeys, vals)))
            rkeys = _unit_rows(m, D_HEAD, g)
            e_rand.append(_recall_error(impl, rkeys, vals, impl.store(rkeys, vals)))
            if has_delta:
                S = torch.zeros(D_HEAD, D_HEAD, dtype=torch.float64)
                for s in range(m):
                    S = impl.delta_write(S, rkeys[s], vals[s])
                e_delta.append(_recall_error(impl, rkeys, vals, S))
        rows.append({"m": m, "m_over_d": m / D_HEAD,
                     "rel_err_orthogonal_keys": statistics.mean(e_orth),
                     "rel_err_random_keys": statistics.mean(e_rand),
                     "rel_err_random_keys_delta": statistics.mean(e_delta) if has_delta else None,
                     "sqrt_interference": ((m - 1) / D_HEAD) ** 0.5})
    at = {r["m"]: r for r in rows}
    ok = (all(at[m]["rel_err_orthogonal_keys"] < TOL_ZERO for m in M_LIST if m <= D_HEAD)
          and at[2 * D_HEAD]["rel_err_orthogonal_keys"] > 0.5
          and at[D_HEAD // 4]["rel_err_random_keys"] > 0.1)
    twice = None
    if has_delta:                                     # 同一個鍵寫兩次
        k = _unit_rows(1, D_HEAD, g)[0]
        v1, v2 = torch.randn(D_HEAD, generator=g, dtype=torch.float64), torch.randn(D_HEAD, generator=g, dtype=torch.float64)
        add = impl.recall(impl.store(torch.stack([k, k]), torch.stack([v1, v2])), k)
        dlt = impl.recall(impl.delta_write(impl.delta_write(torch.zeros(D_HEAD, D_HEAD, dtype=torch.float64), k, v1), k, v2), k)
        twice = {"additive_vs_v1_plus_v2": _rel(add, v1 + v2), "delta_vs_v2": _rel(dlt, v2)}
    msg = (f"orthogonal keys: {at[D_HEAD]['rel_err_orthogonal_keys']:.1e} at m = d_head, {at[D_HEAD + D_HEAD // 4]['rel_err_orthogonal_keys']:.2f} at m = 1.25 d_head; "
           f"random keys: {at[D_HEAD // 4]['rel_err_random_keys']:.2f} already at m = d_head/4, {at[D_HEAD]['rel_err_random_keys']:.2f} at m = d_head"
           + (f"; delta rule {at[D_HEAD]['rel_err_random_keys_delta']:.2f} at m = d_head" if has_delta else "; delta_write not written yet")
           + "\n" + _capacity_table(rows, has_delta))
    if twice:
        msg += (f"\n      same key written twice: additive recall = v1 + v2 (to {twice['additive_vs_v1_plus_v2']:.0e}), "
                f"delta recall = v2 (to {twice['delta_vs_v2']:.0e}) — overwritten, not stacked")
    nums = {"d_head": D_HEAD, "trials": TRIALS, "rows": rows, "same_key_twice": twice}
    return ("ok" if ok else "FAIL"), msg, nums


def _capacity_table(rows, has_delta):
    lines = ["      m      m/d   orthogonal keys   random keys   random + delta   sqrt((m-1)/d)"]
    for r in rows:
        dl = f"{r['rel_err_random_keys_delta']:>14.3f}" if r["rel_err_random_keys_delta"] is not None else f"{'—':>14}"
        lines.append(f"      {r['m']:>4}  {r['m_over_d']:>5.2f}   {r['rel_err_orthogonal_keys']:>15.2e}   {r['rel_err_random_keys']:>11.3f}   {dl}   {r['sqrt_interference']:>13.3f}")
    return "\n".join(lines)


TESTS = [("a", test_a, ("attn_quadratic", "attn_linear")),
         ("b", test_b, ("attn_quadratic", "attn_linear")),
         ("c", test_c, ("step",)),
         ("d", test_d, ("store", "recall"))]

COLOR = {"ok": GREEN, "FAIL": RED, "skip": GREY, "info": AMBER}


def run(impl, rehearse=False, label=""):
    t0 = time.time()
    print(f"{BOLD}{label or impl.__name__}{RESET}  torch {torch.__version__}, {platform.machine()}, CPU, float64, {torch.get_num_threads()} threads")
    results = {}
    for name, fn, needs in TESTS:
        missing = [n for n in needs if not _has(impl, n)]
        if missing:
            status, msg, nums = "skip", f"{missing[0]} not written yet", {}
        else:
            try:
                status, msg, nums = fn(impl)
            except Exception as exc:                          # 現場打錯字：印一行，不要整個炸掉
                status, msg, nums = "FAIL", f"{type(exc).__name__}: {exc}", {}
        print(f"  {COLOR[status]}{status:<4}{RESET} ({name}) {msg}")
        results[name] = {"status": status, "message": msg, **nums}
    elapsed = time.time() - t0
    print(f"{DIM}{elapsed:.2f} s{RESET}")
    if rehearse:
        RUNS.mkdir(parents=True, exist_ok=True)
        rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "impl": label or impl.__name__,
               "torch": torch.__version__, "python": platform.python_version(),
               "machine": platform.machine(), "macos": platform.mac_ver()[0], "device": "cpu", "dtype": "float64",
               "threads": torch.get_num_threads(), "elapsed_s": elapsed, "tests": results}
        path = RUNS / "tests.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{DIM}→ {path.relative_to(HERE)}{RESET}")
    return results


def replay():
    path = RUNS / "tests.json"
    if not path.exists():
        print(f"{RED}沒有 {path.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
        return
    rec = json.loads(path.read_text(encoding="utf-8"))
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {rec['date']}, torch {rec['torch']}, {rec['machine']}, {rec.get('threads', '?')} threads")
    for name, r in rec["tests"].items():
        print(f"  {COLOR.get(r['status'], '')}{r['status']:<4}{RESET} ({name}) {r['message']}")


if __name__ == "__main__":
    import importlib
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--replay" in sys.argv:
        replay()
    else:
        mod = importlib.import_module(args[0] if args else "linear_final")
        run(mod, rehearse="--rehearse" in sys.argv, label=f"{mod.__name__}.py")
