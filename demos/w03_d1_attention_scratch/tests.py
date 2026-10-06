# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 1 的五個測試（投影片 p21 右欄 (a)–(e)）。attention_final.py 與 attention_live.py 共用。

    python3 tests.py attention_live          # 課堂：對 live 版跑（等同 ./present.sh live）；還沒寫的函數會標 skip
    python3 tests.py attention_final         # 等同 python3 attention_final.py

每個測試只依名稱呼叫 impl 的三個函數（簽名見 attention_final.py 開頭）。寫到哪測到哪：
  (a) 需要 scaled_dot_product_attention        (d)、(e) 同
  (b) 另需 causal_mask                          (c) 另需 decode_step
測試 (d) 不需要你把 `/ math.sqrt(d_k)` 註解掉：它把 Q 先乘上 sqrt(d_k) 再丟進去，等於抵銷掉裡面的縮放。
測試 (e) 兩種結果都是「資訊」不是「失敗」：第一次寫一定回 NaN（先讓它壞），修好之後回有限值。
"""
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "rehearsal"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m"

TOL = 1e-5          # (a)、(c) 的容忍誤差，float32
ENTROPY_DK = [16, 128, 1024]   # (d)：與 p18 的圖（w03-data.json 的 variance）同一組 d_k
ENTROPY_KEYS, ENTROPY_TRIALS, ENTROPY_SEED = 64, 200, 3   # 同上：64 個 key、200 次、seed 3


def _has(impl, name):
    return callable(getattr(impl, name, None))


def _unpack(ret):
    """live 版可能只回 out；回 (out, A) 才能跑 (d)。"""
    if isinstance(ret, tuple):
        return ret[0], ret[1]
    return ret, None


def _F_sdpa(Q, K, V, **kw):
    """torch 內建的 SDPA；餵 (1, 1, L, E) 的四維張量（kernel 的檢查對四維最寬鬆），回 (L, E)。"""
    q, k, v = (t.unsqueeze(0).unsqueeze(0) for t in (Q, K, V))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)[0, 0]


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ── (a) 與 F.scaled_dot_product_attention 一致 ─────────────────
def test_a(impl):
    L, d_k, d_v = 8, 16, 8
    Q, K, V = _rand(L, d_k, seed=1), _rand(L, d_k, seed=2), _rand(L, d_v, seed=3)
    out, _ = _unpack(impl.scaled_dot_product_attention(Q, K, V))
    ref = _F_sdpa(Q, K, V)
    err = (out - ref).abs().max().item()
    nums = {"max_abs_err_no_mask": err}
    msg = f"max |ours - F.sdpa| = {err:.2e} (no mask)"
    if _has(impl, "causal_mask"):
        M = impl.causal_mask(L)
        out_c, _ = _unpack(impl.scaled_dot_product_attention(Q, K, V, M))
        ref_c = _F_sdpa(Q, K, V, is_causal=True)
        err_c = (out_c - ref_c).abs().max().item()
        nums["max_abs_err_causal"] = err_c
        msg += f", {err_c:.2e} (causal, vs is_causal=True)"
        err = max(err, err_c)
    return ("ok" if err < TOL else "FAIL"), msg, nums


# ── (b) causal 之下，輸出 t 對 t 之後的輸入梯度為零 ──────────────
def test_b(impl):
    L, d, d_k, d_v = 6, 8, 4, 4
    X = _rand(L, d, seed=4)
    W = {"q": _rand(d, d_k, seed=5) / math.sqrt(d), "k": _rand(d, d_k, seed=6) / math.sqrt(d), "v": _rand(d, d_v, seed=7) / math.sqrt(d)}
    M = impl.causal_mask(L)

    def f(X_):
        return _unpack(impl.scaled_dot_product_attention(X_ @ W["q"], X_ @ W["k"], X_ @ W["v"], M))[0]

    J = torch.autograd.functional.jacobian(f, X)        # (L, d_v, L, d)：J[t, :, s, :] = ∂out_t / ∂x_s
    above = torch.zeros(())
    below = torch.zeros(())
    for t in range(L):
        for s in range(L):
            blk = J[t, :, s, :].abs().max()
            if s > t:
                above = torch.maximum(above, blk)
            else:
                below = torch.maximum(below, blk)
    above, below = above.item(), below.item()
    nums = {"max_abs_jacobian_future": above, "max_abs_jacobian_past": below}
    msg = f"max |∂out_t/∂x_s| for s > t = {above:.1e}; for s ≤ t = {below:.2e} (autograd Jacobian, float32)"
    return ("ok" if above == 0.0 and below > 0 else "FAIL"), msg, nums


# ── (c) 逐步 decode（帶 cache）等於一次 forward ───────────────────
def test_c(impl):
    L, d, d_k, d_v = 8, 16, 8, 8
    X = _rand(L, d, seed=8)
    W = {"q": _rand(d, d_k, seed=9) / math.sqrt(d), "k": _rand(d, d_k, seed=10) / math.sqrt(d), "v": _rand(d, d_v, seed=11) / math.sqrt(d)}
    full, _ = _unpack(impl.scaled_dot_product_attention(X @ W["q"], X @ W["k"], X @ W["v"], impl.causal_mask(L)))
    cache = {"K": torch.empty(0, d_k), "V": torch.empty(0, d_v)}
    steps = []
    for t in range(L):
        steps.append(impl.decode_step(X[t:t + 1], cache, W))
    step = torch.cat(steps, dim=0)
    err = (step - full).abs().max().item()
    k_err = (cache["K"] - X @ W["k"]).abs().max().item()
    nums = {"max_abs_err_step_vs_full": err, "cache_rows": int(cache["K"].shape[0]), "max_abs_err_cached_K": k_err}
    msg = f"max |step-by-step - full forward| = {err:.2e}; cache holds {cache['K'].shape[0]} rows, max |cached K - X W_k| = {k_err:.1e}"
    return ("ok" if err < TOL and cache["K"].shape[0] == L else "FAIL"), msg, nums


# ── (d) 沒有 1/sqrt(d_k) 時，d_k 越大列熵越崩 ──────────────────────
def test_d(impl):
    g = torch.Generator().manual_seed(ENTROPY_SEED)
    rows = []
    for d_k in ENTROPY_DK:
        h_sc, h_un = 0.0, 0.0
        for _ in range(ENTROPY_TRIALS):
            q = torch.randn(1, d_k, generator=g)
            K = torch.randn(ENTROPY_KEYS, d_k, generator=g)
            V = torch.zeros(ENTROPY_KEYS, 1)
            _, A_sc = _unpack(impl.scaled_dot_product_attention(q, K, V))
            _, A_un = _unpack(impl.scaled_dot_product_attention(q * math.sqrt(d_k), K, V))   # 抵銷內部的 /sqrt(d_k)
            if A_sc is None:
                return "skip", "return (out, A) from scaled_dot_product_attention to run this test", {}
            h_sc += -(A_sc * torch.log(A_sc.clamp_min(1e-30))).sum().item()
            h_un += -(A_un * torch.log(A_un.clamp_min(1e-30))).sum().item()
        rows.append({"d_k": d_k, "entropy_scaled": h_sc / ENTROPY_TRIALS, "entropy_unscaled": h_un / ENTROPY_TRIALS})
    max_h = math.log(ENTROPY_KEYS)
    cells = "  ".join(f"d_k={r['d_k']}: {r['entropy_scaled']:.2f} vs {r['entropy_unscaled']:.2f}" for r in rows)
    msg = f"mean row entropy in nats, scaled vs unscaled (max ln {ENTROPY_KEYS} = {max_h:.2f}): {cells}"
    ok = all(r["entropy_scaled"] > 3.3 for r in rows) and rows[-1]["entropy_unscaled"] < 0.4 \
        and all(rows[i]["entropy_unscaled"] > rows[i + 1]["entropy_unscaled"] for i in range(len(rows) - 1))
    return ("ok" if ok else "FAIL"), msg, {"rows": rows, "max_entropy_nats": max_h,
                                           "setup": f"q, k ~ N(0, I); {ENTROPY_KEYS} keys per query; {ENTROPY_TRIALS} trials; seed {ENTROPY_SEED}"}


# ── (e) 全列被 mask → NaN；先讓它壞，再修 ─────────────────────────
def test_e(impl):
    L, d_k, d_v = 5, 8, 4
    Q, K, V = _rand(L, d_k, seed=12), _rand(L, d_k, seed=13), _rand(L, d_v, seed=14)
    M = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    M[2, :] = float("-inf")                                  # 第 2 列整列 -inf：模擬 padding 位置
    out, A = _unpack(impl.scaled_dot_product_attention(Q, K, V, M))
    has_nan = bool(torch.isnan(out).any())
    row = (A[2] if A is not None else out[2]).tolist()
    ref = _F_sdpa(Q, K, V, attn_mask=M)
    ref_nan = bool(torch.isnan(ref).any())
    nums = {"ours_has_nan": has_nan, "row2": row, "F_sdpa_has_nan": ref_nan}
    if has_nan:
        return "info", f"fully masked row → NaN  ← break confirmed; now fix it (F.sdpa on the same input: {'NaN' if ref_nan else 'finite'})", nums
    return "ok", f"fully masked row → finite, row 2 = {[round(x, 3) for x in row]} (fixed; F.sdpa on the same input: {'NaN' if ref_nan else 'finite'})", nums


TESTS = [("a", test_a, ("scaled_dot_product_attention",)),
         ("b", test_b, ("scaled_dot_product_attention", "causal_mask")),
         ("c", test_c, ("scaled_dot_product_attention", "causal_mask", "decode_step")),
         ("d", test_d, ("scaled_dot_product_attention",)),
         ("e", test_e, ("scaled_dot_product_attention",))]

COLOR = {"ok": GREEN, "FAIL": RED, "skip": GREY, "info": AMBER}


def run(impl, rehearse=False, label=""):
    t0 = time.time()
    print(f"{BOLD}{label or impl.__name__}{RESET}  torch {torch.__version__}, {platform.machine()}, CPU")
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
               "machine": platform.machine(), "macos": platform.mac_ver()[0], "device": "cpu",
               "elapsed_s": elapsed, "tests": results}
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
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {rec['date']}, torch {rec['torch']}, {rec['machine']}")
    for name, r in rec["tests"].items():
        print(f"  {COLOR.get(r['status'], '')}{r['status']:<4}{RESET} ({name}) {r['message']}")


if __name__ == "__main__":
    import importlib
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--replay" in sys.argv:
        replay()
    else:
        mod = importlib.import_module(args[0] if args else "attention_final")
        run(mod, rehearse="--rehearse" in sys.argv, label=f"{mod.__name__}.py")
