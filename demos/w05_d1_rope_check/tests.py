# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 1 的三個測試（投影片 p18 右欄 (a)–(c)）。rope_final.py 與 rope_live.py 共用。

    python3 tests.py rope_live            # 課堂：對 live 版跑（等同 ./present.sh live）；還沒寫的函數會標 skip
    python3 tests.py rope_final           # 等同 python3 rope_final.py
    python3 tests.py --replay             # 顯示彩排紀錄 runs/rehearsal/tests.json

每個測試只依名稱呼叫 impl 的三個函數（簽名見 rope_final.py 開頭）。寫到哪測到哪：
  (b)、(c) 需要 rope_angles 與 rotate        (a) 另需 score
三個測試用同一組亂數 q, k, v（float64、d_head = 128、base = 10⁴），m − n = 7 固定、m 從 0 到 900 每 100 一個值。
  (a) RoPE：score 的最大差要到 float64 精度（< 1e-10；實際約 1e-13，角度到 900 rad 時 cos 的捨入）；同一組輸入順便用 float32 再算一次，印出來對照
  (b) 對照組：sinusoidal 的「加法」版 (q + p_m)ᵀ(k + p_n) 在同樣的平移下要**會變**（> 1e-3）；
      順便把三項拆開印：qᵀk 與 p_mᵀp_n 不動，動的全在交叉項 qᵀp_n + p_mᵀk
  (c) 反例：把 v 也旋轉。兩個 key 在位置 n 與 n+3、權重各半，輸出 Σ_j w_j R_{n_j} v 的第一個分量隨 n 變（> 1e-3）；
      範數不變——整個輸出向量被 R_n 轉了一個絕對位置的角度，下游沒有東西把它轉回來
"""
import json
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "rehearsal"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m"

D_HEAD, BASE = 128, 10_000.0
DELTA = 7                                   # m − n 固定
SHIFTS = list(range(0, 1000, 100))          # m 平移十個值
TOL_SAME = 1e-10                            # (a)：float64 下「不變」的門檻（角度到 ~900 rad，cos 的捨入約 1e-13）
TOL_MOVE = 1e-3                             # (b)、(c)：「會變」的門檻
SEEDS = {"q": 1, "k": 2, "v": 3}


def _has(impl, name):
    return callable(getattr(impl, name, None))


def _rand(n, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=dtype)


def _spread(xs):
    t = torch.stack([torch.as_tensor(x, dtype=torch.float64) for x in xs])
    return (t.max() - t.min()).item()


# ── (a) RoPE：(R_m q)ᵀ(R_n k) 只依賴 m − n ───────────────────────────
def test_a(impl):
    th = impl.rope_angles(D_HEAD, BASE)
    q, k = _rand(D_HEAD, SEEDS["q"]), _rand(D_HEAD, SEEDS["k"])
    s64 = [impl.score(q, k, n + DELTA, n, th) for n in SHIFTS]
    spread64 = _spread(s64)
    # 同一組輸入換 float32 再算一次，只當對照印出來（live 版若沒處理 dtype 也不算失敗）
    try:
        th32, q32, k32 = th.float(), q.float(), k.float()
        s32 = [impl.score(q32, k32, n + DELTA, n, th32) for n in SHIFTS]
        spread32 = _spread(s32)
        note32 = f"; the same in float32 moves by {spread32:.1e}"
    except Exception as exc:                      # noqa: BLE001
        spread32, note32 = None, f"; float32 run skipped ({type(exc).__name__})"
    nums = {"score": float(torch.as_tensor(s64[0])), "max_abs_diff_float64": spread64, "max_abs_diff_float32": spread32,
            "d_head": D_HEAD, "base": BASE, "delta": DELTA, "shifts": SHIFTS}
    msg = f"m - n = {DELTA}, m = 0..900: score = {nums['score']:+.6f}, max difference over ten m = {spread64:.1e} (float64){note32}"
    return ("ok" if spread64 < TOL_SAME else "FAIL"), msg, nums


# ── (b) 對照組：加法版 (q + p_m)ᵀ(k + p_n) 會變 ────────────────────────
def _stamp(m, th):
    p = torch.empty(2 * th.numel(), dtype=torch.float64)
    p[0::2] = torch.sin(m * th)                   # 第 2i 維 sin(mθ_i)、第 2i+1 維 cos(mθ_i)：與 RoPE 同一組頻率
    p[1::2] = torch.cos(m * th)
    return p


def test_b(impl):
    th = impl.rope_angles(D_HEAD, BASE)
    q, k = _rand(D_HEAD, SEEDS["q"]), _rand(D_HEAD, SEEDS["k"])
    total, qk, pp, cross = [], [], [], []
    for n in SHIFTS:
        pm, pn = _stamp(n + DELTA, th), _stamp(n, th)
        total.append((q + pm) @ (k + pn))
        qk.append(q @ k)
        pp.append(pm @ pn)                        # 只依賴 m − n：這一項其實是「相對」的
        cross.append(q @ pn + pm @ k)             # 交叉項：依賴絕對位置
    sp_total, sp_qk, sp_pp, sp_cross = _spread(total), _spread(qk), _spread(pp), _spread(cross)
    nums = {"max_abs_diff_total": sp_total, "max_abs_diff_qk": sp_qk, "max_abs_diff_pp": sp_pp, "max_abs_diff_cross": sp_cross,
            "totals": [float(t) for t in total]}
    msg = (f"additive stamp under the same shift: score moves by {sp_total:.2f} "
           f"(q'k and p_m'p_n are constant to {max(sp_qk, sp_pp):.0e}; all of it is in the cross terms q'p_n + p_m'k: {sp_cross:.2f})")
    return ("ok" if sp_total > TOL_MOVE and abs(sp_cross - sp_total) < 1e-9 else "FAIL"), msg, nums


# ── (c) 反例：v 也旋轉，輸出隨絕對位置變 ───────────────────────────────
def test_c(impl):
    th = impl.rope_angles(D_HEAD, BASE)
    v = _rand(D_HEAD, SEEDS["v"])
    gap, w = 3, (0.5, 0.5)                        # 兩個 key 在 n 與 n+3，權重固定各半：只看 v 的旋轉造成什麼
    outs = [w[0] * impl.rotate(v, n, th) + w[1] * impl.rotate(v, n + gap, th) for n in SHIFTS]
    first = [o[0].item() for o in outs]
    norms = [o.norm().item() for o in outs]
    plain = w[0] * v + w[1] * v                   # 不轉 v：輸出與 n 無關（對照）
    sp_first, sp_norm = _spread(first), _spread(norms)
    nums = {"max_abs_diff_component0": sp_first, "max_abs_diff_norm": sp_norm, "component0": first, "norm": norms[0],
            "unrotated_component0": plain[0].item(), "key_positions": f"n, n+{gap}", "weights": list(w)}
    msg = (f"rotate v too: output[0] moves by {sp_first:.3f} as m shifts (unrotated: {plain[0].item():+.3f}, fixed); "
           f"|output| = {norms[0]:.4f} constant to {sp_norm:.0e} — the whole output is rotated by the absolute position")
    return ("ok" if sp_first > TOL_MOVE and sp_norm < 1e-9 else "FAIL"), msg, nums


TESTS = [("a", test_a, ("rope_angles", "rotate", "score")),
         ("b", test_b, ("rope_angles", "rotate")),
         ("c", test_c, ("rope_angles", "rotate"))]

COLOR = {"ok": GREEN, "FAIL": RED, "skip": GREY, "info": AMBER}


def run(impl, rehearse=False, label=""):
    t0 = time.time()
    print(f"{BOLD}{label or impl.__name__}{RESET}  torch {torch.__version__}, {platform.machine()}, CPU, float64")
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
        mod = importlib.import_module(args[0] if args else "rope_final")
        run(mod, rehearse="--rehearse" in sys.argv, label=f"{mod.__name__}.py")
