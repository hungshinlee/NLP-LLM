# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 1 的三個測試（投影片 p16 右欄 (a)–(c)）。block_final.py 與 block_live.py 共用。

    python3 tests.py block_live            # 課堂：對 live 版跑（等同 ./present.sh live）；還沒寫的類別會標 skip
    python3 tests.py block_final           # 等同 python3 block_final.py
    python3 tests.py --replay              # 顯示彩排紀錄（runs/rehearsal/tests.json）

每個測試只依名稱呼叫 impl 的 RMSNorm / SwiGLU / Block / GPT / count_params（簽名見 block_final.py 開頭）。寫到哪測到哪：
  (c) shape 測試逐段推進：有 RMSNorm 就測 RMSNorm、有 SwiGLU 再測 SwiGLU……四段全過才是 ok，中途是 info
  (a) 需要 GPT 與 count_params：count_params(model) 與 tests.py 自己從設定逐項算出的數要「差零個」，再印近似式與相對誤差
  (b) 不看 impl：跑 count_params.py 對 demo_config.toml 裡的兩個模型印三欄，第二欄（逐項精算）要等於第三欄（safetensors 實際）
tiny GPT 的尺寸在 demo_config.toml 的 [tiny]；V 由語料的字元集決定（語料不在就用 65 並標明）。
"""
from __future__ import annotations

import json
import platform
import sys
import time
import tomllib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "rehearsal"
CONFIG = HERE / "demo_config.toml"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m"
FALLBACK_V = 65   # tinyshakespeare 的字元集大小；語料不在本機時用這個（make_demo_w01.py 第一次跑會下載）


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def corpus_vocab() -> tuple[int, str]:
    cfg = load_cfg()
    p = (HERE / cfg["corpus"]["path"]).resolve()
    if p.exists():
        return len(set(p.read_text(encoding="utf-8"))), f"V = {{}} from {p.name}"
    return FALLBACK_V, "V = {} (corpus not found; fallback)"


def tiny_cfg() -> dict:
    cfg = load_cfg()["tiny"]
    V, _ = corpus_vocab()
    return {"V": V, "d": cfg["d"], "n_layer": cfg["n_layer"], "n_head": cfg["n_head"], "d_ff": cfg["d_ff"], "L": cfg["L"]}


def _has(impl, name):
    return callable(getattr(impl, name, None))


# ── 逐項記帳（黑板那一套；tests.py 自己算，不看模型）──────────────
def ledger(cfg: dict, n_kv: int | None = None, head_dim: int | None = None, tie: bool = True, qk_norm: bool = False) -> dict:
    V, d, n_layer, n_head, d_ff = cfg["V"], cfg["d"], cfg["n_layer"], cfg["n_head"], cfg["d_ff"]
    n_kv = n_kv or n_head
    head_dim = head_dim or d // n_head
    rows = [("embedding  V·d", V * d),
            ("per layer  W^Q, W^O: d·n_head·d_head each", 2 * d * n_head * head_dim),
            ("per layer  W^K, W^V: d·n_kv·d_head each", 2 * d * n_kv * head_dim),
            ("per layer  FFN: 3·d·d_ff", 3 * d * d_ff),
            ("per layer  two RMSNorm: 2d", 2 * d)]
    if qk_norm:
        rows.append(("per layer  q_norm, k_norm: 2·d_head", 2 * head_dim))
    per_layer = sum(v for k, v in rows if k.startswith("per layer"))
    rows.append((f"× {n_layer} layers", per_layer * n_layer))
    rows.append(("final RMSNorm  d", d))
    if not tie:
        rows.append(("unembedding  V·d (untied)", V * d))
    exact = V * d + per_layer * n_layer + d + (0 if tie else V * d)
    approx = 12 * n_layer * d * d + (V * d if tie else 2 * V * d)
    return {"rows": rows, "per_layer": per_layer, "exact": exact, "approx": approx,
            "rel_err": (approx - exact) / exact}


def ledger_text(cfg: dict, counted: int | None = None) -> str:
    lg = ledger(cfg)
    lines = [f"{BOLD}term by term{RESET}  (V = {cfg['V']}, d = {cfg['d']}, n_layer = {cfg['n_layer']}, n_head = {cfg['n_head']}, d_ff = {cfg['d_ff']})"]
    for k, v in lg["rows"]:
        lines.append(f"  {k:<44} {v:>12,}")
    lines.append(f"  {'exact':<44} {lg['exact']:>12,}")
    if counted is not None:
        lines.append(f"  {'count_params(model)':<44} {counted:>12,}   {'差零個' if counted == lg['exact'] else f'差 {counted - lg['exact']:+,}'}")
    lines.append(f"  {'approx  12·n_layer·d² + V·d':<44} {lg['approx']:>12,}   rel. error {lg['rel_err']:+.2%}")
    return "\n".join(lines)


# ── (c) shape 逐段推進 ───────────────────────────────────────────
def test_c(impl):
    cfg = tiny_cfg()
    V, d, n_layer, n_head, d_ff, L = cfg["V"], cfg["d"], cfg["n_layer"], cfg["n_head"], cfg["d_ff"], cfg["L"]
    torch.manual_seed(0)
    B, Lx = 2, 16
    x = torch.randn(B, Lx, d)
    parts, nums, failed = [], {}, False
    for name in ("RMSNorm", "SwiGLU", "Block", "GPT"):
        if not _has(impl, name):
            parts.append(f"{name} skip")
            continue
        try:
            if name == "RMSNorm":
                m = impl.RMSNorm(d)
                y = m(x)
                rms = y[0, 0].pow(2).mean().sqrt().item()
                good = tuple(y.shape) == (B, Lx, d) and abs(rms - 1.0) < 1e-3
                nums["rmsnorm_row_rms_after"] = rms
                parts.append(f"RMSNorm {'ok' if good else 'FAIL'} ({tuple(y.shape)}, row rms {rms:.4f})")
            elif name == "SwiGLU":
                m = impl.SwiGLU(d, d_ff)
                y = m(x)
                good = tuple(y.shape) == (B, Lx, d)
                parts.append(f"SwiGLU {'ok' if good else 'FAIL'} ({tuple(y.shape)})")
            elif name == "Block":
                m = impl.Block(d, n_head, d_ff)
                y = m(x)
                good = tuple(y.shape) == (B, Lx, d) and bool(torch.isfinite(y).all())
                parts.append(f"Block {'ok' if good else 'FAIL'} ({tuple(y.shape)})")
            else:
                m = impl.GPT(V, d, n_layer, n_head, d_ff, L)
                m.eval()
                idx = torch.randint(0, V, (B, Lx))
                with torch.no_grad():
                    logits = m(idx)
                finite = bool(torch.isfinite(logits).all())
                good = tuple(logits.shape) == (B, Lx, V) and finite
                nums.update({"gpt_logits_shape": list(logits.shape), "gpt_logits_finite": finite,
                             "gpt_logits_absmax": logits.abs().max().item()})
                parts.append(f"GPT {'ok' if good else 'FAIL'} ({tuple(logits.shape)}, finite {finite})")
            failed = failed or not good
        except Exception as exc:                                     # 現場打錯字：印一行，繼續
            parts.append(f"{name} FAIL {type(exc).__name__}: {exc}")
            failed = True
    msg = " · ".join(parts)
    if failed:
        return "FAIL", msg, nums
    if all(_has(impl, n) for n in ("RMSNorm", "SwiGLU", "Block", "GPT")):
        return "ok", msg, nums
    return "info", msg, nums


# ── (a) count_params 對黑板逐項記帳：差零個 ───────────────────────
def test_a(impl):
    cfg = tiny_cfg()
    _, vnote = corpus_vocab()
    m = impl.GPT(cfg["V"], cfg["d"], cfg["n_layer"], cfg["n_head"], cfg["d_ff"], cfg["L"])
    counted = int(impl.count_params(m))
    lg = ledger(cfg)
    nums = {"count_params": counted, "exact_term_by_term": lg["exact"], "per_layer": lg["per_layer"],
            "approx": lg["approx"], "approx_rel_err": lg["rel_err"], "cfg": cfg, "vocab_note": vnote.format(cfg["V"])}
    diff = counted - lg["exact"]
    msg = (f"count_params = {counted:,}; term by term = {lg['exact']:,} ({'差零個' if diff == 0 else f'差 {diff:+,}'}); "
           f"approx 12·n_layer·d² + V·d = {lg['approx']:,}, rel. error {lg['rel_err']:+.2%}  [{vnote.format(cfg['V'])}]")
    return ("ok" if diff == 0 else "FAIL"), msg, nums


# ── (b) count_params.py 對兩個真實模型：精算欄 = safetensors 欄 ─────
def test_b(impl):
    import count_params as cp
    cfg = load_cfg()
    nums, parts, status = {}, [], "ok"
    for key, m in cfg["models"].items():
        try:
            r = cp.report(key, quiet=True)
        except cp.NotLocal as exc:
            parts.append(f"{key}: skip ({exc})")
            status = "skip" if status == "ok" and not nums else status
            nums[key] = {"status": "skip", "reason": str(exc)}
            continue
        except Exception as exc:
            parts.append(f"{key}: FAIL {type(exc).__name__}: {exc}")
            status = "FAIL"
            nums[key] = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
            continue
        same = r["exact"] == r["actual"]
        status = status if same else "FAIL"
        parts.append(f"{key}: approx {r['approx']:,} / exact {r['exact']:,} / actual {r['actual']:,}"
                     f" ({'exact = actual' if same else f'exact − actual = {r['exact'] - r['actual']:+,}'}; approx rel. error {r['approx_rel_err']:+.1%})")
        nums[key] = {"status": "ok" if same else "FAIL", **r}
    if all(v.get("status") == "skip" for v in nums.values()):
        status = "skip"
    return status, "  ".join(parts), nums


TESTS = [("a", test_a, ("GPT", "count_params")),
         ("b", test_b, ()),
         ("c", test_c, ())]
COLOR = {"ok": GREEN, "FAIL": RED, "skip": GREY, "info": AMBER}


def run(impl, rehearse=False, label=""):
    t0 = time.time()
    print(f"{BOLD}{label or impl.__name__}{RESET}  torch {torch.__version__}, {platform.machine()}, CPU (tests), mps available: {torch.backends.mps.is_available()}")
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
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
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
        mod = importlib.import_module(args[0] if args else "block_final")
        run(mod, rehearse="--rehearse" in sys.argv, label=f"{mod.__name__}.py")
