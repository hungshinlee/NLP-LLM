#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2：speculative decoding——十行、五個測試，再掃 gamma（投影片 p45–p47、p49）。

    ./present.sh live            課堂 (a)：speculative_live.py 寫好之後跑——五個測試，Enter 一個（不載模型）
    ./present.sh live --final    寫錯了：用參考實作 speculative_final.py
    ./present.sh replay-a        現場 (a) 失敗：印彩排的 tests.json（不載模型）
    ./present.sh replay          課堂 (b)：彩排紀錄，Enter 四段（不載模型）
    ./present.sh show            課堂（可選）：一個 prompt，先不用 draft、再用，草稿被留下的 token 上色
    ./present.sh test            不載權重：五個測試、算術對投影片的數；有 mlx 時另用亂數小模型驗量法
    ./present.sh inspect         課前：版本、四個模型、tokenizer 是否相同、mlx-lm 的接受規則那幾行
    ./present.sh probe           彩排前：幾個 prompt 量一次，估全套要多久
    ./present.sh rehearse        彩排全套：(a) → 每個 target 的基準、兩個分布的重疊、驗一趟的時間 → 掃 gamma
                                 （--part a|base|sweep、--target bf16|q4、--force、--note "接電源"）
    ./present.sh tiny [子命令]   亂數權重的小模型走真的流程（不下載；寫到 runs/smoke/）

(b) 的量法：
  基準      target 自己生成（mlx-lm 的 generate_step），同一批 prompt、同一個 sampler。
  有 draft  mlx-lm 的 speculative_generate_step：draft 寫 gamma 個 token，target 一趟驗 gamma + 1 個位置。
            它的接受規則是「target 在每個位置自己取樣，與 draft 的 token 相同才留下」（match-only）。
  時間      兩邊都從第一個 token（第一輪）結束開始計，prefill 不算；遇到 EOS 的那一輪與草擬數不足 gamma 的最後一輪不算。
            tokens/s 是所有 prompt 的 token 數加總 ÷ 秒數加總；加速比是兩個 tokens/s 的比。
  alpha     被接受的草稿 ÷ 真的被驗到的位置（接受的，加上每一輪第一個被拒的）。
            「草稿被留下的比例」（接受的 ÷ 草擬的）另外記：第一個被拒之後的草稿沒被驗就丟了，所以它比 alpha 小。
  c         draft 自己生成時每個 token 的時間 ÷ target 的，同一批 prompt。
  重疊      沿著基準生成的文字，把 target 與 draft 的 logits 各算一次：每個位置的 sum min(p, q) 與 sum p·q。
  驗一趟    context 固定，一次餵 k 個 token 給 target，k = 1 … gamma + 1，各量幾次取中位數。
  預測      Theorem 3.8：Omega(gamma, alpha) / (gamma·c + 1)；
            把分母的 1 換成「驗 gamma + 1 個位置 ÷ 走一步」：用 demo 1 (b) 的曲線，或用這裡直接量的。

相依：(a) 只要 numpy；(b) 要 mlx、mlx-lm（git commit 釘在 ../versions.lock）。replay 與 replay-a 只用標準函式庫。Python 3.11+。
"""
from __future__ import annotations

import argparse
import importlib
import inspect as pyinspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info,  # noqa: E402
                    locked_revision, note, now, peak_memory_gb, reset_peak_memory, versions)
import spec_math as M  # noqa: E402

CONFIG = HERE / "demo_config.toml"
PROMPTS = HERE / "prompts.toml"
RUNS = HERE / "runs"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]   # 離線時 huggingface_hub 會核對 snapshot 是否完整
GB = 1e9
KINDS = ["code", "en", "zh", "tai"]
KIND_LABEL = {"code": "code", "en": "English prose", "zh": "Traditional Chinese", "tai": "Taiwanese (Han characters)"}
KIND_SHORT = {"code": "code", "en": "English", "zh": "Trad. Chinese", "tai": "Taiwanese"}


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def out_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "rehearsal")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def tkey(T: float) -> str:
    return f"{T:g}"


def ckey(gamma: int, T: float) -> str:
    return f"g{gamma}_T{T:g}"


def num(x, fmt: str = ".2f") -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def pause(msg: str) -> bool:
    """Enter 繼續、q 離開；不是終端機時不等。"""
    if not sys.stdin.isatty():
        print(f"\n  {DIM}[{msg}]{RESET}")
        return True
    key = input(f"\n  {DIM}[Enter] {msg}   q = quit{RESET} ").strip().lower()
    print("\033[1A\033[2K", end="")
    return key != "q"


def conditions(text: str | None) -> dict:
    """彩排當下的電源與負載（量時間的 demo，這兩樣要記）。"""
    out = {"note": text or ""}
    try:
        out["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            first = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
            out["power"] = first.strip()
        except Exception:
            pass
    return out


def sweep_cfg(cfg: dict, args, target: str) -> dict:
    """一個 target 的網格；--tiny 換成小的。"""
    s = dict(cfg["sweep"][target])
    m = dict(cfg["measure"])
    out = {"drafts": list(s["drafts"]), "gammas": list(s["gammas"]), "temperatures": list(s["temperatures"]),
           "per_kind": s["per_kind"], "tai_prompts": s["tai_prompts"], "max_tokens": cfg["demo"]["max_tokens"],
           "draft_step_prompts": m["draft_step_prompts"], "verify_context_tokens": m["verify_context_tokens"],
           "verify_repeats": m["verify_repeats"]}
    if args.tiny:
        t = cfg["tiny"]
        out.update({k: t[k] for k in ("gammas", "temperatures", "per_kind", "tai_prompts", "max_tokens", "draft_step_prompts",
                                      "verify_context_tokens", "verify_repeats")})
        if target != "bf16":
            out.update({"gammas": t["gammas"][:2], "temperatures": [0.0], "per_kind": 1, "tai_prompts": 0})
    return out


def load_prompts(per_kind: int, tai: int) -> list[dict]:
    with open(PROMPTS, "rb") as f:
        allp = tomllib.load(f)["prompt"]
    out = []
    for kind in KINDS:
        rows = [p for p in allp if p["kind"] == kind]
        out += rows[: (tai if kind == "tai" else per_kind)]
    return out


def seed_for(base: int, *parts) -> int:
    return (base + zlib.crc32("|".join(str(x) for x in parts).encode())) % (2 ** 31)


# ══ (a) 五個測試 ═════════════════════════════════════════════════
def pick_step(args, rehearsal: bool):
    name = "speculative_final" if (rehearsal or args.final) else "speculative_live"
    return importlib.import_module(name).speculative_step, name + ".py"


def cmd_live(args, cfg, rehearsal: bool = False):
    import numpy as np
    import tests as T
    live = cfg["live"]
    step, src = pick_step(args, rehearsal)
    try:                                                   # 先確認函數本體寫了
        step(np.array(live["p"]), np.array(live["q"]), np.random.default_rng(0))
    except NotImplementedError:
        raise SystemExit(f"{AMBER}speculative_step() 還是空的：在 speculative_live.py 寫函數本體，或 ./present.sh live --final{RESET}")
    banner(f"{cfg['demo']['title']}\n(a) five tests of speculative_step() from {src} · no model")
    print(f"  p = {live['p']}   (the target)\n  q = {live['q']}   (the draft)")
    rows, state, t0 = {}, {}, time.time()
    for k in (1, 2, 3, 4, 5):
        title = f"test {k} — " + T.TITLES[k].format(n=live["n_draws"])
        if rehearsal:
            print(f"\n  {AMBER}{title}{RESET}")
        elif not pause(title):
            return None
        else:
            print(f"  {AMBER}{title}{RESET}")
        rows[str(k)] = T.run(k, step, live, state)
        T.show(k, rows[str(k)])
    T.verdicts(rows)
    rec = {"date": now(), "source": src, "python": platform.python_version(), "numpy": np.__version__,
           "inputs": {k: live[k] for k in ("p", "q", "n_draws", "z", "seed")}, "seconds": round(time.time() - t0, 2), "rows": rows}
    out = (out_dir(args) if rehearsal else RUNS / "live") / "tests.json"
    write_json(out, rec)
    note(f"\n  → {out.relative_to(HERE)}")
    return rec


def cmd_replay_a(args, cfg):
    import tests as T
    p = out_dir(args) / "tests.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse --part a{RESET}")
    d = read_json(p)
    banner(f"{cfg['demo']['title']}\n(a) REPLAY of the rehearsal of {d['date'][:10]} · speculative_step() from {d['source']}", AMBER)
    print(f"  p = {d['inputs']['p']}   (the target)\n  q = {d['inputs']['q']}   (the draft)")
    for k in (1, 2, 3, 4, 5):
        if not pause(f"test {k} — " + T.TITLES[k].format(n=d["inputs"]["n_draws"])):
            return
        print(f"  {AMBER}test {k} — " + T.TITLES[k].format(n=d["inputs"]["n_draws"]) + RESET)
        T.show(k, d["rows"][str(k)])
    T.verdicts(d["rows"])


# ══ (b) 模型與量測 ═══════════════════════════════════════════════
class Lab:
    """一個 target 與它的 draft。"""

    def __init__(self, cfg: dict, args, target: str, drafts: list[str]):
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_map
        self.mx, self.cfg, self.args, self.tiny = mx, cfg, args, bool(args.tiny)
        self.seed = int(cfg["demo"]["seed"])
        self.target_name, self.info = target, {}
        self.drafts, self.draft_toks = {}, {}
        t0 = time.time()
        if self.tiny:
            from mlx_lm.models import qwen3
            mx.random.seed(self.seed)
            raw = dict(model_type="qwen3", hidden_size=128, num_hidden_layers=2, intermediate_size=256, num_attention_heads=4,
                       rms_norm_eps=1e-6, vocab_size=512, num_key_value_heads=2, max_position_embeddings=4096, rope_theta=1e6,
                       head_dim=32, tie_word_embeddings=False)
            self.target = qwen3.Model(qwen3.ModelArgs(**raw))
            self.target.lm_head.weight = self.target.lm_head.weight * 6.0        # 亂數初始化的 logits 太平，放大讓分布有尖峰
            mx.eval(self.target.parameters())
            self.tok, self.eos = None, set()
            self.info[target] = {"label": f"tiny random Qwen3 ({target})", "repo": None, "revision": None}
            for name in drafts:
                sigma = float(cfg["tiny"]["draft_noise"][name])
                d = qwen3.Model(qwen3.ModelArgs(**raw))
                d.update(tree_map(lambda a: a + sigma * mx.sqrt(mx.var(a)) * mx.random.normal(a.shape), self.target.parameters()))
                mx.eval(d.parameters())
                self.drafts[name] = d
                self.info[name] = {"label": f"tiny {name}", "repo": None, "revision": None, "note": f"the target's weights plus noise, sd × {sigma:g}"}
            self.vocab = raw["vocab_size"]
        else:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            need = [("targets", target)] + [("drafts", d) for d in drafts]
            for group, name in need:
                m = cfg[group][name]
                rev = locked_revision(m["repo"], m.get("revision"))
                try:
                    path = snapshot_download(m["repo"], revision=rev, local_files_only=not args.online, ignore_patterns=NON_MODEL_FILES)
                except Exception as e:
                    raise SystemExit(f"{RED}本機沒有 {m['repo']}@{str(rev)[:10]}（{type(e).__name__}）：./present.sh fetch 看缺哪一個，./present.sh fetch --online 補抓{RESET}")
                model, tok = load(path)
                mx.eval(model.parameters())
                raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
                self.info[name] = {"label": m["label"], "repo": m["repo"], "revision": rev, "vocab_size": raw.get("vocab_size"),
                                   "hidden_size": raw.get("hidden_size"), "num_hidden_layers": raw.get("num_hidden_layers"),
                                   "quantization": raw.get("quantization")}
                if group == "targets":
                    self.target, self.tok = model, tok
                else:
                    self.drafts[name], self.draft_toks[name] = model, tok
            self.eos = set(int(x) for x in self.tok.eos_token_ids)
            self.vocab = self.info[target]["vocab_size"]
        for name, model in [(target, self.target)] + list(self.drafts.items()):
            flat = tree_flatten(model.parameters())
            self.info[name]["weight_bytes"] = sum(v.nbytes for _, v in flat)
            self.info[name]["n_arrays"] = len(flat)
        self.load_seconds = round(time.time() - t0, 2)

    # ── prompt ──
    def encode(self, prompt: dict) -> list[int]:
        if self.tok is None:                                             # tiny：亂數 token id，長度由 prompt 的 id 決定
            rng = random.Random(seed_for(self.seed, "prompt", prompt["id"]))
            return [rng.randrange(8, self.vocab - 8) for _ in range(rng.randrange(12, 25))]
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt["text"]}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=bool(self.cfg["demo"]["enable_thinking"]))
        return list(self.tok.encode(text, add_special_tokens=False))

    def text(self, ids) -> str:
        return str(list(ids)) if self.tok is None else self.tok.decode(list(ids))

    def sampler(self, T: float):
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=float(T))

    # ── 不用 draft ──
    def plain(self, model, ids: list[int], T: float, seed: int, max_tokens: int, on_token=None) -> dict:
        from mlx_lm.generate import generate_step
        mx = self.mx
        mx.random.seed(seed)
        toks, stamps, hit = [], [], False
        gen = generate_step(mx.array(ids), model, max_tokens=max_tokens, sampler=self.sampler(T))
        for tok, _ in gen:
            stamps.append(time.perf_counter())
            if int(tok) in self.eos:
                hit = True
                break
            toks.append(int(tok))
            if on_token:
                on_token(int(tok), False)
        gen.close()
        n = len(toks)
        return {"tokens": toks, "eos": hit, "steps": max(n - 1, 0), "seconds": (stamps[n - 1] - stamps[0]) if n >= 2 else 0.0,
                "intervals": [b - a for a, b in zip(stamps[:n - 1], stamps[1:n])]}

    # ── 用 draft ──
    def spec(self, draft, ids: list[int], gamma: int, T: float, seed: int, max_tokens: int, on_token=None) -> dict:
        from mlx_lm.generate import speculative_generate_step
        mx = self.mx
        mx.random.seed(seed)
        toks, rounds, hit = [], [], False
        ntoks, acc, proposed = 0, 0, min(max_tokens, gamma)
        gen = speculative_generate_step(mx.array(ids), self.target, draft, num_draft_tokens=gamma, max_tokens=max_tokens,
                                        sampler=self.sampler(T))
        for tok, _, from_draft in gen:
            t = time.perf_counter()
            ntoks += 1
            if not hit:
                if int(tok) in self.eos:
                    hit = True
                else:
                    toks.append(int(tok))
                    if on_token:
                        on_token(int(tok), bool(from_draft))
            acc += 1 if from_draft else 0
            if (not from_draft) or ntoks == max_tokens:                  # 一輪結束：target 自己的那一個 token，或到了上限
                rounds.append({"acc": acc, "proposed": proposed, "t": t, "eos": hit, "cut": bool(from_draft)})
                if hit or ntoks == max_tokens:
                    break
                acc, proposed = 0, min(max_tokens - ntoks, gamma)
        gen.close()
        return {"tokens": toks, "eos": hit, "rounds": rounds}

    # ── 兩個分布的重疊 ──
    def logits_along(self, model, ids: list[int], out: list[int]):
        """沿著已經生成的文字：回傳預測 out[0], out[1], … 的 logits（len(out) × V，float32）。"""
        mx = self.mx
        seq = mx.array(ids + out[:-1])[None]
        return model(seq)[0, len(ids) - 1:, :].astype(mx.float32)

    def overlap(self, lt, ld, T: float) -> dict:
        mx = self.mx
        agree = (mx.argmax(lt, axis=-1) == mx.argmax(ld, axis=-1)).astype(mx.float32)
        if T > 0:
            p, q = mx.softmax(lt / T, axis=-1), mx.softmax(ld / T, axis=-1)
            smin, spq = mx.minimum(p, q).sum(axis=-1), (p * q).sum(axis=-1)
        else:                                                            # 兩邊都取最大的那一個：兩種規則相同
            smin = spq = agree
        mx.eval(agree, smin, spq)
        return {"n": int(agree.shape[0]), "sum_min": float(smin.sum().item()), "sum_pq": float(spq.sum().item()),
                "agree": float(agree.sum().item())}

    def top2_gap(self, lt) -> list[float]:
        mx = self.mx
        lp = lt - mx.logsumexp(lt, axis=-1, keepdims=True)
        top = mx.sort(mx.topk(lp, 2, axis=-1), axis=-1)
        gap = top[:, 1] - top[:, 0]
        mx.eval(gap)
        return [round(float(x), 5) for x in gap.tolist()]

    # ── 驗一趟的時間 ──
    def verify_times(self, ctx: list[int], feed: list[int], ks: list[int], repeats: int) -> dict:
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        mx = self.mx
        cache = make_prompt_cache(self.target)
        self.target(mx.array(ctx)[None], cache=cache)
        mx.eval([c.state for c in cache])
        out = {}
        for k in ks:
            y = mx.array(feed[:k])[None]
            ts = []
            for r in range(repeats + 1):                                 # 第一次不計
                t0 = time.perf_counter()
                tok = mx.argmax(self.target(y, cache=cache)[:, -k:, :], axis=-1)
                mx.eval(tok)
                dt = time.perf_counter() - t0
                trim_prompt_cache(cache, k)
                if r:
                    ts.append(dt)
            out[str(k)] = 1e3 * M.median(ts)
        del cache
        mx.clear_cache()
        return out

    def close(self) -> None:
        self.target, self.drafts = None, {}
        self.mx.clear_cache()


def clean_rounds(rounds: list[dict], gamma: int) -> list[dict]:
    """從頭數起、完整的那幾輪：草擬了 gamma 個、沒遇到 EOS、不是被上限切斷的。"""
    out = []
    for r in rounds:
        if r["proposed"] != gamma or r["eos"] or r["cut"]:
            break
        out.append(r)
    return out


def tally(rounds: list[dict], gamma: int) -> dict:
    """一個 prompt 的一次生成 → 計數。時間與 token 數不含第一輪（它帶著 prefill）。"""
    c = clean_rounds(rounds, gamma)
    acc_at = [sum(1 for r in c if r["acc"] > j) for j in range(gamma)]
    tested_at = [sum(1 for r in c if r["acc"] >= j) for j in range(gamma)]
    return {"rounds": len(c), "accepted": sum(r["acc"] for r in c), "tested": sum(r["acc"] + (1 if r["acc"] < gamma else 0) for r in c),
            "proposed": gamma * len(c), "tokens": sum(r["acc"] + 1 for r in c[1:]), "seconds": (c[-1]["t"] - c[0]["t"]) if len(c) >= 2 else 0.0,
            "timed_rounds": max(len(c) - 1, 0), "acc_at": acc_at, "tested_at": tested_at}


def add(into: dict, x: dict) -> None:
    for k, v in x.items():
        if isinstance(v, list):
            into[k] = [a + b for a, b in zip(into.get(k, [0] * len(v)), v)]
        else:
            into[k] = into.get(k, 0) + v


def first_divergence(a: list[int], b: list[int]):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


# ── 紀錄檔 ───────────────────────────────────────────────────────
def rec_path(args) -> Path:
    return out_dir(args) / "sweep.json"


def cache_path(args, target: str, T: float) -> Path:
    return RUNS / "cache" / f"{'tiny_' if args.tiny else ''}{target}_T{T:g}.json"


def load_rec(args) -> dict | None:
    p = rec_path(args)
    return read_json(p) if p.exists() else None


def d1_points(cfg: dict, args, target: str) -> dict | None:
    """demo 1 (b)：{B: 一步幾 ms}。沒有紀錄、或是 tiny，就回傳 None。"""
    if args.tiny:
        return None
    p = (HERE / cfg["reference"]["demo1_record"]).resolve()
    if not p.exists():
        return None
    b = (read_json(p).get("models", {}).get(target) or {}).get("b") or {}
    pts = {int(k): v["step_ms"] for k, v in b.items() if isinstance(v, dict) and "step_ms" in v}
    return pts if 1 in pts and len(pts) >= 2 else None


def summarize_target(t: dict) -> dict:
    """一個 target 的所有格子 → 表。只用紀錄裡的數，不載模型。"""
    out = {}
    gammas, d1 = t["grid"]["gammas"], t.get("demo1_step_ms")
    r_d1 = M.ratio_from(d1) if d1 else None
    r_v = M.ratio_from(t["verify_ms"]) if t.get("verify_ms") else None
    for dn, d in t.get("drafts", {}).items():
        out[dn] = {}
        for T in t["grid"]["temperatures"]:
            tk = tkey(T)
            base, step = t.get("baseline", {}).get(tk), d.get("step", {}).get(tk)
            if not base or not step:
                continue
            c = step["c"]
            rows, pool = [], {}
            for g in gammas:
                cell = d.get("cells", {}).get(ckey(g, T))
                if not cell or not cell.get("seconds"):
                    continue
                # 同一批 prompt 的基準
                bs = sum(x[0] for x in base["per_prompt"])
                bt = sum(x[1] for x in base["per_prompt"])
                rate, brate = cell["tokens"] / cell["seconds"], bs / bt
                a = M.alpha_hat(cell["accepted"], cell["tested"])
                per = [(x[0] / x[1]) / (b[0] / b[1]) for x, b in zip(cell["per_prompt"], base["per_prompt"]) if x[1] > 0 and b[1] > 0]
                row = {"gamma": g, "tok_per_s": rate, "speedup": rate / brate, "speedup_q25": M.quantile(per, 0.25), "speedup_q75": M.quantile(per, 0.75),
                       "alpha": a, "kept_share": cell["accepted"] / cell["proposed"] if cell["proposed"] else None,
                       "tokens_per_round": cell["tokens"] / cell["timed_rounds"] if cell.get("timed_rounds") else None,
                       "omega": M.omega(g, a) if a is not None else None,
                       "round_ms": 1e3 * cell["seconds"] / cell["timed_rounds"] if cell.get("timed_rounds") else None,
                       "round_ms_model": (g * step["ms_per_token"] + t["verify_ms"][str(g + 1)]) if t.get("verify_ms", {}).get(str(g + 1)) else None,
                       "S_thm": M.speedup(g, a, c) if a is not None else None,
                       "S_d1": M.speedup(g, a, c, r_d1(g + 1)) if (a is not None and r_d1) else None,
                       "S_verify": M.speedup(g, a, c, r_v(g + 1)) if (a is not None and r_v) else None,
                       "alpha_by_position": [x / y if y else None for x, y in zip(cell["acc_at"], cell["tested_at"])]}
                rows.append(row)
                add(pool, {k: cell[k] for k in ("accepted", "tested")})
                for kind, kc in cell["by_kind"].items():
                    add(pool.setdefault("kind_" + kind, {}), {k: kc[k] for k in ("accepted", "tested")})
            if not rows:
                continue
            a_pool = M.alpha_hat(pool["accepted"], pool["tested"])
            best = max(rows, key=lambda r: r["speedup"])
            gs = [r["gamma"] for r in rows]
            ov = d.get("overlap", {}).get(tk) or {}
            kinds = {}
            bcell = d["cells"][ckey(best["gamma"], T)]
            for kind in KINDS:
                kp = pool.get("kind_" + kind)
                if not kp:
                    continue
                kc, kb = bcell["by_kind"].get(kind), base["by_kind"].get(kind)
                ko = (ov.get("by_kind") or {}).get(kind)
                kinds[kind] = {"alpha": M.alpha_hat(kp["accepted"], kp["tested"]), "tested": kp["tested"],
                               "speedup_at_best": (kc["tokens"] / kc["seconds"]) / (kb[0] / kb[1]) if kc and kb and kc["seconds"] and kb[1] else None,
                               "sum_min": ko["sum_min"] / ko["n"] if ko and ko["n"] else None, "sum_pq": ko["sum_pq"] / ko["n"] if ko and ko["n"] else None,
                               "agree": ko["agree"] / ko["n"] if ko and ko["n"] else None}
            out[dn][tk] = {"c": c, "draft_ms_per_token": step["ms_per_token"], "target_ms_per_token": step["target_ms_per_token"],
                           "alpha_pooled": a_pool, "tested_pooled": pool["tested"],
                           "sum_min": ov["sum_min"] / ov["n"] if ov.get("n") else None, "sum_pq": ov["sum_pq"] / ov["n"] if ov.get("n") else None,
                           "argmax_agree": ov["agree"] / ov["n"] if ov.get("n") else None,
                           "best": {"gamma": best["gamma"], "speedup": best["speedup"]},
                           "thm_best": dict(zip(("gamma", "speedup"), M.best_gamma(a_pool, c, gs))),
                           "d1_best": dict(zip(("gamma", "speedup"), M.best_gamma(a_pool, c, gs, r_d1))) if r_d1 else None,
                           "verify_best": dict(zip(("gamma", "speedup"), M.best_gamma(a_pool, c, gs, r_v))) if r_v else None,
                           "rows": rows, "by_kind": kinds}
    return out


def save_rec(args, rec: dict) -> None:
    rec["updated"] = now()
    for t in rec["targets"].values():
        t["summary"] = summarize_target(t)
    write_json(rec_path(args), rec)


# ── 彩排的三段 ───────────────────────────────────────────────────
def run_base(lab: Lab, args, rec: dict, t: dict, g: dict, prompts: list[dict]) -> dict:
    """基準、draft 的一步、兩個分布的重疊、驗一趟的時間。回傳 {T: 每個 prompt 的 token 與 logit 差}（也存到 runs/cache/）。"""
    mx = lab.mx
    ids = [lab.encode(p) for p in prompts]
    t["prompts"] = {"ids": [p["id"] for p in prompts], "kinds": [p["kind"] for p in prompts], "prompt_tokens": [len(x) for x in ids]}
    caches = {}
    lab.plain(lab.target, ids[0], 0.0, 0, 8)                              # 暖機
    for T in g["temperatures"]:
        tk = tkey(T)
        cp = cache_path(args, lab.target_name, T)
        have = t.get("baseline", {}).get(tk)
        if have and cp.exists() and not args.force and read_json(cp).get("ids") == t["prompts"]["ids"]:
            caches[tk] = read_json(cp)
            continue
        note(f"  baseline, T = {T:g}: {len(prompts)} prompts …")
        per, toks, by_kind, inter, t0 = [], [], {}, [], time.time()
        for i, p in enumerate(prompts):
            r = lab.plain(lab.target, ids[i], T, seed_for(lab.seed, "base", p["id"], tk), g["max_tokens"])
            per.append([r["steps"], r["seconds"], len(r["tokens"]), bool(r["eos"])])
            toks.append(r["tokens"])
            inter += r["intervals"]
            k = by_kind.setdefault(p["kind"], [0, 0.0])
            k[0] += r["steps"]
            k[1] += r["seconds"]
        steps, secs = sum(x[0] for x in per), sum(x[1] for x in per)
        t.setdefault("baseline", {})[tk] = {"T": T, "steps": steps, "seconds": secs, "tok_per_s": steps / secs, "ms_per_token": 1e3 * secs / steps,
                                           "step_ms_median": 1e3 * M.median(inter), "by_kind": by_kind, "per_prompt": per,
                                           "hit_eos": sum(1 for x in per if x[3]), "wall_seconds": round(time.time() - t0, 1), "date": now()}
        caches[tk] = {"ids": t["prompts"]["ids"], "tokens": toks, "gaps": None}
        write_json(cp, caches[tk])
        for d in t.get("drafts", {}).values():                           # 基準重跑了：靠它的東西作廢
            d.get("overlap", {}).pop(tk, None)
        save_rec(args, rec)

    for dn in g["drafts"]:
        d = t.setdefault("drafts", {}).setdefault(dn, {"info": lab.info[dn], "step": {}, "overlap": {}, "cells": {}})
        d["info"] = lab.info[dn]
        # draft 自己生成：每一類輪流取 prompt
        order = [i for j in range(len(prompts)) for i, p in enumerate(prompts) if sum(1 for q in prompts[:i] if q["kind"] == p["kind"]) == j]
        sub = sorted(order[: g["draft_step_prompts"]])
        for T in g["temperatures"]:
            tk = tkey(T)
            if d["step"].get(tk) and not args.force:
                continue
            lab.plain(lab.drafts[dn], ids[0], 0.0, 0, 8)
            steps = secs = 0.0
            for i in sub:
                r = lab.plain(lab.drafts[dn], ids[i], T, seed_for(lab.seed, "draft", dn, prompts[i]["id"], tk), g["max_tokens"])
                steps += r["steps"]
                secs += r["seconds"]
            base = t["baseline"][tk]["per_prompt"]
            bsteps, bsecs = sum(base[i][0] for i in sub), sum(base[i][1] for i in sub)
            d["step"][tk] = {"T": T, "n_prompts": len(sub), "steps": steps, "seconds": secs, "ms_per_token": 1e3 * secs / steps,
                             "target_ms_per_token": 1e3 * bsecs / bsteps, "c": (secs / steps) / (bsecs / bsteps), "date": now()}
            save_rec(args, rec)

    # 重疊：target 的 logits 每個 prompt 只算一次，兩個 draft 共用
    for T in g["temperatures"]:
        tk = tkey(T)
        todo = [dn for dn in g["drafts"] if not t["drafts"][dn]["overlap"].get(tk) or args.force]
        need_gap = T == 0 and caches[tk].get("gaps") is None
        if not todo and not need_gap:
            continue
        note(f"  overlap of the two distributions along the baseline text, T = {T:g} …")
        acc = {dn: {"n": 0, "sum_min": 0.0, "sum_pq": 0.0, "agree": 0.0, "by_kind": {}} for dn in todo}
        gaps = []
        for i, p in enumerate(prompts):
            out = caches[tk]["tokens"][i]
            if len(out) < 2:
                gaps.append([])
                continue
            lt = lab.logits_along(lab.target, ids[i], out)
            if T == 0:
                gaps.append(lab.top2_gap(lt))
            for dn in todo:
                o = lab.overlap(lt, lab.logits_along(lab.drafts[dn], ids[i], out), T)
                add(acc[dn], o)
                add(acc[dn]["by_kind"].setdefault(p["kind"], {}), o)
            del lt
            mx.clear_cache()
        for dn in todo:
            t["drafts"][dn]["overlap"][tk] = {"T": T, **acc[dn], "date": now()}
        if T == 0:
            caches[tk]["gaps"] = gaps
            write_json(cache_path(args, lab.target_name, T), caches[tk])
        save_rec(args, rec)

    if not t.get("verify_ms") or args.force:
        note("  one verification pass: k tokens at once, k = 1 … max gamma + 1 …")
        tk = tkey(g["temperatures"][0])
        i = max(range(len(prompts)), key=lambda j: len(caches[tk]["tokens"][j]))
        seq = ids[i] + caches[tk]["tokens"][i]
        kmax = max(g["gammas"]) + 1
        L = min(g["verify_context_tokens"], len(seq) - kmax)
        t["verify_ms"] = lab.verify_times(seq[:L], seq[L:L + kmax], list(range(1, kmax + 1)), g["verify_repeats"])
        t["verify_context_tokens"] = L
        save_rec(args, rec)
    return caches


def run_cell(lab: Lab, t: dict, g: dict, prompts: list[dict], ids: list[list[int]], caches: dict, dn: str, gamma: int, T: float) -> dict:
    tk = tkey(T)
    tot, by_kind, per, t0 = {}, {}, [], time.time()
    greedy = {"n": 0, "identical": 0, "diverged": []} if T == 0 else None
    reset_peak_memory()
    for i, p in enumerate(prompts):
        r = lab.spec(lab.drafts[dn], ids[i], gamma, T, seed_for(lab.seed, "spec", dn, gamma, p["id"], tk), g["max_tokens"])
        x = tally(r["rounds"], gamma)
        add(tot, x)
        add(by_kind.setdefault(p["kind"], {}), x)
        per.append([x["tokens"], x["seconds"], x["accepted"], x["tested"], x["rounds"]])
        if greedy is not None:
            base = caches[tk]["tokens"][i]
            greedy["n"] += 1
            pos = first_divergence(base, r["tokens"])
            if pos is None:
                greedy["identical"] += 1
            else:
                gp = (caches[tk].get("gaps") or [])
                greedy["diverged"].append({"prompt": p["id"], "position": pos, "of": len(base),
                                           "top2_gap": gp[i][pos] if i < len(gp) and pos < len(gp[i]) else None,
                                           "baseline_token": lab.text(base[pos:pos + 1]), "speculative_token": lab.text(r["tokens"][pos:pos + 1])})
    cell = {"draft": dn, "gamma": gamma, "T": T, "n_prompts": len(prompts), **tot, "by_kind": by_kind, "per_prompt": per,
            "peak_gb": peak_memory_gb(), "wall_seconds": round(time.time() - t0, 1), "date": now()}
    if greedy is not None:
        cell["greedy"] = greedy
    return cell


def new_rec(args, cfg) -> dict:
    return {"date_started": now(), "tiny": bool(args.tiny), "versions": versions(), "device": device_info(), "conditions": [],
            "settings": {"max_tokens": cfg["demo"]["max_tokens"], "enable_thinking": cfg["demo"]["enable_thinking"], "seed": cfg["demo"]["seed"],
                         "rule": "mlx-lm speculative_generate_step: a draft token is kept if the target's own sample at that position is the same token"},
            "targets": {}}


def cmd_rehearse(args, cfg):
    parts = [args.part] if args.part else ["a", "base", "sweep"]
    if "a" in parts:
        cmd_live(args, cfg, rehearsal=True)
    if parts == ["a"]:
        return
    rec = load_rec(args)
    if rec is None or bool(rec.get("tiny")) != bool(args.tiny):
        rec = new_rec(args, cfg)
    rec["versions"], rec["device"] = versions(), device_info()
    rec["conditions"].append({"date": now(), "parts": parts, "target": args.target, **conditions(args.note)})
    targets = [args.target] if args.target else list(cfg["sweep"])
    for tn in targets:
        g = sweep_cfg(cfg, args, tn)
        prompts = load_prompts(g["per_kind"], g["tai_prompts"])
        banner(f"rehearse — target {tn}: {len(prompts)} prompts × drafts {g['drafts']} × gamma {g['gammas']} × T {g['temperatures']}")
        lab = Lab(cfg, args, tn, g["drafts"])
        t = rec["targets"].setdefault(tn, {"drafts": {}})
        if t.get("grid") and (t["grid"].get("max_tokens") != g["max_tokens"] or t.get("prompts", {}).get("ids") != [p["id"] for p in prompts]):
            note("  prompt 或 max_tokens 與紀錄不同：這個 target 的舊紀錄作廢，重量")
            t.clear()
            t["drafts"] = {}
        t["info"] = lab.info[tn]
        t["grid"] = {k: g[k] for k in ("drafts", "gammas", "temperatures", "per_kind", "tai_prompts", "max_tokens")}
        t["demo1_step_ms"] = d1_points(cfg, args, tn)
        try:
            caches = run_base(lab, args, rec, t, g, prompts)
            if "sweep" in parts:
                ids = [lab.encode(p) for p in prompts]
                todo = [(dn, gm, T) for dn in g["drafts"] for T in g["temperatures"] for gm in g["gammas"]
                        if args.force or ckey(gm, T) not in t["drafts"][dn]["cells"]]
                for n, (dn, gm, T) in enumerate(todo, 1):
                    cell = run_cell(lab, t, g, prompts, ids, caches, dn, gm, T)
                    t["drafts"][dn]["cells"][ckey(gm, T)] = cell
                    save_rec(args, rec)
                    row = next((r for r in t["summary"].get(dn, {}).get(tkey(T), {}).get("rows", []) if r["gamma"] == gm), None) or {}
                    print(f"  [{n}/{len(todo)}] {dn}  gamma {gm:>2}  T {T:g}   speed-up {BOLD}{num(row.get('speedup'))}×{RESET}   alpha {num(row.get('alpha'))}"
                          f"   Thm 3.8 {num(row.get('S_thm'))}   {GREY}{cell['wall_seconds']:.0f} s{RESET}")
                if todo:                                                 # 掃完再量一次基準的一部分：這段時間裡機器有沒有變慢
                    T0 = g["temperatures"][0]
                    sub = list(range(0, len(prompts), max(1, len(prompts) // max(1, g["draft_step_prompts"]))))[: g["draft_step_prompts"]]
                    steps = secs = 0.0
                    for i in sub:
                        r = lab.plain(lab.target, ids[i], T0, seed_for(lab.seed, "base", prompts[i]["id"], tkey(T0)), g["max_tokens"])
                        steps += r["steps"]
                        secs += r["seconds"]
                    first = t["baseline"][tkey(T0)]["per_prompt"]
                    f_steps, f_secs = sum(first[i][0] for i in sub), sum(first[i][1] for i in sub)
                    t["baseline_again"] = {"T": T0, "n_prompts": len(sub), "tok_per_s": steps / secs, "first_tok_per_s": f_steps / f_secs,
                                           "ratio": (steps / secs) / (f_steps / f_secs), "date": now()}
                    save_rec(args, rec)
        finally:
            save_rec(args, rec)
            lab.close()
    note(f"\n→ {rec_path(args).relative_to(HERE)}")
    show_all(cfg, rec, stop=False)


# ══ 畫面 ═════════════════════════════════════════════════════════
def bar(x: float | None, hi: float, width: int = 24, mark: float | None = None) -> str:
    if x is None:
        return ""
    n = max(0, min(width, int(round(width * x / hi))))
    s = list("█" * n + " " * (width - n))
    if mark is not None:
        m = max(0, min(width - 1, int(round(width * mark / hi))))
        s[m] = "│"
    return "".join(s)


def header(cfg: dict, rec: dict, what: str) -> None:
    extra = f"\n(b) {what} of {rec.get('updated', '')[:16]}" + ("   [tiny random models — numbers mean nothing]" if rec.get("tiny") else "")
    dev = rec.get("device") or {}
    if dev.get("device_name"):
        extra += f"   {dev['device_name']}"
    banner(cfg["demo"]["title"] + extra, AMBER)


def show_drafts(t: dict) -> None:
    s, base = t["summary"], t.get("baseline", {})
    b0 = base.get("0") or next(iter(base.values()))
    print(f"\n  {BOLD}target: {t['info']['label']}{RESET}, alone: {b0['tok_per_s']:.1f} tokens/s ({b0['ms_per_token']:.1f} ms per token)"
          f"   {GREY}{t['prompts'] and len(t['prompts']['ids'])} prompts, up to {t['grid']['max_tokens']} tokens each{RESET}")
    print(f"\n  {'draft':<14}{'T':>5}{'c':>8}{'alpha':>8}{'Σ p·q':>8}{'Σ min':>8}{'best γ':>9}{'speed-up':>10}   {'Theorem 3.8 says':<20}")
    for dn, byT in s.items():
        for tk, x in byT.items():
            print(f"  {t['drafts'][dn]['info']['label'][:13]:<14}{tk:>5}{x['c']:>8.3f}{BOLD}{num(x['alpha_pooled'], '8.3f')}{RESET}{num(x['sum_pq'], '8.3f')}{num(x['sum_min'], '8.3f')}"
                  f"{x['best']['gamma']:>9}{BOLD}{x['best']['speedup']:>9.2f}×{RESET}   γ = {x['thm_best']['gamma']}, {x['thm_best']['speedup']:.2f}×")
    print(f"  {GREY}c: the draft's time per token ÷ the target's.  alpha: drafts kept ÷ drafts actually checked, pooled over γ.{RESET}")
    print(f"  {GREY}Σ p·q and Σ min(p, q): from the two models' logits along the target's own text. The library keeps a draft only if the target's{RESET}")
    print(f"  {GREY}own sample is the same token, so alpha should sit near Σ p·q (a little below it if agreements come in runs: a rejection is followed{RESET}")
    print(f"  {GREY}at once by another check). Σ min is what the rejection rule of p42 would keep. At T = 0 the two are equal.{RESET}")


def show_curves(t: dict) -> None:
    s = t["summary"]
    has_d1 = bool(t.get("demo1_step_ms"))
    for dn, byT in s.items():
        for tk, x in byT.items():
            hi = max([r["speedup"] for r in x["rows"]] + [r["S_thm"] or 0 for r in x["rows"]]) * 1.05
            print(f"\n  {BOLD}{t['drafts'][dn]['info']['label']} → {t['info']['label']}, T = {tk}{RESET}   c = {x['c']:.3f}   {GREY}█ measured   │ Theorem 3.8{RESET}")
            print(f"  {'γ':>3}{'alpha':>8}{'tok/round':>11}{'Ω':>7}{'speed-up':>10}  {'':<24}{'Thm 3.8':>9}" + (f"{'+ demo 1':>10}" if has_d1 else "") + f"{'+ verify':>10}")
            for r in x["rows"]:
                star = "←" if r["gamma"] == x["best"]["gamma"] else " "
                print(f"  {r['gamma']:>3}{num(r['alpha'], '8.3f')}{num(r['tokens_per_round'], '11.2f')}{num(r['omega'], '7.2f')}{BOLD}{r['speedup']:>9.2f}×{RESET}"
                      f" {star}{bar(r['speedup'], hi, 24, r['S_thm'])}{num(r['S_thm'], '9.2f')}" + (num(r["S_d1"], "10.2f") if has_d1 else "") + num(r["S_verify"], "10.2f"))
            last = x["rows"][-1]
            print(f"  {GREY}alpha by position within a round, γ = {last['gamma']}: " + " ".join(num(a, ".2f") for a in last["alpha_by_position"])
                  + "   (the theorem assumes one alpha for every position)" + RESET)
            print(f"  {GREY}ms per round, measured vs γ × draft step + one verification pass: "
                  + "  ".join(f"γ={r['gamma']}: {num(r['round_ms'], '.1f')} vs {num(r['round_ms_model'], '.1f')}" for r in x["rows"]) + RESET)
    v = t.get("verify_ms") or {}
    if v:
        ks = sorted(int(k) for k in v)
        print(f"\n  {GREY}one verification pass of k tokens ÷ one step (this demo, context {t.get('verify_context_tokens')} tokens): "
              + "  ".join(f"k={k}: {v[str(k)] / v['1']:.2f}" for k in ks) + RESET)
    if has_d1:
        r = M.ratio_from(t["demo1_step_ms"])
        print(f"  {GREY}the same ratio read off demo 1 (b) (k separate sequences, one token each; interpolated): "
              + "  ".join(f"k={g + 1}: {r(g + 1):.2f}" for g in t["grid"]["gammas"]) + RESET)
    print(f"  {GREY}Thm 3.8: Ω(γ, alpha) ÷ (γc + 1).  '+ demo 1' and '+ verify' replace the 1 by the ratio above at k = γ + 1.{RESET}")
    ba = t.get("baseline_again")
    if ba:
        print(f"  {GREY}the target alone, measured again after the sweep on {ba['n_prompts']} prompts: {ba['tok_per_s']:.1f} tokens/s, "
              f"{ba['ratio']:.2f}× the first measurement on the same prompts{RESET}")


def show_kinds(t: dict) -> None:
    s = t["summary"]
    print(f"\n  {BOLD}by kind of text{RESET} — alpha pooled over γ; speed-up at that draft's best γ")
    print(f"  {GREY}each cell: alpha, then the speed-up{RESET}")
    print(f"  {'draft':<14}{'T':>5}  " + "".join(f"{KIND_SHORT[k]:>20}" for k in KINDS))
    for dn, byT in s.items():
        for tk, x in byT.items():
            cells = []
            for k in KINDS:
                y = x["by_kind"].get(k)
                cells.append(f"{'—':>20}" if not y else f"{BOLD}{num(y['alpha'], '11.3f')}{RESET}{num(y['speedup_at_best'], '8.2f')}×")
            print(f"  {t['drafts'][dn]['info']['label'][:13]:<14}{tk:>5}  " + "".join(cells))
    kinds = t["prompts"]["kinds"]
    print(f"  {GREY}prompts per kind: " + ", ".join(f"{KIND_LABEL[k]} {kinds.count(k)}" for k in KINDS if kinds.count(k)) + f".  A small sample for the last one.{RESET}")
    print(f"  {GREY}the model answers in whatever it writes for that prompt; 'kind' is the kind of prompt, not a check of the output's language.{RESET}")


def show_greedy(t: dict) -> None:
    print(f"\n  {BOLD}T = 0: is the speculative output the same tokens as plain greedy decoding?{RESET}")
    found = False
    for dn, d in t["drafts"].items():
        cells = [c for c in d["cells"].values() if c.get("greedy")]
        if not cells:
            continue
        found = True
        n, same = sum(c["greedy"]["n"] for c in cells), sum(c["greedy"]["identical"] for c in cells)
        div = [dict(x, gamma=c["gamma"]) for c in cells for x in c["greedy"]["diverged"]]
        print(f"  {d['info']['label']:<14} {len(cells)} values of γ × {cells[0]['greedy']['n']} prompts = {n} generations:  {BOLD}{same} identical{RESET}, {len(div)} diverge")
        if div:
            pos = sorted(x["position"] for x in div)
            gaps = sorted(x["top2_gap"] for x in div if x["top2_gap"] is not None)
            print(f"      first differing position: earliest {pos[0]}, median {M.median(pos):g}, latest {pos[-1]}")
            if gaps:
                print(f"      gap between the target's two best log-probabilities there: median {M.median(gaps):.4f}, largest {gaps[-1]:.4f}")
            for x in sorted(div, key=lambda x: x["position"])[:4]:
                print(f"      {GREY}{x['prompt']}  γ = {x['gamma']:>2}  token {x['position']:>3} of {x['of']}:  {x['baseline_token']!r} → {x['speculative_token']!r}"
                      f"   gap {num(x['top2_gap'], '.4f')}{RESET}")
    if not found:
        note("  （沒有 T = 0 的格子）")
    print(f"  {GREY}the gap is computed in one pass over the whole text, so it is itself rounded differently from either run.{RESET}")


CAVEAT = ("one machine, one family of models, {n} prompts written for this demo, one run per cell; thermal state and background load are not "
          "controlled; the acceptance rule is the library's (match-only), not the rejection rule of p42.")


def show_all(cfg: dict, rec: dict, stop: bool = True) -> None:
    names = list(rec["targets"])
    main = "bf16" if "bf16" in rec["targets"] else names[0]
    t = rec["targets"][main]
    if not t.get("summary") or not any(t["summary"].values()):
        note("  （還沒有可以印的格子）")
        return
    steps = [("the drafts: c, alpha, the best γ", show_drafts), ("speed-up against γ, with the theorem's curve", show_curves),
             ("by kind of text", show_kinds), ("T = 0: the same tokens as greedy?", show_greedy)]
    for msg, fn in steps:
        if stop and not pause(msg):
            return
        fn(t)
    for other in names:
        if other == main or not rec["targets"][other].get("summary") or not any(rec["targets"][other]["summary"].values()):
            continue
        if stop and not pause(f"another target: {rec['targets'][other]['info']['label']}"):
            return
        show_drafts(rec["targets"][other])
        show_curves(rec["targets"][other])
    print(f"\n  {GREY}{CAVEAT.format(n=len(t['prompts']['ids']))}{RESET}\n")


def cmd_replay(args, cfg):
    rec = load_rec(args)
    if rec is None:
        raise SystemExit(f"{RED}沒有 {rec_path(args).relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    for t in rec["targets"].values():                                    # 表每次都從原始的計數重算
        t["summary"] = summarize_target(t)
    header(cfg, rec, "REPLAY of the rehearsal")
    show_all(cfg, rec)


# ── 其餘的子命令 ─────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    """四個模型都在 versions.lock；這裡只確認本機有。--online 時補抓。"""
    from huggingface_hub import snapshot_download
    for group in ("targets", "drafts"):
        for name, m in cfg[group].items():
            rev = locked_revision(m["repo"], m.get("revision"))
            try:
                p = snapshot_download(m["repo"], revision=rev, local_files_only=not args.online, ignore_patterns=NON_MODEL_FILES)
                print(f"  {GREEN}ok{RESET}  {name:<5} {m['repo']}@{str(rev)[:10]}  →  {p}")
            except Exception as e:
                print(f"  {RED}缺{RESET}  {name:<5} {m['repo']}@{str(rev)[:10]}：{repr(e)[:120]}")


def cmd_inspect(args, cfg):
    G = importlib.import_module("mlx_lm.generate")          # mlx_lm.generate 這個名字在套件裡是函數，模組要這樣拿
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}")
    dev = device_info()
    for k in ("device_name", "memory_size", "max_recommended_working_set_size"):
        if k in dev:
            print(f"  {k:<34} {dev[k] / GB:.1f} GB" if isinstance(dev[k], (int, float)) else f"  {k:<34} {dev[k]}")
    out = {"date": now(), "versions": v, "device": dev, "conditions": conditions(args.note), "tiny": bool(args.tiny)}

    banner("the acceptance rule in the installed mlx-lm")
    src = pyinspect.getsource(G.speculative_generate_step).splitlines()
    hits = [i for i, line in enumerate(src) if "tn != dtn" in line]
    for i in hits:
        for line in src[max(0, i - 2): i + 6]:
            print(f"  {GREY}|{RESET} {line}")
    sig = str(pyinspect.signature(G.speculative_generate_step))
    out["rule"] = {"match_only_line_found": bool(hits), "signature": sig}
    print(f"  {(GREEN + 'match-only: a draft token is kept if the target sampled the same token') if hits else (RED + '找不到 tn != dtn 那一行：接受規則可能改了，讀原始碼')}{RESET}")
    print(f"  {GREY}num_draft_tokens in the signature: {'num_draft_tokens' in sig}{RESET}")

    g = sweep_cfg(cfg, args, "bf16")
    prompts = load_prompts(g["per_kind"], g["tai_prompts"])
    banner("models and tokenizers")
    lab = Lab(cfg, args, "bf16", g["drafts"])
    out["models"] = lab.info
    for name, i in lab.info.items():
        print(f"  {name:<6} {i['label']:<24} weights {i['weight_bytes'] / GB:6.2f} GB   {GREY}{(i.get('repo') or '') + '@' + str(i.get('revision'))[:10] if i.get('repo') else ''}{RESET}")
    if lab.tok is not None:
        probe = [p["text"] for p in prompts[::max(1, len(prompts) // 8)]]
        want = [lab.tok.encode(s, add_special_tokens=False) for s in probe]
        same = {}
        for name in g["drafts"]:
            tk = lab.draft_toks[name]
            same[name] = all(tk.encode(s, add_special_tokens=False) == w for s, w in zip(probe, want)) and lab.info[name]["vocab_size"] == lab.info["bf16"]["vocab_size"]
            print(f"  tokenizer of {name} gives the same ids as the target's on {len(probe)} prompts, same vocabulary size: "
                  f"{(GREEN + 'yes') if same[name] else (RED + 'NO')}{RESET}")
        out["same_tokenizer"] = same
        ids = lab.encode(prompts[0])
        print(f"  prompt {prompts[0]['id']} through the chat template (enable_thinking = {cfg['demo']['enable_thinking']}): {len(ids)} tokens, ends with {lab.text(ids[-8:])!r}")
        print(f"  end-of-sequence ids: {sorted(lab.eos)}")
    lens = {}
    for p in prompts:
        lens.setdefault(p["kind"], []).append(len(lab.encode(p)))
    out["prompt_tokens"] = {k: {"n": len(x), "min": min(x), "median": M.median(x), "max": max(x)} for k, x in lens.items()}
    for k, x in out["prompt_tokens"].items():
        print(f"  prompts, {KIND_LABEL[k]:<28} {x['n']:>3}   tokens min {x['min']}, median {x['median']:g}, max {x['max']}")

    banner("a draft drafting for itself, T = 0: every draft should be kept")
    dn = g["drafts"][0]
    d = lab.drafts[dn]
    keep_target, lab.target = lab.target, d                              # 暫時把 draft 當 target
    try:
        ids = lab.encode(prompts[0])
        base = lab.plain(d, ids, 0.0, 0, g["max_tokens"])
        rows = []
        for gm in sorted({1, 4, max(g["gammas"])}):
            r = lab.spec(d, ids, gm, 0.0, 0, g["max_tokens"])
            x = tally(r["rounds"], gm)
            pos = first_divergence(base["tokens"], r["tokens"])
            rows.append({"gamma": gm, "accepted": x["accepted"], "tested": x["tested"], "first_divergence": pos})
            print(f"  {dn} as its own draft, γ = {gm:>2}: kept {x['accepted']} of {x['tested']} checked"
                  f"   output vs plain greedy: {'identical' if pos is None else 'differs from token ' + str(pos)}")
        out["self_draft"] = rows
    finally:
        lab.target = keep_target
    lab.close()
    write_json(out_dir(args) / "inspect.json", out)
    note(f"\n→ {(out_dir(args) / 'inspect.json').relative_to(HERE)}")


def cmd_probe(args, cfg):
    """幾個 prompt 量一次，估全套要多久。"""
    m = cfg["measure"]
    total = 0.0
    for tn in ([args.target] if args.target else list(cfg["sweep"])):
        g = sweep_cfg(cfg, args, tn)
        full = load_prompts(g["per_kind"], g["tai_prompts"])
        prompts = load_prompts(1 if args.tiny else m["probe_per_kind"], min(1, g["tai_prompts"]))
        gammas = [x for x in (g["gammas"][:2] if args.tiny else m["probe_gammas"]) if x in g["gammas"]] or g["gammas"][:1]
        banner(f"probe — target {tn}: {len(prompts)} prompts, γ = {gammas}, T = 0")
        lab = Lab(cfg, args, tn, g["drafts"])
        ids = [lab.encode(p) for p in prompts]
        lab.plain(lab.target, ids[0], 0.0, 0, 8)
        base = [lab.plain(lab.target, x, 0.0, 0, g["max_tokens"]) for x in ids]
        bs, bt = sum(r["steps"] for r in base), sum(r["seconds"] for r in base)
        print(f"  target alone: {BOLD}{bs / bt:.1f} tokens/s{RESET}   ({sum(len(r['tokens']) for r in base) / len(base):.0f} tokens per prompt, {sum(r['eos'] for r in base)} hit EOS)")
        est = len(full) * len(g["temperatures"]) * g["max_tokens"] / (bs / bt)
        for dn in g["drafts"]:
            lab.plain(lab.drafts[dn], ids[0], 0.0, 0, 8)
            dr = [lab.plain(lab.drafts[dn], x, 0.0, 0, g["max_tokens"]) for x in ids]
            ds, dt_ = sum(r["steps"] for r in dr), sum(r["seconds"] for r in dr)
            c = (dt_ / ds) / (bt / bs)
            print(f"  {dn} alone: {ds / dt_:.1f} tokens/s   c = {BOLD}{c:.3f}{RESET}")
            rates = []
            for gm in gammas:
                tot = {}
                for x in ids:
                    add(tot, tally(lab.spec(lab.drafts[dn], x, gm, 0.0, 0, g["max_tokens"])["rounds"], gm))
                a = M.alpha_hat(tot["accepted"], tot["tested"])
                rate = tot["tokens"] / tot["seconds"] if tot["seconds"] else float("nan")
                rates.append(rate)
                print(f"  {dn}, γ = {gm:>2}: {rate:.1f} tokens/s = {BOLD}{rate / (bs / bt):.2f}×{RESET}   alpha {num(a)}   Thm 3.8 {num(M.speedup(gm, a, c) if a is not None else None)}")
            est += len(full) * len(g["temperatures"]) * len(g["gammas"]) * g["max_tokens"] / (sum(rates) / len(rates))
            est += g["draft_step_prompts"] * len(g["temperatures"]) * g["max_tokens"] / (ds / dt_)
        print(f"  {GREY}rough estimate for the full grid of this target ({len(full)} prompts): {est / 60:.0f} minutes, plus loading and prefill{RESET}")
        total += est
        lab.close()
    print(f"\n  rough total: {BOLD}{total / 60:.0f} minutes{RESET}   {GREY}(fewer prompts: per_kind in demo_config.toml){RESET}")


def cmd_show(args, cfg):
    s = cfg["show"]
    tn, dn, gm, T = args.target or s["target"], args.draft or s["draft"], int(args.gamma or s["gamma"]), float(s["temperature"])
    g = sweep_cfg(cfg, args, tn)
    allp = load_prompts(10 ** 6, 10 ** 6)
    prompt = next((p for p in allp if p["id"] == (args.prompt or s["prompt"])), allp[0])
    banner(f"{cfg['demo']['title']}\n(b) live, optional: one prompt, without and with a draft")
    note(f"載入 {tn} 與 {dn} …")
    lab = Lab(cfg, args, tn, [dn])
    ids = lab.encode(prompt)
    lab.plain(lab.target, ids, 0.0, 0, 8)
    lab.plain(lab.drafts[dn], ids, 0.0, 0, 8)
    print(f"\n  {BOLD}prompt{RESET} ({prompt['id']}): {prompt['text']}")

    def printer():
        det = None if lab.tok is None else lab.tok.detokenizer
        if det is not None:
            det.reset()

        def on(tok: int, from_draft: bool) -> None:
            if det is None:
                seg = f"{tok} "
            else:
                det.add_token(tok)
                seg = det.last_segment
            sys.stdout.write((GREEN + seg + RESET) if from_draft else seg)
            sys.stdout.flush()
        return on

    if not pause(f"{lab.info[tn]['label']} alone"):
        return
    print("  ", end="")
    base = lab.plain(lab.target, ids, T, 1, g["max_tokens"], printer())
    brate = base["steps"] / base["seconds"] if base["seconds"] else float("nan")
    print(f"\n\n  {BOLD}{brate:.1f} tokens/s{RESET}")
    if not pause(f"with {lab.info[dn]['label']} drafting {gm} tokens per round — green: drafted and kept"):
        return
    print("  ", end="")
    r = lab.spec(lab.drafts[dn], ids, gm, T, 1, g["max_tokens"], printer())
    x = tally(r["rounds"], gm)
    rate = x["tokens"] / x["seconds"] if x["seconds"] else float("nan")
    pos = first_divergence(base["tokens"], r["tokens"])
    print(f"\n\n  {BOLD}{rate:.1f} tokens/s = {rate / brate:.2f}×{RESET}   kept {x['accepted']} of {x['tested']} drafts checked (alpha {num(M.alpha_hat(x['accepted'], x['tested']))})"
          f"   {(x['accepted'] + x['rounds']) / max(x['rounds'], 1):.2f} tokens per round")
    if T == 0:
        print(f"  same tokens as without the draft: {'yes' if pos is None else 'no — they differ from token ' + str(pos)}")
    lab.close()


def cmd_test(args, cfg):
    import numpy as np
    import tests as T
    import speculative_final
    fails = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if cond else 1
        print(f"  {GREEN + 'ok  ' if cond else RED + 'FAIL'}{RESET} {label}" + (f"   {GREY}{detail}{RESET}" if detail else ""))

    banner("test — the five tests of (a), on the reference implementation")
    live, ref = cfg["live"], cfg["reference"]
    rows, state = {}, {}
    for k in (1, 2, 3, 4, 5):
        rows[str(k)] = T.run(k, speculative_final.speculative_step, live, state)
        check(f"test {k}: " + T.TITLES[k].format(n=live["n_draws"]), rows[str(k)]["pass"])
    try:
        import speculative_live
        s2 = {}
        got = [T.run(k, speculative_live.speculative_step, live, s2)["pass"] for k in (1, 2, 5)]
        check("speculative_live.py (written): tests 1, 2, 5", all(got))
    except NotImplementedError:
        note("  speculative_live.py 還是骨架（課堂上才寫）：略過")
    rng = np.random.default_rng(1)
    p, q = np.array([0.7, 0.3, 0.0]), np.array([0.0, 0.5, 0.5])
    outs = [speculative_final.speculative_step(p, q, rng)[0] for _ in range(20000)]
    emp = [outs.count(i) / len(outs) for i in range(3)]
    check("zeros on either side: output still p, and a token with p = 0 never comes out", emp[2] == 0 and abs(emp[0] - 0.7) < 0.02, f"{emp}")

    banner("test — the arithmetic against the slides' numbers")
    check("Σ min(p, q) and Σ p·q on the example of p42", abs(M.beta_min(live["p"], live["q"]) - ref["sum_min"]) < 1e-12 and abs(M.beta_match(live["p"], live["q"]) - ref["sum_pq"]) < 1e-12,
          f"{M.beta_min(live['p'], live['q']):.2f}, {M.beta_match(live['p'], live['q']):.2f}")
    w = M.wrong_redraw(live["p"], live["q"])
    check("what the wrong redraw outputs", all(abs(a - b) < 1e-12 for a, b in zip(w, ref["wrong_redraw"])) and abs(sum(w) - 1) < 1e-12, f"{[round(x, 4) for x in w]}")
    check("the residual is a distribution", abs(sum(M.residual(live["p"], live["q"])) - 1) < 1e-12, f"{[round(x, 4) for x in M.residual(live['p'], live['q'])]}")
    e = ref["exercise"]
    got = [M.speedup(g, e["alpha"], e["c"]) for g in (4, 8, 16)]
    check("p48's exercise: S(4), S(8), S(16)", all(abs(a - b) < 0.005 for a, b in zip(got, (e["S4"], e["S8"], e["S16"]))), ", ".join(f"{x:.3f}" for x in got))
    for a, c, bg, bs in ref["curves"]:
        g, s = M.best_gamma(a, c, range(0, 17))
        check(f"p48's curve alpha = {a}, c = {c}: best γ and S", g == bg and abs(s - bs) < 0.005, f"γ = {g}, S = {s:.3f}")
    check("Ω at alpha = 1 is γ + 1; kept share equals alpha at γ = 1", M.omega(5, 1.0) == 6.0 and abs(M.kept_share(1, 0.37) - 0.37) < 1e-12)
    pts = {1: 30.0, 2: 30.0, 4: 33.0, 8: 36.0, 16: 60.0}
    r = M.ratio_from(pts)
    check("interpolating a step-time curve", abs(r(1) - 1) < 1e-12 and abs(r(4) - 1.1) < 1e-12 and 1.2 < r(13) < 2.0 and r(64) == 2.0, f"r(13) = {r(13):.3f}")
    rounds = [{"acc": 2, "proposed": 3, "t": 1.0, "eos": False, "cut": False}, {"acc": 3, "proposed": 3, "t": 2.0, "eos": False, "cut": False},
              {"acc": 0, "proposed": 3, "t": 3.0, "eos": False, "cut": False}, {"acc": 1, "proposed": 2, "t": 4.0, "eos": False, "cut": True}]
    x = tally(rounds, 3)
    check("tallying rounds: accepted, checked, tokens and seconds without the first round",
          (x["rounds"], x["accepted"], x["tested"], x["proposed"], x["tokens"], x["seconds"]) == (3, 5, 7, 9, 5, 2.0) and x["acc_at"] == [2, 2, 1] and x["tested_at"] == [3, 2, 2], f"{x}")
    d1 = d1_points(cfg, args, "bf16")
    if d1:
        check("demo 1's record is readable: step times at B = 1 … 16", all(k in d1 for k in (1, 2, 4, 8, 16)), ", ".join(f"B={k}: {d1[k]:.1f} ms" for k in sorted(d1)[:5]))
    else:
        note("  demo 1 的彩排紀錄讀不到（'+ demo 1' 那一欄會是空的）")

    try:
        import mlx.core as mx
    except Exception as e:
        note(f"\n  mlx 不能用（{type(e).__name__}）：量法的檢查略過")
        sys.exit(1 if fails else 0)
    banner("test — the way of measuring, on tiny random models")
    args.tiny = True
    g = sweep_cfg(cfg, args, "bf16")
    lab = Lab(cfg, args, "bf16", g["drafts"])
    prompts = load_prompts(2, 1)
    ids = [lab.encode(p) for p in prompts]
    n_tok = 40
    base = [lab.plain(lab.target, x, 0.0, 0, n_tok) for x in ids]
    check("plain greedy decoding is repeatable", all(lab.plain(lab.target, x, 0.0, 0, n_tok)["tokens"] == b["tokens"] for x, b in zip(ids, base)))
    for gm in (1, 3, 5):
        rs = [lab.spec(lab.target, x, gm, 0.0, 0, n_tok) for x in ids]
        xs = [tally(r["rounds"], gm) for r in rs]
        check(f"the target drafting for itself, γ = {gm}: everything checked is kept; output = plain greedy",
              all(x["accepted"] == x["tested"] for x in xs) and all(r["tokens"] == b["tokens"] for r, b in zip(rs, base)),
              f"kept {sum(x['accepted'] for x in xs)} of {sum(x['tested'] for x in xs)}")
    dn = g["drafts"][0]
    for gm in (1, 4):
        rs = [lab.spec(lab.drafts[dn], x, gm, 0.0, 0, n_tok) for x in ids]
        same = sum(r["tokens"] == b["tokens"] for r, b in zip(rs, base))
        tot = {}
        for r in rs:
            add(tot, tally(r["rounds"], gm))
            assert sum(rr["acc"] + (0 if rr["cut"] else 1) for rr in r["rounds"]) == len(r["tokens"]), "round bookkeeping"
        check(f"a different draft ({dn}), γ = {gm}, T = 0: some drafts rejected, output still = plain greedy",
              same == len(ids) and 0 < tot["accepted"] < tot["tested"], f"kept {tot['accepted']} of {tot['tested']}; {same}/{len(ids)} identical")
    # T > 0：量到的 alpha 應該在 sum p·q 附近（獨立取樣、相同才留），明顯低於 sum min
    T_, gm, tot, ov = 0.6, 2, {}, {"n": 0, "sum_min": 0.0, "sum_pq": 0.0, "agree": 0.0}
    many = [lab.encode({"id": f"t{i}", "text": ""}) for i in range(24)]
    for i, x in enumerate(many):
        add(tot, tally(lab.spec(lab.drafts[dn], x, gm, T_, 100 + i, 64)["rounds"], gm))
        b = lab.plain(lab.target, x, T_, 500 + i, 64)
        add(ov, lab.overlap(lab.logits_along(lab.target, x, b["tokens"]), lab.logits_along(lab.drafts[dn], x, b["tokens"]), T_))
    a, spq, smin = M.alpha_hat(tot["accepted"], tot["tested"]), ov["sum_pq"] / ov["n"], ov["sum_min"] / ov["n"]
    tol = 4 * math.sqrt(max(spq * (1 - spq), 1e-9) / tot["tested"]) + 0.02
    check("T = 0.6: measured alpha is near Σ p·q from the logits, and Σ p·q ≤ Σ min", abs(a - spq) < tol and spq <= smin + 1e-6,
          f"alpha {a:.3f} over {tot['tested']} checks; Σ p·q {spq:.3f}; Σ min {smin:.3f}")
    v = lab.verify_times(ids[0] + base[0]["tokens"][:8], base[0]["tokens"][8:14], [1, 3, 6], 2)
    check("timing a verification pass leaves numbers for each k", all(v[str(k)] > 0 for k in (1, 3, 6)), ", ".join(f"k={k}: {v[str(k)]:.2f} ms" for k in (1, 3, 6)))
    gap = lab.top2_gap(lab.logits_along(lab.target, ids[0], base[0]["tokens"]))
    check("top-2 gap along the text: one non-negative number per token", len(gap) == len(base[0]["tokens"]) and min(gap) >= 0)
    print(f"\n  {(GREEN + 'all passed') if not fails else (RED + str(fails) + ' failed')}{RESET}")
    sys.exit(1 if fails else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--final", "--ref", dest="final", action="store_true", help="live：用參考實作 speculative_final.py")
    p.add_argument("--target", default=None, help="demo_config.toml 的 [sweep.*] 名稱（bf16 / q4）；不給就是全部")
    p.add_argument("--part", choices=["a", "base", "sweep"], help="rehearse 只做一段（sweep 會先補 base 缺的）")
    p.add_argument("--force", action="store_true", help="rehearse：做過的也重跑")
    p.add_argument("--note", default=None, help="彩排當下的狀態（電源、其他程式），記進紀錄")
    p.add_argument("--draft", default=None, help="show：哪一個 draft")
    p.add_argument("--gamma", default=None, help="show：一輪草擬幾個")
    p.add_argument("--prompt", default=None, help="show：prompts.toml 的 id")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "test", "inspect", "probe", "rehearse", "live", "replay", "replay-a", "show"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "test": cmd_test, "inspect": cmd_inspect, "probe": cmd_probe, "rehearse": cmd_rehearse, "live": cmd_live,
     "replay": cmd_replay, "replay-a": cmd_replay_a, "show": cmd_show}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
