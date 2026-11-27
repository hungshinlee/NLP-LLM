#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 1：兩片屋頂——在這台機器上把 roofline 量出來（投影片 p15–p17、p25）。

    ./present.sh test           不載模型：算術對投影片的數；有 mlx 時另用亂數小模型驗量法（幾秒）
    ./present.sh inspect        課前：版本、機器、兩個模型的形狀與 bytes、KV 的 dtype、4-bit 檔的量化參數；
                                我們的 decode 迴圈對 mlx-lm 自己的 generate 各量一次
    ./present.sh rehearse       彩排全套：(a) → (b) → (c)，兩個模型；做過的點跳過
                                （--part a|b|c、--model bf16|q4、--force、--note "接電源、其他程式都關了"）
    ./present.sh live           課堂 (a)：Enter 三次——prefill、decode、並排
    ./present.sh live-batch 16  課堂（可選）：現場補 (b) 的一個點
    ./present.sh replay         (a) 的彩排紀錄（不載模型）
    ./present.sh replay-batch   (b)(c) 的彩排紀錄，Enter 三段（不載模型）
    ./present.sh tiny [子命令]  亂數權重的小模型走真的流程（不下載；寫到 runs/smoke/）

量法：
  prefill  亂數 token id、長度 L，分塊送進模型，只 eval cache（與 mlx-lm 的 prefill 相同，logits 不算）。
           tokens/s = L / 秒。
  decode   先 prefill 出 context，再一步一個 token、greedy、不掛 logits processor；
           下一步的圖先送出去再等這一步（與 mlx-lm 的 generate_step 相同的排法）。
           一批 B 條、每條的 context 都是 L：B × L 不大時各條用不同的 prompt 一起 prefill，
           太大時只 prefill 一條、把它的 KV 複製成 B 份（表上分別記 distinct／tiled）。
           KV 的緩衝區先配到夠長，計時的那一段不會重新配置。
  每個點先暖機、計時三次、取中位數；每次都從同一個 context 重新開始（把 cache 的 offset 退回去）。
  換算  FLOP/s 與 bytes/s 的式子在 roofs.py；兩種數法（deck／strict）都存。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）。replay 與 test 的算術只用標準函式庫。Python 3.11+。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, append_jsonl, banner, device_info,  # noqa: E402
                    locked_revision, note, now, peak_memory_gb, reset_peak_memory, versions)
import roofs as R  # noqa: E402

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]   # 離線時 huggingface_hub 會核對 snapshot 是否完整
GB = 1e9
STEP = 256          # mlx-lm 的 KVCache 以 256 個 token 為一步配置


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def out_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "rehearsal")


def grids(cfg: dict, args) -> dict:
    """三段的網格；--tiny 換成小的。"""
    a, b, c, t = cfg["a"], cfg["b"], cfg["c"], cfg["tiny"]
    if args.tiny:
        return {"a_lengths": t["prefill_lengths"], "a_prompt": t["decode_prompt_tokens"], "a_tokens": t["decode_tokens"],
                "live_prefill": t["live_prefill_tokens"], "live_decode": t["live_decode_tokens"],
                "b_batches": t["b_batch_sizes"], "b_prompt": t["b_prompt_tokens"], "b_tokens": t["b_decode_tokens"],
                "c_lengths": t["c_context_lengths"], "c_batches": t["c_batch_sizes"], "c_tokens": t["c_decode_tokens"],
                "kv_bits": c["kv_bits"], "kv_group": 32, "budget": t["distinct_token_budget"]}
    return {"a_lengths": a["prefill_lengths"], "a_prompt": a["decode_prompt_tokens"], "a_tokens": a["decode_tokens"],
            "live_prefill": a["live_prefill_tokens"], "live_decode": a["live_decode_tokens"],
            "b_batches": b["batch_sizes"], "b_prompt": b["prompt_tokens"], "b_tokens": b["decode_tokens"],
            "c_lengths": c["context_lengths"], "c_batches": c["batch_sizes"], "c_tokens": c["decode_tokens"],
            "kv_bits": c["kv_bits"], "kv_group": c["kv_group_size"], "budget": cfg["demo"]["distinct_token_budget"]}


# ── 小工具 ───────────────────────────────────────────────────────
def sci(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    e = int(math.floor(math.log10(abs(x)))) if x else 0
    return f"{x / 10 ** e:.2f}e{e}"


def num(x, digits=1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "∞"
    return f"{x:,.{digits}f}"


def pause(msg: str) -> bool:
    """Enter 繼續、q 離開；不是終端機時不等。"""
    if not sys.stdin.isatty():
        print(f"  {DIM}[{msg}]{RESET}")
        return True
    key = input(f"  {DIM}[Enter] {msg}   q = quit{RESET} ").strip().lower()
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


# ── 模型 ─────────────────────────────────────────────────────────
def shape_from_config(raw: dict) -> dict:
    nh = raw["num_attention_heads"]
    return {"d": raw["hidden_size"], "d_ff": raw["intermediate_size"], "n_layer": raw["num_hidden_layers"], "n_head": nh,
            "n_kv": raw.get("num_key_value_heads") or nh, "d_head": raw.get("head_dim") or raw["hidden_size"] // nh,
            "V": raw["vocab_size"], "tied": bool(raw.get("tie_word_embeddings", False)),
            "max_position_embeddings": raw.get("max_position_embeddings")}


class Bench:
    """載一個模型，量 prefill 與 decode。"""

    def __init__(self, cfg: dict, args, name: str):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        self.mx, self.cfg, self.name, self.tiny = mx, cfg, name, bool(args.tiny)
        d = cfg["demo"]
        self.seed, self.warmup, self.repeats = int(d["seed"]), int(d["warmup"]), int(d["repeats"])
        self.prefill_step, self.warm_steps = int(d["prefill_step_size"]), int(d["decode_warm_steps"])
        self.limit = float(d["memory_limit_gb"]) * GB
        t0 = time.time()
        if self.tiny:
            from mlx_lm.models import qwen3
            mx.random.seed(self.seed)
            raw = dict(model_type="qwen3", hidden_size=128, num_hidden_layers=2, intermediate_size=256, num_attention_heads=4,
                       rms_norm_eps=1e-6, vocab_size=512, num_key_value_heads=2, max_position_embeddings=4096, rope_theta=1e6,
                       head_dim=32, tie_word_embeddings=False)
            self.model, self.tok, self.path = qwen3.Model(qwen3.ModelArgs(**raw)), None, None
            self.label, self.repo, self.revision = f"tiny random Qwen3 ({name})", None, None
            self.quantization = None
            if name == "q4":                                             # 小模型也量化一份，讓兩個「模型」真的不同
                import mlx.nn as nn
                nn.quantize(self.model, group_size=64, bits=4)
                self.quantization = {"group_size": 64, "bits": 4}
        else:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            m = cfg["models"][name]
            self.label, self.repo = m["label"], m["repo"]
            self.revision = locked_revision(m["repo"], m.get("revision"))
            self.path = snapshot_download(m["repo"], revision=self.revision, local_files_only=not args.online,
                                          ignore_patterns=NON_MODEL_FILES)
            self.model, self.tok = load(self.path)
            raw = json.loads((Path(self.path) / "config.json").read_text(encoding="utf-8"))
            self.quantization = raw.get("quantization")
        mx.eval(self.model.parameters())
        self.c = shape_from_config(raw)
        self.n = R.count_params(self.c)
        self.W = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        emb = sum(v.nbytes for _, v in tree_flatten(self.model.model.embed_tokens.parameters()))
        self.W_embed = 0 if self.c["tied"] else emb          # tied 時那張表就是 LM head，每一步都整張讀
        self.load_seconds = time.time() - t0
        probe = self.prefill(self.rand_ids(1, 8, 0))
        self.kv_dtype, self.b_kv = str(probe[0].keys.dtype), int(probe[0].keys.dtype.size)
        self.kvt = R.kv_bytes_per_token(self.c, self.b_kv)
        del probe
        mx.clear_cache()

    def info(self) -> dict:
        return {"label": self.label, "repo": self.repo, "revision": self.revision, "config": self.c, "counts": self.n,
                "weight_bytes": self.W, "embedding_bytes": self.W_embed, "bits_per_weight": 8.0 * self.W / self.n["N"],
                "quantization": self.quantization, "kv_dtype": self.kv_dtype, "kv_bytes_per_token": self.kvt,
                "load_seconds": round(self.load_seconds, 2)}

    def rand_ids(self, B: int, L: int, salt: int) -> list[list[int]]:
        rng = random.Random(self.seed * 1000003 + salt)
        V = self.c["V"]
        lo, hi = (8, V - 8) if V < 4096 else (1000, V - 1000)
        return [[rng.randrange(lo, hi) for _ in range(L)] for _ in range(B)]

    # ── prefill ──
    def prefill(self, ids: list[list[int]]):
        from mlx_lm.models.cache import make_prompt_cache
        mx = self.mx
        cache = make_prompt_cache(self.model)
        arr = mx.array(ids)
        for s in range(0, arr.shape[1], self.prefill_step):
            self.model(arr[:, s:s + self.prefill_step], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        return cache

    def measure_prefill(self, L: int) -> dict:
        mx = self.mx
        ids = self.rand_ids(1, L, L)
        for _ in range(self.warmup):
            del_me = self.prefill(ids)
            del del_me
        mx.clear_cache()
        reset_peak_memory()
        secs = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            cache = self.prefill(ids)
            secs.append(time.perf_counter() - t0)
            del cache
            mx.clear_cache()
        s = R.median(secs)
        return {"L": L, "seconds": s, "tok_per_s": L / s, "all_seconds": secs, "peak_gb": peak_memory_gb(), "date": now()}

    # ── decode ──
    def reserve(self, cache, B: int, total: int) -> None:
        """把每一層的 KV 緩衝區配到 total 個 token（256 的倍數），需要時沿 batch 複製成 B 份。"""
        mx = self.mx
        cap = -(-total // STEP) * STEP
        for c in cache:
            k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
            pad = cap - c.offset
            if pad > 0:
                k = mx.concatenate([k, mx.zeros((k.shape[0], k.shape[1], pad, k.shape[3]), k.dtype)], axis=2)
                v = mx.concatenate([v, mx.zeros((v.shape[0], v.shape[1], pad, v.shape[3]), v.dtype)], axis=2)
            if B > k.shape[0]:
                k, v = mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0)
            c.keys, c.values = k, v
            mx.eval(k, v)
        mx.clear_cache()

    def context(self, B: int, L: int, n_new: int, budget: int, kv_bits: int = 0, kv_group: int = 64):
        """B 條序列、各有 L 個 token 的 context（最後一個 token 留給第一步）。回傳 cache、第一步的輸入、做法。"""
        mx = self.mx
        distinct = B == 1 or B * L <= budget
        ids = self.rand_ids(B if distinct else 1, L, 7919 * L + (B if distinct else 1))   # tiled：就是 B = 1 的那個 prompt
        cache = self.prefill([row[:-1] for row in ids]) if L > 1 else None
        if cache is None:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(self.model)
            self.model(mx.array([[row[0]] for row in ids]), cache=cache)   # L = 1：沒有東西可以 prefill，先走一步
            mx.eval([c.state for c in cache])
        last = [row[-1] for row in ids] if distinct else [ids[0][-1]] * B
        self.reserve(cache, B, cache[0].offset + n_new)
        if kv_bits:
            for i, c in enumerate(cache):
                cache[i] = c.to_quantized(group_size=kv_group, bits=kv_bits)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        return cache, mx.array(last), ("distinct" if distinct else "tiled")

    def decode(self, cache, y0, n: int):
        """n 步 greedy。回傳每一步結束的時間戳（n + 1 個）與生成的 token（n 個 (B,) 的陣列）。"""
        mx, model = self.mx, self.model

        def step(y):
            return mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)

        y = step(y0)
        mx.async_eval(y)
        stamps, toks = [time.perf_counter()], []
        for i in range(n):
            nxt = step(y) if i < n - 1 else None
            if nxt is not None:
                mx.async_eval(nxt)
            mx.eval(y)
            stamps.append(time.perf_counter())
            toks.append(y)
            y = nxt
        return stamps, toks

    @staticmethod
    def rewind(cache, base: int) -> None:
        for c in cache:
            c.trim(c.offset - base)

    def measure_decode(self, B: int, L: int, n_gen: int, budget: int, kv_bits: int = 0, kv_group: int = 64) -> dict:
        mx = self.mx
        n_new = max(self.warm_steps, n_gen) + 2
        cap = -(-(L + n_new) // STEP) * STEP
        est = self.W + 1.1 * B * self.kvt * cap
        if est > self.limit:
            return {"B": B, "L_start": L, "n_gen": n_gen, "kv_bits": kv_bits, "skipped": f"estimated {est / GB:.1f} GB > limit {self.limit / GB:.0f} GB",
                    "date": now()}
        mx.clear_cache()
        reset_peak_memory()
        cache, y0, mode = self.context(B, L, n_new, budget, kv_bits, kv_group)
        base = cache[0].offset
        kv_nbytes = sum(int(c.nbytes) for c in cache)
        if self.warm_steps:
            self.decode(cache, y0, self.warm_steps)
            self.rewind(cache, base)
        rates, step_ms, first = [], [], None
        for _ in range(self.repeats):
            stamps, toks = self.decode(cache, y0, n_gen)
            # 第一個時間戳是第一步送出去的時刻；n_gen 步都算進去。
            rates.append(B * n_gen / (stamps[-1] - stamps[0]))
            step_ms.append(1e3 * R.median([b - a for a, b in zip(stamps[1:-1], stamps[2:])] or [stamps[-1] - stamps[0]]))
            if first is None:
                first = [int(t[0].item()) for t in toks[:8]]
            del toks
            self.rewind(cache, base)
        row = {"B": B, "L_start": L, "n_gen": n_gen, "kv_bits": kv_bits, "context": mode,
               "tok_per_s": R.median(rates), "per_seq_tok_per_s": R.median(rates) / B, "step_ms": R.median(step_ms),
               "all_tok_per_s": rates, "kv_allocated_bytes": kv_nbytes, "peak_gb": peak_memory_gb(),
               "first_tokens_row0": first, "date": now()}
        del cache, y0
        mx.clear_cache()
        return row

    def safe_decode(self, *a, **kw) -> dict:
        try:
            return self.measure_decode(*a, **kw)
        except Exception as e:                                           # 記下來、往下跑；表上會標 error
            self.mx.clear_cache()
            return {"B": a[0], "L_start": a[1], "n_gen": a[2], "kv_bits": kw.get("kv_bits", 0), "error": repr(e)[:300], "date": now()}


# ── 紀錄檔 ───────────────────────────────────────────────────────
def rec_path(args) -> Path:
    return out_dir(args) / "roofs.json"


def load_rec(args) -> dict | None:
    p = rec_path(args)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_rec(args, rec: dict) -> None:
    rec["updated"] = now()
    for m in rec["models"].values():
        m["summary"] = summarize(m)
    p = rec_path(args)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def ok(row: dict | None) -> bool:
    return bool(row) and "tok_per_s" in row


def summarize(m: dict) -> dict | None:
    """(a) 做完之後：兩種數法的 F_max、有效頻寬、I*、B*、L_c。"""
    a, i = m.get("a") or {}, m["info"]
    pre = [r for r in (a.get("prefill") or {}).values() if ok(r)]
    dec = a.get("decode")
    if not pre or not ok(dec):
        return None
    return {k: R.roofs_from_measurement(i["config"], i["counts"], i["weight_bytes"], i["embedding_bytes"], i["kv_bytes_per_token"], pre, dec, k)
            for k in ("deck", "strict")}


def model_args(m: dict, count: str = "deck"):
    i = m["info"]
    return i["config"], i["counts"], i["weight_bytes"], i["embedding_bytes"], i["kv_bytes_per_token"]


# ── 畫面 ─────────────────────────────────────────────────────────
def print_model_line(i: dict) -> None:
    c = i["config"]
    print(f"\n  {BOLD}{i['label']}{RESET}   N = {i['counts']['N']:,}   weights {i['weight_bytes'] / GB:.2f} GB"
          f" ({i['bits_per_weight']:.2f} bits/weight)   KV {i['kv_bytes_per_token']:,} bytes/token ({i['kv_dtype'].split('.')[-1]})")
    print(f"  {GREY}{c['n_layer']} layers, d = {c['d']}, {c['n_head']} heads / {c['n_kv']} KV heads, d_head = {c['d_head']}"
          f"{'' if not i.get('repo') else '   ' + i['repo'] + '@' + str(i.get('revision'))[:10]}{RESET}")


def print_prefill_row(i: dict, r: dict, ref: dict, flag: str = "") -> None:
    f = r["tok_per_s"] * R.flops_prefill_per_token(i["config"], i["counts"], r["L"])
    print(f"  {r['L']:>8,}{r['tok_per_s']:>12,.0f}{1e3 / r['tok_per_s']:>12.3f}{sci(f):>12}"
          f"{f / ref['fmax_w3_flops_per_s']:>11.0%}   {flag}")


PRE_HDR = f"  {'L_in':>8}{'tokens/s':>12}{'ms/token':>12}{'FLOP/s':>12}{'vs p13':>11}"
DEC_HDR = f"  {'B':>8}{'tokens/s':>12}{'ms/token':>12}{'bytes/s':>12}{'of 614 GB/s':>13}"


def print_decode_row(i: dict, r: dict, ref: dict, flag: str = "") -> None:
    L = r["L_start"] + r["n_gen"] / 2
    by = r["tok_per_s"] / r["B"] * (i["weight_bytes"] + r["B"] * i["kv_bytes_per_token"] * L)
    print(f"  {r['B']:>8}{r['tok_per_s']:>12,.1f}{1e3 * r['B'] / r['tok_per_s']:>12.2f}{by / GB:>9.0f} GB"
          f"{by / ref['bw_spec_bytes_per_s']:>13.0%}   {flag}")


def print_roofs(m: dict, ref: dict) -> None:
    s = m.get("summary")
    if not s:
        return
    d, st = s["deck"], s["strict"]
    print(f"\n  {BOLD}measured roofs{RESET}:  F_max = {BOLD}{sci(d['F_max'])} FLOP/s{RESET} (at L_in = {d['F_max_at_L']:,})"
          f"     BW_eff = {BOLD}{d['BW_eff'] / GB:.0f} GB/s{RESET}     I* = F_max / BW_eff = {BOLD}{d['I_star']:.0f} FLOP/byte{RESET}")
    print(f"  from these:      B* = {BOLD}{num(d['B_star'], 0)}{RESET} (short context)     L_c = {BOLD}{num(d['L_c'], 0)} tokens{RESET} (B → ∞)")
    print(f"  {GREY}against the slide's inputs: F_max is {d['F_max'] / ref['fmax_w3_flops_per_s']:.2f}× p13's lower bound ({sci(ref['fmax_w3_flops_per_s'])}); "
          f"BW_eff is {d['BW_eff'] / ref['bw_spec_bytes_per_s']:.0%} of the specified {ref['bw_spec_bytes_per_s'] / GB:.0f} GB/s{RESET}")
    print(f"  {GREY}strict count (embedding table not read, no LM head in prefill): F_max {sci(st['F_max'])} (×{st['F_max'] / d['F_max']:.2f}), "
          f"BW_eff {st['BW_eff'] / GB:.0f} GB/s (×{st['BW_eff'] / d['BW_eff']:.2f}), I* {st['I_star']:.0f}, B* {num(st['B_star'], 0)}, L_c {num(st['L_c'], 0)}{RESET}")


def show_a(rec: dict, names: list[str]) -> None:
    ref = rec["reference"]
    for name in names:
        m = rec["models"].get(name)
        if not m or not m.get("a"):
            continue
        i, a = m["info"], m["a"]
        print_model_line(i)
        print(f"\n  {BOLD}prefill{RESET} (read)\n{PRE_HDR}")
        for r in sorted((r for r in a.get("prefill", {}).values() if ok(r)), key=lambda r: r["L"]):
            print_prefill_row(i, r, ref)
        if ok(a.get("decode")):
            print(f"\n  {BOLD}decode{RESET} (write), {a['decode']['n_gen']} tokens after a {a['decode']['L_start']}-token prompt\n{DEC_HDR}")
            print_decode_row(i, a["decode"], ref)
            pre = {r["L"]: r for r in a.get("prefill", {}).values() if ok(r)}
            near = min(pre, key=lambda L: abs(math.log(L / 1000.0))) if pre else None
            if near:
                print(f"\n  per token, writing is {BOLD}{pre[near]['tok_per_s'] / a['decode']['tok_per_s']:,.0f}×{RESET} slower than reading"
                      f" ({near:,}-token prefill against B = 1 decode)")
        print_roofs(m, ref)


def bar(x: float, lo: float, hi: float, width: int = 28) -> str:
    if not x or x <= 0:
        return ""
    f = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) if hi > lo else 1.0
    return "█" * max(1, int(round(f * width)))


def show_b(rec: dict, names: list[str]) -> None:
    ref = rec["reference"]
    allr = [r for n in names for r in (rec["models"].get(n, {}).get("b") or {}).values() if ok(r)]
    if not allr:
        print(f"  {AMBER}(b) has not been rehearsed{RESET}")
        return
    lo, hi = min(r["tok_per_s"] for r in allr) / 1.5, max(r["tok_per_s"] for r in allr)
    for name in names:
        m = rec["models"].get(name)
        rows = sorted((r for r in ((m or {}).get("b") or {}).values()), key=lambda r: r["B"])
        if not rows:
            continue
        i, s = m["info"], m.get("summary")
        print_model_line(i)
        good = [r for r in rows if ok(r)]
        base = next((r for r in good if r["B"] == 1), None)
        print(f"\n  {'B':>5}{'tokens/s':>11}{'× B=1':>8}{'× prev':>8}{'per seq':>9}{'ms/step':>9}{'bound':>10}{'of bound':>9}   (log scale)")
        prev = None
        for r in rows:
            if not ok(r):
                print(f"  {r['B']:>5}   {AMBER}{r.get('skipped') or r.get('error')}{RESET}")
                continue
            bound, side = (None, "")
            if s:
                L = r["L_start"] + r["n_gen"] / 2
                bound, side = R.throughput_upper(r["B"], L, *model_args(m), s["deck"]["F_max"], s["deck"]["BW_eff"])
            gain = "%.2f" % (r["tok_per_s"] / prev["tok_per_s"]) if prev else "—"
            share = "%.0f%%" % (100 * r["tok_per_s"] / bound) if bound else "—"
            print(f"  {r['B']:>5}{r['tok_per_s']:>11,.1f}{(r['tok_per_s'] / base['tok_per_s'] if base else float('nan')):>8.1f}"
                  f"{gain:>8}{r['per_seq_tok_per_s']:>9.1f}{r['step_ms']:>9.1f}"
                  f"{num(bound, 0):>10}{share:>9}   {bar(r['tok_per_s'], lo, hi)} {GREY}{side}{RESET}")
            prev = r
        if s and good:
            L = good[0]["L_start"] + good[0]["n_gen"] / 2
            k = R.kink_B(L, *model_args(m), s["deck"]["F_max"], s["deck"]["BW_eff"])
            print(f"  {GREY}'× prev' is the gain from the last step in B (2.00 = a straight line when B doubles). "
                  f"The bound uses this model's F_max and BW_eff from (a), so (a)'s B = 1 point sits on it by construction. "
                  f"The two roofs meet at B = {num(k, 0)}.{RESET}")


def show_c(rec: dict, names: list[str]) -> None:
    any_rows = False
    for name in names:
        m = rec["models"].get(name)
        rows = [r for r in ((m or {}).get("c") or {}).values()]
        if not rows:
            continue
        any_rows = True
        i, s = m["info"], m.get("summary")
        print_model_line(i)
        Ls, Bs = sorted({r["L_start"] for r in rows}), sorted({r["B"] for r in rows})
        by = {(r["L_start"], r["B"]): r for r in rows}
        short = {r["B"]: r for r in ((m.get("b") or {}).values()) if ok(r)}
        print(f"\n  total tokens/s (× the B = 1 cell of the same row)\n  {'context L':>12}" + "".join(f"{'B = ' + str(B):>20}" for B in Bs) + f"{'KV at B = ' + str(Bs[-1]):>16}")
        if short and all(B in short for B in Bs):
            print(f"  {'short (b)':>12}" + "".join(f"{short[B]['tok_per_s']:>12,.1f} ({short[B]['tok_per_s'] / short[Bs[0]]['tok_per_s']:>4.1f}×)" for B in Bs))
        for L in Ls:
            cells, one = [], by.get((L, Bs[0]))
            for B in Bs:
                r = by.get((L, B))
                if ok(r):
                    cells.append(f"{r['tok_per_s']:>12,.1f} ({(r['tok_per_s'] / one['tok_per_s'] if ok(one) else float('nan')):>4.1f}×)")
                else:
                    cells.append(f"{'skipped' if r and r.get('skipped') else 'error' if r else '—':>20}")
            last = by.get((L, Bs[-1]))
            kv = f"{Bs[-1] * i['kv_bytes_per_token'] * L / GB:>13.1f} GB"
            print(f"  {L:>12,}" + "".join(cells) + kv + (f"   {GREY}{last.get('context', '')}{RESET}" if last else ""))
        if s:
            print(f"  {GREY}predicted from (a): past L_c = {num(s['deck']['L_c'], 0)} tokens no batch size reaches the compute roof{RESET}")
    if not any_rows:
        print(f"  {AMBER}(c) has not been rehearsed{RESET}")


def show_kvq(rec: dict, names: list[str]) -> None:
    for name in names:
        m = rec["models"].get(name)
        rows = [r for r in ((m or {}).get("kvq") or {}).values()]
        if not rows:
            continue
        print_model_line(m["info"])
        print(f"\n  B = 1, the KV stored at fewer bits\n  {'context L':>12}{'KV bits':>10}{'tokens/s':>12}{'ms/step':>10}{'KV allocated':>15}{'× unquantized':>15}")
        base = {r["L_start"]: r for r in rows if ok(r) and not r["kv_bits"]}
        for r in sorted(rows, key=lambda r: (r["L_start"], -(r["kv_bits"] or 99))):
            if not ok(r):
                print(f"  {r['L_start']:>12,}{r['kv_bits'] or 'as is':>10}   {AMBER}{r.get('skipped') or r.get('error')}{RESET}")
                continue
            b = base.get(r["L_start"])
            print(f"  {r['L_start']:>12,}{(str(r['kv_bits']) if r['kv_bits'] else 'as is'):>10}{r['tok_per_s']:>12,.1f}{r['step_ms']:>10.1f}"
                  f"{r['kv_allocated_bytes'] / GB:>12.2f} GB{(r['tok_per_s'] / b['tok_per_s'] if b else float('nan')):>15.2f}")


CAVEAT = ("One machine, one model in two formats, random token ids as prompts, greedy decoding. Each point: warm-up, then the median of "
          "three timed runs. Not controlled: thermal state, background load.")


def header(cfg: dict, rec: dict | None, what: str) -> None:
    extra = ""
    if rec:
        extra = f"\n{what} of {rec.get('updated', '')[:16]}" + ("   [tiny random model — numbers mean nothing]" if rec.get("tiny") else "")
        dev = rec.get("device") or {}
        if dev.get("device_name"):
            extra += f"   {dev['device_name']}"
    banner(cfg["demo"]["title"] + extra, AMBER if rec else "\033[34m")


# ── 子命令 ───────────────────────────────────────────────────────
def model_names(cfg: dict, args) -> list[str]:
    return [args.model] if args.model else list(cfg["models"])


def cmd_fetch(args, cfg):
    """兩個模型都在 versions.lock；這裡只確認本機有。--online 時補抓。"""
    from huggingface_hub import snapshot_download
    for name, m in cfg["models"].items():
        rev = locked_revision(m["repo"], m.get("revision"))
        try:
            p = snapshot_download(m["repo"], revision=rev, local_files_only=not args.online, ignore_patterns=NON_MODEL_FILES)
            print(f"  {GREEN}ok{RESET}  {name:<5} {m['repo']}@{str(rev)[:10]}  →  {p}")
        except Exception as e:
            print(f"  {RED}缺{RESET}  {name:<5} {m['repo']}@{str(rev)[:10]}：{repr(e)[:120]}")


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}")
    dev = device_info()
    for k in ("device_name", "memory_size", "max_recommended_working_set_size", "max_buffer_length"):
        if k in dev:
            print(f"  {k:<34} {dev[k] / GB:.1f} GB" if isinstance(dev[k], (int, float)) else f"  {k:<34} {dev[k]}")
    cond = conditions(args.note)
    print(f"  conditions       {cond}")
    g = grids(cfg, args)
    out = {"date": now(), "versions": v, "device": dev, "conditions": cond, "tiny": bool(args.tiny), "models": {}}
    for name in model_names(cfg, args):
        b = Bench(cfg, args, name)
        i = b.info()
        print_model_line(i)
        n = i["counts"]
        print(f"  parameters: blocks {n['blocks']:,} + norms {n['norms']:,} + embedding {n['embedding']:,} + LM head {n['lm_head']:,} = {n['N']:,}")
        print(f"  weights in memory: {i['weight_bytes']:,} bytes; the input embedding table is {i['embedding_bytes']:,} of them")
        entry = {"info": i}
        if i["quantization"]:
            print(f"  quantization (config.json): {json.dumps(i['quantization'])}")
            lin = b.model.model.layers[0].mlp.gate_proj
            q = {k: (str(getattr(lin, k).dtype), list(getattr(lin, k).shape)) for k in ("weight", "scales", "biases") if hasattr(lin, k)}
            q.update({k: getattr(lin, k) for k in ("bits", "group_size", "mode") if hasattr(lin, k)})
            print(f"  layers[0].mlp.gate_proj: {q}")
            entry["quantized_layer"] = {k: (v if isinstance(v, (int, str, float)) else list(v)) for k, v in q.items()}
        # 我們的迴圈對 mlx-lm 自己的 generate：同一個 prompt 長度、同樣的步數
        ours = b.measure_decode(1, g["a_prompt"], g["a_tokens"], g["budget"])
        print(f"  decode, B = 1, {g['a_tokens']} tokens after a {g['a_prompt']}-token prompt — this demo's loop: {BOLD}{ours['tok_per_s']:.1f} tokens/s{RESET}"
              f"  (three runs: {', '.join(f'{x:.1f}' for x in ours['all_tok_per_s'])})")
        entry["decode_ours"] = ours
        if b.tok is not None:
            from mlx_lm import stream_generate
            ids = b.rand_ids(1, g["a_prompt"], 7919 * g["a_prompt"] + 1)[0]
            last = None
            for _ in range(2):                                          # 第一次當暖機
                for last in stream_generate(b.model, b.tok, prompt=ids, max_tokens=g["a_tokens"]):
                    pass
            print(f"  the same with mlx-lm's stream_generate: {BOLD}{last.generation_tps:.1f} tokens/s{RESET}"
                  f" over {last.generation_tokens} tokens (it stops at an end-of-sequence token; ours does not)")
            entry["decode_mlx_lm"] = {"generation_tps": last.generation_tps, "generation_tokens": last.generation_tokens,
                                      "prompt_tps": last.prompt_tps, "peak_memory_gb": last.peak_memory}
        out["models"][name] = entry
        del b
    p = out_dir(args) / "inspect.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"\n→ {p.relative_to(HERE)}")


def cmd_rehearse(args, cfg):
    g = grids(cfg, args)
    parts = [args.part] if args.part else ["a", "b", "c"]
    rec = load_rec(args) or {"date_started": now(), "models": {}}
    rec.update(versions=versions(), device=device_info(), tiny=bool(args.tiny), reference=cfg["reference"] | {"qwen3_8b": None},
               settings={"demo": cfg["demo"], "grids": g})
    rec["reference"].pop("qwen3_8b", None)
    rec.setdefault("conditions", []).append({"date": now(), "parts": parts, **conditions(args.note)})
    header(cfg, None, "")
    note(CAVEAT)
    for name in model_names(cfg, args):
        b = Bench(cfg, args, name)
        m = rec["models"].setdefault(name, {})
        m["info"] = b.info()
        print_model_line(m["info"])

        def todo(d: dict, key: str) -> bool:
            return args.force or not (ok(d.get(key)) or (d.get(key) or {}).get("skipped"))

        if "a" in parts:
            a = m.setdefault("a", {})
            pre = a.setdefault("prefill", {})
            print(f"\n  {BOLD}(a) prefill{RESET}\n{PRE_HDR}")
            for L in g["a_lengths"]:
                if todo(pre, str(L)):
                    pre[str(L)] = b.measure_prefill(L)
                    save_rec(args, rec)
                print_prefill_row(m["info"], pre[str(L)], rec["reference"])
            if args.force or not ok(a.get("decode")):
                a["decode"] = b.measure_decode(1, g["a_prompt"], g["a_tokens"], g["budget"])
                save_rec(args, rec)
            print(f"\n  {BOLD}(a) decode{RESET}\n{DEC_HDR}")
            print_decode_row(m["info"], a["decode"], rec["reference"])
            print_roofs(m, rec["reference"])
        if "b" in parts:
            bb = m.setdefault("b", {})
            print(f"\n  {BOLD}(b) batch sweep{RESET}, {g['b_tokens']} tokens after a {g['b_prompt']}-token prompt")
            for B in g["b_batches"]:
                if todo(bb, str(B)):
                    bb[str(B)] = b.safe_decode(B, g["b_prompt"], g["b_tokens"], g["budget"])
                    save_rec(args, rec)
                r = bb[str(B)]
                print(f"  B = {B:>4}   " + (f"{r['tok_per_s']:>9,.1f} tokens/s   {r['step_ms']:>7.1f} ms/step   peak {r['peak_gb']:.1f} GB   {r['context']}"
                                            if ok(r) else f"{AMBER}{r.get('skipped') or r.get('error')}{RESET}"))
        if "c" in parts:
            cc, kq = m.setdefault("c", {}), m.setdefault("kvq", {})
            print(f"\n  {BOLD}(c) long context{RESET}, {g['c_tokens']} tokens after an L-token prefill")
            # 小的先跑：最大的那一格出事時，其他的都已經存好
            for L, B in sorted(((L, B) for L in g["c_lengths"] for B in g["c_batches"]), key=lambda t: t[0] * t[1]):
                key = f"{L}:{B}"
                if todo(cc, key):
                    cc[key] = b.safe_decode(B, L, g["c_tokens"], g["budget"])
                    save_rec(args, rec)
                r = cc[key]
                print(f"  L = {L:>6,}  B = {B:>3}   " + (f"{r['tok_per_s']:>9,.1f} tokens/s   {r['step_ms']:>7.1f} ms/step   peak {r['peak_gb']:.1f} GB   {r['context']}"
                                                         if ok(r) else f"{AMBER}{r.get('skipped') or r.get('error')}{RESET}"))
            for L in g["c_lengths"]:
                for bits in g["kv_bits"]:
                    key = f"{L}:{bits}"
                    if not bits and ok(cc.get(f"{L}:1")):
                        kq[key] = cc[f"{L}:1"]                          # 不量化的那一列就是上面 B = 1 的那一格
                    elif todo(kq, key):
                        kq[key] = b.safe_decode(1, L, g["c_tokens"], g["budget"], kv_bits=bits, kv_group=g["kv_group"])
                    save_rec(args, rec)
                    r = kq[key]
                    print(f"  L = {L:>6,}  KV bits = {bits or 'as is':>5}   " + (f"{r['tok_per_s']:>9,.1f} tokens/s   {r['step_ms']:>7.1f} ms/step"
                                                                              if ok(r) else f"{AMBER}{r.get('skipped') or r.get('error')}{RESET}"))
        save_rec(args, rec)
        del b
    note(f"\n→ {rec_path(args).relative_to(HERE)}")


def cmd_replay(args, cfg):
    rec = load_rec(args)
    if not rec:
        print(f"{RED}沒有 {rec_path(args).relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
        return
    header(cfg, rec, "(a) REPLAY of the rehearsal")
    show_a(rec, model_names(cfg, args))
    note("\n" + CAVEAT)


def cmd_replay_batch(args, cfg):
    rec = load_rec(args)
    if not rec:
        print(f"{RED}沒有 {rec_path(args).relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
        return
    names = model_names(cfg, args)
    header(cfg, rec, "(b), (c) REPLAY of the rehearsal")
    print(f"\n  {BOLD}(b) the batch sweep, short context{RESET}")
    show_b(rec, names)
    if not pause("(c): a long context"):
        return
    print(f"\n  {BOLD}(c) decode after a long prefill{RESET}")
    show_c(rec, names)
    if not pause("(c): the KV at fewer bits"):
        return
    show_kvq(rec, names)
    note("\n" + CAVEAT)


def cmd_live(args, cfg):
    g = grids(cfg, args)
    name = args.model or cfg["a"]["live_model"]
    rec = load_rec(args)
    ref = cfg["reference"]
    header(cfg, None, "")
    note(f"載入 {name} …")
    b = Bench(cfg, args, name)
    i = b.info()
    b.measure_prefill(64)                                               # 暖機：按 Enter 之後不該再有第一次執行的編譯
    b.measure_decode(1, g["a_prompt"], 8, g["budget"])
    print_model_line(i)
    log = RUNS / "live" / f"{now()[:10].replace('-', '')}.jsonl"
    if not pause(f"prefill: read {g['live_prefill']:,} tokens"):
        return
    pre = b.measure_prefill(g["live_prefill"])
    print(f"\n  {BOLD}prefill{RESET} (read)\n{PRE_HDR}")
    print_prefill_row(i, pre, ref, f"{GREY}median of {b.repeats}{RESET}")
    append_jsonl(log, {"what": "prefill", "model": name, **pre})
    if not pause(f"decode: write {g['live_decode']} tokens, B = 1"):
        return
    dec = b.measure_decode(1, g["a_prompt"], g["live_decode"], g["budget"])
    print(f"\n  {BOLD}decode{RESET} (write), B = 1\n{DEC_HDR}")
    print_decode_row(i, dec, ref, f"{GREY}median of {b.repeats}{RESET}")
    append_jsonl(log, {"what": "decode", "model": name, **dec})
    if not pause("side by side"):
        return
    live = {"info": i, "a": {"prefill": {"live": pre}, "decode": dec}}
    live["summary"] = summarize(live)
    print(f"\n  per token:  read {1e3 / pre['tok_per_s']:.3f} ms    write {1e3 / dec['tok_per_s']:.1f} ms    →  writing is "
          f"{BOLD}{pre['tok_per_s'] / dec['tok_per_s']:,.0f}×{RESET} slower")
    print_roofs(live, ref)
    note("  (live: F_max from this one prefill length; the rehearsal takes the best of several)")
    if rec and bool(rec.get("tiny")) == bool(args.tiny):
        print(f"\n  {AMBER}from the rehearsal of {rec.get('updated', '')[:16]} (not live):{RESET}")
        show_a(rec, list(rec["models"]))
    print()


def cmd_live_batch(args, cfg):
    g = grids(cfg, args)
    name = args.model or cfg["a"]["live_model"]
    B = int(args.batch)
    rec = load_rec(args)
    header(cfg, None, "")
    b = Bench(cfg, args, name)
    print_model_line(b.info())
    b.measure_decode(1, g["b_prompt"], 8, g["budget"])
    if not pause(f"decode {g['b_tokens']} tokens at B = {B}"):
        return
    one = b.measure_decode(1, g["b_prompt"], g["b_tokens"], g["budget"])
    r = b.safe_decode(B, g["b_prompt"], g["b_tokens"], g["budget"])
    if not ok(r):
        print(f"  {RED}{r.get('skipped') or r.get('error')}{RESET}")
        return
    print(f"\n  B = 1:    {one['tok_per_s']:>9,.1f} tokens/s\n  B = {B:<4}  {r['tok_per_s']:>9,.1f} tokens/s in total   "
          f"{BOLD}{r['tok_per_s'] / one['tok_per_s']:.1f}×{RESET}   ({r['per_seq_tok_per_s']:.1f} per sequence, {r['step_ms']:.1f} ms/step)")
    old = ((rec or {}).get("models", {}).get(name, {}).get("b") or {}).get(str(B))
    if ok(old):
        note(f"  rehearsal ({old['date'][:16]}): {old['tok_per_s']:,.1f} tokens/s")
    append_jsonl(RUNS / "live" / f"{now()[:10].replace('-', '')}.jsonl", {"what": "batch", "model": name, **r})


# ── test ─────────────────────────────────────────────────────────
def cmd_test(args, cfg):
    fails = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if cond else 1
        print(f"  {GREEN + 'ok  ' if cond else RED + 'FAIL'}{RESET} {label}" + (f"   {GREY}{detail}{RESET}" if detail else ""))

    banner("test — the arithmetic against the slides' numbers (no model)")
    ref = cfg["reference"]
    c = dict(ref["qwen3_8b"])
    n = R.count_params(c)
    check("parameter count of Qwen3-8B from its shape", n["N"] == c["N"], f"{n['N']:,}")
    kvt = R.kv_bytes_per_token(c)
    check("KV bytes per token", kvt == 147456, f"{kvt:,}")
    F, BW = ref["fmax_w3_flops_per_s"], ref["bw_spec_bytes_per_s"]
    for label, bits, want_b, want_k in (("bf16", 16.0, 97.98, 106.65), ("4-bit (4.5 bits/weight)", 4.5, 27.56, 29.99)):
        W = n["N"] * bits / 8
        bd = R.boundaries(c, n, W, 0, kvt, F, BW)
        k = R.kink_B(96, c, n, W, 0, kvt, F, BW)
        check(f"{label}: B* and the kink at L = 96, from the slide's inputs", abs(bd["B_star"] - want_b) < 0.02 and abs(k - want_k) < 0.02,
              f"B* {bd['B_star']:.2f}, kink {k:.2f}")
    bd = R.boundaries(c, n, 2.0 * n["N"], 0, kvt, F, BW)
    check("L_c from the slide's inputs", abs(bd["L_c"] - 1182.05) < 0.5 and bd["kv_side_intensity"] == 4.0, f"{bd['L_c']:.1f} tokens; KV-side intensity {bd['kv_side_intensity']:g}")
    t, side = R.throughput_upper(1, 0, c, n, 2.0 * n["N"], 0, kvt, F, BW)
    check("decode bound at B = 1, bf16", abs(t - 37.48) < 0.01 and side == "memory", f"{t:.2f} tokens/s, {side}-bound")
    t2, side2 = R.throughput_upper(512, 96, c, n, 2.0 * n["N"], 0, kvt, F, BW)
    check("decode bound far right of the kink is compute-bound and flat", side2 == "compute" and abs(t2 - 3659.93) < 0.5, f"{t2:.1f} tokens/s")
    pre = [{"L": 1024, "tok_per_s": 3000.0}, {"L": 4096, "tok_per_s": 2900.0}]
    dec = {"L_start": 32, "n_gen": 64, "tok_per_s": 14.0, "B": 1}
    r = R.roofs_from_measurement(c, n, 2.0 * n["N"], n["embedding"] * 2.0, kvt, pre, dec, "deck")
    s = R.roofs_from_measurement(c, n, 2.0 * n["N"], n["embedding"] * 2.0, kvt, pre, dec, "strict")
    back, _ = R.throughput_upper(1, 64, c, n, 2.0 * n["N"], n["embedding"] * 2.0, kvt, r["F_max"], r["BW_eff"])
    check("roofs from a made-up measurement reproduce its B = 1 point", abs(back - 14.0) < 1e-6, f"{back:.4f}")
    check("strict count is smaller on both roofs", s["F_max"] < r["F_max"] and s["BW_eff"] < r["BW_eff"],
          f"F ×{s['F_max'] / r['F_max']:.3f}, BW ×{s['BW_eff'] / r['BW_eff']:.3f}")

    try:
        import mlx.core as mx  # noqa: F401
    except Exception as e:
        note(f"\n  mlx 不能用（{type(e).__name__}）：量法的檢查略過")
        sys.exit(1 if fails else 0)
    banner("test — the way of measuring, on a tiny random model")
    args.tiny = True
    b = Bench(cfg, args, "tiny")
    n_gen, L = 12, 40

    def toks(B, L, budget, kv_bits=0, reserve=True):
        cache, y0, mode = b.context(B, L, n_gen + 2, budget, kv_bits, 32) if reserve else (b.prefill([r[:-1] for r in b.rand_ids(B, L, 7919 * L + B)]),
                                                                                              mx.array([r[-1] for r in b.rand_ids(B, L, 7919 * L + B)]), "plain")
        base = cache[0].offset
        _, t = b.decode(cache, y0, n_gen)
        out = [[int(x) for x in mx.stack(t, axis=1)[i].tolist()] for i in range(B)]
        b.rewind(cache, base)
        _, t2 = b.decode(cache, y0, n_gen)
        again = [[int(x) for x in mx.stack(t2, axis=1)[i].tolist()] for i in range(B)]
        return out, again, mode, cache

    one, again, mode, cache = toks(1, L, 10 ** 6)
    check("a second timed run from the rewound cache writes the same tokens", one == again, f"{one[0][:6]} …")
    check("the reserved buffer is a multiple of 256 and nothing was reallocated", cache[0].keys.shape[2] == 256 and cache[0].offset == L - 1 + n_gen)
    plain, _, _, _ = toks(1, L, 10 ** 6, reserve=False)
    check("reserving the buffer does not change the output", plain == one)
    tiled, _, mode, cache = toks(3, L, 0)
    check("tiled context: every row writes what the single sequence wrote", mode == "tiled" and all(r == one[0] for r in tiled) and cache[0].keys.shape[0] == 3)
    dist, _, mode, _ = toks(3, L, 10 ** 6)
    singles = []
    for i in range(3):
        ids = b.rand_ids(3, L, 7919 * L + 3)[i]
        cch = b.prefill([ids[:-1]])
        _, t = b.decode(cch, mx.array([ids[-1]]), n_gen)
        singles.append([int(x.item()) for x in t])
    same = sum(a == s for a, s in zip(dist, singles))
    check("distinct context: rows are different prompts", mode == "distinct" and len({tuple(r) for r in dist}) > 1, f"{same}/3 rows equal their own single-sequence run")
    q, _, _, cache = toks(1, L, 10 ** 6, kv_bits=8)
    check("quantized KV path runs and keeps its offset", type(cache[0]).__name__ == "QuantizedKVCache" and cache[0].offset == L - 1 + n_gen and len(q[0]) == n_gen,
          f"{sum(a == c2 for a, c2 in zip(q[0], one[0]))}/{n_gen} tokens equal the unquantized run")
    row = b.measure_decode(4, 300, 8, 128)
    check("measure_decode: a tiled batch reports B × the KV of one", row["context"] == "tiled" and row["kv_allocated_bytes"] == 4 * b.kvt * 512, f"{row['kv_allocated_bytes']:,} bytes")
    b.limit = 0
    check("a point over the memory limit is skipped, not run", "skipped" in b.measure_decode(1, 16, 4, 128))
    print(f"\n  {(GREEN + 'all passed') if not fails else (RED + str(fails) + ' failed')}{RESET}")
    sys.exit(1 if fails else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--model", default=None, help="demo_config.toml 的 [models.*] 名稱（bf16 / q4）；不給就是全部（live 是 [a] live_model）")
    p.add_argument("--part", choices=["a", "b", "c"], help="rehearse 只做一段")
    p.add_argument("--force", action="store_true", help="rehearse：做過的點也重跑")
    p.add_argument("--note", default=None, help="彩排當下的狀態（電源、其他程式），記進紀錄")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "test", "inspect", "rehearse", "live", "replay", "replay-batch"):
        sub.add_parser(c)
    lb = sub.add_parser("live-batch")
    lb.add_argument("batch", nargs="?", default="16")
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "test": cmd_test, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "live": cmd_live,
     "live-batch": cmd_live_batch, "replay": cmd_replay, "replay-batch": cmd_replay_batch}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
