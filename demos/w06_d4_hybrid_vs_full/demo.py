#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 4：並聯 hybrid（Falcon-H1-7B）對 full attention（Qwen3-8B）——cache 對 L 的兩條線，以及 key-value retrieval
（投影片 p46「Replay + live: demo 4 — cache against L, and recall」；p45 是前導頁）。

    ./present.sh fetch                 第一次：下載 Falcon-H1-7B-Instruct-4bit 並鎖定 revision（Qwen3-8B-4bit 已在 W3 demo 2 鎖定）
    ./present.sh probe                 課前：每個模型答一個普通問題與 K = 5／20 的小題，看回覆成不成句（彩排後加：分清「模型答不出」與「載入的模型有問題」）
    ./present.sh inspect               課前：兩個模型各載一次，印每層 cache 的類型與每個狀態張量的形狀／dtype、每 token 的 KV 公式、
                                       固定狀態的 bytes、chat template 是否吃 system／enable_thinking、兩個 tokenizer 下兩檔 prompt 的 token 數
    ./present.sh rehearse              課前彩排：每個模型先 memory（1k → 32k 各一次 prefill）再 recall（兩檔 × 五個位置 × Q 題）
    ./present.sh rehearse --part memory          只做 (a)；--part recall 只做 (b)；--model falcon_h1 只做一個模型
    ./present.sh rehearse --part recall --q 2     煙霧測試（每格 2 題，寫到 runs/smoke/）
    ./present.sh                       課堂：載兩個模型；Enter 一題（兩個模型各答一次）、m 印 (a) 的表、r 印 (b) 的表、x 離開
    ./present.sh replay [memory|recall]   現場失敗：只讀 runs/rehearsal/ 的 json，不載模型
    ./present.sh fake                  沒有 mlx 也能演練流程（數字與回答是假的，畫面會標明）

(a) 量法沿 W3 demo 2（`../w03_d2_kv_memory/demo.py`，從檔案路徑 import，不複製）：暖身後記 active 為基線，對 L 個假 token 分塊 prefill、
    只 eval cache 的 state，記 cache 張量的 bytes、Δactive、Δpeak、秒數。這裡多做的一件事：把 cache 樹走一遍——Falcon-H1 每層是
    CacheList(ArraysCache(size=2), KVCache)，ArraysCache 沒有 nbytes——把 KV（keys／values）、SSM 狀態（4 維）、conv 狀態（3 維）
    分開加總。公式：KV 每 token 2 · n_attn · n_kv · d_head · b；固定狀態 = 探針量到的狀態張量 bytes（與 L 無關，逐列核對）。
(b) 任務沿 W5 demo 3（`../w05_d3_lost_in_middle/demo.py`，同樣 import）：Liu et al. §4 的合成 key-value retrieval，K 固定為 W5 的 81／326、
    seed 相同，所以兩個模型答的是 W5 問 Llama 3.1 8B 的同一批題；判定、表、答錯的種類都是 W5 的函數。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）。Python 3.11+（tomllib）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent
sys.path.insert(0, str(DEMOS))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, append_jsonl,  # noqa: E402
                    banner, device_info, locked_revision, note, now, peak_memory_gb,
                    revision_key, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
LIVE = RUNS / "live"
KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"]


def _load_module(name: str, path: Path):
    """從檔案路徑載入模組，不動 sys.path（W4 demo 1 彩排踩過：各資料夾的 tests.py 會撞名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W3 = _load_module("w03_d2_kv_memory_demo", DEMOS / "w03_d2_kv_memory" / "demo.py")
W5 = _load_module("w05_d3_lost_in_middle_demo", DEMOS / "w05_d3_lost_in_middle" / "demo.py")
fmt = W3.fmt


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def model_keys(cfg: dict, args) -> list[str]:
    """要跑哪些模型：--model 指定一個，否則 [demo].order（optional 的模型只在點名時跑）。"""
    if args.model:
        return [args.model]
    return [k for k in cfg["demo"]["order"] if not cfg["models"][k].get("optional")]


def raw_config(path) -> dict:
    return json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))


def contexts(cfg: dict, max_pos: int | None) -> list[tuple[int, str]]:
    """W3 demo 2 的 resolve_contexts 讀 [demo].contexts；這裡 (a) 的設定都在 [memory]，包一層。"""
    return W3.resolve_contexts({"demo": cfg["memory"]}, max_pos)


# ── (a) cache 樹 ─────────────────────────────────────────────────
def walk_cache(cache):
    """走一遍 make_prompt_cache 回傳的 cache 樹，產生 (layer, class, part, array)。
    KVCache 類：keys／values；ArraysCache：cache[0]、cache[1]（Falcon-H1：conv 狀態、SSM 狀態）；CacheList 展開成它的成員。"""
    for i, c in enumerate(cache):
        for p in getattr(c, "caches", [c]):
            cls = type(p).__name__
            if hasattr(p, "keys") and hasattr(p, "values"):
                for nm, a in (("keys", p.keys), ("values", p.values)):
                    if a is not None:
                        yield i, cls, nm, a
            elif hasattr(p, "cache"):
                for j, a in enumerate(p.cache):
                    if a is not None:
                        yield i, cls, f"[{j}]", a


def classify(cls: str, part: str, a) -> str:
    if part in ("keys", "values"):
        return "kv"
    return "ssm_state" if a.ndim == 4 else "conv_state"      # (B, h, d_head, d_state) 對 (B, d_conv − 1, conv_dim)


def cache_bytes(cache) -> dict:
    out = {"kv": 0, "ssm_state": 0, "conv_state": 0}
    for _, cls, part, a in walk_cache(cache):
        out[classify(cls, part, a)] += int(a.nbytes)
    out["total"] = sum(out.values())
    out["fixed"] = out["ssm_state"] + out["conv_state"]
    return out


class HybridBackend(W3.MLXBackend):
    """W3 demo 2 的 MLXBackend，兩處改：探針走整棵 cache 樹（父類別讀 cache[0].keys，CacheList 沒有）；measure 把 bytes 分成 KV／固定狀態。"""

    def _probe(self):
        cache = self._prefill(list(range(1, 9)), 8)
        parts, kv_dtype, kv_shape, n_attn = [], None, None, 0
        for i, cls, part, a in walk_cache(cache):
            kind = classify(cls, part, a)
            if i == 0 or (kind == "kv" and part == "keys" and kv_dtype is None):
                parts.append({"layer": i, "class": cls, "part": part, "kind": kind,
                              "shape": [int(x) for x in a.shape], "dtype": str(a.dtype), "nbytes": int(a.nbytes)})
            if kind == "kv" and part == "keys":
                n_attn += 1
                if kv_dtype is None:
                    kv_dtype, kv_shape = str(a.dtype), tuple(int(x) for x in a.shape)
        self.layer_types = [type(c).__name__ + ("(" + ", ".join(type(p).__name__ for p in c.caches) + ")" if hasattr(c, "caches") else "")
                            for c in cache]
        self.probe_parts, self.n_attn_layers = parts, n_attn
        self.fixed_bytes = cache_bytes(cache)["fixed"]        # 探針（8 個 token）之後的固定狀態：與 L 無關
        del cache
        self._clear()
        return kv_dtype or "(no KV cache)", kv_shape or ()

    def kv_per_token(self) -> int:
        c = self.config
        if not self.n_attn_layers:
            return 0
        return 2 * self.n_attn_layers * c["num_key_value_heads"] * c["head_dim"] * self.bytes_per_elem()

    def measure(self, L: int, step: int, seed: int) -> dict:
        import random
        rng = random.Random(seed)
        vocab = self.config["vocab_size"] or 32000
        ids = [rng.randrange(1000, vocab - 1000) for _ in range(L)]
        self._clear()
        W3.reset_peak_memory()
        a0 = self._active()
        t0 = time.time()
        cache = self._prefill(ids, step)
        seconds = time.time() - t0
        parts = cache_bytes(cache)
        a1, pk = self._active(), self._peak()
        del cache
        self._clear()
        return {"L": L, "cache_nbytes": parts["total"], "kv_nbytes": parts["kv"], "ssm_state_nbytes": parts["ssm_state"],
                "conv_state_nbytes": parts["conv_state"], "active_delta": a1 - a0, "peak_delta": pk - a0,
                "seconds": seconds, "prefill_tps": L / seconds if seconds else None, "fake": False}


class FakeHybridBackend(W3.FakeBackend):
    """--fake：沒有 mlx 也能演練。數字是公式值加一點雜訊，畫面會標明。"""

    def __init__(self, repo, revision, offline, spec=None):
        super().__init__(repo, revision, offline)
        hybrid = "falcon" in repo.lower()
        self.config = {"model_type": "falcon_h1" if hybrid else "qwen3", "num_hidden_layers": 44 if hybrid else 36,
                       "num_attention_heads": 12 if hybrid else 32, "num_key_value_heads": 2 if hybrid else 8, "hidden_size": 3072 if hybrid else 4096,
                       "head_dim": 128, "head_dim_in_config": True, "max_position_embeddings": 262144 if hybrid else 40960,
                       "vocab_size": 130049 if hybrid else 151936, "rope_scaling": None, "quantization": {"bits": 4, "group_size": 64}}
        self.weight_bytes = 4_270_000_000 if hybrid else 4_610_000_000
        self.n_attn_layers = self.config["num_hidden_layers"]
        self.layer_types = (["CacheList(ArraysCache, KVCache)"] if hybrid else ["KVCache"]) * self.n_attn_layers
        self.kv_dtype, self.kv_shape = "mlx.core.float16 (fake)", (1, self.config["num_key_value_heads"], 256, 128)
        self.probe_parts = ([{"layer": 0, "class": "ArraysCache", "part": "[0]", "kind": "conv_state", "shape": [1, 3, 3584], "dtype": "float16 (fake)", "nbytes": 21504},
                             {"layer": 0, "class": "ArraysCache", "part": "[1]", "kind": "ssm_state", "shape": [1, 24, 128, 256], "dtype": "float32 (fake)", "nbytes": 3145728}]
                            if hybrid else []) + [{"layer": 0, "class": "KVCache", "part": "keys", "kind": "kv", "shape": list(self.kv_shape), "dtype": "float16 (fake)", "nbytes": 0}]
        self.fixed_bytes = (21504 + 3145728) * 44 if hybrid else 0

    def kv_per_token(self) -> int:
        return 2 * self.n_attn_layers * self.config["num_key_value_heads"] * self.config["head_dim"] * 2

    def measure(self, L, step, seed):
        per = self.kv_per_token()
        time.sleep(min(0.3 + L / 200_000, 1.5))
        kv = per * L
        return {"L": L, "cache_nbytes": kv + self.fixed_bytes, "kv_nbytes": kv, "ssm_state_nbytes": self.fixed_bytes * 0.993 if self.fixed_bytes else 0,
                "conv_state_nbytes": self.fixed_bytes * 0.007 if self.fixed_bytes else 0, "active_delta": int((kv + self.fixed_bytes) * 1.002),
                "peak_delta": int((kv + self.fixed_bytes) * 1.05 + 300 * MiB), "seconds": L / (1800 if self.fixed_bytes else 2500),
                "prefill_tps": 1800.0 if self.fixed_bytes else 2500.0, "fake": True}


def make_backend(key: str, cfg: dict, args):
    m = cfg["models"][key]
    revision = locked_revision(m["repo"], m.get("revision"))
    note(f"載入 {m['label']}  ←  {m['repo']}" + (f"@{revision[:10]}" if revision else ""))
    b = (FakeHybridBackend if args.fake else HybridBackend)(m["repo"], revision, offline=not args.online)
    note(f"載入完成：{b.load_seconds:.1f} 秒")
    b.key, b.label, b.short, b.repo, b.revision = key, m["label"], m.get("short", m["label"]), m["repo"], revision
    return b


def ssm_formula(raw: dict, n_layer: int, b_state: int) -> dict | None:
    """config 的 mamba_* 鍵算固定狀態（SSM 狀態 + conv 狀態）的公式值；沒有那些鍵就 None。"""
    if "mamba_n_heads" not in raw:
        return None
    h, dh, ds = int(raw["mamba_n_heads"]), int(raw["mamba_d_head"]), int(raw["mamba_d_state"])
    g, dconv, dssm = int(raw.get("mamba_n_groups", 1)), int(raw.get("mamba_d_conv", 4)), int(raw.get("mamba_d_ssm", raw.get("intermediate_size", 0)))
    conv_dim = dssm + 2 * g * ds
    return {"n_layer": n_layer, "mamba_n_heads": h, "mamba_d_head": dh, "mamba_d_state": ds, "ssm_elems_per_layer": h * dh * ds,
            "ssm_bytes_per_layer": h * dh * ds * b_state, "ssm_bytes_total": n_layer * h * dh * ds * b_state, "state_bytes_per_elem": b_state,
            "conv_dim": conv_dim, "conv_elems_per_layer": (dconv - 1) * conv_dim}


def bytes_of_dtype(dtype: str) -> int:
    for name, b in (("float32", 4), ("float16", 2), ("bfloat16", 2), ("float8", 1), ("uint8", 1)):
        if name in dtype:
            return b
    return 2


def print_memory_header(b, cfg: dict) -> dict:
    c = b.config
    bpe = b.bytes_per_elem()
    per = b.kv_per_token()
    raw = raw_config(b.path) if not b.fake else {}
    banner(f"(a) cache against L  ·  {b.label}" + (f"  {RED}[FAKE]{RESET}" if b.fake else ""))
    print(f"  config.json ({c['model_type']}): num_hidden_layers = {BOLD}{c['num_hidden_layers']}{RESET}, "
          f"num_key_value_heads = {BOLD}{c['num_key_value_heads']}{RESET} (num_attention_heads = {c['num_attention_heads']}), "
          f"head_dim = {BOLD}{c['head_dim']}{RESET}, max_position_embeddings = {c['max_position_embeddings']}"
          + (f", attn_layer_indices = {json.dumps(raw['attn_layer_indices'])}" if "attn_layer_indices" in raw else ""))
    kinds = {}
    for t in b.layer_types:
        kinds[t] = kinds.get(t, 0) + 1
    print("  cache per layer: " + ", ".join(f"{n} × {t}" for t, n in kinds.items()) + f"   → {BOLD}{b.n_attn_layers} layers hold a KV cache{RESET}")
    for p in b.probe_parts:
        print(f"    layer {p['layer']} {p['class'] + (p['part'] if p['part'].startswith('[') else '.' + p['part']):<20} {p['kind']:<11} shape {p['shape']}  dtype {p['dtype']}  ({fmt(p['nbytes']).strip()} after an 8-token probe)")
    print(f"  KV per token: 2 × {b.n_attn_layers} × {c['num_key_value_heads']} × {c['head_dim']} × {bpe} = {BOLD}{per:,} bytes = {per / KiB:g} KiB{RESET}"
          if per else f"  {BOLD}no KV cache{RESET}: this model pays nothing per token")
    state_dtype = next((p["dtype"] for p in b.probe_parts if p["kind"] == "ssm_state"), None)
    formula = ssm_formula(raw, c["num_hidden_layers"], bytes_of_dtype(state_dtype or "float32")) if raw else None
    if b.fixed_bytes:
        print(f"  fixed state (SSM + conv, all layers, measured after the probe): {BOLD}{fmt(b.fixed_bytes).strip()}{RESET} = {b.fixed_bytes:,} bytes"
              f"  — does not grow with L" + (f"   dtype of the SSM state: {state_dtype}" if state_dtype else ""))
        if formula:
            print(f"    config formula: {formula['n_layer']} × {formula['mamba_n_heads']} × {formula['mamba_d_head']} × {formula['mamba_d_state']} × {formula['state_bytes_per_elem']}"
                  f" = {formula['ssm_bytes_total']:,} bytes SSM state (+ conv {formula['conv_elems_per_layer']:,} elements/layer)")
    rows = contexts(cfg, c["max_position_embeddings"])
    print(f"\n  {'context':<14}{'predicted (KV·L + fixed)':>26}")
    for L, label in rows:
        print(f"  {label + ' (' + f'{L:,}' + ')':<14}{fmt(per * L + b.fixed_bytes):>26}")
    print()
    return {"config": c, "raw_keys": {k: raw[k] for k in raw if k.startswith("mamba_") or k in ("attn_layer_indices", "max_position_embeddings")},
            "bytes_per_elem": bpe, "kv_dtype": b.kv_dtype, "kv_probe_shape": list(b.kv_shape), "n_attn_layers": b.n_attn_layers,
            "layer_types": kinds, "probe_parts": b.probe_parts, "weight_bytes": b.weight_bytes, "kv_bytes_per_token": per,
            "fixed_state_bytes": b.fixed_bytes, "ssm_state_dtype": state_dtype, "fixed_state_formula": formula,
            "contexts": [{"L": L, "label": lab} for L, lab in rows]}


ROW_HDR = f"  {'context':<14}{'predicted':>12}{'cache total':>13}{'of which KV':>13}{'fixed':>11}{'Δactive':>11}{'Δpeak':>11}   {'time':<22}"


def print_mem_row(r: dict, per: int, fixed: int, flag: str = "") -> None:
    L = r["L"]
    t = f"{r['seconds']:.1f} s ({r['prefill_tps']:,.0f} tok/s)" if r.get("prefill_tps") else f"{r['seconds']:.1f} s"
    fx = r["ssm_state_nbytes"] + r["conv_state_nbytes"]
    print(f"  {r.get('label', '') + ' (' + f'{L:,}' + ')':<14}{fmt(per * L + fixed):>12}{fmt(r['cache_nbytes']):>13}{fmt(r['kv_nbytes']):>13}"
          f"{(fmt(fx) if fx else '—'):>11}{fmt(r['active_delta']):>11}{fmt(r['peak_delta']):>11}   {t:<22}{flag}")


def slope_line(rows: list[dict], per: int) -> str | None:
    ok = [r for r in rows if r.get("kv_nbytes") is not None]
    if len(ok) < 2:
        return None
    a, z = ok[0], ok[-1]
    slope = (z["cache_nbytes"] - a["cache_nbytes"]) / (z["L"] - a["L"])
    intercept = a["cache_nbytes"] - slope * a["L"]
    s = f"slope of the cache total between {a['L']:,} and {z['L']:,} tokens: {slope / KiB:.2f} KiB/token"
    if per:
        s += f" vs. formula {per / KiB:g} KiB/token ({slope / per * 100:.1f}%)"
    s += f"; intercept {fmt(intercept).strip()}"
    return s


def run_memory(b, cfg: dict) -> dict:
    head = print_memory_header(b, cfg)
    per, fixed = head["kv_bytes_per_token"], head["fixed_state_bytes"]
    rows = contexts(cfg, b.config["max_position_embeddings"])
    base = b.baseline()
    note(f"基線 active = {fmt(base).strip()}（權重 {fmt(b.weight_bytes).strip()} + 其他）\n")
    step, seed = int(cfg["memory"]["prefill_step_size"]), int(cfg["memory"]["seed"])
    print(ROW_HDR)
    out = []
    for i, (L, label) in enumerate(rows):
        r = b.measure(L, step, seed + i)
        r["label"] = label
        print_mem_row(r, per, fixed, f"{RED}[FAKE]{RESET}" if r.get("fake") else "")
        out.append({**r, "date": now()})
    s = slope_line(out, per)
    if s:
        print(f"\n  {s}")
    fx = {r["ssm_state_nbytes"] + r["conv_state_nbytes"] for r in out}
    if fixed and len(fx) == 1:
        print(f"  {GREEN}fixed state is the same at every L: {fmt(fx.pop()).strip()}{RESET}")
    elif fixed:
        print(f"  {AMBER}fixed state varies with L: {sorted(fx)}{RESET}")
    print()
    return {"model": b.repo, "revision": b.revision, "label": b.label, "short": b.short, "key": b.key, "fake": b.fake, "baseline_active_bytes": base,
            "prefill_step_size": step, **head, "rows": out}


# ── (b) recall：W5 demo 3 的 harness，同 K、同 seed ─────────────────
class RecallModel(W5.MLXModel):
    """W5 的 MLXModel，多一條退路：chat template 不吃 system 角色（Falcon-H1 待實測）就把 system 併進 user 那一輪。"""

    def render(self, cfg: dict, user: str) -> str:
        try:
            s = super().render(cfg, user)
            self.system_merged = False
            return s
        except Exception as e:        # jinja 的 TemplateError 不是 TypeError，父類別不接
            self.system_merged = True
            self.render_error = f"{type(e).__name__}: {str(e)[:120]}"
            msgs = [{"role": "user", "content": cfg["prompt"]["system"] + "\n\n" + user}]
            try:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class FakeRecallModel(W5.FakeModel):
    def __init__(self, key, spec, offline, weights=True):
        super().__init__(key, spec, offline, weights)
        self.system_merged = False
        self.rng.seed(sum(map(ord, key)))   # 兩個假模型答得不一樣


def make_recall_model(key: str, cfg: dict, args, weights: bool = True):
    spec = cfg["models"][key]
    rev = locked_revision(spec["repo"], spec.get("revision"))
    note(f"載入 {spec['label']}{'（只載 tokenizer）' if not weights else ''}  ←  {spec['repo']}" + (f"@{rev[:10]}" if rev else ""))
    m = (FakeRecallModel if args.fake else RecallModel)(key, spec, offline=not args.online, weights=weights)
    note(f"  完成：{m.load_seconds:.1f} 秒")
    m.short = spec.get("short", spec["label"])
    return m


def sizes_for(cfg: dict, only: str | None = None) -> list[dict]:
    return [s for s in cfg["sizes"] if not only or s["label"] == only]


def run_recall(m, cfg: dict, Q: int, sizes: list[dict]) -> dict:
    for s in sizes:
        assert int(s.get("k") or 0) > 0, "K 固定在 demo_config.toml 的 [[sizes]].k（與 W5 demo 3 相同的題），不校準"
    t0 = time.time()
    cells = W5.run_cells(m, cfg, sizes, Q)
    summary = W5.summarize(cells)
    summary.update({"date": now(), "model": m.repo, "revision": m.revision, "label": m.label, "short": m.short, "key": m.key, "fake": m.fake, "q": Q,
                    "seconds_total": round(time.time() - t0), "peak_memory_gb": peak_memory_gb() if not m.fake else None,
                    "max_position_embeddings": m.max_pos, "rope_scaling": m.rope_scaling, "think_supported": m.think_supported,
                    "system_merged": getattr(m, "system_merged", None), "render_error": getattr(m, "render_error", None),
                    "system": cfg["prompt"]["system"], "instruction": cfg["prompt"]["instruction"], "temperature": cfg["demo"]["temperature"]})
    W5.print_table(summary, [s["label"] for s in sizes], tag=f"{AMBER}彩排 {summary['date'][:16]} · {m.label}{RESET}")
    full = {k: v for k, v in summary.items() if k != "cells"}
    full["cells"] = cells
    return summary, full


def print_slide_table(summaries: dict, cfg: dict) -> None:
    """p46 的十格：一個模型一列（兩檔並排），五個位置一欄。"""
    sizes = [s["label"] for s in cfg["sizes"]]
    Ks = " · ".join(str(s["k"]) for s in cfg["sizes"])
    print(f"  {'exact match /Q, K = ' + Ks:<34}" + "".join(f"{lab:>14}" for _, lab, _ in W5.POSITIONS))
    for key, summ in summaries.items():
        cells = {(c["size"], c["pos_name"]): c for c in summ["cells"]}
        row = f"  {summ.get('short', summ['label'])[:22] + ', K = ' + Ks:<34}"
        for name, _, _ in W5.POSITIONS:
            parts = []
            for s in sizes:
                c = cells.get((s, name))
                parts.append(f"{c['em']}/{c['n']}" if c else "—")
            row += f"{' · '.join(parts):>14}"
        print(row)
    print()


# ── 彩排紀錄 ─────────────────────────────────────────────────────
def mem_path(fake: bool, out_dir: Path = REHEARSAL) -> Path:
    return out_dir / ("memory.fake.json" if fake else "memory.json")


def recall_path(key: str, fake: bool, out_dir: Path = REHEARSAL) -> Path:
    return out_dir / (f"recall_{key}.fake.json" if fake else f"recall_{key}.json")


def recall_summary_path(fake: bool, out_dir: Path = REHEARSAL) -> Path:
    return out_dir / ("recall_summary.fake.json" if fake else "recall_summary.json")


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_memory(rec: dict, fake: bool, out_dir: Path = REHEARSAL) -> Path:
    """memory.json 以模型 key 為鍵；一次只跑一個模型時保留另一個的紀錄。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = mem_path(fake, out_dir)
    cur = load_json(p) or {"models": {}}
    cur["models"][rec["key"]] = rec
    cur.update({"date": now(), "versions": versions(), "device": device_info() if not fake else {}, "fake": fake})
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def save_recall(key: str, summary: dict, full: dict, fake: bool, out_dir: Path = REHEARSAL) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    recall_path(key, fake, out_dir).write_text(json.dumps(full, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    p = recall_summary_path(fake, out_dir)
    cur = load_json(p) or {"models": {}}
    cur["models"][key] = summary
    cur.update({"date": now(), "versions": versions(), "device": device_info() if not fake else {}, "fake": fake})
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def replay_memory(cfg: dict, fake: bool, interactive: bool = False) -> bool:
    rec = load_json(mem_path(fake))
    if not rec:
        print(f"{RED}沒有 runs/rehearsal/memory.json：先跑 ./present.sh rehearse --part memory{RESET}")
        return False
    banner(f"{AMBER}REPLAY (a) — 課前彩排的紀錄，不是現場{RESET}  {rec['date']}")
    v = rec.get("versions") or {}
    print(f"  mlx-lm {v.get('mlx-lm')} @ {str(v.get('mlx_lm_commit'))[:10]}, mlx {v.get('mlx')}; {rec.get('device', {}).get('device_name', '')}")
    for key in cfg["demo"]["order"]:
        m = rec["models"].get(key)
        if not m:
            continue
        if interactive:
            k = input(f"\n  {DIM}Enter = {m['label']}   q = 跳過{RESET} ").strip().lower()
            print("\033[1A\033[2K", end="")
            if k == "q":
                break
        c = m["config"]
        per, fixed = m["kv_bytes_per_token"], m["fixed_state_bytes"]
        print(f"\n  {BOLD}{m['label']}{RESET}  {m['model']}")
        print(f"  {m['n_attn_layers']} layers with KV × {c['num_key_value_heads']} KV heads × head_dim {c['head_dim']} × b = {m['bytes_per_elem']}"
              f"  →  {per / KiB:g} KiB / token" + (f";  fixed state {fmt(fixed).strip()} ({m.get('ssm_state_dtype')})" if fixed else ";  no fixed state")
              + f"   (weights {fmt(m['weight_bytes']).strip()})")
        print(ROW_HDR)
        for r in m["rows"]:
            print_mem_row(r, per, fixed, f"{AMBER}彩排 {r['date'][:16]}{RESET}")
        s = slope_line(m["rows"], per)
        if s:
            print(f"  {s}")
    keys = [k for k in cfg["demo"]["order"] if k in rec["models"] and rec["models"][k]["kv_bytes_per_token"]]
    if len(keys) >= 2:
        a, b_ = rec["models"][keys[0]], rec["models"][keys[1]]
        print(f"\n  slope ratio {a.get('short', a['label'])} / {b_.get('short', b_['label'])}: formula {a['kv_bytes_per_token'] / b_['kv_bytes_per_token']:.2f}"
              f", measured {(a['rows'][-1]['cache_nbytes'] - a['rows'][0]['cache_nbytes']) / (b_['rows'][-1]['cache_nbytes'] - b_['rows'][0]['cache_nbytes']):.2f}")
    print()
    return True


def replay_recall(cfg: dict, fake: bool) -> bool:
    rec = load_json(recall_summary_path(fake))
    if not rec:
        print(f"{RED}沒有 runs/rehearsal/recall_summary.json：先跑 ./present.sh rehearse --part recall{RESET}")
        return False
    banner(f"{AMBER}REPLAY (b) — 課前彩排的紀錄，不是現場{RESET}  {rec['date']}")
    sizes = [s["label"] for s in cfg["sizes"]]
    ordered = {k: rec["models"][k] for k in cfg["demo"]["order"] if k in rec["models"]}
    for key, summ in ordered.items():
        print(f"\n  {BOLD}{summ['label']}{RESET}  Q = {summ['q']} per cell, {summ.get('seconds_total')} s total"
              + (f", peak memory {summ['peak_memory_gb']:.1f} GB" if summ.get("peak_memory_gb") else "")
              + (f"  {AMBER}(system prompt merged into the user turn){RESET}" if summ.get("system_merged") else ""))
        W5.print_table(summ, sizes, tag=f"{AMBER}彩排 {summ['date'][:16]}{RESET}")
    print(f"  {BOLD}the slide's table{RESET}")
    print_slide_table(ordered, cfg)
    return True


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for name, m in cfg["models"].items():
        if args.only and name != args.only:
            continue
        if m.get("optional") and not args.only:
            note(f"略過 {name}（optional；要抓就 --only {name}）")
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {name}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha, ignore_patterns=NON_MODEL_FILES)
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已寫進 versions.lock{RESET}")


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"  {RED}這台不是 Apple silicon：只能用 --fake 演練{RESET}")
    info = device_info() if not args.fake else {}
    rec = {"date": now(), "versions": v, "device": info, "models": {}}
    for key in model_keys(cfg, args):
        b = make_backend(key, cfg, args)
        head = print_memory_header(b, cfg)
        entry = {"model": b.repo, "revision": b.revision, "label": b.label, **{k: v_ for k, v_ in head.items() if k != "contexts"}}
        del b
        # (b) 的 prompt：只載 tokenizer，兩檔固定的 K 在這個 tokenizer 下各幾個 token、template 吃不吃 system
        m = make_recall_model(key, cfg, args, weights=False)
        cal = [W5.calibrate(m, cfg, s) for s in cfg["sizes"]]
        q = W5.make_question(cal[0]["K"], W5.question_seeds(cfg, cal[0]["label"], 1)[0])
        prompt = m.render(cfg, W5.render_user(cfg, W5.arrange(q, cal[0]["K"] // 2), q["answer"][0]))
        print(f"  (b) prompts under this tokenizer: " + ", ".join(f"{c['label']}: K = {c['K']} → {c['tokens_at_K']:,} tokens" for c in cal)
              + f"; max_position_embeddings {m.max_pos:,}" + (f", rope_scaling {json.dumps(m.rope_scaling)}" if m.rope_scaling else ""))
        print(f"      chat template: " + ("system merged into the user turn (template rejected the system role: "
              + str(getattr(m, 'render_error', '')) + ")" if getattr(m, "system_merged", False) else "system role accepted")
              + (f"; enable_thinking accepted: {m.think_supported}" if m.think_supported is not None else ""))
        print(f"      first line of the rendered prompt: {GREY}{prompt.splitlines()[0][:100]!r}{RESET}\n")
        entry.update({"recall_calibration": cal, "system_merged": getattr(m, "system_merged", None), "render_error": getattr(m, "render_error", None),
                      "think_supported": m.think_supported, "max_position_embeddings": m.max_pos, "rope_scaling": m.rope_scaling})
        rec["models"][key] = entry
        del m
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    path = REHEARSAL / ("inspect.fake.json" if args.fake else "inspect.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def cmd_probe(args, cfg):
    """每個模型：一個普通問題 + 一題 K = 5 的 key-value retrieval，印原始回覆。回覆不成句就是載入的模型有問題，不是 recall。"""
    for key in model_keys(cfg, args):
        m = make_recall_model(key, cfg, args)
        banner(f"probe  ·  {m.label}" + (f"  {RED}[FAKE]{RESET}" if m.fake else ""))
        for q in cfg["demo"].get("probe_questions", ["What is the capital of France? Answer in one word."]):
            msgs = [{"role": "user", "content": q}]
            if m.fake:
                text, st = "(fake) Paris", {"seconds": 0.0}
            else:
                try:
                    prompt = m.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    prompt = m.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                text, st = m.answer(cfg, prompt)
            print(f"  {BOLD}Q:{RESET} {q}\n  {BOLD}A:{RESET} {text.strip()[:200]!r}   {GREY}{st.get('seconds')} s{RESET}")
        for K in (5, 20):
            q = W5.make_question(K, int(cfg["demo"]["seed"]) + 7 * K)
            r = W5.ask(m, cfg, q, max(1, K // 2))
            print(f"  {BOLD}K = {K}, answer at {max(1, K // 2)}:{RESET} {W5.mark(r)}  reply {r['reply'][:120]!r}   {GREY}{r['seconds']} s, {r.get('prompt_tokens')} tokens{RESET}"
                  + (f"   {AMBER}{W5.describe_wrong(r)}{RESET}" if not r["em"] and W5.describe_wrong(r) else ""))
        print()
        del m


def cmd_rehearse(args, cfg):
    parts = [args.part] if args.part else ["memory", "recall"]
    Q = int(args.q or cfg["demo"]["q"])
    smoke = args.q is not None and Q < int(cfg["demo"]["q"])
    out_dir = RUNS / "smoke" if smoke else REHEARSAL
    sizes = sizes_for(cfg, args.size)
    t_all = time.time()
    for key in model_keys(cfg, args):
        spec = cfg["models"][key]
        allowed = spec.get("parts") or ["memory", "recall"]
        if "memory" in parts and "memory" in allowed:
            b = make_backend(key, cfg, args)
            rec = run_memory(b, cfg)
            p = save_memory(rec, args.fake, out_dir)
            note(f"→ {p.relative_to(HERE)}")
            del b
            try:
                W3._mx_fn("clear_cache")()
            except Exception:
                pass
        if "recall" in parts and "recall" in allowed:
            m = make_recall_model(key, cfg, args)
            banner(f"(b) key-value retrieval  ·  {m.label}" + (f"  {AMBER}[smoke test, Q = {Q}]{RESET}" if smoke else ""))
            summary, full = run_recall(m, cfg, Q, sizes)
            p = save_recall(key, summary, full, args.fake, out_dir)
            note(f"→ {p.relative_to(HERE)}  ({summary['seconds_total']} s)")
            del m
    note(f"\n全部 {time.time() - t_all:.0f} s")
    if "recall" in parts and not args.model:
        rec = load_json(recall_summary_path(args.fake, out_dir))
        if rec and len(rec["models"]) >= 2:
            print_slide_table({k: rec["models"][k] for k in cfg["demo"]["order"] if k in rec["models"]}, cfg)


def wait(msg: str) -> str:
    try:
        key = input(f"\n{GREY}[{msg} · m = (a) 的表 · r = (b) 的表 · x = 離開]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        key = "x"
    print()
    return key


def cmd_show(args, cfg):
    """課堂：兩個模型都載著；每按一次 Enter 出一題，兩個模型各答一次（同一題、同一個位置）。"""
    keys = model_keys(cfg, args)
    models = {k: make_recall_model(k, cfg, args) for k in keys}
    banner(f"{cfg['demo']['title']}" + (f"  {RED}[FAKE]{RESET}" if args.fake else ""))
    live = [(s, p) for s, p in cfg["demo"]["live_questions"]]
    Ks = {s["label"]: int(s["k"]) for s in cfg["sizes"]}
    print("  live questions (not the rehearsed ones): " + ", ".join(f"{s} answer at {p}" for s, p in live)
          + "; each is put to " + " then ".join(models[k].label for k in keys) + ".")
    LIVE.mkdir(parents=True, exist_ok=True)
    log = LIVE / f"{now()[:10].replace('-', '')}.jsonl"
    i = 0
    while True:
        key = wait(f"Enter = question {i + 1}/{len(live)}: {live[i][0]}, answer at {live[i][1]}" if i < len(live) else "no more questions")
        if key == "x":
            break
        if key == "m":
            replay_memory(cfg, args.fake, interactive=True)
            continue
        if key == "r":
            replay_recall(cfg, args.fake)
            continue
        if i >= len(live):
            continue
        size, pos_name = live[i]
        K = Ks[size]
        seed = W5.question_seeds(cfg, size, len(live), live=True)[i]
        q = W5.make_question(K, seed)
        label, pos = dict((n, (lab, p)) for n, lab, p in W5.positions_for(K))[pos_name]
        pairs = W5.arrange(q, pos)
        head = json.dumps(dict(pairs[:1]))[:-1] + ", … (" + str(K) + " pairs) … }"
        print(f"  {BOLD}JSON data:{RESET}  {GREY}{head}{RESET}")
        print(f"  {BOLD}Key:{RESET} {json.dumps(q['answer'][0])}   {GREY}(pair {pos} of {K} — {label}; ≈ {size}){RESET}")
        for k in keys:
            m = models[k]
            r = W5.ask(m, cfg, q, pos)
            r.update({"size": size, "pos_name": pos_name, "model_key": k, "date": now()})
            tps = f", prefill {r['prompt_tps']:,.0f} tok/s" if r.get("prompt_tps") else ""
            print(f"  {BOLD}{m.short:<16}{RESET} {W5.mark(r)}  {r['reply']!r:<42} {GREY}{r['seconds']:.1f} s, {r.get('prompt_tokens') or 0:,} prompt tokens{tps}{RESET}"
                  + (f"   {AMBER}{W5.describe_wrong(r)}{RESET}" if not r["em"] and W5.describe_wrong(r) else ""))
            append_jsonl(log, {"model": m.repo, **r})
        print(f"  {GREY}expected {q['answer'][1]}{RESET}")
        i += 1
    print()


def cmd_replay(args, cfg):
    part = args.part or "all"
    if part in ("memory", "all"):
        replay_memory(cfg, args.fake, interactive=(part == "memory"))
    if part in ("recall", "all"):
        replay_recall(cfg, args.fake)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（數字與回答是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--model", help="只跑 demo_config.toml 的 [models.*] 其中一個（qwen3_8b / falcon_h1 / mamba2）")
    p.add_argument("--only", help="fetch：只抓一個模型")
    p.add_argument("--part", choices=["memory", "recall"], help="rehearse／replay：只做 (a) memory 或 (b) recall")
    p.add_argument("--size", help="rehearse：只做一檔（4k / 16k）")
    p.add_argument("--q", type=int, help="rehearse：每格幾題；小於 demo_config 的 q 就是煙霧測試，寫到 runs/smoke/")
    sub = p.add_subparsers(dest="cmd", required=True)   # present.sh 把選項放在子命令前面，所以選項全掛在主 parser
    for c in ("fetch", "inspect", "probe", "rehearse", "show", "replay"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "probe": cmd_probe, "rehearse": cmd_rehearse, "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
