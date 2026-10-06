#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 2：KV cache 的記憶體實測對照公式（投影片 p25「Live: predict the curve from config.json, then measure it」）。

    ./present.sh inspect    課前：讀 config.json 的三個數、探 KV 的 dtype、印每 token 的 bytes 與各 context 的預測值
    ./present.sh rehearse   課前彩排：每個 context 各跑一次 prefill，記 cache 張量大小、Δactive、Δpeak、耗時 → runs/rehearsal/curve.json
    ./present.sh            課堂：Enter 一列一列量（彩排時太慢的列改顯示彩排值並標「非現場」）；q 離開
    ./present.sh replay     現場失敗：只顯示彩排的表
    ./present.sh fetch      模型已在 W2 demo 3 下載並鎖定，通常不必；換模型時才跑
    ./present.sh fake       沒有 mlx 也能演練流程（數字是假的，畫面會標明）

量法（統一記憶體下量到的是整個行程的配置，所以一定要扣基線）：
  1. 載入權重後跑一次 8 個 token 的暖身，丟掉 cache，mx.clear_cache()，reset_peak_memory()，記 active 為基線；
  2. 對長度 L 的假 token（亂數 id）分塊 prefill（prefill_step_size，與 mlx-lm generate_step 相同），每塊只 eval cache 的 state
     ——logits 從頭到尾不算，與 mlx-lm 的 prefill 一樣；
  3. 三個實測數：cache 張量本身的 bytes（KVCache.nbytes，配置以 256 個 token 為一步，L 是 256 的倍數時恰等於公式）、
     Δactive（prefill 結束時仍配置著的 − 基線 ≈ KV）、Δpeak（過程中的峰值 − 基線 = KV + 計算時的暫存）。
  公式：M_KV = 2 · n_layer · n_kv · d_head · L · b，b 由 cache 張量的 dtype 決定（4-bit 模型的 KV 仍是 16-bit，inspect 會印出來）。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）。Python 3.11+（tomllib）。
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, _mx_fn, append_jsonl,  # noqa: E402
                    banner, device_info, note, now, read_lock, reset_peak_memory,
                    revision_key, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
REHEARSAL = HERE / "runs" / "rehearsal"
LIVE = HERE / "runs" / "live"
KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


# ── config.json 的三個數 ──────────────────────────────────────────
def read_model_config(path: Path) -> dict:
    """讀 config.json；形狀可能藏在 text_config 底下（Qwen3.8 那種多模態包裝，W2 demo 1 踩過）。"""
    raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    tc = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}

    def get(*keys, default=None):
        for k in keys:
            for src in (raw, tc):
                if k in src and src[k] is not None:
                    return src[k]
        return default

    n_layer = get("num_hidden_layers", "n_layer", "num_layers")
    n_head = get("num_attention_heads", "n_head")
    n_kv = get("num_key_value_heads", default=n_head)
    hidden = get("hidden_size", "d_model")
    head_dim = get("head_dim") or (hidden // n_head if hidden and n_head else None)
    return {"model_type": get("model_type"), "num_hidden_layers": n_layer, "num_attention_heads": n_head,
            "num_key_value_heads": n_kv, "hidden_size": hidden, "head_dim": head_dim,
            "head_dim_in_config": get("head_dim") is not None,
            "max_position_embeddings": get("max_position_embeddings"), "vocab_size": get("vocab_size"),
            "rope_scaling": get("rope_scaling"), "quantization": raw.get("quantization")}


def kv_bytes_per_token(c: dict, b: int) -> int:
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * c["head_dim"] * b


def fmt(nbytes: float) -> str:
    if nbytes >= GiB:
        return f"{nbytes / GiB:6.3f} GiB"
    if nbytes >= MiB:
        return f"{nbytes / MiB:6.1f} MiB"
    return f"{nbytes / KiB:6.1f} KiB"


def resolve_contexts(cfg: dict, max_pos: int | None) -> list[tuple[int, str]]:
    out = []
    cap = cfg["demo"].get("max_context") or 0
    for c in cfg["demo"]["contexts"]:
        if c == "max":
            if not max_pos:
                continue
            L = min(max_pos, cap) if cap else max_pos
            out.append((int(L), "model maximum" + (f" (capped from {max_pos})" if cap and cap < max_pos else "")))
        else:
            out.append((int(c), f"{int(c) // 1024}k"))
    return out


# ── 後端 ─────────────────────────────────────────────────────────
class MLXBackend:
    fake = False

    def __init__(self, repo: str, revision: str | None, offline: bool):
        from huggingface_hub import snapshot_download
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load
        self.mx = mx
        t0 = time.time()
        # ignore_patterns：離線時 huggingface_hub ≥1.3 會核對 snapshot 完整性（W2 demo 1 踩過）
        self.path = snapshot_download(repo, revision=revision or None, local_files_only=offline,
                                      ignore_patterns=["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"])
        self.model, self.tokenizer = load(self.path)
        self.config = read_model_config(self.path)
        self.weight_bytes = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        self.load_seconds = time.time() - t0
        self.kv_dtype, self.kv_shape = self._probe()
        self.active0 = None

    # mlx 的記憶體 API 在 0.2x 從 mx.metal 搬到 mx 頂層；_mx_fn 兩邊都找（common.py）。
    def _active(self) -> int:
        fn = _mx_fn("get_active_memory")
        return int(fn()) if fn else 0

    def _peak(self) -> int:
        fn = _mx_fn("get_peak_memory")
        return int(fn()) if fn else 0

    def _clear(self) -> None:
        fn = _mx_fn("clear_cache")
        if fn:
            fn()

    def _new_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def _prefill(self, ids: list[int], step: int):
        cache = self._new_cache()
        for s in range(0, len(ids), step):
            chunk = self.mx.array(ids[s:s + step])[None]
            self.model(chunk, cache=cache)                  # 與 mlx-lm generate_step 的 prefill 相同：
            self.mx.eval([c.state for c in cache])          # 只 eval cache，logits 不算
            self._clear()
        return cache

    def _probe(self):
        cache = self._prefill(list(range(1, 9)), 8)
        c = cache[0]
        dtype, shape = str(c.keys.dtype), tuple(int(x) for x in c.keys.shape)
        del cache
        self._clear()
        return dtype, shape

    def bytes_per_elem(self) -> int:
        d = self.kv_dtype
        for name, b in (("float32", 4), ("float16", 2), ("bfloat16", 2), ("float8", 1), ("uint8", 1)):
            if name in d:
                return b
        return 2

    def baseline(self) -> int:
        self._clear()
        reset_peak_memory()
        self.active0 = self._active()
        return self.active0

    def measure(self, L: int, step: int, seed: int) -> dict:
        rng = random.Random(seed)
        vocab = self.config["vocab_size"] or 32000
        ids = [rng.randrange(1000, vocab - 1000) for _ in range(L)]
        self._clear()
        reset_peak_memory()
        a0 = self._active()
        t0 = time.time()
        cache = self._prefill(ids, step)
        seconds = time.time() - t0
        nbytes = sum(int(getattr(c, "nbytes", 0)) for c in cache)
        a1, pk = self._active(), self._peak()
        del cache
        self._clear()
        return {"L": L, "cache_nbytes": nbytes, "active_delta": a1 - a0, "peak_delta": pk - a0,
                "seconds": seconds, "prefill_tps": L / seconds if seconds else None, "fake": False}


class FakeBackend:
    """--fake：沒有 mlx 也能演練流程。數字是假的（預測值加一點雜訊），畫面會標明。"""
    fake = True

    def __init__(self, repo: str, revision: str | None, offline: bool):
        self.path, self.load_seconds = "(fake)", 0.0
        self.config = {"model_type": "fake", "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
                       "hidden_size": 4096, "head_dim": 128, "head_dim_in_config": False,
                       "max_position_embeddings": 131072, "vocab_size": 128256, "rope_scaling": None, "quantization": None}
        self.weight_bytes = 4_500_000_000
        self.kv_dtype, self.kv_shape = "mlx.core.float16 (fake)", (1, 8, 256, 128)
        self.active0 = None

    def bytes_per_elem(self) -> int:
        return 2

    def baseline(self) -> int:
        self.active0 = self.weight_bytes
        return self.active0

    def measure(self, L: int, step: int, seed: int) -> dict:
        per = kv_bytes_per_token(self.config, 2)
        time.sleep(min(0.3 + L / 200_000, 1.5))
        return {"L": L, "cache_nbytes": per * L, "active_delta": int(per * L * 1.002),
                "peak_delta": int(per * L * 1.05 + 300 * MiB), "seconds": L / 2500, "prefill_tps": 2500.0, "fake": True}


def make_backend(args, cfg: dict):
    m = cfg["models"][args.model]
    revision = m.get("revision") or read_lock().get(revision_key(m["repo"])) or None
    cls = FakeBackend if args.fake else MLXBackend
    note(f"載入 {m['label']}  ←  {m['repo']}" + (f"@{revision[:10]}" if revision else ""))
    b = cls(m["repo"], revision, offline=not args.online)
    note(f"載入完成：{b.load_seconds:.1f} 秒")
    b.label, b.repo, b.revision = m["label"], m["repo"], revision
    return b


# ── 畫面 ─────────────────────────────────────────────────────────
def print_header(b, cfg: dict) -> dict:
    """三個數、每 token 的 bytes、各 context 的預測值。回傳給 json 用的 dict。"""
    c = b.config
    bpe = b.bytes_per_elem()
    per = kv_bytes_per_token(c, bpe)
    banner(f"{cfg['demo']['title']}  ·  {b.label}" + (f"  {RED}[FAKE]{RESET}" if b.fake else ""))
    print(f"  config.json ({c['model_type']}):")
    print(f"    num_hidden_layers     = {BOLD}{c['num_hidden_layers']}{RESET}")
    print(f"    num_key_value_heads   = {BOLD}{c['num_key_value_heads']}{RESET}"
          f"   (num_attention_heads = {c['num_attention_heads']}"
          f" → {'MHA' if c['num_key_value_heads'] == c['num_attention_heads'] else 'GQA' if c['num_key_value_heads'] > 1 else 'MQA'})")
    print(f"    head_dim              = {BOLD}{c['head_dim']}{RESET}"
          + ("" if c["head_dim_in_config"] else f"   (not in config: hidden_size {c['hidden_size']} / {c['num_attention_heads']} heads)"))
    print(f"    max_position_embeddings = {c['max_position_embeddings']}"
          + (f"   rope_scaling = {json.dumps(c['rope_scaling'])}" if c["rope_scaling"] else ""))
    print(f"  KV cache tensor: dtype {BOLD}{b.kv_dtype}{RESET} → b = {bpe} bytes"
          f"   (shape of layer 0 keys after an 8-token probe: {b.kv_shape}; the cache grows in steps of 256 tokens)")
    print(f"  weights in memory: {fmt(b.weight_bytes)}" + (f"   quantization = {json.dumps(c['quantization'])}" if c["quantization"] else ""))
    print(f"\n  per token: 2 × {c['num_hidden_layers']} × {c['num_key_value_heads']} × {c['head_dim']} × {bpe}"
          f" = {BOLD}{per:,} bytes = {per / KiB:g} KiB{RESET}")
    rows = resolve_contexts(cfg, c["max_position_embeddings"])
    print(f"\n  {'context':<24}{'predicted M_KV':>16}")
    for L, label in rows:
        print(f"  {label + ' (' + f'{L:,}' + ')':<24}{fmt(per * L):>16}")
    print()
    return {"config": c, "bytes_per_elem": bpe, "kv_dtype": b.kv_dtype, "kv_probe_shape": list(b.kv_shape),
            "weight_bytes": b.weight_bytes, "kv_bytes_per_token": per, "contexts": [{"L": L, "label": lab} for L, lab in rows]}


ROW_HDR = f"  {'context':<22}{'predicted':>12}{'cache tensors':>15}{'Δactive':>12}{'Δpeak':>12}{'transient':>12}   {'time':<22}"


def print_row(label: str, per: int, r: dict, flag: str = "") -> None:
    L = r["L"]
    pred = per * L
    trans = r["peak_delta"] - r["cache_nbytes"]
    t = f"{r['seconds']:.1f} s ({r['prefill_tps']:,.0f} tok/s)" if r.get("prefill_tps") else f"{r['seconds']:.1f} s"
    print(f"  {label + ' (' + f'{L:,}' + ')':<22}{fmt(pred):>12}{fmt(r['cache_nbytes']):>15}{fmt(r['active_delta']):>12}"
          f"{fmt(r['peak_delta']):>12}{fmt(trans):>12}   {t:<22}{flag}")


def slope_check(per: int, rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("cache_nbytes")]
    if len(ok) < 2:
        return
    a, z = ok[0], ok[-1]
    slope = (z["active_delta"] - a["active_delta"]) / (z["L"] - a["L"])
    print(f"\n  slope of Δactive between {a['L']:,} and {z['L']:,} tokens: {slope / KiB:.1f} KiB/token"
          f"   vs. formula {per / KiB:g} KiB/token  ({slope / per * 100:.1f}%)")


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for name, m in cfg["models"].items():
        if args.only and name != args.only:
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {name}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha,
                          ignore_patterns=["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"])
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已寫進 versions.lock{RESET}")


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"  {RED}這台不是 Apple silicon：只能用 --fake 演練{RESET}")
    info = device_info() if not args.fake else {}
    for k in ("device_name", "memory_size", "max_recommended_working_set_size"):
        if k in info:
            val = info[k]
            print(f"  {k:<16} {fmt(val) if isinstance(val, (int, float)) else val}")
    b = make_backend(args, cfg)
    head = print_header(b, cfg)
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    rec = {"date": now(), "model": b.repo, "revision": b.revision, "versions": v, "device": info, **head}
    path = REHEARSAL / ("inspect.fake.json" if b.fake else "inspect.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def _run_rows(b, cfg, rows, per, on_row, rehearsed: dict | None = None, live_limit: float = 0.0, interactive=False):
    step = int(cfg["demo"]["prefill_step_size"])
    seed = int(cfg["demo"]["seed"])
    out = []
    print(ROW_HDR)
    for i, (L, label) in enumerate(rows):
        if interactive:
            key = input(f"  {DIM}Enter = {label} ({L:,})   q = 離開{RESET} ").strip().lower()
            if key == "q":
                break
            print("\033[1A\033[2K", end="")
        prev = (rehearsed or {}).get(str(L))
        if prev and live_limit and prev["seconds"] > live_limit:
            print_row(label, per, prev, f"{AMBER}彩排值 {prev['date'][:10]}，非現場（彩排耗時 {prev['seconds']:.0f} s > {live_limit:g} s）{RESET}")
            out.append({**prev, "replayed": True})
            continue
        r = b.measure(L, step, seed + i)
        r["label"] = label
        flag = f"{RED}[FAKE]{RESET}" if r.get("fake") else ""
        if prev and not r.get("fake"):
            flag += f"{GREY}(彩排 Δactive {fmt(prev['active_delta']).strip()}){RESET}"
        print_row(label, per, r, flag)
        out.append(r)
        on_row(r)
    slope_check(per, out)
    return out


def cmd_rehearse(args, cfg):
    b = make_backend(args, cfg)
    head = print_header(b, cfg)
    per = head["kv_bytes_per_token"]
    rows = resolve_contexts(cfg, b.config["max_position_embeddings"])
    base = b.baseline()
    note(f"基線 active = {fmt(base).strip()}（權重 {fmt(b.weight_bytes).strip()} + 其他）\n")
    results = _run_rows(b, cfg, rows, per, on_row=lambda r: None)
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    rec = {"date": now(), "model": b.repo, "revision": b.revision, "versions": versions(), "device": device_info() if not b.fake else {},
           "fake": b.fake, "baseline_active_bytes": base, "prefill_step_size": cfg["demo"]["prefill_step_size"], **head,
           "rows": [{**r, "date": now()} for r in results]}
    path = REHEARSAL / ("curve.fake.json" if b.fake else "curve.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"\n→ {path.relative_to(HERE)}")


def _load_rehearsal(fake: bool) -> dict | None:
    path = REHEARSAL / ("curve.fake.json" if fake else "curve.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_show(args, cfg):
    b = make_backend(args, cfg)
    head = print_header(b, cfg)
    per = head["kv_bytes_per_token"]
    rows = resolve_contexts(cfg, b.config["max_position_embeddings"])
    reh = _load_rehearsal(b.fake)
    rehearsed = {str(r["L"]): r for r in reh["rows"]} if reh and reh.get("model") == b.repo else {}
    if reh and not rehearsed:
        note(f"彩排紀錄是另一個模型（{reh.get('model')}），不拿來對照")
    base = b.baseline()
    note(f"基線 active = {fmt(base).strip()}。每列先請學生填預測欄，再按 Enter。\n")
    LIVE.mkdir(parents=True, exist_ok=True)
    log = LIVE / f"{now()[:10].replace('-', '')}.jsonl"
    _run_rows(b, cfg, rows, per, on_row=lambda r: append_jsonl(log, {"date": now(), "model": b.repo, **r}),
              rehearsed=rehearsed, live_limit=float(cfg["demo"].get("live_limit_s") or 0), interactive=True)
    print()


def cmd_replay(args, cfg):
    reh = _load_rehearsal(args.fake)
    if not reh:
        print(f"{RED}沒有 runs/rehearsal/curve.json：先跑 ./present.sh rehearse{RESET}")
        return
    per = reh["kv_bytes_per_token"]
    c = reh["config"]
    banner(f"{AMBER}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {reh['date']}  {reh['model']}")
    print(f"  {c['num_hidden_layers']} layers × {c['num_key_value_heads']} KV heads × head_dim {c['head_dim']} × b = {reh['bytes_per_elem']}"
          f"  →  {per / KiB:g} KiB / token   (KV dtype {reh['kv_dtype']}; weights {fmt(reh['weight_bytes']).strip()})")
    print(ROW_HDR)
    for r in reh["rows"]:
        print_row(r.get("label", ""), per, r, f"{AMBER}彩排 {r['date'][:16]}{RESET}")
    slope_check(per, reh["rows"])
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（數字是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--model", default="primary", help="demo_config.toml 的 [models.*] 名稱（primary / backup）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "inspect", "rehearse", "show", "replay"):
        sp = sub.add_parser(c)
        sp.add_argument("--only")
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
