#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 4：StreamingLLM 的 sink 對照（投影片 p43「Replay (if time): demo 4 — a sliding window with and without the sink」）。

    ./present.sh inspect    課前：只載 tokenizer，數文字的 token、看 mlx-lm 的 RotatingKVCache 怎麼留 token、
                            驗 RoPE 的旋轉可以合成與反轉（重編位置靠它）→ runs/rehearsal/inspect.json
    ./present.sh rehearse   課前彩排：同一段文字、四種 cache，逐 token 的 NLL → runs/rehearsal/curve.{json,svg}
    ./present.sh            課堂：replay——每條線 Enter 一次（分格的 perplexity 方塊圖），最後印 p43 的表
    ./present.sh replay     同上
    ./present.sh fetch      模型已在 W2 demo 3 下載並鎖定，通常不必；換模型時才跑
    ./present.sh fake       沒有 mlx 也能演練流程（曲線是假的，畫面會標明）

四種 cache（同一段文字、同一個模型、同一個 tokenizer，只換 KV cache 的留法）：
  full                 全留（分段 prefill）
  window               只留最近 W 個 token 的 KV；第一個 token 在第 W 步被丟掉——sink 跟著沒了
  window_sink_abs      留最近 W 個 + 最前面 n_sink 個；位置是原始位置。**這就是 mlx-lm 的 RotatingKVCache(keep=n_sink)**
                       （llama.py 在 key 進 cache 之前就用 cache.offset 旋轉，cache 從頭到尾沒看過位置）
  window_sink_reindex  同上，但 sink 的位置改成 cache 內的相對位置（StreamingLLM §3 的做法）——自己寫的 SinkWindowCache，
                       每一步把 sink 的 key 重新旋轉到「離 query 正好 cache 長度」的位置

量的是逐 token 的 NLL（nats）；表格報 token 位置 < W 與 ≥ W 兩段的 perplexity = exp(mean NLL)。
兩萬字元遠在 Llama 3.1 的名義長度之內，所以 window 那條若崩，崩的原因是 sink 被丟掉，不是位置超過訓練長度。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）。Python 3.11+（tomllib）。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect as pyinspect
import json
import math
import platform
import random
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info,  # noqa: E402
                    locked_revision, note, now, peak_memory_gb, reset_peak_memory, revision_key,
                    versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "config.json",
                   "generation_config.json"]

# 四種 cache，依投影片 p43 表格的列序（第四列是建 demo 時加的：庫的 keep 與論文的重編不是同一件事）。
VARIANTS = [
    ("full", "full cache"),
    ("window", "window W, no sink"),
    ("window_sink_abs", "window W + n_sink, original positions (library keep)"),
    ("window_sink_reindex", "window W + n_sink, re-indexed (StreamingLLM §3)"),
]
BLOCKS = "▁▂▃▄▅▆▇█"


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def load_text(cfg: dict) -> tuple[str, dict]:
    p = (HERE / cfg["text"]["path"]).resolve()
    if not p.exists():
        sys.exit(f"{RED}找不到文字 {p}：W4 demo 1 的 README 寫了去哪下載 tinyshakespeare.txt{RESET}")
    raw = p.read_text(encoding="utf-8")
    text = raw[: int(cfg["text"]["chars"])]
    meta = {"path": cfg["text"]["path"], "chars": len(text), "sha256_of_slice": hashlib.sha256(text.encode()).hexdigest()[:16],
            "first_line": text.splitlines()[0][:60] if text else ""}
    return text, meta


# ── 自己寫的 cache：sink + 視窗，位置可重編 ─────────────────────────
class SinkWindowCache:
    """一層的 KV cache：留最前面 n_sink 個 token 與最近 window 個 token。一次只吃一個 token（S = 1）。

    llama.py 在呼叫 update_and_fetch 之前已經用 cache.offset（＝絕對位置 t）旋轉了 q 與 k，所以：
      - 視窗裡的 key 停在它們的絕對位置，與 query 的相對距離本來就對；
      - sink 的 key 到貨時是旋轉到位置 j（j < n_sink）的。reindex=False 就這樣留著（＝ mlx-lm 的 keep）；
        reindex=True 時先用負的 offset 轉回未旋轉的 key，之後每一步把 n_sink 個 sink 當成一段序列
        用 offset = t − C + 1 一次旋轉（C = 現在 cache 裡的 token 數），第 j 個 sink 就落在位置 t − (C − 1 − j)——
        也就是它在 cache 裡的相對位置，與 StreamingLLM §3「用 cache 內的位置」一致。
        旋轉可以合成（rope(rope(x, a), b) = rope(x, a + b)），inspect 會驗。
    """

    def __init__(self, window: int, n_sink: int, reindex: bool, rope=None):
        self.window, self.n_sink, self.reindex, self.rope = int(window), int(n_sink), bool(reindex), rope
        if self.reindex and self.n_sink > 0 and rope is None:
            raise ValueError("reindex=True 需要該層的 rope 模組")
        self.offset = 0            # llama.py 讀這個當 RoPE 的位置：下一個 token 的絕對位置
        self.sink_k = self.sink_v = None   # (B, H, n_sink, D)；reindex 時存的是未旋轉的 key
        self.win_k = self.win_v = None     # (B, H, ≤ window, D)，最舊的在前
        self.evicted = 0

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        if keys.shape[2] != 1:
            raise ValueError("SinkWindowCache 一次只吃一個 token")
        t = self.offset
        if t < self.n_sink:
            k = self.rope(keys, offset=-t) if (self.reindex and t > 0) else keys   # 轉回位置 0
            self.sink_k = k if self.sink_k is None else mx.concatenate([self.sink_k, k], axis=2)
            self.sink_v = values if self.sink_v is None else mx.concatenate([self.sink_v, values], axis=2)
        else:
            self.win_k = keys if self.win_k is None else mx.concatenate([self.win_k, keys], axis=2)
            self.win_v = values if self.win_v is None else mx.concatenate([self.win_v, values], axis=2)
            if self.win_k.shape[2] > self.window:
                self.win_k, self.win_v = self.win_k[..., 1:, :], self.win_v[..., 1:, :]
                self.evicted += 1
        self.offset = t + 1
        if self.sink_k is None:
            return self.win_k, self.win_v
        if self.win_k is None:
            sk = self.rope(self.sink_k, offset=0) if self.reindex else self.sink_k
            return sk, self.sink_v
        if self.reindex:
            C = self.sink_k.shape[2] + self.win_k.shape[2]
            sk = self.rope(self.sink_k, offset=t - C + 1)   # 第 j 個 sink → 位置 t − (C − 1 − j)
        else:
            sk = self.sink_k
        return mx.concatenate([sk, self.win_k], axis=2), mx.concatenate([self.sink_v, self.win_v], axis=2)

    # llama.py 的 create_attention_mask 會呼叫這個；S = 1 不需要 mask
    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        if N != 1:
            raise ValueError("SinkWindowCache 一次只吃一個 token")
        return None

    def size(self) -> int:
        return (0 if self.sink_k is None else self.sink_k.shape[2]) + (0 if self.win_k is None else self.win_k.shape[2])

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self):
        return self.sink_k, self.sink_v, self.win_k, self.win_v, self.offset

    @state.setter
    def state(self, v):
        self.sink_k, self.sink_v, self.win_k, self.win_v, self.offset = v


# ── 後端 ─────────────────────────────────────────────────────────
class MLXModel:
    fake = False

    def __init__(self, key: str, spec: dict, offline: bool, weights: bool = True):
        from huggingface_hub import snapshot_download
        t0 = time.time()
        rev = locked_revision(spec["repo"], spec.get("revision"))
        kw = {"ignore_patterns": NON_MODEL_FILES} if weights else {"allow_patterns": TOKENIZER_FILES}
        self.path = snapshot_download(spec["repo"], revision=rev, local_files_only=offline, **kw)
        self.key, self.label, self.repo, self.revision = key, spec["label"], spec["repo"], rev
        if weights:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.path)
        else:
            from transformers import AutoTokenizer
            self.model, self.tokenizer = None, AutoTokenizer.from_pretrained(self.path)
        self.load_seconds = time.time() - t0
        self.config = json.loads((Path(self.path) / "config.json").read_text(encoding="utf-8"))
        self.max_pos = self.config.get("max_position_embeddings")
        self.rope_scaling = self.config.get("rope_scaling")

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)
        return [int(i) for i in ids]

    def bos(self):
        return getattr(self.tokenizer, "bos_token_id", None)

    def layers(self):
        return self.model.model.layers

    def rope_of(self, layer):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        rope = getattr(attn, "rope", None)
        if rope is None:
            raise RuntimeError("找不到這層的 rope 模組（layer.self_attn.rope）；換了模型就要看它的原始碼")
        return rope

    def make_rope_from_config(self):
        """不載權重也能拿到同一種 RoPE（inspect 用）。"""
        from mlx_lm.models.rope_utils import initialize_rope
        c = self.config
        head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        return initialize_rope(head_dim, c.get("rope_theta", 10000.0), bool(c.get("rope_traditional", False)),
                               c.get("rope_scaling"), c.get("max_position_embeddings")), head_dim

    # 四種 cache
    def make_caches(self, name: str, cfg: dict):
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        W, n_sink = int(cfg["demo"]["window"]), int(cfg["demo"]["n_sink"])
        if name == "full":
            return [KVCache() for _ in self.layers()]
        if name == "window":
            return [RotatingKVCache(max_size=W, keep=0) for _ in self.layers()]
        if name == "window_sink_abs":
            return [RotatingKVCache(max_size=W + n_sink, keep=n_sink) for _ in self.layers()]
        if name == "window_sink_reindex":
            return [SinkWindowCache(W, n_sink, reindex=True, rope=self.rope_of(l)) for l in self.layers()]
        if name == "window_sink_abs_mine":   # --check：自己的 class 關掉重編，應與庫的 keep 一致
            return [SinkWindowCache(W, n_sink, reindex=False) for _ in self.layers()]
        raise ValueError(name)

    def nll_full(self, ids: list[int], cfg: dict, progress) -> list[float]:
        import mlx.core as mx
        caches = self.make_caches("full", cfg)
        x = mx.array(ids)
        chunk = int(cfg["demo"]["chunk"])
        out = [float("nan")]   # 位置 0 沒有預測
        for s in range(0, len(ids) - 1, chunk):
            e = min(s + chunk, len(ids) - 1)
            logits = self.model(x[s:e][None], cache=caches)[0].astype(mx.float32)       # (S, V)
            tgt = x[s + 1:e + 1]
            lse = mx.logsumexp(logits, axis=-1)
            nll = lse - mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0]
            mx.eval(nll)
            out.extend(nll.tolist())
            progress(e)
        return out

    def nll_stepwise(self, ids: list[int], caches, cfg: dict, progress) -> list[float]:
        import mlx.core as mx
        x = mx.array(ids)
        out = [float("nan")]
        for t in range(len(ids) - 1):
            logits = self.model(x[t:t + 1][None], cache=caches)[0, -1].astype(mx.float32)   # (V,)
            nll = mx.logsumexp(logits) - logits[int(ids[t + 1])]
            mx.eval(nll)
            out.append(nll.item())
            if (t + 1) % 256 == 0:
                try:
                    mx.clear_cache()
                except AttributeError:
                    pass
            progress(t + 1)
        return out

    def rope_checks(self, rope, head_dim: int) -> dict:
        """重編位置靠三件事：旋轉可合成、負 offset 可反轉、score 只看相對位置。"""
        import mlx.core as mx
        x = mx.random.normal((1, 2, 1, head_dim))
        a = mx.abs(rope(rope(x, offset=3), offset=7) - rope(x, offset=10)).max().item()
        try:
            b = mx.abs(rope(rope(x, offset=5), offset=-5) - x).max().item()
            neg_ok = True
        except Exception as e:   # noqa: BLE001
            b, neg_ok = str(e), False
        blk = mx.random.normal((1, 2, 4, head_dim))
        per = mx.concatenate([rope(blk[..., j:j + 1, :], offset=11 + j) for j in range(4)], axis=2)
        c = mx.abs(rope(blk, offset=11) - per).max().item()
        q, k = mx.random.normal((1, 2, 1, head_dim)), mx.random.normal((1, 2, 1, head_dim))
        s1 = (rope(q, offset=100) * rope(k, offset=93)).sum().item()
        s2 = (rope(q, offset=5000) * rope(k, offset=4993)).sum().item()
        return {"compose_max_abs_err": a, "negative_offset_inverts": neg_ok, "invert_max_abs_err": b,
                "block_offset_is_per_element_max_abs_err": c, "score_rel_only_pair": [s1, s2],
                "score_rel_only_abs_err": abs(s1 - s2), "rope_class": type(rope).__name__}


class FakeModel:
    """--fake：沒有 Apple silicon 也能演練。曲線是假的：window 過了 W 之後爆、留 sink 的平；畫面會標明。"""
    fake = True

    def __init__(self, key, spec, offline, weights=True):
        self.key, self.label, self.repo, self.revision = key, spec["label"] + " [FAKE]", spec["repo"], None
        self.path, self.load_seconds = "(fake)", 0.0
        self.config = {"max_position_embeddings": 131072, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}
        self.max_pos, self.rope_scaling = 131072, self.config["rope_scaling"]
        self.rng = random.Random(0)

    def encode(self, text: str) -> list[int]:
        return [128000] + [self.rng.randrange(1000, 120000) for _ in range(int(len(text) / 4.0))]

    def bos(self):
        return 128000

    def fake_curve(self, name: str, n: int, cfg: dict) -> list[float]:
        W, n_sink = int(cfg["demo"]["window"]), int(cfg["demo"]["n_sink"])
        out = [float("nan")]
        base = {"full": 0.0, "window": 0.0, "window_sink_abs": 0.08, "window_sink_reindex": 0.05, "window_sink_abs_mine": 0.08}[name]
        for t in range(1, n):
            rng = random.Random(t)                      # 同一個位置同一份底噪，W 之前四條線才會像真的一樣重合
            v = 2.0 + rng.gauss(0, 0.35) + (0.6 if t < 32 else 0.0) + 0.01 * rng.random()
            if name == "window" and t >= W:
                v += 4.0 + 0.0005 * (t - W) + self.rng.gauss(0, 0.6)
            elif name != "full" and name != "window" and t >= W + n_sink:
                v += base + self.rng.gauss(0, 0.05)
            out.append(max(0.05, v))
            time.sleep(0.0002)
        return out


def make_model(key: str, cfg: dict, args, weights: bool = True):
    spec = cfg["models"][key]
    cls = FakeModel if args.fake else MLXModel
    rev = locked_revision(spec["repo"], spec.get("revision"))
    note(f"載入 {spec['label']}{'（只載 tokenizer）' if not weights else ''}  ←  {spec['repo']}" + (f"@{rev[:10]}" if rev else ""))
    m = cls(key, spec, offline=not args.online, weights=weights)
    note(f"  完成：{m.load_seconds:.1f} 秒")
    return m


# ── 統計 ─────────────────────────────────────────────────────────
def mean_nll(nll: list[float], lo: int, hi: int) -> float | None:
    xs = [v for v in nll[lo:hi] if v == v]
    return sum(xs) / len(xs) if xs else None


def ppl(x: float | None) -> float | None:
    return math.exp(x) if x is not None else None


def fmt(x: float | None) -> str:
    return f"{x:,.2f}" if x is not None else "—"


def bins_of(nll: list[float], size: int) -> list[dict]:
    out = []
    for s in range(1, len(nll), size):
        e = min(s + size, len(nll))
        m = mean_nll(nll, s, e)
        out.append({"from": s, "to": e - 1, "nll": round(m, 4) if m is not None else None, "ppl": round(ppl(m), 2) if m is not None else None})
    return out


def summarize(curves: list[dict], cfg: dict) -> dict:
    W, bsz = int(cfg["demo"]["window"]), int(cfg["demo"]["bin"])
    rows = []
    for c in curves:
        nll = c["nll"]
        n = len(nll)
        lo, hi = mean_nll(nll, 1, W), mean_nll(nll, W, n)
        b = bins_of(nll, bsz)
        worst = max((x for x in b if x["ppl"] is not None), key=lambda x: x["ppl"], default=None)
        rows.append({"name": c["name"], "label": c["label"], "n_tokens": n, "cache_tokens": c.get("cache_tokens"),
                     "nll_lt_W": round(lo, 4) if lo is not None else None, "ppl_lt_W": round(ppl(lo), 2) if lo is not None else None,
                     "nll_ge_W": round(hi, 4) if hi is not None else None, "ppl_ge_W": round(ppl(hi), 2) if hi is not None else None,
                     "worst_bin": worst, "bins": b, "seconds": c.get("seconds"), "tok_per_s": c.get("tok_per_s"),
                     "note": c.get("note")})
    return {"window": W, "n_sink": int(cfg["demo"]["n_sink"]), "bin": bsz, "rows": rows}


def ascii_row(bins: list[dict], lo: float, hi: float) -> str:
    s = ""
    for b in bins:
        if b["ppl"] is None:
            s += " "
            continue
        v = math.log10(b["ppl"])
        k = 0 if hi <= lo else int(round((v - lo) / (hi - lo) * (len(BLOCKS) - 1)))
        s += BLOCKS[max(0, min(len(BLOCKS) - 1, k))]
    return s


def print_table(summary: dict, tag: str = "") -> None:
    W = summary["window"]
    print(f"  {'cache':<54}{'ppl, tokens < W':>18}{'ppl, tokens ≥ W':>18}{'worst bin':>14}{'cache tokens':>14}{'time':>10}   {tag}")
    for r in summary["rows"]:
        wb = r["worst_bin"]
        worst = f"{wb['ppl']:,.1f} @{wb['from']}" if wb else "—"
        lt = f"{r['ppl_lt_W']:.2f}" if r["ppl_lt_W"] is not None else "—"
        ge = f"{r['ppl_ge_W']:,.2f}" if r["ppl_ge_W"] is not None else "—"
        ct = str(r["cache_tokens"]) if r["cache_tokens"] is not None else "—"
        tm = f"{r['seconds']:.0f} s" if r.get("seconds") is not None else "—"
        print(f"  {r['label'][:53]:<54}{lt:>18}{ge:>18}{worst:>14}{ct:>14}{tm:>10}")
    print(f"\n  {DIM}W = {W}, n_sink = {summary['n_sink']}; perplexity = exp(mean NLL) over token positions 1..W−1 and W..end; "
          f"worst bin = the {summary['bin']}-token bin with the highest perplexity (position of its first token){RESET}\n")


def write_svg(summary: dict, meta: dict, path: Path, fake: bool) -> None:
    rows = summary["rows"]
    W = summary["window"]
    n = max(r["n_tokens"] for r in rows)
    vals = [b["ppl"] for r in rows for b in r["bins"] if b["ppl"] is not None]
    if not vals:
        return
    ylo, yhi = math.floor(math.log10(min(vals)) * 2) / 2, math.ceil(math.log10(max(vals)) * 2) / 2
    if yhi <= ylo:
        yhi = ylo + 1
    Wd, Hd, L, R, T, B = 900, 420, 70, 20, 40, 60
    def X(t): return L + (Wd - L - R) * t / n
    def Y(v): return T + (Hd - T - B) * (1 - (math.log10(v) - ylo) / (yhi - ylo))
    colors = {"full": "#1f3a93", "window": "#c0392b", "window_sink_abs": "#e67e22", "window_sink_reindex": "#27ae60", "window_sink_abs_mine": "#8e44ad"}
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Hd}" viewBox="0 0 {Wd} {Hd}" font-family="system-ui, sans-serif" font-size="12">',
         f'<rect width="{Wd}" height="{Hd}" fill="white"/>',
         f'<text x="{L}" y="22" font-size="14" font-weight="bold">per-token perplexity under four KV caches · {meta.get("model", "")}</text>']
    y = ylo
    while y <= yhi + 1e-9:
        s.append(f'<line x1="{L}" y1="{Y(10 ** y):.1f}" x2="{Wd - R}" y2="{Y(10 ** y):.1f}" stroke="#ddd"/>')
        s.append(f'<text x="{L - 6}" y="{Y(10 ** y) + 4:.1f}" text-anchor="end">{10 ** y:.3g}</text>')
        y += 0.5
    s.append(f'<line x1="{X(W):.1f}" y1="{T}" x2="{X(W):.1f}" y2="{Hd - B}" stroke="#555" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{X(W) + 4:.1f}" y="{T + 12}" fill="#555">W = {W}</text>')
    for t in range(0, n + 1, 1000):
        s.append(f'<text x="{X(t):.1f}" y="{Hd - B + 16}" text-anchor="middle">{t}</text>')
    s.append(f'<text x="{(L + Wd - R) / 2:.0f}" y="{Hd - B + 34}" text-anchor="middle">token position</text>')
    s.append(f'<text x="16" y="{(T + Hd - B) / 2:.0f}" transform="rotate(-90 16 {(T + Hd - B) / 2:.0f})" text-anchor="middle">perplexity (log)</text>')
    for i, r in enumerate(rows):
        pts = " ".join(f"{X((b['from'] + b['to']) / 2):.1f},{Y(b['ppl']):.1f}" for b in r["bins"] if b["ppl"] is not None)
        col = colors.get(r["name"], "#333")
        s.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts}"/>')
        lx = L + 10
        s.append(f'<rect x="{lx}" y="{T + 26 + 16 * i}" width="14" height="3" fill="{col}"/>')
        s.append(f'<text x="{lx + 20}" y="{T + 32 + 16 * i}">{r["label"]}</text>')
    s.append("</svg>")
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for name, m in cfg["models"].items():
        if args.only and name != args.only:
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {name}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha, ignore_patterns=NON_MODEL_FILES)
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已寫進 versions.lock{RESET}")


def source_lines(obj, needles: tuple[str, ...]) -> list[str]:
    try:
        src = pyinspect.getsource(obj).splitlines()
    except (OSError, TypeError):
        return []
    return [ln.strip() for ln in src if any(nd in ln for nd in needles)]


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"  {RED}這台不是 Apple silicon：只能用 --fake 演練{RESET}")
    text, meta = load_text(cfg)
    m = make_model(args.model, cfg, args, weights=False)
    banner(f"{cfg['demo']['title']}  ·  {m.label}")
    ids = m.encode(text)
    W, n_sink = int(cfg["demo"]["window"]), int(cfg["demo"]["n_sink"])
    print(f"  text: {meta['path']} first {meta['chars']:,} chars → {len(ids):,} tokens (first token {ids[0]}"
          + (" = BOS" if m.bos() is not None and ids[0] == m.bos() else ", no BOS") + f"); first line {meta['first_line']!r}")
    print(f"  W = {W}, n_sink = {n_sink}: {max(0, len(ids) - W):,} token positions lie past the window; "
          f"config.json max_position_embeddings = {m.max_pos:,}"
          + (f", rope_scaling {json.dumps(m.rope_scaling)}" if m.rope_scaling else ""))
    print(f"  the three windowed caches run one token at a time: {3 * (len(ids) - 1):,} forward passes in all")

    rec = {"date": now(), "model": m.repo, "revision": m.revision, "versions": v, "text": meta, "n_tokens": len(ids),
           "first_token": ids[0], "bos_token_id": m.bos(), "window": W, "n_sink": n_sink,
           "max_position_embeddings": m.max_pos, "rope_scaling": m.rope_scaling}

    # mlx-lm 的 keep 到底留了什麼、位置在哪裡決定——看原始碼，不看 README（附錄 C 的規則）
    banner("mlx-lm: where positions are assigned, and what RotatingKVCache(keep) keeps", AMBER)
    if m.fake:
        print(f"  {AMBER}--fake：沒有 mlx，跳過原始碼與數值檢查{RESET}")
    else:
        import mlx.core as mx
        from mlx_lm.models import cache as cache_mod
        from mlx_lm.models import llama as llama_mod
        attn_lines = source_lines(llama_mod.Attention.__call__, ("self.rope(", "update_and_fetch"))
        mpc_lines = source_lines(cache_mod.make_prompt_cache, ("RotatingKVCache(",))
        print(f"  {BOLD}mlx_lm/models/llama.py Attention.__call__:{RESET}")
        for ln in attn_lines:
            print(f"    {GREY}{ln}{RESET}")
        print(f"  → q and k are rotated at cache.offset — the absolute position — before the cache sees them; the cache never sees a position.")
        print(f"  {BOLD}mlx_lm/models/cache.py make_prompt_cache (what --max-kv-size gives you):{RESET}")
        for ln in mpc_lines:
            print(f"    {GREY}{ln}{RESET}")
        # 微測試：keep=2、max_size=6，餵 10 個 token，看留下哪些絕對位置
        rk = cache_mod.RotatingKVCache(max_size=6, keep=2)
        for i in range(10):
            k = mx.full((1, 1, 1, 4), float(i))
            rk.update_and_fetch(k, k)
        kept = sorted(int(x) for x in rk.keys_and_values()[0][0, 0, :, 0].tolist())
        print(f"  RotatingKVCache(max_size=6, keep=2) after 10 single tokens keeps absolute positions {kept}"
              f" — the first {rk.keep} and the last {rk.max_size - rk.keep}, at their original positions")
        print(f"  → StreamingLLM §3 re-indexes positions inside the cache; the library's keep does not. "
              f"Variant 3 is the library's keep as is; variant 4 re-indexes the sinks by hand (SinkWindowCache).")
        rec["mlx_lm_source"] = {"llama_attention": attn_lines, "make_prompt_cache": mpc_lines, "rotating_keep_test": kept}

        banner("RoPE checks (re-indexing relies on all three)", AMBER)
        rope, head_dim = m.make_rope_from_config()
        chk = m.rope_checks(rope, head_dim)
        ok = lambda cond: f"{GREEN}ok{RESET}" if cond else f"{RED}FAIL{RESET}"
        print(f"  {chk['rope_class']}, head_dim {head_dim}")
        print(f"  rope(rope(x, 3), 7) = rope(x, 10):        max |Δ| = {chk['compose_max_abs_err']:.2e}   {ok(chk['compose_max_abs_err'] < 1e-2)}")
        if chk["negative_offset_inverts"]:
            print(f"  rope(rope(x, 5), −5) = x:                max |Δ| = {chk['invert_max_abs_err']:.2e}   {ok(chk['invert_max_abs_err'] < 1e-2)}")
        else:
            print(f"  rope with a negative offset:             {RED}raised: {chk['invert_max_abs_err']}{RESET}  → variant 4 cannot run as written")
        print(f"  rope on a 4-token block = per element:   max |Δ| = {chk['block_offset_is_per_element_max_abs_err']:.2e}   {ok(chk['block_offset_is_per_element_max_abs_err'] < 1e-2)}")
        print(f"  score depends on m − n only:             |Δ| = {chk['score_rel_only_abs_err']:.2e} between (100, 93) and (5000, 4993)   {ok(chk['score_rel_only_abs_err'] < 1e-2 * max(1.0, abs(chk['score_rel_only_pair'][0])))}")
        print(f"  {DIM}(fp16 rotations: 1e-3-ish differences are rounding; the model's own attention runs in the same precision){RESET}")
        rec["rope_checks"] = chk
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    path = REHEARSAL / ("inspect.fake.json" if m.fake else "inspect.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def run_variant(m, name: str, label: str, ids: list[int], cfg: dict) -> dict:
    W, n_sink, bsz = int(cfg["demo"]["window"]), int(cfg["demo"]["n_sink"]), int(cfg["demo"]["bin"])
    n = len(ids)
    banner(f"{name}: {label}")
    t0 = time.time()
    last = {"t": 0, "printed": 0}

    def progress(t):
        if t - last["printed"] >= bsz or t >= n - 1:
            el = time.time() - t0
            print(f"    {t:>6,}/{n - 1:,} tokens   {t / el if el else 0:7.1f} tok/s   {el:6.0f} s", end="\r", flush=True)
            last["printed"] = t

    if m.fake:
        nll = m.fake_curve(name, n, cfg)
        cache_tokens = {"full": n, "window": W, "window_sink_abs": W + n_sink, "window_sink_reindex": W + n_sink, "window_sink_abs_mine": W + n_sink}[name]
    elif name == "full":
        nll = m.nll_full(ids, cfg, progress)
        cache_tokens = n
    else:
        caches = m.make_caches(name, cfg)
        nll = m.nll_stepwise(ids, caches, cfg, progress)
        c0 = caches[0]
        cache_tokens = int(c0.size()) if hasattr(c0, "size") else None
    sec = time.time() - t0
    print()
    lo, hi = mean_nll(nll, 1, W), mean_nll(nll, W, n)
    print(f"  ppl < W: {fmt(ppl(lo))}   ppl ≥ W: {fmt(ppl(hi))}   {sec:.0f} s ({(n - 1) / sec if sec else 0:.1f} tok/s), cache holds {cache_tokens} tokens at the end")
    return {"name": name, "label": label, "nll": [round(v, 4) if v == v else None for v in nll], "seconds": round(sec, 1),
            "tok_per_s": round((n - 1) / sec, 1) if sec else None, "cache_tokens": cache_tokens}


def cmd_rehearse(args, cfg):
    text, meta = load_text(cfg)
    if args.chars:
        text = text[: int(args.chars)]
        meta = {**meta, "chars": len(text), "smoke": True}
    smoke = bool(args.chars)
    out_dir = RUNS / "smoke" if smoke else REHEARSAL
    m = make_model(args.model, cfg, args)
    ids = m.encode(text)
    W = int(cfg["demo"]["window"])
    banner(f"{cfg['demo']['title']}  ·  {m.label}" + (f"  {AMBER}[smoke test, {len(text):,} chars]{RESET}" if smoke else ""))
    print(f"  {len(ids):,} tokens, W = {W}, n_sink = {cfg['demo']['n_sink']}; {max(0, len(ids) - W):,} positions past the window")
    if len(ids) <= W + 64:
        print(f"  {AMBER}文字比視窗長不了多少（{len(ids)} tokens vs W = {W}）：過了 W 的那段會很短{RESET}")
    if not m.fake:
        rope, head_dim = m.make_rope_from_config()
        chk = m.rope_checks(rope, head_dim)
        if not chk["negative_offset_inverts"] or chk["invert_max_abs_err"] > 1e-2:
            print(f"  {RED}RoPE 的負 offset 反轉沒過（{chk['invert_max_abs_err']}）：window_sink_reindex 這條跳過；見 README〈彩排時要確認的事〉{RESET}")
            skip_reindex = True
        else:
            skip_reindex = False
    else:
        skip_reindex = False
    wanted = [v for v in VARIANTS if not args.variants or v[0] in args.variants.split(",")]
    if args.check:
        wanted.append(("window_sink_abs_mine", "window W + n_sink, original (own cache, no re-index)"))
    if not m.fake:
        reset_peak_memory()
    t0 = time.time()
    curves = []
    for name, label in wanted:
        if name == "window_sink_reindex" and skip_reindex:
            continue
        curves.append(run_variant(m, name, label, ids, cfg))
    # 一致性：W 之前所有變體應與 full 幾乎相同（分段 prefill vs. 一次一個 token 的浮點差）
    by = {c["name"]: c for c in curves}
    checks = {}
    if "full" in by:
        for name, c in by.items():
            if name == "full":
                continue
            lim = W if name == "window" else W + int(cfg["demo"]["n_sink"])
            d = max(abs(a - b) for a, b in zip(by["full"]["nll"][1:lim], c["nll"][1:lim]) if a is not None and b is not None)
            checks[f"max_abs_dnll_vs_full_before_eviction__{name}"] = round(d, 4)
            print(f"  {name:<22} before its first eviction: max |NLL − full| = {d:.4f}  ({'same computation, fp noise' if d < 0.25 else AMBER + 'larger than fp noise — look' + RESET})")
    if args.check and "window_sink_abs" in by and "window_sink_abs_mine" in by:
        d = max(abs(a - b) for a, b in zip(by["window_sink_abs"]["nll"][1:], by["window_sink_abs_mine"]["nll"][1:]) if a is not None and b is not None)
        checks["max_abs_dnll_library_keep_vs_mine"] = round(d, 4)
        print(f"  library keep vs. SinkWindowCache(reindex=False), all positions: max |ΔNLL| = {d:.4f}")
    summary = summarize(curves, cfg)
    summary.update({"date": now(), "model": m.repo, "revision": m.revision, "versions": versions(),
                    "device": device_info() if not m.fake else {}, "fake": m.fake, "text": meta, "n_tokens": len(ids),
                    "bos_token_id": m.bos(), "first_token": ids[0], "chunk": int(cfg["demo"]["chunk"]),
                    "seconds_total": round(time.time() - t0), "peak_memory_gb": peak_memory_gb() if not m.fake else None,
                    "max_position_embeddings": m.max_pos, "rope_scaling": m.rope_scaling, "checks": checks, "smoke": smoke})
    print()
    print_table(summary, tag=f"{AMBER}彩排 {summary['date'][:16]}{RESET}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = ".fake" if m.fake else ""
    (out_dir / f"curve{sfx}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    full = {k: v for k, v in summary.items() if k != "rows"}
    full["curves"] = curves
    (out_dir / f"tokens{sfx}.json").write_text(json.dumps(full, ensure_ascii=False) + "\n", encoding="utf-8")
    write_svg(summary, {"model": m.label}, out_dir / f"curve{sfx}.svg", m.fake)
    note(f"→ {out_dir.relative_to(HERE)}/curve{sfx}.json (the table and the bins), tokens{sfx}.json (every token's NLL), curve{sfx}.svg   ({summary['seconds_total']} s total)")


def load_summary(fake: bool) -> dict | None:
    p = REHEARSAL / ("curve.fake.json" if fake else "curve.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wait(msg: str) -> str:
    try:
        key = input(f"\n{GREY}[{msg} · x = 離開]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        key = "x"
    return key


def cmd_replay(args, cfg):
    summary = load_summary(args.fake)
    if not summary:
        print(f"{RED}沒有 runs/rehearsal/curve.json：先跑 ./present.sh rehearse{RESET}")
        return
    pause = bool(cfg["demo"].get("pause", True)) and not args.no_pause
    v = summary.get("versions") or {}
    banner(f"{AMBER}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {summary['date']}  {summary['model']}" + (f"  {RED}[FAKE]{RESET}" if summary.get("fake") else ""))
    tx = summary.get("text", {})
    print(f"  {tx.get('path')} first {tx.get('chars'):,} chars = {summary['n_tokens']:,} tokens; W = {summary['window']}, n_sink = {summary['n_sink']}; "
          f"{summary.get('seconds_total')} s total" + (f", peak memory {summary['peak_memory_gb']:.1f} GB" if summary.get("peak_memory_gb") else "")
          + f"; mlx-lm {v.get('mlx-lm')}, mlx {v.get('mlx')}")
    rows = summary["rows"]
    vals = [math.log10(b["ppl"]) for r in rows for b in r["bins"] if b["ppl"] is not None]
    lo, hi = min(vals), max(vals)
    bsz = summary["bin"]
    nb = max(len(r["bins"]) for r in rows)
    W = summary["window"]
    wpos = min(nb - 1, max(0, (W - 1) // bsz))
    print(f"\n  each cell = {bsz} tokens, height = log perplexity (shared scale {10 ** lo:.1f} … {10 ** hi:,.0f}); ↑ marks the bin where the window first slides")
    print(f"  {'':<54}{' ' * wpos}↑ W")
    for r in rows:
        if pause:
            key = wait(f"Enter = {r['label'][:60]}")
            if key == "x":
                print()
                return
        row = ascii_row(r["bins"], lo, hi)
        print(f"  {r['label'][:53]:<54}{row}   ppl < W {fmt(r['ppl_lt_W'])}, ≥ W {fmt(r['ppl_ge_W'])}")
    if pause:
        key = wait("Enter = p43 的表")
        if key == "x":
            print()
            return
    print()
    print_table(summary, tag=f"{AMBER}彩排 {summary['date'][:16]}{RESET}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（曲線是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--model", default="primary", help="demo_config.toml 的 [models.*] 名稱（primary / backup）")
    p.add_argument("--only", help="fetch：只抓一個模型")
    p.add_argument("--chars", type=int, help="rehearse：只取前幾個字元（煙霧測試，寫到 runs/smoke/）")
    p.add_argument("--variants", help="rehearse：只跑哪幾種 cache，逗號分隔（full,window,window_sink_abs,window_sink_reindex）")
    p.add_argument("--check", action="store_true", help="rehearse：多跑一條自己的 class 關掉重編的版本，對庫的 keep")
    p.add_argument("--no-pause", action="store_true", help="replay：不等 Enter")
    sub = p.add_subparsers(dest="cmd", required=True)   # present.sh 把選項放在子命令前面，所以選項全掛在主 parser
    for c in ("fetch", "inspect", "rehearse", "show", "replay"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "show": cmd_replay, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
