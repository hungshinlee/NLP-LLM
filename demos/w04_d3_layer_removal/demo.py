#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 3：逐層跳過一層、量 perplexity（投影片 p32「Live: skip one layer at a time, and measure the perplexity」）。

    ./present.sh fetch      下載 Qwen3-0.6B（與備用的 1.7B）；W3 demo 3 已 fetch 過的話只會確認 ../versions.lock 的 revision（要網路）
    ./present.sh inspect    課前：版本、device、decoder layer 的回傳格式、hook 恆等化是否真的「沒改動」、一次前向幾秒 → runs/rehearsal/inspect.json
    ./present.sh rehearse   課前彩排：baseline + 逐層跳過（window 1）+ 連續跳過 k 層（window 2、4）→ runs/rehearsal/curve.json 與 curve.svg
    ./present.sh            課堂：載模型 → 印 baseline → 停下來等學生預測 → Enter → 28 個點一個一個長出來 → 總表
    ./present.sh replay     現場失敗：用彩排的紀錄走同一套畫面（畫面標明非現場）
    ./present.sh fake       沒有模型也能演練流程（數字是假的，畫面會標明）

做法（大綱〈課堂 demo 建議〉3）：對一段固定文字（W1 demo 2 的 zh_city.txt + W2 demo 1 的英文版 zh_city_en.txt）量 teacher-forcing 的
perplexity，然後把第 i 層的 forward 換成恆等——**參數沒有刪，只是那一層這次不往 residual stream 上加東西**——再量一次，28 次。

「換成恆等」的實作：在 `model.model.layers[i]` 上掛 forward hook，把該層的輸出換成它的輸入 hidden_states。
  - 不換掉模組：Qwen3Model.forward 會讀 `decoder_layer.attention_type` 挑 mask，換成 nn.Identity 會壞；
  - 不依賴回傳格式：decoder layer 在 transformers 4.x 回 tuple、5.x 多半直接回 tensor，hook 照原格式只換第一項（inspect 會印出實測到的格式）；
  - 代價：那一層仍然算了一次（結果被丟掉）。0.6B、兩段各約兩百 token 的文字，多算一層不到一秒。

除了 perplexity，每個點另記兩個「不同性質的指標」：top-1 next-token accuracy（對真正的下一個 token）與 top-1 agreement
（與 baseline 的預測是否一致）。perplexity 對擾動敏感、accuracy 類指標遲鈍——這是 p33「只在某類指標上」那一列的同一份資料版本。

相依：torch、transformers（版本釘在 ../versions.lock）。Python 3.11+（tomllib）。
"""
from __future__ import annotations

import argparse
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
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, append_jsonl,  # noqa: E402
                    banner, locked_revision, note, now, revision_key, write_lock)

CONFIG = HERE / "demo_config.toml"
REHEARSAL = HERE / "runs" / "rehearsal"
LIVE = HERE / "runs" / "live"
IGNORE = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "*.gguf", "*.onnx", "*.msgpack", "*.h5"]


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for t in cfg["texts"]:
        p = (HERE / t["path"]).resolve()
        t["abs_path"] = str(p)
        t["text"] = p.read_text(encoding="utf-8").strip()
    return cfg


def versions() -> dict:
    from importlib import metadata
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── 後端 ─────────────────────────────────────────────────────────
class TorchBackend:
    fake = False

    def __init__(self, repo: str, revision: str | None, offline: bool, device: str, attn: str = "sdpa"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(0)
        t0 = time.time()
        self.path = snapshot_download(repo, revision=revision or None, local_files_only=offline, ignore_patterns=IGNORE)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        kw = dict(attn_implementation=attn)
        try:                      # transformers 5 叫 dtype，4.x 叫 torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.float32, **kw)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float32, **kw)
        if device == "mps" and not torch.backends.mps.is_available():
            note("MPS 不可用，改用 CPU")
            device = "cpu"
        self.device = device
        self.model.to(device).eval()
        self.attn = attn
        self.load_seconds = time.time() - t0
        self.layers = self._find_layers()
        c = self.model.config
        self.config = {"model_type": c.model_type, "num_hidden_layers": c.num_hidden_layers,
                       "num_attention_heads": c.num_attention_heads,
                       "num_key_value_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
                       "hidden_size": c.hidden_size, "vocab_size": c.vocab_size,
                       "tie_word_embeddings": bool(getattr(c, "tie_word_embeddings", False)),
                       "attn_implementation": getattr(c, "_attn_implementation", attn),
                       "layers_found": len(self.layers)}
        self.n_layer = len(self.layers)
        self._encoded: dict[str, "torch.Tensor"] = {}
        self.layer_output_format: str | None = None

    def _find_layers(self):
        """找 decoder layer 的 ModuleList：Qwen3／Llama 都是 model.model.layers；其他架構退回找第一個長度 = num_hidden_layers 的 ModuleList。"""
        import torch.nn as nn
        m = self.model
        inner = getattr(m, "model", None)
        if inner is not None and isinstance(getattr(inner, "layers", None), nn.ModuleList):
            return inner.layers
        for mod in m.modules():
            if isinstance(mod, nn.ModuleList) and len(mod) == m.config.num_hidden_layers:
                return mod
        raise RuntimeError("找不到 decoder layer 的 ModuleList")

    def encode(self, text: str):
        if text not in self._encoded:
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            self._encoded[text] = ids.to(self.device)
        return self._encoded[text]

    def n_tokens(self, text: str) -> int:
        return int(self.encode(text).shape[1])

    def _skip_hook(self, record_format: bool = False):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            if record_format:
                self.layer_output_format = (f"tuple[{len(output)}]" if isinstance(output, tuple)
                                            else type(output).__name__)
            if isinstance(output, tuple):
                return (x,) + tuple(output[1:])
            return x
        return hook

    def score(self, text: str, skip: list[int] | None = None) -> dict:
        """一次 teacher-forcing 前向。回傳 nll 總和、預測的 token 數、top-1 正確數、每個位置的 argmax（算 agreement 用）。"""
        torch = self.torch
        ids = self.encode(text)
        handles = []
        for i in (skip or []):
            handles.append(self.layers[i].register_forward_hook(self._skip_hook(), with_kwargs=True))
        t0 = time.time()
        try:
            with torch.no_grad():
                logits = self.model(input_ids=ids, use_cache=False).logits[0].float()
        finally:
            for h in handles:
                h.remove()
        seconds = time.time() - t0
        tgt = ids[0, 1:]
        lp = torch.log_softmax(logits[:-1], dim=-1)
        nll = -lp.gather(1, tgt[:, None])[:, 0]
        pred = logits[:-1].argmax(-1)
        return {"nll_sum": float(nll.sum()), "n": int(tgt.numel()),
                "top1_correct": int((pred == tgt).sum()), "pred": pred.cpu().tolist(),
                "finite": bool(torch.isfinite(nll).all()), "seconds": seconds}

    def probe_identity(self, text: str) -> dict:
        """inspect 用：hook 把輸出換成輸入之後，拿掉 hook 的結果要與沒掛過 hook 的完全一致；並記回傳格式。"""
        torch = self.torch
        ids = self.encode(text)
        with torch.no_grad():
            ref = self.model(input_ids=ids, use_cache=False).logits
            h = self.layers[0].register_forward_hook(self._skip_hook(record_format=True), with_kwargs=True)
            try:
                skipped = self.model(input_ids=ids, use_cache=False).logits
            finally:
                h.remove()
            again = self.model(input_ids=ids, use_cache=False).logits
        return {"layer_output_format": self.layer_output_format,
                "max_abs_diff_after_removing_hook": float((ref - again).abs().max()),
                "max_abs_diff_skip_layer0": float((ref - skipped).abs().max())}


class FakeBackend:
    fake = True

    def __init__(self, repo: str, revision: str | None, offline: bool, device: str, attn: str = "sdpa"):
        self.path, self.load_seconds, self.device, self.attn = "(fake)", 0.0, device, attn
        self.n_layer = 28
        self.config = {"model_type": "fake", "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                       "hidden_size": 1024, "vocab_size": 151936, "tie_word_embeddings": True,
                       "attn_implementation": "fake", "layers_found": 28}
        self.rng = random.Random(4)
        self.layer_output_format = "fake"

    def n_tokens(self, text: str) -> int:
        return max(8, len(text) // 2)

    def score(self, text: str, skip: list[int] | None = None) -> dict:
        n = self.n_tokens(text) - 1
        base = 2.3 if any(ord(c) > 0x2E80 for c in text) else 2.0
        bump = 0.0
        for i in (skip or []):
            t = i / (self.n_layer - 1)
            bump += 1.6 * math.exp(-(t / 0.08) ** 2) + 1.0 * math.exp(-((1 - t) / 0.08) ** 2) + 0.03 + self.rng.uniform(0, 0.04)
        pred = [self.rng.randrange(1000) for _ in range(n)]
        time.sleep(0.05)
        return {"nll_sum": (base + bump) * n, "n": n, "top1_correct": int(n * max(0.1, 0.55 - 0.3 * bump)),
                "pred": pred, "finite": True, "seconds": 0.05}

    def probe_identity(self, text: str) -> dict:
        return {"layer_output_format": "fake", "max_abs_diff_after_removing_hook": 0.0, "max_abs_diff_skip_layer0": 9.9}


def make_backend(args, cfg: dict):
    m = cfg["models"][args.model]
    revision = locked_revision(m["repo"], m.get("revision") or None)
    cls = FakeBackend if args.fake else TorchBackend
    device = args.device or cfg["demo"]["device"]
    note(f"載入 {m['label']}  ←  {m['repo']}" + (f"@{revision[:10]}" if revision else "") + f"  device={device}  attn={cfg['demo']['attn_implementation']}")
    b = cls(m["repo"], revision, offline=not args.online, device=device, attn=cfg["demo"]["attn_implementation"])
    note(f"載入完成：{b.load_seconds:.1f} 秒；{b.n_layer} 層")
    b.label, b.repo, b.revision = m["label"], m["repo"], revision
    return b


# ── 量測 ─────────────────────────────────────────────────────────
def measure(b, texts: list[dict], skip: list[int] | None, baseline_preds: dict | None = None) -> dict:
    """對所有文字量一次，回傳合併與逐篇的 perplexity、accuracy、agreement。"""
    per = {}
    nll, n, correct, agree, seconds, finite = 0.0, 0, 0, 0, 0.0, True
    for t in texts:
        s = b.score(t["text"], skip)
        per[t["name"]] = {"ppl": math.exp(s["nll_sum"] / s["n"]) if s["finite"] else float("inf"),
                          "nll_per_token": s["nll_sum"] / s["n"], "n": s["n"],
                          "top1_acc": s["top1_correct"] / s["n"], "pred": s["pred"]}
        if baseline_preds is not None:
            a = sum(int(p == q) for p, q in zip(s["pred"], baseline_preds[t["name"]]))
            per[t["name"]]["top1_agreement"] = a / s["n"]
            agree += a
        nll += s["nll_sum"]; n += s["n"]; correct += s["top1_correct"]; seconds += s["seconds"]; finite &= s["finite"]
    out = {"skip": list(skip or []), "ppl": math.exp(nll / n) if finite else float("inf"), "nll_per_token": nll / n,
           "n_tokens": n, "top1_acc": correct / n, "seconds": seconds, "finite": finite, "per_text": per}
    if baseline_preds is not None:
        out["top1_agreement"] = agree / n
    return out


def strip_preds(r: dict) -> dict:
    """寫進 json 前把逐位置的 argmax 拿掉（只在記憶體裡算 agreement 用）。"""
    r = dict(r)
    r["per_text"] = {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in r["per_text"].items()}
    return r


# ── 畫面 ─────────────────────────────────────────────────────────
BAR_MAX = 44     # 長條最長幾格
LOG_CAP = 4.0    # 長條代表 log2(ppl / baseline)，封頂 2^4 = 16 倍


def bar(ratio: float) -> str:
    if not math.isfinite(ratio):
        return f"{RED}{'█' * BAR_MAX}▶{RESET}"
    v = max(0.0, min(LOG_CAP, math.log2(max(ratio, 1e-9))))
    k = int(round(v / LOG_CAP * BAR_MAX))
    color = GREEN if ratio < 1.15 else (AMBER if ratio < 2 else RED)
    return f"{color}{'█' * k}{RESET}{'·' * (BAR_MAX - k)}" + ("▶" if v >= LOG_CAP else " ")


def fmt_ppl(x: float) -> str:
    return "  inf" if not math.isfinite(x) else (f"{x:7.1f}" if x < 1000 else f"{x:7.0f}")


def print_baseline(base: dict, texts: list[dict], b) -> None:
    print(f"\n  {BOLD}baseline (no layer skipped){RESET}   perplexity {BOLD}{base['ppl']:.2f}{RESET}"
          f"   top-1 accuracy {base['top1_acc']*100:.1f}%   {base['n_tokens']} predicted tokens   {base['seconds']:.2f} s")
    for t in texts:
        p = base["per_text"][t["name"]]
        print(f"  {GREY}  {t['name']:<8} ppl {p['ppl']:7.2f}   acc {p['top1_acc']*100:5.1f}%   {p['n']} tokens   {t.get('note','')}{RESET}")


def header_line() -> None:
    print(f"\n  {DIM}{'skip':<6}{'ppl':>8}  {'× base':>7}  {'acc':>6}  {'agree':>6}  log₂(ppl / baseline), one block = {LOG_CAP / BAR_MAX:.2f}, ▶ = capped at ×{2 ** LOG_CAP:.0f}{RESET}")


def point_line(r: dict, base_ppl: float, label: str | None = None) -> str:
    ratio = r["ppl"] / base_ppl
    lab = label if label is not None else (str(r["skip"][0]) if len(r["skip"]) == 1 else f"{r['skip'][0]}–{r['skip'][-1]}")
    ag = f"{r['top1_agreement']*100:5.1f}%" if "top1_agreement" in r else "   —  "
    return f"  {lab:<6}{fmt_ppl(r['ppl']):>8}  {('×' + f'{ratio:5.2f}') if math.isfinite(ratio) else '  ×inf':>7}  {r['top1_acc']*100:5.1f}%  {ag}  {bar(ratio)}"


def print_summary(base: dict, pts: list[dict], n_layer: int) -> None:
    fin = [r for r in pts if math.isfinite(r["ppl"])]
    ratios = sorted(r["ppl"] / base["ppl"] for r in fin)
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    cheapest = sorted(fin, key=lambda r: r["ppl"])[:5]
    dearest = sorted(pts, key=lambda r: -r["ppl"] if math.isfinite(r["ppl"]) else -1e99)[:3]
    first, last = pts[0], pts[-1]
    mid = pts[min(len(pts) - 1, n_layer // 2)]
    k = len(pts[0]["skip"])
    lab = (lambda r: str(r["skip"][0])) if k == 1 else (lambda r: f"{r['skip'][0]}–{r['skip'][-1]}")
    print(f"\n  {BOLD}summary{RESET}  (window = {k} layer{'s' if k > 1 else ''})")
    print(f"  the three guesses:   first ({lab(first)})  ×{first['ppl']/base['ppl']:.2f}     "
          f"middle ({lab(mid)})  ×{mid['ppl']/base['ppl']:.2f}     last ({lab(last)})  ×{last['ppl']/base['ppl']:.2f}")
    print(f"  median over all {len(pts)} points: ×{med:.2f};   cheapest five: "
          + ", ".join(f"{lab(r)} (×{r['ppl']/base['ppl']:.2f})" for r in cheapest)
          + f";   most expensive: " + ", ".join(f"{lab(r)} (×{(r['ppl']/base['ppl']) if math.isfinite(r['ppl']) else float('inf'):.1f})" for r in dearest))
    accs = [r["top1_acc"] for r in fin]
    print(f"  {GREY}top-1 accuracy: baseline {base['top1_acc']*100:.1f}%, skipped-layer range {min(accs)*100:.1f}%–{max(accs)*100:.1f}%  "
          f"— the slow metric; perplexity is the fast one (p33){RESET}")


# ── SVG（不用 matplotlib；退路與投影用）─────────────────────────
def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_curve(base: dict, windows: dict[str, list[dict]], title: str, path: Path) -> None:
    """x = 跳過的（第一）層，y = log 尺度的 ppl / baseline；每個 window 一條線；右軸 top-1 accuracy（虛線）。"""
    W, H = 1100, 520
    x0, y0, w, h = 90, 60, 820, 380
    n_layer = max(max(r["skip"]) for pts in windows.values() for r in pts) + 1
    vals = [r["ppl"] / base["ppl"] for pts in windows.values() for r in pts if math.isfinite(r["ppl"])]
    ymax = max(2.0, min(10 ** 4, max(vals) * 1.15)) if vals else 2.0
    lo, hi = math.log10(1.0), math.log10(ymax)

    def X(i): return x0 + 30 + (w - 60) * i / max(1, n_layer - 1)
    def Y(ratio):
        v = math.log10(max(1.0, min(ymax, ratio))) if math.isfinite(ratio) else hi
        return y0 + h - (v - lo) / (hi - lo) * h

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="13">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{x0}" y="30" font-size="16" font-weight="600">{esc(title)}</text>',
           f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>']
    # y 格線（1, 2, 5, 10, ...）
    tick = 1.0
    while tick <= ymax:
        for m in (1, 2, 5):
            t = tick * m
            if 1.0 <= t <= ymax:
                out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#eee"/>')
                out.append(f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" fill="#555">×{t:g}</text>')
        tick *= 10
    out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(1):.1f}" y2="{Y(1):.1f}" stroke="#666" stroke-dasharray="4 4"/>')
    for i in range(0, n_layer, 2):
        out.append(f'<text x="{X(i):.1f}" y="{y0 + h + 18}" text-anchor="middle" fill="#555">{i}</text>')
    out.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 42}" text-anchor="middle" fill="#333">first skipped layer i  (window of k consecutive layers i … i+k−1)</text>')
    out.append(f'<text transform="translate(28,{y0 + h / 2}) rotate(-90)" text-anchor="middle" fill="#333">perplexity / baseline (log scale)</text>')
    colors = ["#1f5fbf", "#d97706", "#b91c1c", "#6b21a8"]
    for ci, (k, pts) in enumerate(windows.items()):
        col = colors[ci % len(colors)]
        poly = " ".join(f"{X(r['skip'][0]):.1f},{Y(r['ppl'] / base['ppl']):.1f}" for r in pts)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for r in pts:
            cx, cy = X(r["skip"][0]), Y(r["ppl"] / base["ppl"])
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{col}"/>')
            if not math.isfinite(r["ppl"]):
                out.append(f'<text x="{cx:.1f}" y="{cy - 8:.1f}" text-anchor="middle" fill="{col}">inf</text>')
        out.append(f'<rect x="{x0 + w + 20}" y="{y0 + 10 + 24 * ci}" width="14" height="4" fill="{col}"/>')
        out.append(f'<text x="{x0 + w + 40}" y="{y0 + 16 + 24 * ci}" fill="#333">skip {k} layer{"s" if int(k) > 1 else ""}</text>')
    # 右軸：window 1 的 top-1 accuracy
    if "1" in windows:
        pts = windows["1"]
        def Ya(a): return y0 + h - a * h
        poly = " ".join(f"{X(r['skip'][0]):.1f},{Ya(r['top1_acc']):.1f}" for r in pts)
        out.append(f'<polyline points="{poly}" fill="none" stroke="#059669" stroke-width="1.8" stroke-dasharray="6 4"/>')
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            out.append(f'<text x="{x0 + w + 8}" y="{Ya(a) + 4:.1f}" fill="#059669">{int(a * 100)}%</text>')
        out.append(f'<rect x="{x0 + w + 20}" y="{y0 + 10 + 24 * len(windows)}" width="14" height="4" fill="#059669"/>')
        out.append(f'<text x="{x0 + w + 40}" y="{y0 + 16 + 24 * len(windows)}" fill="#333">top-1 acc (skip 1)</text>')
        out.append(f'<text x="{x0 + w + 40}" y="{y0 + 34 + 24 * len(windows)}" fill="#059669">baseline {base["top1_acc"] * 100:.1f}%</text>')
    out.append(f'<text x="{x0}" y="{H - 14}" fill="#555">baseline perplexity {base["ppl"]:.2f} over {base["n_tokens"]} predicted tokens; '
               f'a skipped layer\'s forward is replaced by the identity (parameters untouched)</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for name, m in cfg["models"].items():
        if args.only and name != args.only:
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {name}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha, ignore_patterns=IGNORE)
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已在 ../versions.lock{RESET}")


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    rec = {"date": now(), "versions": v}
    if not args.fake:
        import torch
        rec["mps_available"] = bool(torch.backends.mps.is_available())
        print(f"  mps available    {rec['mps_available']}")
    b = make_backend(args, cfg)
    c = b.config
    print(f"\n  config: {c['model_type']}  {c['num_hidden_layers']} layers ({c['layers_found']} found in the ModuleList)  d = {c['hidden_size']}"
          f"  V = {c['vocab_size']}  tied embeddings = {c['tie_word_embeddings']}  attn = {BOLD}{c['attn_implementation']}{RESET}")
    texts = cfg["texts"]
    for t in texts:
        print(f"  {t['name']:<8} {b.n_tokens(t['text'])} tokens  ({len(t['text'].encode('utf-8'))} bytes)  {t['abs_path']}")
    pr = b.probe_identity(texts[0]["text"])
    print(f"  decoder layer returns: {BOLD}{pr['layer_output_format']}{RESET}")
    print(f"  logits after removing the hook vs. before: max |Δ| = {pr['max_abs_diff_after_removing_hook']:.3g}  "
          f"{'ok' if pr['max_abs_diff_after_removing_hook'] == 0 else AMBER + '← 非零，hook 沒拆乾淨或 device 非決定' + RESET}")
    print(f"  logits with layer 0 skipped vs. baseline:  max |Δ| = {pr['max_abs_diff_skip_layer0']:.3g}  "
          f"{'ok（有改動）' if pr['max_abs_diff_skip_layer0'] > 0 else RED + '← 零，hook 沒生效' + RESET}")
    base = measure(b, texts, None)
    print_baseline(base, texts, b)
    one = measure(b, texts, [b.n_layer // 2], {k: v["pred"] for k, v in base["per_text"].items()})
    print(f"  one skipped-layer measurement (layer {b.n_layer // 2}): ppl {one['ppl']:.2f} (×{one['ppl'] / base['ppl']:.3f}), {one['seconds']:.2f} s"
          f"  → {b.n_layer} points ≈ {one['seconds'] * b.n_layer:.0f} s on {b.device}")
    rec.update({"model": b.repo, "revision": b.revision, "device": b.device, "config": c, "probe": pr,
                "texts": [{"name": t["name"], "path": t["path"], "bytes": len(t["text"].encode("utf-8")), "tokens": b.n_tokens(t["text"])} for t in texts],
                "baseline": strip_preds(base), "one_point": strip_preds(one),
                "estimated_seconds_window1": one["seconds"] * b.n_layer})
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    path = REHEARSAL / ("inspect.fake.json" if b.fake else "inspect.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def run_windows(b, cfg, windows: list[int], live: bool, pause: bool, log=None, pace: float = 0.0) -> tuple[dict, dict]:
    texts = cfg["texts"]
    banner(f"{cfg['demo']['title']}  ·  {b.label}" + (f"  {RED}[FAKE]{RESET}" if b.fake else ""))
    note(f"{b.n_layer} layers.  Fixed text: " + " + ".join(f"{t['name']} ({b.n_tokens(t['text'])} tokens)" for t in texts)
         + f".  Skipping = that layer's forward replaced by the identity; parameters untouched.")
    base = measure(b, texts, None)
    print_baseline(base, texts, b)
    preds = {k: v["pred"] for k, v in base["per_text"].items()}
    if pause:
        input(f"\n  {AMBER}{BOLD}Before running: from the bus picture, which layers are cheap to skip, which are not? "
              f"Three guesses on the board — first / a middle / last.{RESET}   {DIM}Enter = start skipping, one layer at a time{RESET} ")
        print("\033[1A\033[2K", end="")
    results: dict[str, list[dict]] = {}
    for k in windows:
        pts = []
        print(f"\n  {BOLD}skip {k} consecutive layer{'s' if k > 1 else ''}{RESET}" if k > 1 else f"\n  {BOLD}skip one layer at a time{RESET}")
        header_line()
        for i in range(0, b.n_layer - k + 1):
            r = measure(b, texts, list(range(i, i + k)), preds)
            pts.append(r)
            print(point_line(r, base["ppl"]), flush=True)
            if pace:
                time.sleep(pace)   # 彩排實測 28 點 0.22 s，不放慢就看不到「一個點一個點長出來」
            if log:
                log({"date": now(), "window": k, **strip_preds(r)})
        print_summary(base, pts, b.n_layer)
        results[str(k)] = pts
        if pause and k != windows[-1]:
            key = input(f"\n  {DIM}Enter = also skip {windows[windows.index(k) + 1]} consecutive layers   q = 結束{RESET} ").strip().lower()
            print("\033[1A\033[2K", end="")
            if key == "q":
                break
    return base, results


def save_run(b, cfg, base, results, where: Path, suffix: str = "") -> Path:
    where.mkdir(parents=True, exist_ok=True)
    rec = {"date": now(), "model": b.repo, "revision": b.revision, "versions": versions(), "device": b.device,
           "attn_implementation": b.config["attn_implementation"], "config": b.config, "fake": b.fake,
           "texts": [{"name": t["name"], "path": t["path"], "bytes": len(t["text"].encode("utf-8")), "tokens": b.n_tokens(t["text"])} for t in cfg["texts"]],
           "skip_method": "forward hook on model.model.layers[i] returning the input hidden_states (identity); parameters untouched",
           "layer_output_format": b.layer_output_format,
           "baseline": strip_preds(base), "windows": {k: [strip_preds(r) for r in pts] for k, pts in results.items()}}
    path = where / f"curve{suffix}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    svg_curve(base, results, f"{cfg['demo']['title']} · {b.label} · {rec['date'][:10]}" + ("  [FAKE]" if b.fake else ""), where / f"curve{suffix}.svg")
    note(f"\n→ {path.relative_to(HERE)}  與 curve{suffix}.svg")
    return path


def cmd_rehearse(args, cfg):
    b = make_backend(args, cfg)
    windows = [int(x) for x in (args.windows.split(",") if args.windows else cfg["demo"]["rehearse_windows"])]
    t0 = time.time()
    base, results = run_windows(b, cfg, windows, live=False, pause=False)
    note(f"total {time.time() - t0:.0f} s on {b.device}")
    save_run(b, cfg, base, results, REHEARSAL, ".fake" if b.fake else "")


def cmd_show(args, cfg):
    b = make_backend(args, cfg)
    windows = [int(x) for x in (args.windows.split(",") if args.windows else cfg["demo"]["show_windows"])]
    LIVE.mkdir(parents=True, exist_ok=True)
    stamp = now().replace(":", "").replace("-", "")
    log = LIVE / f"{stamp[:8]}.jsonl"
    pace = cfg["demo"].get("show_pace", 0.0) if args.pace is None else args.pace
    base, results = run_windows(b, cfg, windows, live=True, pause=not args.no_pause,
                                log=lambda rec: append_jsonl(log, {"model": b.repo, **rec}), pace=0.0 if args.no_pause else pace)
    save_run(b, cfg, base, results, LIVE, f"-{stamp}" + (".fake" if b.fake else ""))
    print()


def cmd_replay(args, cfg):
    path = REHEARSAL / ("curve.fake.json" if args.fake else "curve.json")
    if not path.exists():
        print(f"{RED}沒有 {path.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
        return
    reh = json.loads(path.read_text(encoding="utf-8"))
    base = reh["baseline"]
    banner(f"{AMBER}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {reh['date']}  {reh['model']}  ({reh['attn_implementation']}, {reh['device']})")
    note(f"{reh['config']['num_hidden_layers']} layers.  Fixed text: " + " + ".join(f"{t['name']} ({t['tokens']} tokens)" for t in reh["texts"]))
    print(f"\n  {BOLD}baseline (no layer skipped){RESET}   perplexity {BOLD}{base['ppl']:.2f}{RESET}"
          f"   top-1 accuracy {base['top1_acc']*100:.1f}%   {base['n_tokens']} predicted tokens")
    for name, p in base["per_text"].items():
        print(f"  {GREY}  {name:<8} ppl {p['ppl']:7.2f}   acc {p['top1_acc']*100:5.1f}%   {p['n']} tokens{RESET}")
    windows = [k for k in (args.windows.split(",") if args.windows else cfg["demo"]["show_windows"]) if str(k) in reh["windows"]]
    windows = [str(k) for k in windows] or list(reh["windows"])
    if not args.no_pause:
        input(f"\n  {AMBER}{BOLD}Before running: three guesses on the board — first / a middle / last.{RESET}   {DIM}Enter = replay, one point at a time{RESET} ")
        print("\033[1A\033[2K", end="")
    for k in windows:
        pts = reh["windows"][k]
        print(f"\n  {BOLD}skip {k} consecutive layers{RESET}" if int(k) > 1 else f"\n  {BOLD}skip one layer at a time{RESET}  {AMBER}(replay){RESET}")
        header_line()
        for r in pts:
            print(point_line(r, base["ppl"]), flush=True)
            time.sleep(0 if args.no_pause else 0.25)
        print_summary(base, pts, reh["config"]["num_hidden_layers"])
        if k != windows[-1] and not args.no_pause:
            key = input(f"\n  {DIM}Enter = next window   q = 結束{RESET} ").strip().lower()
            print("\033[1A\033[2K", end="")
            if key == "q":
                break
    note(f"\nSVG：{(REHEARSAL / ('curve.fake.svg' if args.fake else 'curve.svg')).relative_to(HERE)}")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（數字是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--model", default="primary", help="demo_config.toml 的 [models.*] 名稱（primary / backup）")
    p.add_argument("--device", default=None, help="覆蓋 demo_config.toml 的 device（cpu / mps）")
    p.add_argument("--windows", default=None, help="覆蓋要跑的 window 大小，逗號分隔，例如 1,2,4")
    p.add_argument("--no-pause", action="store_true", help="show／replay 不停下來等 Enter，也不放慢")
    p.add_argument("--pace", type=float, default=None, help="show 每印一點停幾秒（覆蓋 demo_config.toml 的 show_pace；MPS 上 28 點 0.22 s，不放慢看不到曲線長出來）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "inspect", "rehearse", "show", "replay"):
        sp = sub.add_parser(c)
        sp.add_argument("--only")
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
