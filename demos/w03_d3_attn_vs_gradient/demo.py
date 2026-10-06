#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 3：attention 熱圖 vs. gradient × input vs. attention rollout（投影片 p45「Live: the same sentence, three rankings」）。

    ./present.sh fetch      下載 Qwen3-0.6B（與備用的 1.7B），revision 鎖進 ../versions.lock（要網路）
    ./present.sh inspect    課前：transformers／torch 版本、eager 與 sdpa 兩種載法對 output_attentions 的行為、層數／head 數、一次前向＋反向的耗時
    ./present.sh rehearse   課前彩排：每句、每個 head 都算一遍，寫 runs/rehearsal/rankings.json 與 *.svg（講稿與 p45 表格從這裡抄）
    ./present.sh            課堂：Enter 逐欄揭露（attention → gradient × input → rollout → 換 head …）；n 換句子；q 離開
    ./present.sh replay     現場失敗：只顯示彩排的表（畫面標明非現場）
    ./present.sh fake       沒有模型也能演練流程（數字是假的，畫面會標明）

三個問題都是「對最後一個位置的輸出，哪些輸入 token 重要」，三種答案：
  (i)   attention：某一層某個 head 的 attention 矩陣最後一列（權重加總為 1）——需要 attn_implementation="eager"，
        SDPA 路徑不回傳權重（inspect 會實測釘住的 transformers 版本怎麼處理 output_attentions=True）；
  (ii)  gradient × input：對輸入 embedding 開梯度，目標是最後一個位置預測第一名 token 的 log-prob，
        每個 token 的分數 = ⟨∂target/∂e_s, e_s⟩（一階泰勒：「把這個 token 的 embedding 拿掉，target 變多少」），排序用絕對值、正規化成佔比；
  (iii) attention rollout（Abnar & Zuidema 2020）：每層各 head 平均的 attention 矩陣 A_l，Ã_l = (1−r) A_l + r I 後列正規化，
        R = Ã_L ⋯ Ã_1，取最後一列——把 residual 算進去之後「資訊從哪個輸入流到輸出」的比例。
三欄之間印 Spearman 秩相關；同一層各 head 的第一名各是誰也印出來（換 head 第一欄就整個換掉，這是 p45 的第 6 步）。

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
                    banner, locked_revision, note, now, read_lock, revision_key, write_lock)

CONFIG = HERE / "demo_config.toml"
REHEARSAL = HERE / "runs" / "rehearsal"
LIVE = HERE / "runs" / "live"
IGNORE = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "*.gguf", "*.onnx", "*.msgpack", "*.h5"]
COLS = ("attention", "gradient × input", "rollout")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def versions() -> dict:
    from importlib import metadata
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── 純 python 的小工具 ────────────────────────────────────────────
def ranks(xs: list[float]) -> list[float]:
    """平均秩（大的排前面）。"""
    order = sorted(range(len(xs)), key=lambda i: -xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def normalize(xs: list[float]) -> list[float]:
    s = sum(xs)
    return [x / s for x in xs] if s else xs


def top(xs: list[float], k: int) -> list[int]:
    return sorted(range(len(xs)), key=lambda i: -xs[i])[:k]


def clean_token(tok: str) -> str:
    return tok.replace("Ġ", "␣").replace("Ċ", "⏎").replace(" ", "␣")


# ── 後端 ─────────────────────────────────────────────────────────
class TorchBackend:
    fake = False

    def __init__(self, repo: str, revision: str | None, offline: bool, device: str, attn: str = "eager"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
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
        c = self.model.config
        self.config = {"model_type": c.model_type, "num_hidden_layers": c.num_hidden_layers,
                       "num_attention_heads": c.num_attention_heads,
                       "num_key_value_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
                       "hidden_size": c.hidden_size, "head_dim": getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
                       "vocab_size": c.vocab_size, "attn_implementation": getattr(c, "_attn_implementation", attn)}

    def tokens(self, text: str) -> tuple[list[int], list[str]]:
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        toks = [clean_token(self.tok.convert_tokens_to_string([t])) for t in self.tok.convert_ids_to_tokens(ids)]
        return ids, toks

    def analyze(self, text: str, layer: int, heads: list[int], residual: float) -> dict:
        """一次前向（開梯度）＋一次反向。回傳三種分數與各 head 的最後一列。"""
        torch = self.torch
        ids, toks = self.tokens(text)
        L = len(ids)
        x = torch.tensor([ids], device=self.device)
        emb = self.model.get_input_embeddings()(x).detach().requires_grad_(True)
        t0 = time.time()
        with torch.enable_grad():
            out = self.model(inputs_embeds=emb, output_attentions=True, use_cache=False)
            logits = out.logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt_id = int(logits.argmax())
            target = logp[tgt_id]
            target.backward()
        seconds = time.time() - t0
        atts = out.attentions
        if atts is None or any(a is None for a in atts):
            raise RuntimeError(f"attn_implementation={self.attn!r} 沒有回傳 attention 權重（output_attentions=True 被忽略）；用 eager")
        atts = [a[0].detach().float().cpu() for a in atts]           # 每層 [n_head, L, L]
        # (ii) gradient × input：每個位置的 embedding 與它的梯度做內積
        gxi = (emb.grad[0] * emb[0]).sum(-1).detach().float().cpu().tolist()
        gxi_abs = [abs(v) for v in gxi]
        # (iii) rollout
        eye = torch.eye(L)
        R = None
        per_layer_mean = []
        for A in atts:
            Am = A.mean(0)
            per_layer_mean.append(Am)
            At = (1 - residual) * Am + residual * eye
            At = At / At.sum(-1, keepdim=True)
            R = At if R is None else At @ R
        rollout_last = R[-1].tolist()
        # (i) 指定層各 head 的最後一列，順便記全部 head 的第一名
        layer_att = atts[layer]                                       # [n_head, L, L]
        head_rows = {h: layer_att[h, -1].tolist() for h in range(layer_att.shape[0])}
        top1_by_head = [int(torch.argmax(layer_att[h, -1])) for h in range(layer_att.shape[0])]
        top5 = torch.topk(logp, 5)
        return {
            "text": text, "tokens": toks, "ids": ids, "L": L,
            "target_token": clean_token(self.tok.decode([tgt_id])), "target_logprob": float(target),
            "next_top5": [(clean_token(self.tok.decode([int(i)])), float(p)) for p, i in zip(top5.values, top5.indices)],
            "layer": layer, "heads": heads,
            "attention_rows": {str(h): head_rows[h] for h in heads},
            "attention_matrix": {str(h): layer_att[h].tolist() for h in heads},
            "layer_mean_matrix": per_layer_mean[layer].tolist(),
            "top1_by_head": top1_by_head,
            "gradient_x_input": gxi, "gradient_x_input_share": normalize(gxi_abs),
            "rollout": rollout_last, "rollout_matrix": R.tolist(),
            "seconds": seconds, "fake": False,
        }


class FakeBackend:
    fake = True

    def __init__(self, repo: str, revision: str | None, offline: bool, device: str, attn: str = "eager"):
        self.path, self.load_seconds, self.device, self.attn = "(fake)", 0.0, device, attn
        self.config = {"model_type": "fake", "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                       "hidden_size": 1024, "head_dim": 128, "vocab_size": 151936, "attn_implementation": "fake"}
        self.rng = random.Random(3)

    def tokens(self, text: str):
        toks = text.split() if " " in text else list(text)
        return list(range(len(toks))), toks

    def analyze(self, text: str, layer: int, heads: list[int], residual: float) -> dict:
        ids, toks = self.tokens(text)
        L = len(ids)
        rng = self.rng

        def row(sink=0.4):
            r = [rng.random() for _ in range(L)]
            r[0] += sink * L
            return normalize(r)

        rows = {str(h): row() for h in heads}
        mats = {str(h): [normalize([rng.random() for _ in range(i + 1)] + [0.0] * (L - i - 1)) for i in range(L)] for h in heads}
        gxi = [rng.uniform(-1, 1) for _ in range(L)]
        time.sleep(0.3)
        return {"text": text, "tokens": toks, "ids": ids, "L": L, "target_token": "␣were", "target_logprob": -0.7,
                "next_top5": [("␣were", -0.7), ("␣have", -1.9), ("␣are", -2.4), ("␣had", -3.0), ("␣was", -3.3)],
                "layer": layer, "heads": heads, "attention_rows": rows, "attention_matrix": mats,
                "layer_mean_matrix": mats[str(heads[0])], "top1_by_head": [rng.randrange(L) for _ in range(16)],
                "gradient_x_input": gxi, "gradient_x_input_share": normalize([abs(v) for v in gxi]),
                "rollout": row(0.2), "rollout_matrix": mats[str(heads[0])], "seconds": 0.3, "fake": True}


def make_backend(args, cfg: dict, attn: str = "eager"):
    m = cfg["models"][args.model]
    revision = locked_revision(m["repo"], m.get("revision") or None)
    cls = FakeBackend if args.fake else TorchBackend
    note(f"載入 {m['label']}  ←  {m['repo']}" + (f"@{revision[:10]}" if revision else "") + f"  device={cfg['demo']['device']}  attn={attn}")
    b = cls(m["repo"], revision, offline=not args.online, device=cfg["demo"]["device"], attn=attn)
    note(f"載入完成：{b.load_seconds:.1f} 秒")
    b.label, b.repo, b.revision = m["label"], m["repo"], revision
    return b


def resolve_layer(cfg: dict, n_layer: int) -> int:
    l = cfg["demo"]["layer"]
    return n_layer // 2 if l == "mid" else int(l) % n_layer


# ── 畫面 ─────────────────────────────────────────────────────────
def shade(v: float, vmax: float) -> str:
    """0..vmax → 背景色（白 → 深藍），24-bit ANSI。"""
    t = 0.0 if vmax <= 0 else max(0.0, min(1.0, v / vmax))
    r, g, b = int(255 - 200 * t), int(255 - 170 * t), int(255 - 80 * t)
    fg = "38;2;20;20;20" if t < 0.55 else "38;2;240;240;240"
    return f"\033[48;2;{r};{g};{b}m\033[{fg}m"


def print_heatmap(tokens: list[str], M: list[list[float]], title: str, width: int = 4) -> None:
    """L×L 矩陣的終端機熱圖（每格一個顏色塊）；token 太多就只印最後 24 個。"""
    L = len(tokens)
    lo = max(0, L - 24)
    vmax = max(max(r[lo:]) for r in M[lo:]) or 1.0
    print(f"  {DIM}{title}{RESET}")
    lab = [t[:width - 1].ljust(width - 1) for t in tokens]
    for i in range(lo, L):
        cells = "".join(f"{shade(M[i][j], vmax)}{' ' * width}{RESET}" for j in range(lo, L))
        print(f"  {tokens[i][:10]:>10} {cells}")
    print(f"  {'':>10} " + "".join(l.ljust(width) for l in lab[lo:]))


def print_bar_row(tokens: list[str], xs: list[float], title: str) -> None:
    vmax = max(xs) or 1.0
    print(f"  {DIM}{title}{RESET}")
    print("  " + "".join(f"{shade(v, vmax)} {t[:6]:^6} {RESET}" for t, v in zip(tokens, xs)))


def print_table(r: dict, head: int, revealed: int, k: int) -> None:
    """三欄並列，revealed = 已揭露幾欄。"""
    toks = r["tokens"]
    cols = [r["attention_rows"][str(head)], r["gradient_x_input_share"], r["rollout"]]
    hdr = [f"attention (layer {r['layer']}, head {head})", "gradient × input (|·|, share)", "rollout (Abnar & Zuidema)"]
    print(f"\n  {'rank':<5}" + "".join(f"{h:<34}" for h in hdr[:revealed]))
    for i in range(k):
        line = f"  {i + 1:<5}"
        for c in cols[:revealed]:
            idx = top(c, k)[i]
            line += f"{BOLD}{toks[idx][:14]:<14}{RESET}{c[idx]:6.3f}  [{idx:>2}]      "
        print(line)
    if revealed >= 2:
        pairs = [(0, 1)] + ([(0, 2), (1, 2)] if revealed >= 3 else [])
        s = "   ".join(f"ρ({COLS[a]} , {COLS[b]}) = {spearman(cols[a], cols[b]):+.2f}" for a, b in pairs)
        print(f"  {GREY}Spearman over all {len(toks)} tokens:  {s}{RESET}")


def print_sentence_header(r: dict) -> None:
    print(f"\n  {BOLD}{r['text']}{RESET}")
    print(f"  {len(r['tokens'])} tokens: " + " ".join(f"{GREY}{i}{RESET}{t}" for i, t in enumerate(r["tokens"])))
    n5 = ", ".join(f"{t} ({math.exp(p):.2f})" for t, p in r["next_top5"])
    print(f"  next-token prediction at the last position: {BOLD}{r['target_token']}{RESET}  (p = {math.exp(r['target_logprob']):.2f});  top-5: {n5}")
    print(f"  {GREY}all three columns ask: which input tokens matter for that prediction?{RESET}")


def print_head_spread(r: dict) -> None:
    toks = r["tokens"]
    counts: dict[int, int] = {}
    for i in r["top1_by_head"]:
        counts[i] = counts.get(i, 0) + 1
    spread = ", ".join(f"{toks[i]}×{n}" for i, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    print(f"  {GREY}layer {r['layer']}: top-1 token of each of the {len(r['top1_by_head'])} heads → {spread}{RESET}")


# ── SVG（不用 matplotlib；退路用）────────────────────────────────
def svg_heatmap(tokens: list[str], M: list[list[float]], title: str, path: Path) -> None:
    L = len(tokens)
    cell, left, topm = 22, 110, 40
    W, H = left + L * cell + 20, topm + L * cell + 110
    vmax = max(max(r) for r in M) or 1.0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>', f'<text x="{left}" y="24" font-size="14" font-weight="600">{esc(title)}</text>']
    for i in range(L):
        for j in range(L):
            t = max(0.0, min(1.0, M[i][j] / vmax))
            col = f"rgb({int(255 - 200 * t)},{int(255 - 170 * t)},{int(255 - 80 * t)})"
            out.append(f'<rect x="{left + j * cell}" y="{topm + i * cell}" width="{cell}" height="{cell}" fill="{col}"/>')
        out.append(f'<text x="{left - 6}" y="{topm + i * cell + 15}" text-anchor="end">{esc(tokens[i][:12])}</text>')
    for j in range(L):
        x, y = left + j * cell + 11, topm + L * cell + 8
        out.append(f'<text transform="translate({x},{y}) rotate(60)" text-anchor="start">{esc(tokens[j][:12])}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_bars(r: dict, head: int, path: Path) -> None:
    toks = r["tokens"]
    L = len(toks)
    cols = [(f"attention (layer {r['layer']}, head {head})", r["attention_rows"][str(head)]),
            ("gradient × input (|·|, share)", r["gradient_x_input_share"]), ("attention rollout", r["rollout"])]
    cw, bh, left = 300, 14, 90
    W, H = left + 3 * (cw + 30), 40 + L * bh + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for c, (title, xs) in enumerate(cols):
        x0 = left + c * (cw + 30)
        vmax = max(xs) or 1.0
        out.append(f'<text x="{x0}" y="22" font-size="13" font-weight="600">{esc(title)}</text>')
        for i, v in enumerate(xs):
            y = 34 + i * bh
            if c == 0:
                out.append(f'<text x="{x0 - 6}" y="{y + 11}" text-anchor="end">{esc(toks[i][:12])}</text>')
            out.append(f'<rect x="{x0}" y="{y + 2}" width="{max(1.0, (cw - 60) * v / vmax):.1f}" height="{bh - 4}" fill="#3b6fb6"/>')
            out.append(f'<text x="{x0 + (cw - 60) * v / vmax + 4:.1f}" y="{y + 11}" fill="#555">{v:.3f}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
        print(f"{GREEN}完成，revision 已寫進 ../versions.lock{RESET}")


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    rec = {"date": now(), "versions": v, "device": cfg["demo"]["device"]}
    if not args.fake:
        import torch
        rec["mps_available"] = bool(torch.backends.mps.is_available())
        print(f"  mps available    {rec['mps_available']}")
    # 1. eager：正式用的載法
    b = make_backend(args, cfg, attn="eager")
    c = b.config
    print(f"\n  config: {c['model_type']}  {c['num_hidden_layers']} layers × {c['num_attention_heads']} heads"
          f" ({c['num_key_value_heads']} KV heads, head_dim {c['head_dim']})   attn_implementation = {BOLD}{c['attn_implementation']}{RESET}")
    layer = resolve_layer(cfg, c["num_hidden_layers"])
    heads = [h % c["num_attention_heads"] for h in cfg["demo"]["heads"]]
    s = cfg["sentences"][0]["text"]
    r = b.analyze(s, layer, heads, cfg["demo"]["rollout_residual"])
    print(f"  eager: output_attentions=True → {len(r['attention_matrix'])} heads kept of layer {layer}; matrix {r['L']}×{r['L']};"
          f" rows sum to {sum(r['attention_rows'][str(heads[0])]):.4f};  forward+backward {r['seconds']:.2f} s on {b.device}")
    rec.update({"model": b.repo, "revision": b.revision, "config": c, "layer": layer, "heads": heads,
                "eager_ok": True, "eager_seconds": r["seconds"], "tokens": r["tokens"], "target_token": r["target_token"]})
    # 2. sdpa：確認「不回傳權重」到底是回 None、警告後退回 eager、還是報錯（大綱說待實測）
    if not args.fake:
        try:
            b2 = make_backend(args, cfg, attn="sdpa")
            try:
                r2 = b2.analyze(s, layer, heads, cfg["demo"]["rollout_residual"])
                rec["sdpa_behaviour"] = f"returned attentions (model reports attn_implementation={b2.config['attn_implementation']})"
            except RuntimeError as e:
                rec["sdpa_behaviour"] = f"no attentions: {e}"
        except Exception as e:  # noqa: BLE001
            rec["sdpa_behaviour"] = f"load or forward failed: {type(e).__name__}: {e}"
        print(f"  sdpa:  {AMBER}{rec['sdpa_behaviour']}{RESET}")
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    path = REHEARSAL / ("inspect.fake.json" if b.fake else "inspect.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def cmd_rehearse(args, cfg):
    b = make_backend(args, cfg)
    c = b.config
    layer = resolve_layer(cfg, c["num_hidden_layers"])
    heads = [h % c["num_attention_heads"] for h in cfg["demo"]["heads"]]
    k = int(cfg["demo"]["top_k"])
    banner(f"{cfg['demo']['title']}  ·  {b.label}" + (f"  {RED}[FAKE]{RESET}" if b.fake else ""))
    results = []
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    suffix = ".fake" if b.fake else ""
    for si, s in enumerate(cfg["sentences"]):
        r = b.analyze(s["text"], layer, heads, cfg["demo"]["rollout_residual"])
        r["note"] = s.get("note", "")
        print_sentence_header(r)
        for h in heads:
            print_table(r, h, 3, k)
        print_head_spread(r)
        print_heatmap(r["tokens"], r["attention_matrix"][str(heads[0])], f"attention, layer {layer}, head {heads[0]}")
        print_heatmap(r["tokens"], r["rollout_matrix"], "rollout, all layers")
        print_bar_row(r["tokens"], r["gradient_x_input_share"], "gradient × input (|·|, share)")
        for h in heads:
            svg_heatmap(r["tokens"], r["attention_matrix"][str(h)], f"attention · layer {layer} · head {h} · {s['text']}",
                        REHEARSAL / f"s{si}_attention_l{layer}h{h}{suffix}.svg")
            svg_bars(r, h, REHEARSAL / f"s{si}_rankings_l{layer}h{h}{suffix}.svg")
        svg_heatmap(r["tokens"], r["rollout_matrix"], f"attention rollout · {s['text']}", REHEARSAL / f"s{si}_rollout{suffix}.svg")
        r["spearman"] = {"attention_vs_gxi": {str(h): spearman(r["attention_rows"][str(h)], r["gradient_x_input_share"]) for h in heads},
                         "attention_vs_rollout": {str(h): spearman(r["attention_rows"][str(h)], r["rollout"]) for h in heads},
                         "gxi_vs_rollout": spearman(r["gradient_x_input_share"], r["rollout"])}
        results.append(r)
    rec = {"date": now(), "model": b.repo, "revision": b.revision, "versions": versions(), "device": b.device,
           "attn_implementation": c["attn_implementation"], "config": c, "layer": layer, "heads": heads,
           "rollout_residual": cfg["demo"]["rollout_residual"], "target": cfg["demo"]["target"], "fake": b.fake, "sentences": results}
    path = REHEARSAL / f"rankings{suffix}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"\n→ {path.relative_to(HERE)}  與同資料夾的 *.svg")


def _reveal_loop(results: list[dict], heads: list[int], k: int, live: bool, on_row=None) -> None:
    """課堂用的逐步揭露。Enter：attention → gxi → rollout → 換 head（循環）；n：下一句；q：離開。"""
    si = 0
    while si < len(results):
        r = results[si]
        print_sentence_header(r)
        stage, hi = 0, 0
        while True:
            head = heads[hi % len(heads)]
            if stage == 0:
                prompt = f"Enter = column 1: attention (layer {r['layer']}, head {head})"
            elif stage == 1:
                prompt = "Enter = column 2: gradient × input"
            elif stage == 2:
                prompt = "Enter = column 3: attention rollout"
            else:
                prompt = f"Enter = another head (head {heads[(hi + 1) % len(heads)]})"
            key = input(f"  {DIM}{prompt}   n = 下一句   q = 離開{RESET} ").strip().lower()
            print("\033[1A\033[2K", end="")
            if key == "q":
                return
            if key == "n":
                break
            if stage >= 3:
                hi += 1
                head = heads[hi % len(heads)]
                print(f"\n  {AMBER}head {head} — column 1 changes, columns 2 and 3 do not{RESET}")
                print_table(r, head, 3, k)
                print_head_spread(r)
                print_heatmap(r["tokens"], r["attention_matrix"][str(head)], f"attention, layer {r['layer']}, head {head}")
            else:
                stage += 1
                print_table(r, head, stage, k)
                if stage == 1:
                    print_heatmap(r["tokens"], r["attention_matrix"][str(head)], f"attention, layer {r['layer']}, head {head}")
                elif stage == 2:
                    print_bar_row(r["tokens"], r["gradient_x_input_share"], "gradient × input (|·|, share)")
                elif stage == 3:
                    print_heatmap(r["tokens"], r["rollout_matrix"], "rollout, all layers")
            if on_row:
                on_row({"date": now(), "sentence": r["text"], "stage": stage, "head": head})
        si += 1


def cmd_show(args, cfg):
    b = make_backend(args, cfg)
    c = b.config
    layer = resolve_layer(cfg, c["num_hidden_layers"])
    heads = [h % c["num_attention_heads"] for h in cfg["demo"]["heads"]]
    k = int(cfg["demo"]["top_k"])
    banner(f"{cfg['demo']['title']}  ·  {b.label}" + (f"  {RED}[FAKE]{RESET}" if b.fake else ""))
    note(f"{c['num_hidden_layers']} layers × {c['num_attention_heads']} heads; column 1 starts at layer {layer}, heads {heads}. 先算好所有句子再逐欄揭露。")
    results = []
    for s in cfg["sentences"]:
        r = b.analyze(s["text"], layer, heads, cfg["demo"]["rollout_residual"])
        results.append(r)
        note(f"  {s['text'][:40]}…  {r['L']} tokens, {r['seconds']:.2f} s")
    LIVE.mkdir(parents=True, exist_ok=True)
    log = LIVE / f"{now()[:10].replace('-', '')}.jsonl"
    _reveal_loop(results, heads, k, live=True, on_row=lambda rec: append_jsonl(log, {"model": b.repo, **rec}))
    print()


def cmd_replay(args, cfg):
    path = REHEARSAL / ("rankings.fake.json" if args.fake else "rankings.json")
    if not path.exists():
        print(f"{RED}沒有 {path.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
        return
    reh = json.loads(path.read_text(encoding="utf-8"))
    banner(f"{AMBER}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {reh['date']}  {reh['model']}  ({reh['attn_implementation']}, {reh['device']})")
    _reveal_loop(reh["sentences"], reh["heads"], int(cfg["demo"]["top_k"]), live=False)
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
