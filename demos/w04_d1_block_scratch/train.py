#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 1 的訓練（彩排用，不在課堂從頭跑）、課堂上的短跑、MPS 探針、以及 replay。

    ./present.sh rehearse-train     訓到 [train].steps（彩排）：loss 曲線、held-out 的 nats/char 與 bits-per-byte、步數與秒數
                                    → runs/rehearsal/train.json 與 loss.svg；p22 的 replay 與講稿裡的數字從這裡抄
    ./present.sh peek               課堂（有時間才做）：現場跑最初 [train].live_steps 步，看 loss 開始掉；不存檔
    ./present.sh probe              課前：torch 版本、MPS 可不可用、內建 F.scaled_dot_product_attention 在 MPS 上
                                    dropout_p > 0 是報錯還是靜默接受、手刻的 attention dropout 在 MPS 上前向＋反向能不能跑
                                    → runs/rehearsal/probe.json（附錄 C 那一條「待實測」在這裡變成實測）
    ./present.sh replay train       現場：印彩排的曲線（終端機）與最後的 BPB 對 W1 四個 n-gram 的 BPB；loss.svg 用 open 開

同一把尺要說清楚（README 與 p22 的【提示】也寫了）：
  語料與 held-out 文字和 W1 相同（W1 把 regex token 的前 90% 當 train，這裡取同一個切點的字元起點），
  但 W1 的 n-gram 只預測詞與標點（regex 切出來的 token，空白、換行、連字號、引號不在裡面），分母是 ' '.join(tokens) 的 bytes；
  這裡的字元模型預測原文的每一個字元，分母是原文的 UTF-8 bytes。兩個分母都印出來（bpb_raw、bpb_w1_denominator），
  但「預測目標不同」這件事分母換不掉——這正是 W1 p27 講的：BPB 可比的前提是同一條位元組流。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "demo_config.toml"
REHEARSAL = HERE / "runs" / "rehearsal"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


# ── 語料：字元級，切點對齊 W1 ────────────────────────────────────
def load_corpus(cfg: dict) -> dict:
    cc = cfg["corpus"]
    p = (HERE / cc["path"]).resolve()
    if not p.exists():
        raise FileNotFoundError(f"沒有語料 {p}：先跑一次 python3 nlp_llm/slides/scripts/make_demo_w01.py（它會下載 tinyshakespeare）")
    text = p.read_text(encoding="utf-8")
    toks = list(re.finditer(cc["w1_token_regex"], text))
    split = int(len(toks) * cc["w1_split"])                 # W1：split = int(len(toks)*0.9)
    cut = toks[split].start()                                # 第 split 個 token 的字元起點
    train_text, held_text = text[:cut], text[cut:]
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    held_tokens = [m.group(0) for m in toks[split:]]
    return {"path": str(p), "chars": chars, "stoi": stoi, "V": len(chars),
            "train_text": train_text, "held_text": held_text, "cut_char": cut,
            "w1_tokens_total": len(toks), "w1_split_token": split,
            "held_bytes_raw": len(held_text.encode("utf-8")),
            "held_bytes_w1": len(" ".join(held_tokens).encode("utf-8")),   # W1 的分母：bytes_held = len(' '.join(held).encode('utf-8'))
            "held_chars": len(held_text), "train_chars": len(train_text)}


def w1_bpb(cfg: dict) -> dict | None:
    p = (HERE / cfg["corpus"]["w1_data"]).resolve()
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"bpb": d.get("bpb"), "corpus": d.get("corpus")}


def pick_device(name: str):
    import torch
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "mps":
        print(f"{AMBER}MPS 不可用，改用 CPU{RESET}")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: dict, V: int, **override):
    import block_final as bf
    t = cfg["tiny"]
    kw = dict(V=V, d=t["d"], n_layer=t["n_layer"], n_head=t["n_head"], d_ff=t["d_ff"], L=t["L"], dropout=t["dropout"])
    kw.update(override)
    return bf.GPT(**kw)


def evaluate(model, data, L: int, stride: int, device, batch: int = 32) -> dict:
    """held-out：視窗 L、每次前進 stride；第一個視窗的 L 個目標全計，之後每個視窗只計還沒被計過的（最後 stride 個）目標，
    所以 held-out 的每個字元（第一個除外，它沒有上文）剛好被預測一次，前面的字元當上文。回 nats 總和與字元數。"""
    import torch
    model.eval()
    n = data.numel()
    last = n - L - 1                                          # 最後一個合法的視窗起點：x = data[s:s+L]、y = data[s+1:s+L+1]
    starts = list(range(0, max(1, last), stride))
    if starts[-1] < last:
        starts.append(last)                                   # 收尾：最後一個視窗貼齊結尾
    todo, prev_end = [], 0                                    # 每個視窗要計幾個目標（目標的索引是 s+1 … s+L）
    for s in starts:
        k = (s + L + 1) - max(prev_end, s + 1)
        todo.append(max(0, k))
        prev_end = s + L + 1
    total_nll, total_chars = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            x = torch.stack([data[s:s + L] for s in chunk])
            y = torch.stack([data[s + 1:s + L + 1] for s in chunk])
            logits = model(x.to(device))
            nll = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.to(device).view(-1),
                                                    reduction="none").view(len(chunk), L)
            for j in range(len(chunk)):
                k = todo[i + j]
                if k:
                    total_nll += nll[j, L - k:].sum().item()
                    total_chars += k
    model.train()
    return {"nll_nats": total_nll, "chars": total_chars, "nats_per_char": total_nll / total_chars}


def bpb_from(ev: dict, corpus: dict) -> dict:
    bits = ev["nll_nats"] / math.log(2)
    return {"bits_total": bits, "bpc": bits / ev["chars"],
            "bpb_raw": bits / corpus["held_bytes_raw"], "bpb_w1_denominator": bits / corpus["held_bytes_w1"]}


def lr_at(step: int, cfg: dict) -> float:
    t = cfg["train"]
    if step < t["warmup"]:
        return t["lr"] * (step + 1) / t["warmup"]
    frac = (step - t["warmup"]) / max(1, t["steps"] - t["warmup"])
    return t["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))   # cosine 到 0.1·lr


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and "emb" not in n else no_decay).append(p)     # 矩陣 decay；Norm 的 γ 與 embedding 不 decay
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def train(cfg: dict, steps: int, save: bool, label: str) -> dict:
    import torch
    import block_final as bf
    t, tiny = cfg["train"], cfg["tiny"]
    torch.manual_seed(t["seed"])
    corpus = load_corpus(cfg)
    device = pick_device(t["device"])
    enc = lambda s: torch.tensor([corpus["stoi"][c] for c in s], dtype=torch.long)
    train_data, held_data = enc(corpus["train_text"]), enc(corpus["held_text"])
    model = build_model(cfg, corpus["V"]).to(device)
    n_params = bf.count_params(model)
    import tests
    lg = tests.ledger({"V": corpus["V"], "d": tiny["d"], "n_layer": tiny["n_layer"], "n_head": tiny["n_head"], "d_ff": tiny["d_ff"]})
    opt = torch.optim.AdamW(param_groups(model, t["weight_decay"]), lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8)
    L, B = tiny["L"], t["batch"]
    print(f"{BOLD}{label}{RESET}  torch {torch.__version__}, device {device}, {platform.machine()}")
    print(f"  corpus {Path(corpus['path']).name}: V = {corpus['V']} chars; train {corpus['train_chars']:,} chars, held-out {corpus['held_chars']:,} chars "
          f"(cut at W1 token {corpus['w1_split_token']:,}/{corpus['w1_tokens_total']:,} = char {corpus['cut_char']:,})")
    print(f"  tiny GPT: d {tiny['d']}, n_layer {tiny['n_layer']}, n_head {tiny['n_head']}, d_ff {tiny['d_ff']}, L {L}, dropout {tiny['dropout']}; "
          f"params {n_params:,} (term by term {lg['exact']:,}; approx {lg['approx']:,}, {lg['rel_err']:+.2%})")
    print(f"  AdamW lr {t['lr']}, betas {t['betas']}, wd {t['weight_decay']} (decoupled, matrices only), warmup {t['warmup']}, cosine → 0.1·lr, clip {t['grad_clip']}; "
          f"batch {B} × {L} = {B * L:,} tokens/step; {steps} steps")
    g = torch.Generator().manual_seed(t["seed"])
    curve, evals = [], []
    model.train()
    t0 = time.time()
    tok_seen = 0
    for step in range(steps):
        ix = torch.randint(0, train_data.numel() - L - 1, (B,), generator=g)
        x = torch.stack([train_data[i:i + L] for i in ix]).to(device)
        y = torch.stack([train_data[i + 1:i + L + 1] for i in ix]).to(device)
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, cfg)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, corpus["V"]), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
        opt.step()
        tok_seen += B * L
        if step % t["log_every"] == 0 or step == steps - 1:
            el = time.time() - t0
            curve.append({"step": step, "loss": loss.item(), "grad_norm": gn, "lr": lr_at(step, cfg), "elapsed_s": el})
            print(f"  step {step:>5}  loss {loss.item():.4f}  ({loss.item() / math.log(2):.3f} bits/char)  |g| {gn:.2f}  lr {lr_at(step, cfg):.2e}  {el:6.1f} s  {tok_seen / max(el, 1e-9):,.0f} tok/s")
        if (step % t["eval_every"] == 0 and step > 0) or step == steps - 1:
            ev = evaluate(model, held_data, L, t["eval_stride"], device)
            bb = bpb_from(ev, corpus)
            evals.append({"step": step, "elapsed_s": time.time() - t0, **ev, **bb})
            print(f"  {BLUE}held-out @ {step}: {ev['nats_per_char']:.4f} nats/char = {bb['bpc']:.3f} bits/char; "
                  f"BPB {bb['bpb_raw']:.3f} (raw bytes) / {bb['bpb_w1_denominator']:.3f} (W1's denominator){RESET}")
    seconds = time.time() - t0
    # 生成幾個字元看看（不是評估）
    model.eval()
    idx = torch.tensor([[corpus["stoi"]["\n"]]] if "\n" in corpus["stoi"] else [[0]], device=device)
    with torch.no_grad():
        for _ in range(t["sample_chars"]):
            logits = model(idx[:, -L:])[:, -1, :]
            nxt = torch.multinomial(torch.softmax(logits.float(), -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
    sample = "".join(corpus["chars"][i] for i in idx[0].tolist())
    final = evals[-1] if evals else None
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "torch": torch.__version__, "device": str(device),
           "python": platform.python_version(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
           "tiny": tiny, "train": {**t, "steps": steps}, "corpus": {k: v for k, v in corpus.items() if k not in ("chars", "stoi", "train_text", "held_text")},
           "params": {"count": n_params, "exact": lg["exact"], "approx": lg["approx"], "approx_rel_err": lg["rel_err"]},
           "seconds": seconds, "tokens_seen": tok_seen, "tokens_per_s": tok_seen / seconds,
           "curve": curve, "evals": evals, "final": final, "w1": w1_bpb(cfg), "sample": sample}
    print(f"\n  {BOLD}{steps} steps in {seconds:.0f} s ({seconds / 60:.1f} min), {tok_seen / seconds:,.0f} tokens/s{RESET}")
    if final:
        print_bpb_table(rec)
    print(f"\n  sample ({t['sample_chars']} chars, temperature 1):\n{GREY}{sample}{RESET}")
    if save:
        REHEARSAL.mkdir(parents=True, exist_ok=True)
        (REHEARSAL / "train.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        svg_curve(rec, REHEARSAL / "loss.svg")
        print(f"{DIM}→ runs/rehearsal/train.json, runs/rehearsal/loss.svg{RESET}")
    return rec


def print_bpb_table(rec: dict) -> None:
    f = rec["final"]
    w1 = rec.get("w1") or {}
    print(f"\n  {BOLD}bits-per-byte on the same held-out text (the last 10% of W1's tokens: The Tempest){RESET}")
    w1d = '" ".join(tokens)'
    print(f"    {'model':<40} {'predicts':<28} {'bytes in denominator':>22} {'BPB':>8}")
    if w1.get("bpb"):
        for n, v in sorted(w1["bpb"].items(), key=lambda kv: int(kv[0])):
            lab = f"W1 {n}-gram, stupid backoff (w01-data.json)"
            print(f"    {lab:<40} {'words + punctuation':<28} {w1d:>22} {v:>8.3f}")
    print(f"    {'this tiny GPT, char-level':<40} {'every character':<28} {'raw UTF-8':>22} {f['bpb_raw']:>8.3f}")
    print(f"    {'  same bits, W1 denominator':<40} {'':<28} {w1d:>22} {f['bpb_w1_denominator']:>8.3f}")
    print(f"    {GREY}held-out: {rec['corpus']['held_chars']:,} chars = {rec['corpus']['held_bytes_raw']:,} raw bytes; W1 denominator {rec['corpus']['held_bytes_w1']:,} bytes. "
          f"Not the same ruler: the n-gram never pays for whitespace, line breaks or the characters its regex drops (W1 p27).{RESET}")


# ── MPS 探針 ─────────────────────────────────────────────────────
def probe(cfg: dict, save: bool = True) -> dict:
    import torch
    import torch.nn.functional as F
    import block_final as bf
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__, "python": platform.python_version(),
           "macos": platform.mac_ver()[0], "machine": platform.machine(), "mps_available": torch.backends.mps.is_available(),
           "mps_built": torch.backends.mps.is_built()}
    print(f"{BOLD}probe{RESET}  torch {torch.__version__}, macOS {rec['macos']}, mps available {rec['mps_available']}")
    dev = torch.device("mps") if rec["mps_available"] else torch.device("cpu")
    B, H, L, dh = 2, 4, 16, 32

    def try_sdpa(name, **kw):
        q = torch.randn(B, H, L, dh, device=dev, requires_grad=True)
        k, v = torch.randn(B, H, L, dh, device=dev), torch.randn(B, H, L, dh, device=dev)
        try:
            out = F.scaled_dot_product_attention(q, k, v, **kw)
            out.sum().backward()
            r = {"status": "ok", "grad_finite": bool(torch.isfinite(q.grad).all())}
        except Exception as exc:
            r = {"status": "error", "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"}
        print(f"  F.scaled_dot_product_attention on {dev} {kw}: {GREEN if r['status'] == 'ok' else RED}{r['status']}{RESET} {r.get('error', '')}")
        return r

    rec["builtin_sdpa"] = {"dropout_0": try_sdpa("dropout_0", dropout_p=0.0, is_causal=True),
                           "dropout_0.1_train": try_sdpa("dropout_0.1", dropout_p=0.1, is_causal=True)}
    # 手刻：W3 的函數 + dropout(A) @ V，在同一個裝置上前向＋反向
    try:
        attn = bf.CausalSelfAttention(64, 4, dropout=0.1).to(dev).train()
        x = torch.randn(B, L, 64, device=dev, requires_grad=True)
        attn(x).sum().backward()
        rec["handwritten_attention_dropout"] = {"status": "ok", "grad_finite": bool(torch.isfinite(x.grad).all())}
    except Exception as exc:
        rec["handwritten_attention_dropout"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"}
    r = rec["handwritten_attention_dropout"]
    print(f"  hand-written attention (W3's function, dropout on A) on {dev}, forward + backward: {GREEN if r['status'] == 'ok' else RED}{r['status']}{RESET} {r.get('error', '')}")
    # 一步訓練的時間量級（tiny 設定，10 步）
    try:
        t = cfg["tiny"]
        m = bf.GPT(65, t["d"], t["n_layer"], t["n_head"], t["d_ff"], t["L"], dropout=t["dropout"]).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        Bt = cfg["train"]["batch"]
        x = torch.randint(0, 65, (Bt, t["L"]), device=dev)
        for i in range(12):
            if i == 2:
                if dev.type == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
            loss = torch.nn.functional.cross_entropy(m(x).float().view(-1, 65), x.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        per = (time.time() - t0) / 10
        rec["step_time_s"] = per
        rec["est_minutes_for_steps"] = per * cfg["train"]["steps"] / 60
        print(f"  one training step (batch {Bt} × {t['L']}, tiny GPT) on {dev}: {per:.3f} s → {cfg['train']['steps']} steps ≈ {rec['est_minutes_for_steps']:.1f} min")
    except Exception as exc:
        rec["step_time_s"] = None
        print(f"  {RED}timing failed: {type(exc).__name__}: {exc}{RESET}")
    if save:
        REHEARSAL.mkdir(parents=True, exist_ok=True)
        (REHEARSAL / "probe.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{DIM}→ runs/rehearsal/probe.json{RESET}")
    return rec


# ── replay ───────────────────────────────────────────────────────
def replay(cfg: dict) -> None:
    p = REHEARSAL / "train.json"
    if not p.exists():
        print(f"{RED}沒有 runs/rehearsal/train.json：先跑 ./present.sh rehearse-train{RESET}")
        return
    rec = json.loads(p.read_text(encoding="utf-8"))
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {rec['date']}, torch {rec['torch']}, {rec['device']}, {rec['machine']}")
    t, tiny = rec["train"], rec["tiny"]
    print(f"  tiny GPT {rec['params']['count']:,} params (d {tiny['d']}, {tiny['n_layer']} layers, {tiny['n_head']} heads, d_ff {tiny['d_ff']}, L {tiny['L']}); "
          f"{t['steps']} steps × {t['batch']} × {tiny['L']} tokens = {rec['tokens_seen']:,} tokens in {rec['seconds']:.0f} s ({rec['seconds'] / 60:.1f} min, {rec['tokens_per_s']:,.0f} tok/s)")
    print(f"\n  {BOLD}train loss (nats/char){RESET}")
    pts = [(c["step"], c["loss"]) for c in rec["curve"]]
    print(ascii_curve(pts, width=72, height=12))
    print(f"\n  {BOLD}held-out{RESET}")
    for e in rec["evals"]:
        print(f"    step {e['step']:>5}  {e['nats_per_char']:.4f} nats/char  {e['bpc']:.3f} bits/char  BPB {e['bpb_raw']:.3f} (raw) / {e['bpb_w1_denominator']:.3f} (W1 denom.)")
    print_bpb_table(rec)
    print(f"\n  {DIM}loss.svg: open {REHEARSAL / 'loss.svg'}{RESET}")


def ascii_curve(pts: list[tuple[int, float]], width: int = 72, height: int = 12) -> str:
    if not pts:
        return "    (no points)"
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    lo, hi = min(ys), max(ys)
    hi = hi if hi > lo else lo + 1e-9
    grid = [[" "] * width for _ in range(height)]
    for x, y in pts:
        col = int((x - xs[0]) / max(1, xs[-1] - xs[0]) * (width - 1))
        row = int((hi - y) / (hi - lo) * (height - 1))
        grid[row][col] = "●"
    lines = []
    for i, row in enumerate(grid):
        lab = f"{hi - (hi - lo) * i / (height - 1):5.2f} │" if i in (0, height // 2, height - 1) else "      │"
        lines.append("    " + lab + "".join(row))
    lines.append("    " + "      └" + "─" * width)
    lines.append("    " + f"       step {xs[0]}" + " " * max(1, width - 18 - len(str(xs[-1]))) + f"step {xs[-1]}")
    return "\n".join(lines)


def svg_curve(rec: dict, path: Path) -> None:
    W, H, l, r, tp, bt = 760, 360, 60, 20, 30, 50
    pts = [(c["step"], c["loss"]) for c in rec["curve"]]
    ev = [(e["step"], e["nats_per_char"]) for e in rec["evals"]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + [p[1] for p in ev]
    x0, x1 = 0, max(xs) or 1
    y0, y1 = min(ys) * 0.95, max(ys) * 1.02
    X = lambda x: l + (x - x0) / (x1 - x0) * (W - l - r)
    Y = lambda y: tp + (y1 - y) / (y1 - y0) * (H - tp - bt)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{l}" y="18" font-size="14" font-weight="600">tiny GPT on tinyshakespeare (char-level), {rec["params"]["count"]:,} params — train loss and held-out nats/char</text>']
    for k in range(5):
        y = y0 + (y1 - y0) * k / 4
        out.append(f'<line x1="{l}" y1="{Y(y):.1f}" x2="{W - r}" y2="{Y(y):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{l - 6}" y="{Y(y) + 4:.1f}" text-anchor="end" fill="#555">{y:.2f}</text>')
    for k in range(5):
        x = x0 + (x1 - x0) * k / 4
        out.append(f'<text x="{X(x):.1f}" y="{H - bt + 16}" text-anchor="middle" fill="#555">{int(x)}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.1f}" y="{H - 8}" text-anchor="middle" fill="#555">step ({rec["train"]["batch"]} × {rec["tiny"]["L"]} tokens each)</text>')
    out.append('<polyline fill="none" stroke="#3b6fb6" stroke-width="1.5" points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts) + '"/>')
    if ev:
        out.append('<polyline fill="none" stroke="#d9534f" stroke-width="2" points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in ev) + '"/>')
        for x, y in ev:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="#d9534f"/>')
        f = rec["final"]
        out.append(f'<text x="{W - r}" y="{tp + 14}" text-anchor="end" fill="#d9534f">held-out {f["nats_per_char"]:.3f} nats/char = {f["bpc"]:.3f} bits/char; BPB {f["bpb_raw"]:.3f} (raw bytes)</text>')
    out.append(f'<text x="{W - r}" y="{tp + 30}" text-anchor="end" fill="#3b6fb6">train loss (per {rec["train"]["log_every"]} steps)</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rehearse", "peek", "probe", "replay"])
    ap.add_argument("--steps", type=int, default=None, help="覆蓋 [train].steps（rehearse）或 [train].live_steps（peek）")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "probe":
        probe(cfg, save=not a.no_save)
    elif a.cmd == "replay":
        replay(cfg)
    elif a.cmd == "peek":
        train(cfg, a.steps or cfg["train"]["live_steps"], save=False, label="peek — live, first steps only, not saved")
    else:
        train(cfg, a.steps or cfg["train"]["steps"], save=not a.no_save, label="rehearse — full training run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
