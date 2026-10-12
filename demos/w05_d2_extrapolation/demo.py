#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 2：長度外推——NoPE vs. RoPE，然後 PI 與 NTK-aware 不重訓（彩排預錄、課堂 replay）。投影片 p24。

上週的 tiny GPT（字元級、L = 256、NoPE）再訓一個 RoPE 版（底 10^4），同一份語料、同一個 held-out、同一組超參數、同樣 3000 步；
對 held-out 以 L_eval ∈ {256, 512, 1024} 各做一次前向，記**每個位置的平均 loss**（nats/char 對位置）；
再對訓好的 RoPE 版套 position interpolation（s = 4）與 NTK-aware（改底）**不重訓**各量一次。四條曲線、四組數字。

  ./present.sh rehearse            全套：訓 nope、訓 rope（各約 9–10 分鐘，W4 實測）→ runs/weights/*.pt；量四組 → runs/rehearsal/*.json + curves.svg
  ./present.sh train rope          只訓一個版本（nope|rope）
  ./present.sh measure             只量：讀 runs/weights/ 的兩個權重，量 nope、rope、rope+PI、rope+NTK → json + svg
  ./present.sh replay              課堂（p24）：Enter 一條線——nope → rope → PI → NTK；最後印 p24 那張表（位置 < 256 / 256–1024）與每個 L_eval 的平均
  ./present.sh show                只印表（投影機上字太小時用；也是 replay 的退路）
  ./present.sh estimate            不訓練，用 W4 demo 1 probe.json 的每步秒數估 rehearse 要跑多久
  ./present.sh rehearse --steps 30 煙霧測試：30 步 + 全套量測，寫到 runs/smoke/（不碰 runs/rehearsal/）

因果模型的一個事實（replay 會印出來）：位置 p 的 loss 只看得到 0…p，與視窗多長無關。所以 L_eval = 1024 那一次前向的前 256 個位置
**就是** L_eval = 256 的曲線（視窗起點不同、數字到小數第三位才會不同）。三個長度的「平均 loss」會變，是因為平均的範圍變了，不是因為
模型在短視窗裡表現不同。這是「逐位置看」比「報一個 perplexity 對 L」更誠實的原因。

誠實的限定（每次 replay 都印）：字元級 tiny GPT 的長距離依賴很弱（它主要在學拼字與格式）；曲線的形狀是機制的形狀，不是任何 7B 模型的門檻值；
所有結論只對這個設定成立。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "demo_config.toml"
REHEARSAL = HERE / "runs" / "rehearsal"
WEIGHTS = HERE / "runs" / "weights"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"
RUNS = ("nope", "rope", "rope_pi", "rope_ntk")
LABEL = {"nope": "NoPE (W4's model, retrained)", "rope": "RoPE, b = 10^4", "rope_pi": "RoPE + PI (s = 4)", "rope_ntk": "RoPE + NTK-aware (b → b·s^(d/(d−2)))"}
LIMIT = ("這是字元級的 tiny GPT（4,763,136 個參數，W4 demo 1 精算的那個數；RoPE 不加參數），主要在學拼字與格式，長距離依賴本來就弱。曲線的形狀是機制的形狀，"
         "不是任何 7B 模型的門檻值；所有結論只對這個設定（同語料、同 held-out、同超參數、3000 步）成立。")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def demo1(cfg: dict):
    import rope_gpt
    return rope_gpt.Demo1(cfg["shared"]["demo1"])


def pairs_under_one_turn(d_head: int, base: float, L: int) -> int:
    """L_train 內轉不滿一圈的對數：λ_i = 2π/θ_i > L。與 demo 1 的 wavelengths.py 同一條公式。"""
    return sum(1 for i in range(d_head // 2) if 2 * math.pi / (base ** (-2 * i / d_head)) > L)


def out_dir(args) -> Path:
    return (HERE / "runs" / "smoke") if args.steps else REHEARSAL


# ── 訓練：沿 W4 demo 1 的迴圈（同超參數、同 seed、同評估）────────────
def train_variant(d1, cfg: dict, pos: str, steps: int | None, save_weights: bool, label: str) -> dict:
    import torch
    t4, tiny = d1.cfg["train"], d1.cfg["tiny"]
    steps = steps or t4["steps"]
    corpus = d1.corpus()
    device = d1.tr.pick_device(t4["device"])
    enc = lambda s: torch.tensor([corpus["stoi"][c] for c in s], dtype=torch.long)
    train_data, held_data = enc(corpus["train_text"]), enc(corpus["held_text"])
    model = d1.build(corpus["V"], pos, cfg["rope"]["base"], seed=t4["seed"]).to(device)
    n_params = d1.bf.count_params(model)
    opt = torch.optim.AdamW(d1.tr.param_groups(model, t4["weight_decay"]), lr=t4["lr"], betas=tuple(t4["betas"]), eps=1e-8)
    L, B = tiny["L"], t4["batch"]
    cfg4 = {"train": {**t4, "steps": steps}}                                   # lr_at 讀 cfg["train"]
    base_note = "" if pos == "nope" else f", base {cfg['rope']['base']:g}"
    print(f"\n{BOLD}{label}{RESET}  pos = {pos}{base_note}  torch {torch.__version__}, device {device}, {platform.machine()}")
    print(f"  corpus {Path(corpus['path']).name}: V = {corpus['V']}; train {corpus['train_chars']:,} chars, held-out {corpus['held_chars']:,} chars (W4's cut)")
    print(f"  tiny GPT: d {tiny['d']}, {tiny['n_layer']} layers, {tiny['n_head']} heads (d_head {tiny['d'] // tiny['n_head']}), d_ff {tiny['d_ff']}, L {L}, dropout {tiny['dropout']}; params {n_params:,}")
    print(f"  AdamW lr {t4['lr']}, betas {t4['betas']}, wd {t4['weight_decay']}, warmup {t4['warmup']}, cosine → 0.1·lr, clip {t4['grad_clip']}; batch {B} × {L}; {steps} steps; seed {t4['seed']}")
    if pos == "rope":
        n_under = pairs_under_one_turn(tiny["d"] // tiny["n_head"], cfg["rope"]["base"], L)
        print(f"  RoPE: {tiny['d'] // tiny['n_head'] // 2} pairs; {n_under} of them never complete one turn within L_train = {L}")
    g = torch.Generator().manual_seed(t4["seed"])
    curve, evals = [], []
    model.train()
    t0 = time.time()
    for step in range(steps):
        ix = torch.randint(0, train_data.numel() - L - 1, (B,), generator=g)
        x = torch.stack([train_data[i:i + L] for i in ix]).to(device)
        y = torch.stack([train_data[i + 1:i + L + 1] for i in ix]).to(device)
        lr = d1.tr.lr_at(step, cfg4)
        for grp in opt.param_groups:
            grp["lr"] = lr
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, corpus["V"]), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t4["grad_clip"]).item()
        opt.step()
        if step % t4["log_every"] == 0 or step == steps - 1:
            el = time.time() - t0
            curve.append({"step": step, "loss": loss.item(), "grad_norm": gn, "lr": lr, "elapsed_s": el})
            print(f"  step {step:>5}  loss {loss.item():.4f}  ({loss.item() / math.log(2):.3f} bits/char)  |g| {gn:.2f}  lr {lr:.2e}  {el:6.1f} s")
        if (step % t4["eval_every"] == 0 and step > 0) or step == steps - 1:
            ev = d1.tr.evaluate(model, held_data, L, t4["eval_stride"], device)      # W4 的尺：視窗 L、stride 128
            evals.append({"step": step, **ev, "bpc": ev["nats_per_char"] / math.log(2)})
            print(f"  {BLUE}held-out @ {step}: {ev['nats_per_char']:.4f} nats/char = {evals[-1]['bpc']:.3f} bits/char (W4's windowed measure){RESET}")
    seconds = time.time() - t0
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "pos": pos, "base": cfg["rope"]["base"] if pos == "rope" else None,
           "torch": torch.__version__, "device": str(device), "python": platform.python_version(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
           "tiny": tiny, "train": {**t4, "steps": steps}, "params": n_params, "seconds": seconds,
           "corpus": {k: v for k, v in corpus.items() if k not in ("chars", "stoi", "train_text", "held_text")},
           "curve": curve, "evals": evals, "final": evals[-1] if evals else None}
    print(f"  {BOLD}{steps} steps in {seconds:.0f} s ({seconds / 60:.1f} min){RESET}; final held-out {rec['final']['bpc']:.3f} bits/char (W4's windowed measure)")
    if save_weights:
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        p = WEIGHTS / f"{pos}{'' if not steps or steps == t4['steps'] else f'-{steps}'}.pt"
        torch.save({"state_dict": model.state_dict(), "pos": pos, "base": rec["base"], "tiny": tiny, "V": corpus["V"], "steps": steps, "date": rec["date"]}, p)
        rec["weights"] = str(p.relative_to(HERE))
        print(f"{DIM}→ {rec['weights']}{RESET}")
    return rec


def load_variant(d1, cfg: dict, pos: str, steps: int | None):
    import torch
    t4 = d1.cfg["train"]
    p = WEIGHTS / f"{pos}{'' if not steps or steps == t4['steps'] else f'-{steps}'}.pt"
    if not p.exists():
        raise FileNotFoundError(f"沒有 {p.relative_to(HERE)}：先跑 ./present.sh train {pos}（或 rehearse）")
    ck = torch.load(p, map_location="cpu")
    device = d1.tr.pick_device(t4["device"])
    model = d1.build(ck["V"], ck["pos"], ck["base"] or cfg["rope"]["base"], seed=t4["seed"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, device, ck


# ── 量：逐位置的平均 loss ─────────────────────────────────────────
def measure(d1, cfg: dict, model, device, name: str, corpus: dict) -> dict:
    """對 held-out 以每個 L_eval 做不重疊視窗的前向；回每個位置的平均 nll（nats/char）與整段平均。"""
    import torch
    import rope_gpt
    ev = cfg["eval"]
    model.eval()
    model.L = max(ev["lengths"])                                                  # 上週的 GPT 會 assert L ≤ self.L；只是把上限放寬，沒有別的作用
    data = torch.tensor([corpus["stoi"][c] for c in corpus["held_text"]], dtype=torch.long)
    out = {"name": name, "label": LABEL[name], "rope": rope_gpt.rope_state(model), "by_length": {}}
    for L in ev["lengths"]:
        n_win = (data.numel() - 1) // L
        starts = [i * L for i in range(n_win)]
        acc = torch.zeros(L, dtype=torch.float64)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, n_win, ev["batch"]):
                chunk = starts[i:i + ev["batch"]]
                x = torch.stack([data[s:s + L] for s in chunk]).to(device)
                y = torch.stack([data[s + 1:s + L + 1] for s in chunk]).to(device)
                logits = model(x)
                nll = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="none").view(len(chunk), L)
                acc += nll.sum(0).cpu().double()                  # 先回 CPU 再升 float64：MPS 沒有 double（與 demo 1 釘 CPU 同一個理由）
        per_pos = (acc / n_win).tolist()
        mean = float(acc.sum() / (n_win * L))
        split = ev["split_at"]
        rec = {"L_eval": L, "windows": n_win, "chars": n_win * L, "mean_nats": mean, "mean_bits": mean / math.log(2),
               "seconds": time.time() - t0, "by_position": per_pos,
               "mean_below_split": sum(per_pos[:split]) / min(L, split),
               "mean_from_split": (sum(per_pos[split:]) / (L - split)) if L > split else None}
        out["by_length"][str(L)] = rec
        tail = f", {split}–{L}: {rec['mean_from_split']:.4f}" if rec["mean_from_split"] is not None else ""
        print(f"  {name:<9} L_eval {L:>5}: {n_win:>4} windows, mean {mean:.4f} nats/char ({mean / math.log(2):.3f} bits); positions < {split}: {rec['mean_below_split']:.4f}{tail}   {rec['seconds']:.1f} s")
    model.train()
    return out


def measure_all(d1, cfg: dict, steps: int | None, outd: Path, train_recs: dict | None = None) -> dict:
    import torch
    import rope_gpt
    sc = cfg["scale"]
    corpus = d1.corpus()
    results = {}
    print(f"\n{BOLD}measure{RESET}  held-out {corpus['held_chars']:,} chars; L_eval {cfg['eval']['lengths']}; non-overlapping windows; per-position mean loss")
    m_nope, dev, ck_n = load_variant(d1, cfg, "nope", steps)
    results["nope"] = measure(d1, cfg, m_nope, dev, "nope", corpus)
    del m_nope
    m_rope, dev, ck_r = load_variant(d1, cfg, "rope", steps)
    base, d_head = m_rope.blocks[0].attn.base, m_rope.blocks[0].attn.head_dim
    results["rope"] = measure(d1, cfg, m_rope, dev, "rope", corpus)
    rope_gpt.set_rope(m_rope, base=base, scale=sc["s"])
    print(f"  {DIM}PI: positions m → m/{sc['s']:g}  (θ_i → θ_i/{sc['s']:g}); weights unchanged{RESET}")
    results["rope_pi"] = measure(d1, cfg, m_rope, dev, "rope_pi", corpus)
    b_ntk = rope_gpt.ntk_base(base, sc["s"], d_head)
    rope_gpt.set_rope(m_rope, base=b_ntk, scale=1.0)
    print(f"  {DIM}NTK-aware: base {base:g} → {b_ntk:,.1f} = b·s^({d_head}/{d_head - 2}); weights unchanged{RESET}")
    results["rope_ntk"] = measure(d1, cfg, m_rope, dev, "rope_ntk", corpus)
    summary = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": platform.machine(), "macos": platform.mac_ver()[0],
               "torch": torch.__version__, "device": str(dev),
               "steps": ck_n["steps"], "weights": {"nope": ck_n["date"], "rope": ck_r["date"]},
               "rope_base": base, "d_head": d_head, "pairs_under_one_turn_at_L_train": pairs_under_one_turn(d_head, base, d1.cfg["tiny"]["L"]),
               "scale_s": sc["s"], "ntk_base": b_ntk, "lengths": cfg["eval"]["lengths"], "split_at": cfg["eval"]["split_at"], "bin": cfg["eval"]["bin"],
               "w4_reference": w4_reference(d1),
               "train": {k: {kk: v[kk] for kk in ("date", "seconds", "params", "final", "device", "torch") if kk in v} for k, v in (train_recs or {}).items()},
               "table": {n: {"below": r["by_length"][str(max(cfg["eval"]["lengths"]))]["mean_below_split"],
                             "from": r["by_length"][str(max(cfg["eval"]["lengths"]))]["mean_from_split"],
                             "means": {L: r["by_length"][L]["mean_nats"] for L in r["by_length"]}} for n, r in results.items()},
               "limit": LIMIT}
    outd.mkdir(parents=True, exist_ok=True)
    for n, r in results.items():
        (outd / f"{n}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (outd / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    svg_curves(results, summary, outd / "curves.svg")
    print(f"{DIM}→ {outd.relative_to(HERE)}/{{nope,rope,rope_pi,rope_ntk,summary}}.json, curves.svg{RESET}")
    print_tables(results, summary)
    return summary


def w4_reference(d1) -> dict | None:
    """上週 NoPE 版的 held-out bits/char（W4 demo 1 的 train.json），replay 時並排——重訓的 NoPE 應該落在附近。"""
    p = d1.dir / "runs" / "rehearsal" / "train.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    f = r.get("final") or {}
    return {"date": r.get("date"), "bpc": f.get("bpc"), "nats_per_char": f.get("nats_per_char"), "seconds": r.get("seconds"), "steps": r["train"]["steps"]}


# ── 表與圖 ────────────────────────────────────────────────────────
def bins_of(per_pos: list[float], size: int) -> list[float]:
    return [sum(per_pos[i:i + size]) / len(per_pos[i:i + size]) for i in range(0, len(per_pos), size)]


def spark(vals: list[float], lo: float, hi: float) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, max(0, int((v - lo) / max(hi - lo, 1e-9) * 7.999)))] for v in vals)


def print_tables(results: dict, summary: dict) -> None:
    Lmax = str(max(summary["lengths"]))
    split = summary["split_at"]
    print(f"\n  {BOLD}p24's table — mean loss (nats/char) from the L_eval = {Lmax} forward pass{RESET}")
    print(f"    {'':<32}" + "".join(f"{LABEL[n].split(',')[0].split(' (')[0]:>22}" for n in RUNS))
    print(f"    {f'positions < {split}':<32}" + "".join(f"{summary['table'][n]['below']:>22.4f}" for n in RUNS))
    print(f"    {f'positions {split}–{Lmax}':<32}" + "".join(f"{summary['table'][n]['from']:>22.4f}" for n in RUNS))
    print(f"\n  {BOLD}mean loss over the whole window, per L_eval{RESET}")
    for L in summary["lengths"]:
        print(f"    {f'L_eval = {L}':<32}" + "".join(f"{summary['table'][n]['means'][str(L)]:>22.4f}" for n in RUNS))
    ref = summary.get("w4_reference")
    if ref and ref.get("nats_per_char"):
        print(f"\n  {GREY}W4 demo 1's NoPE ({ref['date'][:10]}, windowed measure, stride 128): {ref['nats_per_char']:.4f} nats/char = {ref['bpc']:.3f} bits/char; "
              f"the retrained NoPE here: {summary['train'].get('nope', {}).get('final', {}).get('nats_per_char', float('nan')):.4f} (same measure){RESET}")
    print(f"  {GREY}因果模型：位置 p 的 loss 只看 0…p，與視窗長度無關——L_eval = {Lmax} 的前 {split} 個位置就是 L_eval = {split} 的曲線（視窗起點不同，小數第三位才會差）。{RESET}")
    print(f"  {GREY}{LIMIT}{RESET}")


def replay(cfg: dict, pause: bool) -> None:
    d = REHEARSAL
    if not (d / "summary.json").exists():
        print(f"{RED}沒有 runs/rehearsal/summary.json：先跑 ./present.sh rehearse{RESET}")
        return
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    results = {n: json.loads((d / f"{n}.json").read_text(encoding="utf-8")) for n in RUNS}
    Lmax = str(max(summary["lengths"]))
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {summary['date']}, torch {summary['torch']}, {summary['device']}, {summary['machine']}")
    tr = summary.get("train", {})
    print(f"  tiny GPT, char-level, L_train 256, {summary['steps']} steps each; "
          + "; ".join(f"{k}: {v['seconds'] / 60:.1f} min, held-out {v['final']['bpc']:.3f} bits/char" for k, v in tr.items() if v.get("final")))
    print(f"  RoPE base {summary['rope_base']:g}, d_head {summary['d_head']}: {summary['pairs_under_one_turn_at_L_train']} of {summary['d_head'] // 2} pairs never complete a turn within 256; "
          f"PI s = {summary['scale_s']:g}; NTK-aware base → {summary['ntk_base']:,.0f}")
    size = summary["bin"]
    allb = [v for n in RUNS for v in bins_of(results[n]["by_length"][Lmax]["by_position"], size)]
    lo, hi = min(allb), max(allb)
    print(f"\n  {BOLD}per-position loss, L_eval = {Lmax}, one block per {size} positions (▁ = {lo:.2f} … █ = {hi:.2f} nats/char); │ marks position 256{RESET}")
    nb = len(allb) // len(RUNS)
    mark = 256 // size
    for n in RUNS:
        if pause:
            input(f"  {DIM}Enter → {LABEL[n]}{RESET}")
        b = bins_of(results[n]["by_length"][Lmax]["by_position"], size)
        s = spark(b, lo, hi)
        t = summary["table"][n]
        print(f"  {LABEL[n]:<40} {s[:mark]}│{s[mark:]}   < 256: {t['below']:.3f}   256–{Lmax}: {t['from']:.3f}")
    if pause:
        input(f"  {DIM}Enter → 表{RESET}")
    print_tables(results, summary)
    print(f"\n  {DIM}curves.svg: open {d / 'curves.svg'}{RESET}")


def svg_curves(results: dict, summary: dict, path: Path) -> None:
    W, H, l, r, tp, bt = 900, 420, 64, 24, 36, 56
    Lmax = str(max(summary["lengths"]))
    size = summary["bin"]
    colors = {"nope": "#6b6b6b", "rope": "#3b6fb6", "rope_pi": "#d9534f", "rope_ntk": "#2e8b57"}
    series = {n: bins_of(results[n]["by_length"][Lmax]["by_position"], size) for n in RUNS}
    ys = [v for s in series.values() for v in s]
    y0, y1 = min(ys) * 0.95, max(ys) * 1.05
    x1 = int(Lmax)
    X = lambda x: l + x / x1 * (W - l - r)
    Y = lambda y: tp + (y1 - y) / (y1 - y0) * (H - tp - bt)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{l}" y="20" font-size="14" font-weight="600">tiny GPT (char-level, trained at L = 256): per-position loss on held-out text, L_eval = {Lmax}, mean per {size} positions</text>']
    for k in range(5):
        y = y0 + (y1 - y0) * k / 4
        out.append(f'<line x1="{l}" y1="{Y(y):.1f}" x2="{W - r}" y2="{Y(y):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{l - 6}" y="{Y(y) + 4:.1f}" text-anchor="end" fill="#555">{y:.2f}</text>')
    for x in range(0, x1 + 1, 256):
        out.append(f'<text x="{X(x):.1f}" y="{H - bt + 16}" text-anchor="middle" fill="#555">{x}</text>')
    out.append(f'<line x1="{X(256):.1f}" y1="{tp}" x2="{X(256):.1f}" y2="{H - bt}" stroke="#999" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{X(256) + 4:.1f}" y="{tp + 12}" fill="#666">L_train = 256</text>')
    out.append(f'<text x="{(l + W - r) / 2:.1f}" y="{H - 10}" text-anchor="middle" fill="#555">position in the window</text>')
    out.append(f'<text x="14" y="{(tp + H - bt) / 2:.1f}" text-anchor="middle" fill="#555" transform="rotate(-90 14 {(tp + H - bt) / 2:.1f})">loss (nats/char)</text>')
    for n in RUNS:
        pts = " ".join(f"{X((i + 0.5) * size):.1f},{Y(v):.1f}" for i, v in enumerate(series[n]))
        out.append(f'<polyline fill="none" stroke="{colors[n]}" stroke-width="2" points="{pts}"/>')
    for i, n in enumerate(RUNS):
        yy = tp + 14 + 16 * i
        out.append(f'<line x1="{W - r - 300}" y1="{yy}" x2="{W - r - 276}" y2="{yy}" stroke="{colors[n]}" stroke-width="2"/>')
        out.append(f'<text x="{W - r - 270}" y="{yy + 4}" fill="#333">{LABEL[n]}</text>')
    out.append(f'<text x="{W - r}" y="{H - bt + 34}" text-anchor="end" fill="#888" font-size="11">rehearsal {summary["date"][:10]} · {summary["machine"]} · torch {summary["torch"]} · {summary["steps"]} steps each · not a measurement of any large model</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def estimate(d1, cfg: dict) -> None:
    p = d1.dir / "runs" / "rehearsal" / "probe.json"
    per = json.loads(p.read_text(encoding="utf-8")).get("step_time_s") if p.exists() else None
    steps = d1.cfg["train"]["steps"]
    if per is None:
        print(f"{AMBER}W4 demo 1 沒有 probe.json（./present.sh probe 在那邊跑），無法估{RESET}")
        return
    tr = json.loads((d1.dir / "runs" / "rehearsal" / "train.json").read_text(encoding="utf-8")) if (d1.dir / "runs" / "rehearsal" / "train.json").exists() else None
    print(f"{BOLD}estimate{RESET}  W4 demo 1 probe.json: {per:.3f} s/step (NoPE) → {steps} steps ≈ {per * steps / 60:.1f} min per model, two models ≈ {2 * per * steps / 60:.1f} min")
    if tr:
        print(f"  W4 demo 1's full run took {tr['seconds'] / 60:.1f} min for {tr['train']['steps']} steps (includes {len(tr['evals'])} held-out evaluations)")
    print(f"  RoPE adds two elementwise multiplies per layer; expect the same order. Measuring: four variants × three lengths, well under a minute each.")
    print(f"  {DIM}The estimate assumes the step time scales with nothing here (same model size); the real number is only known after rehearse.{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rehearse", "train", "measure", "replay", "show", "estimate"])
    ap.add_argument("variant", nargs="?", choices=["nope", "rope"], help="train 時：訓哪一個")
    ap.add_argument("--steps", type=int, default=None, help="覆蓋步數（煙霧測試）；有這個就寫到 runs/smoke/，權重檔名帶步數")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "replay":
        replay(cfg, pause=True); return 0
    if a.cmd == "show":
        replay(cfg, pause=False); return 0
    d1 = demo1(cfg)
    if a.cmd == "estimate":
        estimate(d1, cfg); return 0
    if a.cmd == "train":
        if not a.variant:
            ap.error("train 要指定 nope 或 rope")
        train_variant(d1, cfg, a.variant, a.steps, cfg["train"]["save_weights"], label=f"train — {a.variant}")
        return 0
    outd = out_dir(a)
    if a.steps:
        print(f"{AMBER}--steps {a.steps}: 煙霧測試，輸出到 runs/smoke/，不碰 runs/rehearsal/{RESET}")
    recs = {}
    if a.cmd == "rehearse":
        for v in cfg["train"]["variants"]:
            recs[v] = train_variant(d1, cfg, v, a.steps, True, label=f"rehearse — train {v}")
        outd.mkdir(parents=True, exist_ok=True)
        (outd / "train.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        p = outd / "train.json"
        if p.exists():
            recs = json.loads(p.read_text(encoding="utf-8"))
    measure_all(d1, cfg, a.steps, outd, recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
