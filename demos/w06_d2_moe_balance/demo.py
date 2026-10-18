#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 2：MoE 的三種平衡法——expert 負載與 held-out loss（彩排預錄、課堂 replay）。投影片 p17（p16 前導頁）。

W4 demo 1 的 tiny GPT，每一層的 FFN 換成 MoE 層（n_E = 8、k = 2、d_E = d_ff/2；moe_gpt.py），同一份語料、同一個 seed，三組設定各訓一次：
  (i)   none     沒有平衡項
  (ii)  switch   Switch 的 L_aux = α · n_E · Σ_i f_i P_i，α = 10^-2
  (iii) bias     aux-loss-free 的偏置：選集用 TopK(r + u)，u_i ← u_i + γ sign(c̄ − c_i)，不進 loss
每 25 步對同一批 held-out token（開頭 32 個視窗）做一次前向，記 loss 與每一層逐 expert 的份額 f_i；兩個指標：
最大的 f_i 對 1/n_E、份額 < 1% 的 expert 數（「餓死」）。另每 500 步與最後跑 W4 的完整 held-out 尺（好跟 W4 的 2.184 bits/char 對）。

  ./present.sh rehearse            全套：三組各 3000 步 → runs/rehearsal/{none,switch,bias}.json + summary.json + curves.svg
  ./present.sh train switch        只訓一組（none|switch|bias）→ runs/rehearsal/<setting>.json（不動其他兩組；之後 ./present.sh summarize）
  ./present.sh summarize           只從三個 json 重寫 summary.json 與 curves.svg
  ./present.sh replay              課堂（p17）：Enter 一組——(i) → (ii) → (iii)：最後的指標、每層 8 格負載長條、兩條曲線；最後印 p17 的表
  ./present.sh show                只印表（投影機上字太小時用；也是 replay 的退路）
  ./present.sh estimate            不訓練，用 W4 demo 1 probe.json 的每步秒數估 rehearse 要跑多久（MoE 一層跑 8 個 expert，實際會比它慢）
  ./present.sh rehearse --steps 30 煙霧測試：30 步 × 三組 + 全套紀錄，寫到 runs/smoke/（不碰 runs/rehearsal/）

誠實的限定（每次 replay 都印）：字元級 tiny GPT、八個 expert、一個 seed、α 與 γ 照原文沒調；曲線的形狀是機制的形狀，
不是任何 256 個 expert 的旗艦的數字。三組在這個尺度分不開，正確的結論是「這個尺度看不到差別」，不是「三種方法一樣」。
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
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"
LABEL = {"none": "(i) no balancing term", "switch": "(ii) Switch auxiliary loss, α = 10⁻²", "bias": "(iii) auxiliary-loss-free bias"}
SHORT = {"none": "(i) none", "switch": "(ii) Switch L_aux", "bias": "(iii) loss-free bias"}
LIMIT = ("這是字元級的 tiny GPT 換上 8 個 expert（每 token 選 2 個），一個 seed、α 與 γ 照原文沒調、3000 步。"
         "曲線的形狀是機制的形狀，不是任何 256 個 expert 的旗艦的數字；三組分不開時正確的結論是「這個尺度看不到差別」，不是「三種方法一樣」。")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def demo1(cfg: dict):
    import moe_gpt
    return moe_gpt.Demo1(cfg["shared"]["demo1"])


def out_dir(args) -> Path:
    return (HERE / "runs" / "smoke") if args.steps else REHEARSAL


def settings(cfg: dict) -> list[str]:
    return list(cfg["balance"]["settings"])


# ── 訓練：沿 W4 demo 1 的迴圈（同超參數、同 seed、同評估），多一個 held-out 子集的負載紀錄 ──
def train_setting(d1, cfg: dict, balance: str, steps: int | None, label: str) -> dict:
    import torch
    import moe_gpt
    t4, tiny, moe, bal, rc = d1.cfg["train"], d1.cfg["tiny"], cfg["moe"], cfg["balance"], cfg["record"]
    steps = steps or t4["steps"]
    corpus = d1.corpus()
    device = d1.tr.pick_device(t4["device"])
    enc = lambda s: torch.tensor([corpus["stoi"][c] for c in s], dtype=torch.long)
    train_data, held_data = enc(corpus["train_text"]), enc(corpus["held_text"])
    L, B = tiny["L"], t4["batch"]
    model = d1.build(corpus["V"], balance, moe, seed=t4["seed"]).to(device)
    split = moe_gpt.count_params_split(model)
    opt = torch.optim.AdamW(d1.tr.param_groups(model, t4["weight_decay"]), lr=t4["lr"], betas=tuple(t4["betas"]), eps=1e-8)
    cfg4 = {"train": {**t4, "steps": steps}}                                   # lr_at 讀 cfg["train"]
    # held-out 子集：開頭 held_windows 個不重疊視窗，每次同一批 token
    nw = rc["held_windows"]
    hx = torch.stack([held_data[i * L:(i + 1) * L] for i in range(nw)]).to(device)
    hy = torch.stack([held_data[i * L + 1:(i + 1) * L + 1] for i in range(nw)]).to(device)
    print(f"\n{BOLD}{label}{RESET}  balance = {balance}  torch {torch.__version__}, device {device}, {platform.machine()}")
    print(f"  corpus {Path(corpus['path']).name}: V = {corpus['V']}; train {corpus['train_chars']:,} chars, held-out {corpus['held_chars']:,} chars (W4's cut); load measured on the first {nw} held-out windows ({nw * L:,} tokens, {nw * L * moe['k']:,} expert slots; 1% = {nw * L * moe['k'] // 100} slots)")
    print(f"  tiny GPT: d {tiny['d']}, {tiny['n_layer']} layers, {tiny['n_head']} heads, L {L}, dropout {tiny['dropout']}; FFN → MoE: n_E {moe['n_E']}, k {moe['k']}, d_E {moe['d_E']} (W4's d_ff {tiny['d_ff']})")
    print(f"  params: total {split['total']:,} (experts {split['experts']:,}, routers {split['router']:,}, rest {split['rest']:,}); active per token {split['active']:,} (ρ = {split['rho']:.3f}); expert params per token per layer {split['per_layer_expert_active']:,} = W4's FFN 3·d·d_ff = {3 * tiny['d'] * tiny['d_ff']:,}")
    print(f"  AdamW lr {t4['lr']}, betas {t4['betas']}, wd {t4['weight_decay']}, warmup {t4['warmup']}, cosine → 0.1·lr, clip {t4['grad_clip']}; batch {B} × {L}; {steps} steps; seed {t4['seed']}")
    if balance == "switch":
        print(f"  L_aux = α · n_E · Σ f_i P_i per layer, summed over {tiny['n_layer']} layers, α = {bal['alpha']} (uniform → {bal['alpha']} per layer)")
    if balance == "bias":
        print(f"  u_i ← u_i + γ · sign(c̄ − c_i) after every step, γ = {moe['bias_step']}; selection uses r + u, gates use r")
    g = torch.Generator().manual_seed(t4["seed"])
    curve, evals = [], []
    t0 = time.time()

    def record(step: int, train_loss: float, aux_train: float, gn: float, lr: float):
        model.eval()
        with torch.no_grad():
            logits = model(hx)
            hl = torch.nn.functional.cross_entropy(logits.float().view(-1, corpus["V"]), hy.view(-1)).item()
            st = moe_gpt.load_stats(model)
            aux_h = float(moe_gpt.aux_sum(model))
        model.train()
        rec = {"step": step, "train_loss": train_loss, "aux_train": aux_train, "grad_norm": gn, "lr": lr, "elapsed_s": time.time() - t0,
               "held_loss": hl, "aux_held": aux_h, "max_f": st["max_f"], "starved": st["starved"], "f": st["f"], "P": st["P"],
               "u": st["u"] if balance == "bias" else None}
        curve.append(rec)
        mf, sv = sum(st["max_f"]) / len(st["max_f"]), sum(st["starved"]) / len(st["starved"])
        print(f"  step {step:>5}  loss {train_loss:.4f}  held {hl:.4f} ({hl / math.log(2):.3f} bits/char)  "
              f"max f {mf:.3f} (worst layer {max(st['max_f']):.3f})  starved {sv:.1f}/{moe['n_E']} (worst {max(st['starved'])})  "
              f"n_E·Σf·P {aux_h / tiny['n_layer']:.3f}  |g| {gn:.2f}  lr {lr:.2e}  {rec['elapsed_s']:6.1f} s")

    model.train()
    for step in range(steps):
        ix = torch.randint(0, train_data.numel() - L - 1, (B,), generator=g)
        x = torch.stack([train_data[i:i + L] for i in ix]).to(device)
        y = torch.stack([train_data[i + 1:i + L + 1] for i in ix]).to(device)
        lr = d1.tr.lr_at(step, cfg4)
        for grp in opt.param_groups:
            grp["lr"] = lr
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, corpus["V"]), y.view(-1))
        aux = moe_gpt.aux_sum(model)                                             # n_E · Σ f_i P_i 加總（三組都算）
        total = loss + bal["alpha"] * aux if balance == "switch" else loss       # 只有 (ii) 進 loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t4["grad_clip"]).item()
        opt.step()
        for layer in moe_gpt.moe_layers(model):                                  # (iii)：每步之後動偏置（其他設定是 no-op）
            layer.update_bias()
        if step % rc["every"] == 0 or step == steps - 1:
            record(step, loss.item(), float(aux.detach()), gn, lr)
        if (step % rc["full_eval_every"] == 0 and step > 0) or step == steps - 1:
            ev = d1.tr.evaluate(model, held_data, L, t4["eval_stride"], device)   # W4 的尺：視窗 L、stride 128
            evals.append({"step": step, **ev, "bpc": ev["nats_per_char"] / math.log(2)})
            print(f"  {BLUE}held-out @ {step} (W4's windowed measure): {ev['nats_per_char']:.4f} nats/char = {evals[-1]['bpc']:.3f} bits/char{RESET}")
    seconds = time.time() - t0
    last = curve[-1]
    tail = curve[-4:]                                                             # 最後 100 步（4 筆）的平均，看終值不被單一筆雜訊帶走
    final = {"step": last["step"], "held_loss": last["held_loss"], "held_bits": last["held_loss"] / math.log(2),
             "max_f_mean_layers": sum(last["max_f"]) / len(last["max_f"]), "max_f_worst_layer": max(last["max_f"]),
             "starved_mean_layers": sum(last["starved"]) / len(last["starved"]), "starved_worst_layer": max(last["starved"]),
             "starved_total": sum(last["starved"]), "f_final": last["f"],
             "tail100": {"held_loss": sum(r["held_loss"] for r in tail) / len(tail),
                         "max_f_mean_layers": sum(sum(r["max_f"]) / len(r["max_f"]) for r in tail) / len(tail),
                         "starved_mean_layers": sum(sum(r["starved"]) / len(r["starved"]) for r in tail) / len(tail)},
             "aux_held_per_layer": last["aux_held"] / tiny["n_layer"], "u_final": last["u"], "w4_measure": evals[-1] if evals else None}
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "balance": balance, "label": LABEL[balance],
           "torch": torch.__version__, "device": str(device), "python": platform.python_version(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
           "tiny": tiny, "moe": moe, "balance_cfg": bal, "record": rc, "train": {**t4, "steps": steps}, "params": split, "seconds": seconds,
           "held_subset": {"windows": nw, "tokens": nw * L, "slots": nw * L * moe["k"]},
           "corpus": {k: v for k, v in corpus.items() if k not in ("chars", "stoi", "train_text", "held_text")},
           "curve": curve, "evals": evals, "final": final}
    print(f"  {BOLD}{steps} steps in {seconds:.0f} s ({seconds / 60:.1f} min){RESET}; final held-out (subset) {final['held_loss']:.4f} nats/char; "
          f"max f {final['max_f_mean_layers']:.3f} mean over layers / {final['max_f_worst_layer']:.3f} worst; starved {final['starved_mean_layers']:.1f} mean / {final['starved_worst_layer']} worst"
          + (f"; W4's measure {final['w4_measure']['bpc']:.3f} bits/char" if final["w4_measure"] else ""))
    return rec


# ── 總表 ──────────────────────────────────────────────────────────
def summarize(cfg: dict, outd: Path, recs: dict | None = None) -> dict:
    recs = recs or {s: json.loads((outd / f"{s}.json").read_text(encoding="utf-8")) for s in settings(cfg) if (outd / f"{s}.json").exists()}
    if not recs:
        raise FileNotFoundError(f"{outd.relative_to(HERE)} 裡沒有任何一組的 json：先跑 ./present.sh rehearse")
    any_rec = next(iter(recs.values()))
    w4 = w4_reference(cfg)
    summary = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": any_rec["machine"], "macos": any_rec["macos"], "torch": any_rec["torch"], "device": any_rec["device"],
               "steps": any_rec["train"]["steps"], "moe": any_rec["moe"], "balance_cfg": any_rec["balance_cfg"], "record": any_rec["record"],
               "params": any_rec["params"], "n_layer": any_rec["tiny"]["n_layer"], "held_subset": any_rec["held_subset"],
               "runs": {s: {"date": r["date"], "seconds": r["seconds"], "final": r["final"]} for s, r in recs.items()},
               "table": {s: {"max_f": r["final"]["max_f_mean_layers"], "max_f_worst": r["final"]["max_f_worst_layer"],
                             "starved": r["final"]["starved_mean_layers"], "starved_worst": r["final"]["starved_worst_layer"], "starved_total": r["final"]["starved_total"],
                             "held_loss": r["final"]["held_loss"], "w4_bpc": (r["final"]["w4_measure"] or {}).get("bpc")} for s, r in recs.items()},
               "w4_reference": w4, "limit": LIMIT}
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    svg_curves(recs, summary, outd / "curves.svg")
    print(f"{DIM}→ {outd.relative_to(HERE)}/{{{','.join(recs)}}}.json, summary.json, curves.svg{RESET}")
    return summary


def w4_reference(cfg: dict) -> dict | None:
    """上週 dense FFN 版的 held-out bits/char（W4 demo 1 的 train.json），表裡並排。"""
    p = (HERE / cfg["shared"]["demo1"]).resolve() / "runs" / "rehearsal" / "train.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    f = r.get("final") or {}
    return {"date": r.get("date"), "bpc": f.get("bpc"), "nats_per_char": f.get("nats_per_char"), "seconds": r.get("seconds"), "steps": r["train"]["steps"], "params": (r.get("params") or {}).get("count")}


# ── 表、長條、曲線 ────────────────────────────────────────────────
def spark(vals: list[float], lo: float, hi: float) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, max(0, int((v - lo) / max(hi - lo, 1e-9) * 7.999)))] for v in vals)


def bar8(f: list[float], width: int = 48) -> str:
    """一層的 n_E 個份額畫成一行：每格的寬度 ∝ f_i（全部加起來 width 格），份額 < 1% 的格子標 ·。"""
    return " ".join("·" if v < 0.01 else "█" * max(1, int(round(v * width))) for v in f)


def print_table(summary: dict) -> None:
    S = [s for s in ("none", "switch", "bias") if s in summary["table"]]
    n_E = summary["moe"]["n_E"]
    print(f"\n  {BOLD}p17's table — final values (step {summary['steps'] - 1}; load on the fixed held-out subset; mean over {summary['n_layer']} layers, worst layer in parentheses){RESET}")
    print(f"    {'':<46}" + "".join(f"{SHORT[s]:>26}" for s in S))
    print(f"    {f'largest expert load, final (ideal 1/{n_E} = {1 / n_E:.3f}; max 1/{summary['moe']['k']})':<46}" + "".join(f"{summary['table'][s]['max_f']:>18.3f} ({summary['table'][s]['max_f_worst']:.3f})" for s in S))
    print(f"    {'experts below 1% of slots, final (per layer)':<46}" + "".join(f"{summary['table'][s]['starved']:>18.1f} ({summary['table'][s]['starved_worst']})" for s in S))
    print(f"    {'held-out loss, subset (nats/char)':<46}" + "".join(f"{summary['table'][s]['held_loss']:>26.4f}" for s in S))
    if all(summary["table"][s]["w4_bpc"] for s in S):
        print(f"    {'held-out, W4 windowed measure (bits/char)':<46}" + "".join(f"{summary['table'][s]['w4_bpc']:>26.3f}" for s in S))
    ref = summary.get("w4_reference")
    if ref and ref.get("bpc"):
        times = f" and {summary['params']['total'] / ref['params']:.1f}× the parameters in total" if ref.get("params") else ""
        print(f"  {GREY}W4 demo 1's dense-FFN tiny GPT ({ref['date'][:10]}, same measure): {ref['bpc']:.3f} bits/char — the MoE versions have the same expert parameters per token{times}{RESET}")
    print(f"  {GREY}{LIMIT}{RESET}")


def replay(cfg: dict, pause: bool) -> None:
    d = REHEARSAL
    if not (d / "summary.json").exists():
        print(f"{RED}沒有 runs/rehearsal/summary.json：先跑 ./present.sh rehearse{RESET}")
        return
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    S = [s for s in ("none", "switch", "bias") if (d / f"{s}.json").exists()]
    recs = {s: json.loads((d / f"{s}.json").read_text(encoding="utf-8")) for s in S}
    n_E, n_layer = summary["moe"]["n_E"], summary["n_layer"]
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {summary['date']}, torch {summary['torch']}, {summary['device']}, {summary['machine']}")
    print(f"  W4's tiny GPT, FFN → MoE (n_E = {n_E}, k = {summary['moe']['k']}, d_E = {summary['moe']['d_E']}); {summary['params']['total']:,} params, {summary['params']['active']:,} active per token; "
          f"{summary['steps']} steps each; " + "; ".join(f"{s}: {summary['runs'][s]['seconds'] / 60:.1f} min" for s in S))
    print(f"  load = share of the {summary['held_subset']['slots']:,} expert slots on a fixed held-out subset ({summary['held_subset']['tokens']:,} tokens), recorded every {summary['record']['every']} steps; 1/n_E = {1 / n_E:.3f}; 'starved' = below 1%")
    allf = [sum(r["max_f"]) / n_layer for s in S for r in recs[s]["curve"]]
    alll = [r["held_loss"] for s in S for r in recs[s]["curve"]]
    k = summary["moe"]["k"]
    flo, fhi, llo, lhi = 1 / n_E, 1 / k, min(alll), max(alll)          # 一個 expert 最多拿 1/k 的名額（k 個名額一定給 k 個不同的 expert）
    print(f"\n  {BOLD}sparklines over training: max_i f_i (mean over layers; ▁ = 1/n_E = {flo:.3f} … █ = 1/k = {fhi:.2f}, every token picking the same {k} experts) and held-out loss (▁ = {llo:.2f} … █ = {lhi:.2f} nats/char){RESET}")
    for s in S:
        if pause:
            input(f"  {DIM}Enter → {LABEL[s]}{RESET}")
        r = recs[s]
        fin = r["final"]
        mf = [sum(c["max_f"]) / n_layer for c in r["curve"]]
        hl = [c["held_loss"] for c in r["curve"]]
        print(f"\n  {BOLD}{LABEL[s]}{RESET}")
        print(f"    max f   {spark(mf, flo, fhi)}")
        print(f"    loss    {spark(hl, llo, lhi)}")
        print(f"    final: largest load {fin['max_f_mean_layers']:.3f} mean over layers ({fin['max_f_worst_layer']:.3f} worst; ideal {1 / n_E:.3f}); "
              f"starved {fin['starved_mean_layers']:.1f} of {n_E} per layer ({fin['starved_worst_layer']} worst; {fin['starved_total']} over all layers); held-out {fin['held_loss']:.4f} nats/char"
              + (f"; W4's measure {fin['w4_measure']['bpc']:.3f} bits/char" if fin.get("w4_measure") else ""))
        print(f"    final load per layer (one cell per expert, width ∝ share; · = below 1%):")
        for li, f in enumerate(fin["f_final"]):
            print(f"      layer {li}  {bar8(f)}   max {max(f):.3f}")
        if s == "bias" and fin.get("u_final"):
            print(f"    final bias u, layer 0: " + " ".join(f"{v:+.3f}" for v in fin["u_final"][0]) + "   (selection logit offsets; gates never see them)")
    if pause:
        input(f"\n  {DIM}Enter → 表{RESET}")
    print_table(summary)
    print(f"\n  {DIM}curves.svg: open {d / 'curves.svg'}{RESET}")


def svg_curves(recs: dict, summary: dict, path: Path) -> None:
    """兩個 panel：左 max_i f_i（mean over layers）對步數，右 held-out loss 對步數；三條線。給 README 與課前看，投影片的圖另由 make_figs_w06.py 產。"""
    W, H, l, r, tp, bt, gap = 1000, 400, 60, 20, 40, 56, 60
    S = [s for s in ("none", "switch", "bias") if s in recs]
    colors = {"none": "#dc2626", "switch": "#0b5cad", "bias": "#15803d"}
    n_E, n_layer = summary["moe"]["n_E"], summary["n_layer"]
    pw = (W - l - r - gap) / 2
    steps = summary["steps"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{l}" y="22" font-size="14" font-weight="600">W4\'s tiny GPT with an MoE layer (n_E = {n_E}, k = {summary["moe"]["k"]}): expert load and held-out loss under three balancing rules</text>']
    k = summary["moe"]["k"]
    panels = [("max_i f_i (mean over layers)", lambda c: sum(c["max_f"]) / n_layer, 0.0, min(1.0, 1.2 / k)),
              ("held-out loss (nats/char, fixed subset)", lambda c: c["held_loss"], None, None)]
    for pi, (title, fn, y0, y1) in enumerate(panels):
        px = l + pi * (pw + gap)
        vals = [fn(c) for s in S for c in recs[s]["curve"]]
        if y0 is None:
            y0, y1 = min(vals) * 0.95, max(vals) * 1.05
        X = lambda x: px + x / steps * pw
        Y = lambda y: tp + (y1 - y) / (y1 - y0) * (H - tp - bt)
        for k in range(5):
            y = y0 + (y1 - y0) * k / 4
            out.append(f'<line x1="{px}" y1="{Y(y):.1f}" x2="{px + pw:.1f}" y2="{Y(y):.1f}" stroke="#eee"/>')
            out.append(f'<text x="{px - 6}" y="{Y(y) + 4:.1f}" text-anchor="end" fill="#555">{y:.2f}</text>')
        for x in range(0, steps + 1, max(1, steps // 4)):
            out.append(f'<text x="{X(x):.1f}" y="{H - bt + 16}" text-anchor="middle" fill="#555">{x}</text>')
        out.append(f'<text x="{px + pw / 2:.1f}" y="{H - 10}" text-anchor="middle" fill="#555">training step</text>')
        out.append(f'<text x="{px + 4}" y="{tp + 14}" fill="#333" font-weight="600">{title}</text>')
        if pi == 0:
            out.append(f'<line x1="{px}" y1="{Y(1 / n_E):.1f}" x2="{px + pw:.1f}" y2="{Y(1 / n_E):.1f}" stroke="#999" stroke-dasharray="4 3"/>')
            out.append(f'<text x="{px + pw - 4:.1f}" y="{Y(1 / n_E) - 4:.1f}" text-anchor="end" fill="#666">1/n_E = {1 / n_E:.3f}</text>')
            out.append(f'<line x1="{px}" y1="{Y(1 / k):.1f}" x2="{px + pw:.1f}" y2="{Y(1 / k):.1f}" stroke="#dc2626" stroke-dasharray="3 4"/>')
            out.append(f'<text x="{px + pw - 4:.1f}" y="{Y(1 / k) - 4:.1f}" text-anchor="end" fill="#dc2626">1/k = {1 / k:.2f} (every token picks the same {k} experts)</text>')
        for s in S:
            pts = " ".join(f"{X(c['step']):.1f},{Y(fn(c)):.1f}" for c in recs[s]["curve"])
            out.append(f'<polyline fill="none" stroke="{colors[s]}" stroke-width="2" points="{pts}"/>')
    lx = l + pw + gap + pw - 250                                         # 圖例放右 panel 右上（loss 從左上往右下走，那裡是空的）
    for i, s in enumerate(S):
        yy = tp + 34 + 16 * i
        out.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 24}" y2="{yy}" stroke="{colors[s]}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 30}" y="{yy + 4}" fill="#333">{LABEL[s]}</text>')
    out.append(f'<text x="{W - r}" y="{H - bt + 34}" text-anchor="end" fill="#888" font-size="11">rehearsal {summary["date"][:10]} · {summary["machine"]} · torch {summary["torch"]} · {steps} steps each · one seed · not a measurement of any large model</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def estimate(d1, cfg: dict) -> None:
    p = d1.dir / "runs" / "rehearsal" / "probe.json"
    per = json.loads(p.read_text(encoding="utf-8")).get("step_time_s") if p.exists() else None
    steps = d1.cfg["train"]["steps"]
    n = len(settings(cfg))
    if per is None:
        print(f"{AMBER}W4 demo 1 沒有 probe.json（./present.sh probe 在那邊跑），無法估{RESET}")
        return
    tr = json.loads((d1.dir / "runs" / "rehearsal" / "train.json").read_text(encoding="utf-8")) if (d1.dir / "runs" / "rehearsal" / "train.json").exists() else None
    print(f"{BOLD}estimate{RESET}  W4 demo 1 probe.json: {per:.3f} s/step (dense FFN) → {steps} steps ≈ {per * steps / 60:.1f} min per run, {n} runs ≈ {n * per * steps / 60:.1f} min **if a step cost the same**")
    if tr:
        print(f"  W4 demo 1's full run took {tr['seconds'] / 60:.1f} min for {tr['train']['steps']} steps (includes {len(tr['evals'])} held-out evaluations)")
    print(f"  The MoE layer runs {cfg['moe']['n_E']} experts in a Python loop (each on ~{cfg['moe']['k']}/{cfg['moe']['n_E']} of the tokens) and records the held-out subset every {cfg['record']['every']} steps: expect a step to cost more than W4's, by a factor only rehearse can tell.")
    print(f"  {DIM}Run ./present.sh rehearse --steps 30 first: it prints the seconds for 30 steps of each setting, which is the real estimate.{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rehearse", "train", "summarize", "replay", "show", "estimate"])
    ap.add_argument("setting", nargs="?", choices=["none", "switch", "bias"], help="train 時：訓哪一組")
    ap.add_argument("--steps", type=int, default=None, help="覆蓋步數（煙霧測試）；有這個就寫到 runs/smoke/")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "replay":
        replay(cfg, pause=True); return 0
    if a.cmd == "show":
        replay(cfg, pause=False); return 0
    outd = out_dir(a)
    if a.cmd == "summarize":
        summary = summarize(cfg, outd); print_table(summary); return 0
    d1 = demo1(cfg)
    if a.cmd == "estimate":
        estimate(d1, cfg); return 0
    if a.steps:
        print(f"{AMBER}--steps {a.steps}: 煙霧測試，輸出到 runs/smoke/，不碰 runs/rehearsal/{RESET}")
    outd.mkdir(parents=True, exist_ok=True)
    if a.cmd == "train":
        if not a.setting:
            ap.error("train 要指定 none、switch 或 bias")
        rec = train_setting(d1, cfg, a.setting, a.steps, label=f"train — {a.setting}")
        (outd / f"{a.setting}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{DIM}→ {outd.relative_to(HERE)}/{a.setting}.json（之後 ./present.sh summarize 重寫 summary.json 與 curves.svg）{RESET}")
        return 0
    recs = {}
    for s in settings(cfg):
        recs[s] = train_setting(d1, cfg, s, a.steps, label=f"rehearse — {s}")
        (outd / f"{s}.json").write_text(json.dumps(recs[s], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = summarize(cfg, outd, recs)
    print_table(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
