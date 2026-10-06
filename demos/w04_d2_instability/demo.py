#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 2：訓練不穩定的復現（彩排預錄、課堂 replay）。投影片 p29（設定 (i)）與 p44（設定 (ii)(iii)）。

同一個 tiny GPT（demo 1 的 block_final.GPT）、同一份語料，每 log_every 步記四個統計量：
  loss、max|s_ij|（softmax 前的 attention logit，所有層所有 head 的最大絕對值）、輸出層的 log Z（logsumexp 的平均）、全模型梯度範數，
另記每一層的梯度範數（p29 的「哪一層的梯度先不對」）。

  ./present.sh rehearse            三組全部跑（幾十分鐘，看開頭印的估計）→ runs/rehearsal/{i,ii,iii}.json + *.svg
  ./present.sh rehearse i          只跑一組；--depth 24、--lr 3e-3、--steps、--mult 1,3,10,30、--z-loss 1e-4、--clip 1.0 覆蓋 demo_config.toml
  ./present.sh replay i            課堂（p29）：pre vs. post 的表、兩條 loss 與梯度範數曲線、第 0 步逐層梯度範數
  ./present.sh replay ii           課堂（p44）：三個 LR 的表（spike 步數、spike 時與 500 步前的 max|s_ij|、log Z）與四條曲線
  ./present.sh replay iii          課堂（p44 第四列）：10× 開 QK-norm，對 (ii) 的 10× 並排
  ./present.sh show i|ii|iii       只印表格（投影機上字太小時用；也是 replay 的退路）
  ./present.sh estimate            不訓練，只用 demo 1 probe.json 的每步秒數估三組要跑多久
  ./present.sh baselines           字元級 unigram／bigram 對同一份 held-out 的 cross-entropy（nats/char）→ runs/rehearsal/baselines.json；
                                   replay 的表用它標「loss 卡在哪個水準」（彩排實測：post-LN 卡在 unigram、10× 卡在 bigram）

誠實的限定（每次 replay 都印）：這是 proxy 的 proxy——Wortsman et al. 的 proxy 是數千萬到數億參數，這裡是幾百萬；
復現的是機制的形狀，沒有任何一個數字是真實模型的門檻值。spike 的判定（loss 比目前最小值高 spike_jump nats）是這個 demo 自己的操作型定義。
"""
from __future__ import annotations

import argparse
import importlib.util
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
SETTINGS = ("i", "ii", "iii")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


# ── demo 1 的東西：從檔案路徑載，不動 sys.path（demo 1 彩排踩過：各資料夾的 tests.py 會撞名）──
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Demo1:
    def __init__(self, cfg: dict):
        self.dir = (HERE / cfg["shared"]["demo1"]).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(f"找不到 demo 1 的資料夾 {self.dir}")
        with open(self.dir / "demo_config.toml", "rb") as f:
            self.cfg = tomllib.load(f)                       # [tiny] 與 [corpus] 從這裡來
        self.bf = _load(self.dir / "block_final.py", "w04_block_final")      # GPT；它自己會從檔案載 W3 的 attention
        self.tr = _load(self.dir / "train.py", "w04_d1_train")               # load_corpus / pick_device / param_groups（top-level 不 import torch）

    def corpus(self) -> dict:
        return self.tr.load_corpus(self.cfg)                 # 路徑相對 demo 1 的資料夾解析

    def probe_step_time(self) -> float | None:
        p = self.dir / "runs" / "rehearsal" / "probe.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("step_time_s")
        return None


# ── 一條訓練：回傳整條紀錄 ───────────────────────────────────────
def run_one(d1: Demo1, cfg: dict, *, label: str, setting: str, norm: str, n_layer: int, lr: float, warmup: int,
            steps: int, qk_norm: bool = False, z_loss: float = 0.0, grad_clip: float | None = None, d_ff: int | None = None) -> dict:
    import torch
    c, tiny = cfg["common"], d1.cfg["tiny"]
    clip = c["grad_clip"] if grad_clip is None else grad_clip
    torch.manual_seed(c["seed"])
    corpus = d1.corpus()
    device = d1.tr.pick_device(c["device"])
    enc = lambda s: torch.tensor([corpus["stoi"][ch] for ch in s], dtype=torch.long)
    train_data = enc(corpus["train_text"])
    L, B, V = tiny["L"], c["batch"], corpus["V"]
    model = d1.bf.GPT(V=V, d=tiny["d"], n_layer=n_layer, n_head=tiny["n_head"], d_ff=d_ff or tiny["d_ff"], L=L,
                      dropout=c["dropout"], norm=norm, qk_norm=qk_norm).to(device)
    n_params = d1.bf.count_params(model)
    opt = torch.optim.AdamW(d1.tr.param_groups(model, c["weight_decay"]), lr=lr, betas=tuple(c["betas"]), eps=1e-8)
    lr_at = lambda s: lr * (s + 1) / warmup if s < warmup else lr        # 線性 warmup → 常數（不 cosine）
    print(f"\n{BOLD}{label}{RESET}  setting ({setting}) · {norm}-LN · {n_layer} layers · lr {lr:.1e} · warmup {warmup} · "
          f"{'QK-norm · ' if qk_norm else ''}{f'z-loss {z_loss:g} · ' if z_loss else ''}clip {clip or 'off'} · dropout {c['dropout']} · {steps} steps")
    print(f"  {n_params:,} params, device {device}, torch {torch.__version__}; batch {B} × {L} tokens; stats every {c['log_every']} steps")
    g = torch.Generator().manual_seed(c["seed"])
    curve, grad_by_layer = [], []
    diverged_at = None
    model.train()
    t0 = time.time()
    for step in range(steps):
        ix = torch.randint(0, train_data.numel() - L - 1, (B,), generator=g)
        x = torch.stack([train_data[i:i + L] for i in ix]).to(device)
        y = torch.stack([train_data[i + 1:i + L + 1] for i in ix]).to(device)
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        log = step % c["log_every"] == 0 or step == steps - 1
        logits = model(x, record_stats=log)
        lf = logits.float()
        loss = torch.nn.functional.cross_entropy(lf.view(-1, V), y.view(-1))
        total = loss
        if z_loss:
            total = loss + z_loss * torch.logsumexp(lf, dim=-1).pow(2).mean()      # PaLM 的 z-loss：α (log Z)²
        opt.zero_grad(set_to_none=True)
        total.backward()
        if log:
            gn_total = math.sqrt(sum(p.grad.float().pow(2).sum().item() for p in model.parameters() if p.grad is not None))
            per_layer = [math.sqrt(sum(p.grad.float().pow(2).sum().item() for p in blk.parameters() if p.grad is not None)) for blk in model.blocks]
            rec = {"step": step, "loss": loss.item(), "grad_norm": gn_total, "lr": lr_at(step),
                   "max_abs_logit": model.stats.get("max_abs_logit"), "log_Z": model.stats.get("log_Z"), "elapsed_s": time.time() - t0}
            curve.append(rec)
            grad_by_layer.append({"step": step, "norms": per_layer})
            flag = f" {RED}NaN/inf{RESET}" if not math.isfinite(rec["loss"]) else ""
            print(f"  step {step:>5}  loss {rec['loss']:.4f}  max|s| {rec['max_abs_logit']:>8.2f}  log Z {rec['log_Z']:>7.2f}  |g| {gn_total:>9.3f}  lr {rec['lr']:.1e}  {rec['elapsed_s']:6.1f} s{flag}")
            if not math.isfinite(rec["loss"]) or not math.isfinite(gn_total):
                diverged_at = step
                print(f"  {RED}diverged at step {step}: stopping this run{RESET}")
                break
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    seconds = time.time() - t0
    rec = {"label": label, "setting": setting, "norm": norm, "n_layer": n_layer, "d": tiny["d"], "n_head": tiny["n_head"],
           "d_ff": d_ff or tiny["d_ff"], "L": L, "params": n_params, "lr": lr, "warmup": warmup, "steps": steps, "steps_done": (curve[-1]["step"] + 1) if curve else 0,
           "qk_norm": qk_norm, "z_loss": z_loss, "grad_clip": clip, "dropout": c["dropout"], "batch": B, "seconds": seconds,
           "device": str(device), "torch": torch.__version__, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "diverged_at": diverged_at, "curve": curve, "grad_by_layer": grad_by_layer}
    rec["spike"] = detect_spike(rec, cfg)
    rec["summary"] = summarize(rec, cfg)
    print(f"  {BOLD}{rec['steps_done']} steps in {seconds:.0f} s{RESET}  " + one_line(rec))
    return rec


def detect_spike(rec: dict, cfg: dict) -> dict | None:
    """第一個「loss 比目前最小值高 spike_jump nats」或 NaN 的紀錄點。另抓 spike 時與 earlier 步前的 max|s_ij| 與 log Z。"""
    c = cfg["common"]
    pts = rec["curve"]
    run_min = math.inf
    for i, p in enumerate(pts):
        bad = not math.isfinite(p["loss"]) or (p["loss"] - run_min > c["spike_jump"])
        if bad:
            before = [q for q in pts if q["step"] <= p["step"] - c["earlier"]]
            b = before[-1] if before else None
            return {"step": p["step"], "loss": p["loss"], "running_min": run_min, "max_abs_logit": p["max_abs_logit"], "log_Z": p["log_Z"],
                    "earlier_step": b["step"] if b else None, "earlier_max_abs_logit": b["max_abs_logit"] if b else None,
                    "earlier_log_Z": b["log_Z"] if b else None, "kind": "nan" if not math.isfinite(p["loss"]) else "jump"}
        run_min = min(run_min, p["loss"])
    return None


def summarize(rec: dict, cfg: dict) -> dict:
    pts = [p for p in rec["curve"] if math.isfinite(p["loss"])]
    w = cfg["i"]["grad_norm_window"]
    early = [p["grad_norm"] for p in rec["curve"] if p["step"] < w and math.isfinite(p["grad_norm"])]
    tail = pts[-4:] if pts else []
    g0 = rec["grad_by_layer"][0]["norms"] if rec["grad_by_layer"] else []
    return {"final_loss": sum(p["loss"] for p in tail) / len(tail) if tail else None,     # 最後四個紀錄點的平均（約 100 步）
            "min_loss": min((p["loss"] for p in pts), default=None),
            "max_abs_logit_final": tail[-1]["max_abs_logit"] if tail else None,
            "max_abs_logit_peak": max((p["max_abs_logit"] for p in pts if p["max_abs_logit"] is not None), default=None),
            "log_Z_final": tail[-1]["log_Z"] if tail else None,
            "grad_norm_first_window": {"window": w, "mean": sum(early) / len(early) if early else None, "max": max(early) if early else None},
            "grad_by_layer_step0": g0, "argmax_layer_step0": (max(range(len(g0)), key=lambda i: g0[i]) if g0 else None),
            "diverged": rec["diverged_at"] is not None}


def one_line(rec: dict) -> str:
    s, sp = rec["summary"], rec["spike"]
    out = f"final loss {s['final_loss']:.3f}" if s["final_loss"] is not None else "no finite loss"
    out += f", max|s| peak {s['max_abs_logit_peak']:.1f}" if s["max_abs_logit_peak"] is not None else ""
    if sp:
        out += f"; {RED}spike at step {sp['step']}{RESET} ({sp['kind']}; max|s| {fmt(sp['max_abs_logit'])}, {rec_earlier(sp)})"
    else:
        out += f"; {GREEN}no spike{RESET}"
    if rec["diverged_at"] is not None:
        out += f"; {RED}diverged at {rec['diverged_at']}{RESET}"
    return out


def rec_earlier(sp: dict) -> str:
    return (f"{sp['step'] - sp['earlier_step']} steps earlier (step {sp['earlier_step']}): {fmt(sp['earlier_max_abs_logit'])}"
            if sp.get("earlier_step") is not None else "no logged point far enough before it")


def fmt(v, nd=2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN"
    return f"{v:.{nd}f}"


# ── 三組設定 ─────────────────────────────────────────────────────
def rehearse(setting: str, cfg: dict, a: argparse.Namespace) -> dict:
    d1 = Demo1(cfg)
    runs = []
    if setting == "i":
        s = cfg["i"]
        depths = [a.depth] if a.depth else s["depths"]
        lr, steps = a.lr or s["lr"], a.steps or s["steps"]
        for depth in depths:
            for norm in ("pre", "post"):
                runs.append(run_one(d1, cfg, label=f"(i) {norm}-LN, {depth} layers, no warmup", setting="i", norm=norm, n_layer=depth,
                                    lr=lr, warmup=s["warmup"], steps=steps, grad_clip=a.clip, d_ff=s.get("d_ff")))
    elif setting == "ii":
        s = cfg["ii"]
        mults = [float(m) for m in a.mult.split(",")] if a.mult else s["multipliers"]
        steps = a.steps or s["steps"]
        for m in mults:
            runs.append(run_one(d1, cfg, label=f"(ii) lr {m:g}× base", setting="ii", norm="pre", n_layer=d1.cfg["tiny"]["n_layer"],
                                lr=s["base_lr"] * m, warmup=s["warmup"], steps=steps, grad_clip=a.clip))
            runs[-1]["multiplier"] = m
    else:
        s = cfg["iii"]
        m = float(a.mult) if a.mult else s["multiplier"]
        steps = a.steps or s["steps"]
        zl = a.z_loss if a.z_loss is not None else s["z_loss"]
        base = cfg["ii"]["base_lr"]
        runs.append(run_one(d1, cfg, label=f"(iii) lr {m:g}× base + QK-norm", setting="iii", norm="pre", n_layer=d1.cfg["tiny"]["n_layer"],
                            lr=base * m, warmup=s["warmup"], steps=steps, qk_norm=s["qk_norm"], grad_clip=a.clip))
        runs[-1]["multiplier"] = m
        if zl:
            runs.append(run_one(d1, cfg, label=f"(iii) lr {m:g}× base + QK-norm + z-loss {zl:g}", setting="iii", norm="pre",
                                n_layer=d1.cfg["tiny"]["n_layer"], lr=base * m, warmup=s["warmup"], steps=steps, qk_norm=s["qk_norm"],
                                z_loss=zl, grad_clip=a.clip))
            runs[-1]["multiplier"] = m
    rec = {"setting": setting, "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": platform.machine(), "macos": platform.mac_ver()[0],
           "python": platform.python_version(), "config": {"common": cfg["common"], setting: cfg[setting]}, "runs": runs}
    if not a.no_save:
        REHEARSAL.mkdir(parents=True, exist_ok=True)
        p = REHEARSAL / f"{setting}.json"
        if p.exists() and setting == "iii" and not zl:
            # 之前存過 z-loss 那條就保留它（只重跑 QK-norm 那條時不要把 z-loss 的紀錄弄丟）
            old = json.loads(p.read_text(encoding="utf-8")).get("runs", [])
            runs.extend(r for r in old if r.get("z_loss"))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        svg_setting(rec, cfg, REHEARSAL / f"{setting}.svg")
        print(f"{DIM}→ runs/rehearsal/{setting}.json, runs/rehearsal/{setting}.svg{RESET}")
    return rec


def baselines(cfg: dict, save: bool = True) -> dict:
    """同一份語料、同一個切點：held-out 在 train 的字元 unigram 與 bigram（add-0.5 平滑）下的 cross-entropy，nats/char。
    這兩個數是 replay 表「loss 卡在哪個水準」的尺：post-LN 不加 warmup 停在 unigram 附近、10× 的 entropy collapse 停在 bigram 附近（2026-10-06 彩排）。
    純標準函式庫，一秒內。"""
    import collections
    d1 = Demo1(cfg)
    c = d1.corpus()
    tr, he, V = c["train_text"], c["held_text"], c["V"]
    uni = collections.Counter(tr)
    n = len(tr)
    h_uni_train = -sum(v / n * math.log(v / n) for v in uni.values())
    xent_uni = -sum(math.log(uni[ch] / n) for ch in he) / len(he)                       # held-out 字元全部出現在 train 裡（V 由全文定）
    big, ctx = collections.Counter(zip(tr, tr[1:])), collections.Counter(tr[:-1])
    xent_big = sum(-math.log((big[(a, b)] + 0.5) / (ctx[a] + 0.5 * V)) for a, b in zip(he, he[1:])) / (len(he) - 1)
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "V": V, "train_chars": len(tr), "held_chars": len(he), "cut_char": c["cut_char"],
           "unigram_entropy_train": h_uni_train, "held_xent_unigram": xent_uni, "held_xent_bigram_add0.5": xent_big,
           "note": "nats/char; same split as demo 1 (W1's 90% token cut); bigram uses add-0.5 smoothing over V symbols"}
    print(f"{BOLD}baselines{RESET}  V = {V}; train {len(tr):,} chars, held-out {len(he):,} chars (same cut as demo 1)")
    print(f"  char unigram entropy of train        {h_uni_train:.3f} nats/char")
    print(f"  held-out under train unigram         {xent_uni:.3f} nats/char   ← post-LN without warmup stalls here")
    print(f"  held-out under train bigram (+0.5)   {xent_big:.3f} nats/char   ← the 10× run's plateau is here")
    if save:
        REHEARSAL.mkdir(parents=True, exist_ok=True)
        (REHEARSAL / "baselines.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{DIM}→ runs/rehearsal/baselines.json{RESET}")
    return rec


def load_baselines() -> dict | None:
    p = REHEARSAL / "baselines.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def level_of(loss: float | None, bl: dict | None) -> str:
    """把一個 loss 對到最近的 n-gram 水準（差 0.15 nats 以內才標）。"""
    if loss is None or bl is None:
        return "—"
    for lab, v in (("≈ unigram", bl["held_xent_unigram"]), ("≈ bigram", bl["held_xent_bigram_add0.5"])):
        if abs(loss - v) < 0.15:
            return f"{lab} ({v:.2f})"
    return "below bigram" if loss < bl["held_xent_bigram_add0.5"] else "above unigram"


def estimate(cfg: dict) -> None:
    d1 = Demo1(cfg)
    st = d1.probe_step_time()
    base_layers = d1.cfg["tiny"]["n_layer"]
    if st is None:
        print(f"{AMBER}demo 1 還沒有 runs/rehearsal/probe.json；先在那邊跑 ./present.sh probe{RESET}")
        return
    print(f"{BOLD}estimate{RESET}  demo 1 的 probe：{base_layers} 層一步 {st:.3f} s（batch {cfg['common']['batch']}）；假設每步時間與層數成正比、與 dropout／統計量記錄無關（都是假設，跑了才知道）")
    total = 0.0
    for depth in cfg["i"]["depths"]:
        t = 2 * cfg["i"]["steps"] * st * depth / base_layers
        total += t
        print(f"  (i)   {depth} 層 × 2 條 × {cfg['i']['steps']} 步 ≈ {t / 60:5.1f} min")
    t = len(cfg["ii"]["multipliers"]) * cfg["ii"]["steps"] * st
    total += t
    print(f"  (ii)  {len(cfg['ii']['multipliers'])} 條 × {cfg['ii']['steps']} 步 ≈ {t / 60:5.1f} min（提早發散的會更短）")
    t = (2 if cfg["iii"]["z_loss"] else 1) * cfg["iii"]["steps"] * st
    total += t
    print(f"  (iii) {2 if cfg['iii']['z_loss'] else 1} 條 × {cfg['iii']['steps']} 步 ≈ {t / 60:5.1f} min")
    print(f"  {BOLD}合計 ≈ {total / 60:.0f} min{RESET}")


# ── replay／show ─────────────────────────────────────────────────
def load_setting(setting: str) -> dict | None:
    p = REHEARSAL / f"{setting}.json"
    if not p.exists():
        print(f"{RED}沒有 runs/rehearsal/{setting}.json：先跑 ./present.sh rehearse {setting}{RESET}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def banner(rec: dict) -> None:
    r0 = rec["runs"][0] if rec["runs"] else {}
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {rec['date']}, torch {r0.get('torch', '?')}, {r0.get('device', '?')}, {rec['machine']}")
    print(f"{GREY}  a proxy of a proxy: tiny GPT {r0.get('params', 0):,} params (d {r0.get('d')}, {r0.get('n_head')} heads, d_ff {r0.get('d_ff')}, L {r0.get('L')}); "
          f"Wortsman et al.'s proxies are 10–1000× larger. The shape of the mechanism only — no number here is a real model's threshold.{RESET}")


def table_i(rec: dict) -> None:
    c = rec["config"]["common"]
    print(f"\n  {BOLD}setting (i): pre-LN vs. post-LN, no warmup, lr {rec['runs'][0]['lr']:.0e}, clip {rec['runs'][0]['grad_clip'] or 'off'}{RESET}")
    bl = load_baselines()
    print(f"    {'':<10} {'depth':>5} {'final loss':>11} {'stuck at':>18} {'grad norm, first ' + str(rec['config']['i']['grad_norm_window']) + ' steps (mean / max)':>42} {'diverged?':>14} {'grad @ step 0: layer 0 … last':>30}")
    for r in rec["runs"]:
        s = r["summary"]
        gw = s["grad_norm_first_window"]
        div = f"yes, step {r['diverged_at']}" if r["diverged_at"] is not None else (f"spike @ {r['spike']['step']}" if r["spike"] else "no")
        g0 = s["grad_by_layer_step0"]
        g0s = f"{g0[0]:.2f} … {g0[-1]:.2f} (max at layer {s['argmax_layer_step0']})" if g0 else "—"
        print(f"    {r['norm'] + '-LN':<10} {r['n_layer']:>5} {fmt(s['final_loss'], 3):>11} {level_of(s['final_loss'], bl):>18} {fmt(gw['mean']) + ' / ' + fmt(gw['max']):>42} {div:>14} {g0s:>30}")
    bl_note = f"'stuck at' compares with the held-out text under a char unigram ({bl['held_xent_unigram']:.2f}) / bigram ({bl['held_xent_bigram_add0.5']:.2f}), baselines.json" if bl else "'stuck at' needs ./present.sh baselines"
    print(f"    {GREY}final loss = mean of the last four logged train-loss points; grad norms are the whole model's, before any clipping; the last column is each Block's gradient norm at step 0 (Xiong et al. predict post-LN's last layers largest at init). {bl_note}{RESET}")


def curves_i(rec: dict) -> None:
    for key, title in (("loss", "train loss (nats/char)"), ("grad_norm", "gradient norm (whole model, log scale)")):
        series = []
        for r, glyph in zip(rec["runs"], "●○▲△"):
            pts = [(p["step"], p[key]) for p in r["curve"] if p[key] is not None and math.isfinite(p[key]) and p[key] > 0]
            if key == "grad_norm":
                pts = [(x, math.log10(y)) for x, y in pts]
            series.append((f"{r['norm']}-LN {r['n_layer']}L", pts, glyph))
        print(f"\n  {BOLD}{title}{RESET}   " + "   ".join(f"{g} {lab}" for lab, _, g in series))
        print(ascii_multi(series, log10_labels=(key == "grad_norm")))
    print(f"\n  {BOLD}gradient norm per Block at step 0{RESET} (layer 0 = nearest the embedding)")
    for r in rec["runs"]:
        g0 = r["summary"]["grad_by_layer_step0"]
        if g0:
            top = max(g0)
            print(f"    {r['norm'] + '-LN':<8} " + " ".join(f"{v:6.2f}" for v in g0))
            print(f"    {'':<8} " + " ".join(("▇" * max(1, round(6 * v / top))).rjust(6) if top > 0 else "     ." for v in g0))


def table_ii(rec: dict, extra: list[dict] | None = None) -> None:
    c = rec["config"]["common"]
    runs = rec["runs"] + (extra or [])
    bl = load_baselines()
    print(f"\n  {BOLD}settings (ii)/(iii): learning rate × base ({rec['config'].get('ii', {}).get('base_lr', '?')}), {runs[0]['steps'] if runs else '?'} steps{RESET}")
    print(f"    {'run':<24} {'final loss':>11} {'stuck at':>18} {'max|s| peak':>12} {'max|s| final':>13} {'log Z final':>12} {'spike?':>22}")
    for r in runs:
        sp, s = r["spike"], r["summary"]
        name = f"{r.get('multiplier', '?'):g}×" + (" + QK-norm" if r["qk_norm"] else "") + (f" + z-loss {r['z_loss']:g}" if r["z_loss"] else "")
        spike = (f"step {sp['step']}" + (" (NaN)" if sp["kind"] == "nan" else "")) if sp else "no spike"
        print(f"    {name:<24} {fmt(s['final_loss'], 3):>11} {level_of(s['final_loss'], bl):>18} {fmt(s['max_abs_logit_peak'], 1):>12} {fmt(s['max_abs_logit_final'], 1):>13} {fmt(s['log_Z_final']):>12} {spike:>22}")
    if any(r["spike"] for r in runs):
        print(f"    {BOLD}spikes{RESET}: " + "; ".join(f"{r.get('multiplier', '?'):g}×{' QK' if r['qk_norm'] else ''}: step {r['spike']['step']}, max|s| {fmt(r['spike']['max_abs_logit'])} "
                                              f"({rec_earlier(r['spike'])}), log Z {fmt(r['spike']['log_Z'])}" for r in runs if r["spike"]))
    bl_note = (f"'stuck at' compares the final loss with the same held-out text under a char unigram ({bl['held_xent_unigram']:.2f}) and bigram ({bl['held_xent_bigram_add0.5']:.2f}) — baselines.json"
               if bl else "'stuck at' needs runs/rehearsal/baselines.json (./present.sh baselines)")
    print(f"    {GREY}final loss = mean of the last four logged train-loss points; max|s| = largest |q·k/√d_head| over all layers and heads in that step's forward pass; "
          f"log Z = mean logsumexp of the output logits; spike = loss exceeds its running minimum by {c['spike_jump']} nats, or NaN (this demo's operational definition). {bl_note}{RESET}")


def curves_ii(rec: dict, extra: list[dict] | None = None) -> None:
    runs = rec["runs"] + (extra or [])
    for key, title, logscale in (("loss", "train loss (nats/char)", False), ("max_abs_logit", "max |s_ij| — the attention logit before softmax", False),
                                 ("log_Z", "log Z of the output logits", False), ("grad_norm", "gradient norm (log scale)", True)):
        series = []
        for r, glyph in zip(runs, "●○▲△■□"):
            pts = [(p["step"], p[key]) for p in r["curve"] if p[key] is not None and math.isfinite(p[key])]
            if logscale:
                pts = [(x, math.log10(y)) for x, y in pts if y > 0]
            name = f"{r.get('multiplier', '?'):g}×" + (" QK" if r["qk_norm"] else "") + (" z" if r["z_loss"] else "")
            series.append((name, pts, glyph))
        print(f"\n  {BOLD}{title}{RESET}   " + "   ".join(f"{g} {lab}" for lab, _, g in series))
        print(ascii_multi(series, log10_labels=logscale))
        for r in runs:
            if r["spike"]:
                print(f"    {GREY}{r.get('multiplier', '?'):g}×{' QK' if r['qk_norm'] else ''}: spike at step {r['spike']['step']}{RESET}", end="")
        print()


def replay(setting: str, show_only: bool) -> None:
    rec = load_setting(setting)
    if rec is None:
        return
    banner(rec)
    if setting == "i":
        table_i(rec)
        if not show_only:
            curves_i(rec)
    elif setting == "ii":
        table_ii(rec)
        if not show_only:
            curves_ii(rec)
    else:
        # (iii) 並排 (ii) 的同一個倍數那條，看 QK-norm 壓住了什麼
        other = load_setting("ii") if (REHEARSAL / "ii.json").exists() else None
        m = rec["runs"][0].get("multiplier")
        base = [r for r in (other or {}).get("runs", []) if r.get("multiplier") == m]
        rec_cmp = {**rec, "runs": base, "config": {**rec["config"], "ii": (other or {}).get("config", {}).get("ii", {})}}
        table_ii(rec_cmp, extra=rec["runs"])
        if not show_only:
            curves_ii(rec_cmp, extra=rec["runs"])
    print(f"\n  {DIM}{setting}.svg: open {REHEARSAL / (setting + '.svg')}{RESET}")


# ── 畫圖：終端機 ASCII 與 SVG ────────────────────────────────────
def ascii_multi(series: list[tuple[str, list[tuple[int, float]], str]], width: int = 76, height: int = 12, log10_labels: bool = False) -> str:
    allpts = [p for _, pts, _ in series for p in pts]
    if not allpts:
        return "    (no points)"
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    x0, x1, lo, hi = min(xs), max(xs), min(ys), max(ys)
    hi = hi if hi > lo else lo + 1e-9
    grid = [[" "] * width for _ in range(height)]
    for _, pts, glyph in series:
        for x, y in pts:
            col = int((x - x0) / max(1, x1 - x0) * (width - 1))
            row = int((hi - y) / (hi - lo) * (height - 1))
            grid[row][col] = glyph
    lines = []
    for i, row in enumerate(grid):
        v = hi - (hi - lo) * i / (height - 1)
        lab = (f"{10 ** v:8.3g} │" if log10_labels else f"{v:8.2f} │") if i in (0, height // 2, height - 1) else "         │"
        lines.append("    " + lab + "".join(row))
    lines.append("    " + "         └" + "─" * width)
    lines.append("    " + f"          step {x0}" + " " * max(1, width - 22 - len(str(x1))) + f"step {x1}")
    return "\n".join(lines)


def svg_setting(rec: dict, cfg: dict, path: Path) -> None:
    """四個 panel（loss、max|s|、log Z、梯度範數），每條 run 一色；spike 畫虛線。純手寫 SVG，不依賴 matplotlib。"""
    runs = rec["runs"]
    colors = ["#3b6fb6", "#d9534f", "#2e8b57", "#b8860b", "#7b4fa0", "#555"]
    panels = [("loss", "train loss (nats/char)", False), ("max_abs_logit", "max |s_ij| (attention logit before softmax)", False),
              ("log_Z", "log Z of output logits", False), ("grad_norm", "gradient norm (log10)", True)]
    PW, PH, l, r, tp, bt, gap = 440, 240, 56, 16, 34, 40, 24
    W, H = 2 * PW + gap, 2 * PH + gap + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="8" y="16" font-size="13" font-weight="600">W4 demo 2, setting ({rec["setting"]}) — rehearsed {rec["date"][:10]}; tiny GPT, '
           f'{runs[0]["params"]:,} params; a proxy of a proxy: the shape only</text>']
    legend_x = 8
    for i, run in enumerate(runs):
        name = f"{run['norm']}-LN {run['n_layer']}L" if rec["setting"] == "i" else (f"lr {run.get('multiplier', '?'):g}×" + (" + QK-norm" if run["qk_norm"] else "") + (f" + z-loss" if run["z_loss"] else ""))
        out.append(f'<rect x="{legend_x}" y="22" width="12" height="3" fill="{colors[i % len(colors)]}"/>')
        out.append(f'<text x="{legend_x + 16}" y="26">{name}{" — spike @ " + str(run["spike"]["step"]) if run["spike"] else ""}</text>')
        legend_x += 16 + 7 * (len(name) + (12 if run["spike"] else 0)) + 14
    for pi, (key, title, logscale) in enumerate(panels):
        ox, oy = (pi % 2) * (PW + gap), 40 + (pi // 2) * (PH + gap)
        pts_all = []
        series = []
        for run in runs:
            pts = [(p["step"], p[key]) for p in run["curve"] if p[key] is not None and math.isfinite(p[key])]
            if logscale:
                pts = [(x, math.log10(y)) for x, y in pts if y > 0]
            series.append(pts)
            pts_all += pts
        if not pts_all:
            continue
        xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
        x0, x1, y0, y1 = 0, max(xs) or 1, min(ys), max(ys)
        y0, y1 = (y0 - 0.05 * (y1 - y0 or 1)), (y1 + 0.05 * (y1 - y0 or 1))
        X = lambda x: ox + l + (x - x0) / (x1 - x0) * (PW - l - r)
        Y = lambda y: oy + tp + (y1 - y) / (y1 - y0) * (PH - tp - bt)
        out.append(f'<text x="{ox + l}" y="{oy + 14}" font-weight="600">{title}</text>')
        for k in range(5):
            y = y0 + (y1 - y0) * k / 4
            out.append(f'<line x1="{ox + l}" y1="{Y(y):.1f}" x2="{ox + PW - r}" y2="{Y(y):.1f}" stroke="#eee"/>')
            out.append(f'<text x="{ox + l - 5}" y="{Y(y) + 4:.1f}" text-anchor="end" fill="#555">{(10 ** y) if logscale else y:.3g}</text>')
        for k in range(5):
            x = x0 + (x1 - x0) * k / 4
            out.append(f'<text x="{X(x):.1f}" y="{oy + PH - bt + 14}" text-anchor="middle" fill="#555">{int(x)}</text>')
        out.append(f'<text x="{ox + (l + PW - r) / 2:.1f}" y="{oy + PH - 6}" text-anchor="middle" fill="#555">step</text>')
        for i, (run, pts) in enumerate(zip(runs, series)):
            if pts:
                out.append(f'<polyline fill="none" stroke="{colors[i % len(colors)]}" stroke-width="1.5" points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts) + '"/>')
            if run["spike"] and x0 <= run["spike"]["step"] <= x1:
                out.append(f'<line x1="{X(run["spike"]["step"]):.1f}" y1="{oy + tp}" x2="{X(run["spike"]["step"]):.1f}" y2="{oy + PH - bt}" stroke="{colors[i % len(colors)]}" stroke-dasharray="3 3"/>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rehearse", "replay", "show", "estimate", "baselines"])
    ap.add_argument("setting", nargs="?", choices=SETTINGS + ("all",), default=None)
    ap.add_argument("--depth", type=int, help="(i) 覆蓋 depths，只跑一個深度")
    ap.add_argument("--lr", type=float, help="(i) 覆蓋 lr")
    ap.add_argument("--steps", type=int, help="覆蓋該設定的 steps")
    ap.add_argument("--mult", help="(ii) 逗號分隔的倍數，如 1,3,10,30；(iii) 單一倍數")
    ap.add_argument("--z-loss", dest="z_loss", type=float, default=None, help="(iii) 多跑一條加 z-loss 的（PaLM 用 1e-4）")
    ap.add_argument("--clip", type=float, default=None, help="覆蓋 [common].grad_clip（預設 0 = 關）")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "estimate":
        estimate(cfg)
    elif a.cmd == "baselines":
        baselines(cfg, save=not a.no_save)
    elif a.cmd == "rehearse":
        which = SETTINGS if a.setting in (None, "all") else (a.setting,)
        if len(which) > 1:
            estimate(cfg)
        for s in which:
            rehearse(s, cfg, a)
    else:
        if a.setting in (None, "all"):
            print("replay／show 要指定設定：i、ii 或 iii", file=sys.stderr)
            return 2
        replay(a.setting, show_only=(a.cmd == "show"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
