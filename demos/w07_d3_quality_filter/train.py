#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 3 的 (b) 半邊：同一個模型、兩份等 byte 的語料各訓一次，在幾個 held-out 上分開量。

模型、超參數、射線全部來自 demo 2（../w07_d2_scaling_fit/train.py 與它的 demo_config.toml，從檔案路徑載入，不複製）：
  Demo2(cfg)                      載 demo 2 的 train 模組與設定；.d1 是 W4 demo 1（GPT、pick_device、param_groups、lr_at、evaluate）
  load_heldouts(cfg, d2, corpora) 四個 held-out → {key: {"label", "bytes", "source", ...}}
  train_run(cfg, d2, corpus, heldouts, steps, tag, save)   訓一個 run：每 record_every 步在**每個** held-out 的固定子集上量一次；訓完每個整段跑 W4 的 evaluate

跟 demo 2 的 train_size 只差一件事：held-out 不只一個——demo 2 只記自己語料的尾端，這裡要兩個 run 在同幾段文字上比，所以曲線與終值都對每個 held-out 各記一份。
訓練迴圈本身（不重疊視窗、每個 epoch 打亂、warmup → cosine、clip）與 demo 2 逐行相同。
"""
from __future__ import annotations

import importlib.util
import math
import platform
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Demo2:
    """demo 2 的 train.py（fetch_corpus、load_corpus、Demo1、build_model、count_split、lr_for、warmup_for、generate）與 demo_config.toml。"""

    def __init__(self, cfg: dict):
        self.dir = (HERE / cfg["shared"]["demo2"]).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(f"找不到 demo 2 的資料夾 {self.dir}")
        with open(self.dir / "demo_config.toml", "rb") as f:
            self.cfg2 = tomllib.load(f)
        self.T = load_module(self.dir / "train.py", "w07_d2_train")
        self.d1 = self.T.Demo1(str(self.dir / self.cfg2["shared"]["demo1"]))      # 絕對路徑：Demo1 的 rel 是對 demo 2 的 HERE 寫的，從這裡載要先解析
        self.size = next(s for s in self.cfg2["sizes"] if s["key"] == cfg["train"]["size"])

    def load_corpus(self, which: str) -> dict:
        """demo 2 的 load_corpus 讀 [corpus].dir（相對 demo 2 的資料夾）；它用自己的 HERE 解析，所以直接呼叫即可。"""
        return self.T.load_corpus(self.cfg2, which)


def load_heldouts(cfg: dict, d2: Demo2, corpora: dict, S) -> dict:
    """[[heldouts]] → {key: {label, bytes, source, chars, ...}}。S 是 sample 模組（load_wikipedia）。"""
    out = {}
    for h in cfg["heldouts"]:
        if h["source"] == "file":
            p = (HERE / h["path"]).resolve()
            b = p.read_bytes()
            info = {"path": str(p.relative_to(HERE.parent)) if p.is_relative_to(HERE.parent) else str(p)}
        elif h["source"] == "wikipedia":
            b, meta = S.load_wikipedia(cfg, d2.cfg2)
            info = {k: meta.get(k) for k in ("repo", "path", "license", "row_group", "seed", "articles", "date")}
        elif h["source"] == "corpus_tail":
            c = corpora[h["corpus"]]
            b = c["held"]
            info = {"corpus": h["corpus"], "repo": c["repo"], "license": c["license"], "held_docs": c["held_docs"]}
        else:
            raise ValueError(h["source"])
        out[h["key"]] = {"key": h["key"], "label": h["label"], "source": h["source"], "bytes": b, "n_bytes": len(b), **info}
    return out


def subset_tensors(held: bytes, L: int, max_windows: int, device):
    """固定子集：開頭最多 max_windows 個不重疊視窗（短的 held-out 有幾個算幾個，至少 1 個——754 bytes 的段落有 2 個）。"""
    import torch
    t = torch.frombuffer(bytearray(held), dtype=torch.uint8).long()
    n = max(1, min(max_windows, (t.numel() - 1) // L))
    if t.numel() < L + 1:                                                   # 比一個視窗還短：補到 L + 1（不會發生在這四個，保險）
        t = torch.cat([t, torch.zeros(L + 1 - t.numel(), dtype=torch.long)])
    hx = torch.stack([t[i * L:(i + 1) * L] for i in range(n)]).to(device)
    hy = torch.stack([t[i * L + 1:(i + 1) * L + 1] for i in range(n)]).to(device)
    return hx, hy, n


def train_run(cfg: dict, d2: Demo2, corpus: dict, heldouts: dict, steps: int | None, tag: str, save_weights: Path | None = None) -> dict:
    import torch
    T, d1, t, ray, s = d2.T, d2.d1, d2.cfg2["train"], d2.cfg2["ray"], d2.size
    L, B = t["L"], t["batch"]
    device = d1.tr.pick_device(t["device"])
    model = T.build_model(d1, s, L, t["dropout"], t["seed"], t["d_head"]).to(device)
    split = T.count_split(model)
    N = split[ray["count"]]
    D_target = ray["tokens_per_param"] * N
    full_steps = max(1, round(D_target / (B * L)))
    steps = steps or (cfg["train"]["steps"] or full_steps)
    lr, warm = T.lr_for(s["d"], t), T.warmup_for(steps, t)
    cfg_lr = {"train": {"lr": lr, "warmup": warm, "steps": steps}}
    opt = torch.optim.AdamW(d1.tr.param_groups(model, t["weight_decay"]), lr=lr, betas=tuple(t["betas"]), eps=1e-8)
    train_u8 = torch.frombuffer(bytearray(corpus["train"]), dtype=torch.uint8)
    subsets = {k: subset_tensors(h["bytes"], L, cfg["train"]["subset_windows"], device) for k, h in heldouts.items()}
    n_windows = (train_u8.numel() - 1) // L
    g = torch.Generator().manual_seed(t["seed"])
    perm, pos, epoch = torch.randperm(n_windows, generator=g), 0, 0
    every = cfg["train"]["record_every"]
    print(f"\n{BOLD}{tag} — trained on {corpus['which']} ({Path(corpus['path']).name}){RESET}  torch {torch.__version__}, device {device}, {platform.machine()}")
    print(f"  corpus {corpus['repo']} ({corpus['license']}): {corpus['docs']:,} docs, train {corpus['train_bytes'] / 1e6:.1f} MB = {n_windows:,} windows of {L}")
    print(f"  byte-level GPT {s['key']} (V = {T.V_BYTES}): {s['n_layer']} layers, d {s['d']}, {s['d'] // t['d_head']} heads, d_ff {s['d_ff']}, L {L}; params {split['total']:,} (non-embedding {split['nonemb']:,})")
    print(f"  {steps:,} steps of {B}×{L} = {steps * B * L:,} tokens" + (f"  {AMBER}(overridden; the ray would be {full_steps:,}){RESET}" if steps != full_steps else f" = {ray['tokens_per_param']}·N")
          + f"; {steps * B / n_windows:.2f} epochs; AdamW lr {lr:.2e}, warmup {warm}, cosine → 0.1·lr, clip {t['grad_clip']}, wd {t['weight_decay']}, dropout {t['dropout']}, seed {t['seed']}")
    print(f"  held-outs (subset windows for the curve): " + ", ".join(f"{k} {n}" for k, (_, _, n) in subsets.items()))
    curve, t0, tok_seen = [], time.time(), 0

    def record(step: int, train_loss: float, gn: float, lr_now: float):
        model.eval()
        rec = {"step": step, "tokens_seen": tok_seen, "train_loss": train_loss, "grad_norm": gn, "lr": lr_now, "elapsed_s": time.time() - t0, "held": {}}
        with torch.no_grad():
            for k, (hx, hy, _) in subsets.items():
                hl = torch.nn.functional.cross_entropy(model(hx).float().view(-1, T.V_BYTES), hy.view(-1)).item()
                rec["held"][k] = hl / math.log(2)                                   # bits/byte（子集）
        model.train()
        curve.append(rec)
        if step % d2.cfg2["record"]["log_every"] == 0 or step == steps - 1:
            print(f"  step {step:>6}/{steps}  loss {train_loss:.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in rec["held"].items())
                  + f"  |g| {gn:.2f}  lr {lr_now:.2e}  {rec['elapsed_s']:7.1f} s  {tok_seen / max(rec['elapsed_s'], 1e-9):,.0f} tok/s")

    model.train()
    for step in range(steps):
        if pos + B > n_windows:
            perm, pos, epoch = torch.randperm(n_windows, generator=g), 0, epoch + 1
        ix = perm[pos:pos + B] * L
        pos += B
        x = torch.stack([train_u8[i:i + L] for i in ix.tolist()]).long().to(device)
        y = torch.stack([train_u8[i + 1:i + L + 1] for i in ix.tolist()]).long().to(device)
        lr_now = d1.tr.lr_at(step, cfg_lr)
        for grp in opt.param_groups:
            grp["lr"] = lr_now
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, T.V_BYTES), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
        opt.step()
        tok_seen += B * L
        if step % every == 0 or step == steps - 1:
            record(step, loss.item(), gn, lr_now)
    seconds = time.time() - t0
    smoke = steps != full_steps
    final = {}
    for k, h in heldouts.items():
        data = torch.frombuffer(bytearray(h["bytes"]), dtype=torch.uint8).long()
        if smoke:
            data = data[: min(data.numel(), 131_072)]                           # 煙霧測試只量前 128 KiB
        ev = d1.tr.evaluate(model, data, L, t["eval_stride"], device)            # W4 的尺：整段、視窗 L、stride 128，每個 byte 預測一次
        final[k] = {"nats_per_byte": ev["nats_per_char"], "bpb": ev["nats_per_char"] / math.log(2), "bytes_scored": ev["chars"], "subset_bpb": curve[-1]["held"][k]}
    print(f"  {BOLD}{steps:,} steps in {seconds:.0f} s ({seconds / 60:.1f} min), {tok_seen / seconds:,.0f} tok/s{RESET}")
    for k, f in final.items():
        print(f"    {BLUE}{heldouts[k]['label']:<40}{RESET} {f['bpb']:.4f} bits/byte  {DIM}({f['bytes_scored']:,} bytes scored; subset {f['subset_bpb']:.4f}){RESET}")
    sample = T.generate(model, device, t["sample_bytes"], corpus)
    if save_weights:
        save_weights.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_weights / f"{corpus['which']}.pt")
    return {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "trained_on": corpus["which"], "size": {**s, "n_head": s["d"] // t["d_head"], "V": T.V_BYTES, "L": L},
            "torch": torch.__version__, "device": str(device), "python": platform.python_version(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
            "params": split, "N": N, "steps": steps, "steps_for_ray": full_steps, "tokens": tok_seen, "epochs": tok_seen / (n_windows * L), "lr": lr, "warmup": warm,
            "train": dict(t), "ray": ray,
            "corpus": {k: v for k, v in corpus.items() if k not in ("train", "held")}, "n_windows": n_windows,
            "heldouts": {k: {kk: vv for kk, vv in h.items() if kk != "bytes"} for k, h in heldouts.items()},
            "curve": curve, "final": final, "seconds": seconds, "tokens_per_s": tok_seen / seconds, "sample": sample}
