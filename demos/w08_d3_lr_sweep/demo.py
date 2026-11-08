#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 3：LR 掃描——LoRA 與 full fine-tuning 各掃一組 learning rate（投影片 p48 前導頁、p49 replay）。

    ./present.sh inspect      課前：版本、模型與資料在不在、兩組 LR 網格（含轉成 bf16 之後的值、PEFT 單位的換算）、N 與 N_LoRA 對 w08-data.json
    ./present.sh probe        彩排前：每個候選 LR 只跑 30 步，看會不會發散、一步幾秒；印出整個掃描要多久 → runs/rehearsal/probe.json
    ./present.sh rehearse     彩排：base 的 held-out loss → 兩組各五個 run（每個 run 另開子行程）→ 固定 LR 的交叉 run → curves.json、best.json；
                              已完成的 run 會跳過（--redo 重跑）；--group lora|full 只做一組；--no-cross 不做交叉
    ./present.sh replay       課堂：Enter 第一張圖（最終 loss 對 LR 的兩條曲線）→ Enter 第二張圖（固定 LR 對各自最佳的訓練曲線）→ p49 的表
    ./present.sh run ...      一個 run（rehearse 在子行程裡呼叫的就是它）：--group lora|full|base --lr 1.2e-5 [--iters 30 --probe]
    ./present.sh aggregate    只重算 curves.json／best.json（例如手動補跑了一個 run 之後）
    ./present.sh fake ...     沒有 mlx 也能演練流程（輸出是假的，畫面會標明）

訓練走 mlx-lm 自己的 `train_model`（與 `mlx_lm.lora` 命令列同一組參數，程式把等價的命令列印出來）；LoRA 的 rank／scale 從 lora.yaml 給。
每個 run 在自己的子行程裡跑：記憶體乾淨（full FT 的 Adam 狀態不會留在 allocator 裡影響下一個 run 的 peak）、中途掛掉可以接著跑。

比例的單位：mlx 的 LoRA 是 y = Wx + scale · BAx，scale 20 = Hu 寫法的 α/r；PEFT 常見的 α = 2r（α/r = 2）比它小 10 倍。
Adam 的更新量 ∝ scale · lr，所以「PEFT 單位的最佳 LR」= 這裡量到的 lr × (scale / 2) = × 10。報比例時兩種都印。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）、transformers（tokenizer）、huggingface_hub。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10（Mac 側 VM）：pip install tomli
    import tomli as tomllib   # type: ignore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info,  # noqa: E402
                    locked_revision, note, now, peak_memory_gb, reset_peak_memory, versions)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
RUN_DIR = REHEARSAL / "runs"       # 每個 run 一個 json（進版控）
PROBE_DIR = REHEARSAL / "probe"    # probe 的每個 run 一個 json（進版控）
WEIGHTS = RUNS / "weights"         # 不進版控
LORA_YAML = HERE / "lora.yaml"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]
GROUPS = ("full", "lora")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def data_dir(cfg: dict) -> Path:
    return (HERE / cfg["data"]["dir"]).resolve()


def lr_tag(lr: float) -> str:
    return f"{lr:.1e}"      # 1.2e-05、2.0e-04：run 的名字與權重資料夾都用它


def run_name(group: str, lr: float) -> str:
    return f"{group}_lr{lr_tag(lr)}"


def fmt_lr(lr: float) -> str:
    return f"{lr:.1e}".replace("e-0", "e-")


def pause(prompt: str = "") -> str:
    try:
        return input(f"{DIM}{prompt}{RESET}")
    except EOFError:
        return ""


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def lr_in(lr: float, lrs: list[float]) -> bool:
    return any(abs(math.log(lr) - math.log(x)) < 1e-6 for x in lrs)


# ── 參數物件與命令列 ─────────────────────────────────────────────
def lora_parameters(cfg: dict) -> dict:
    lo = cfg["lora"]
    return {"rank": int(lo["rank"]), "dropout": float(lo["dropout"]), "scale": float(lo["scale"]), "keys": list(lo["keys"])}


def write_yaml(cfg: dict) -> None:
    p = lora_parameters(cfg)
    body = ["# 由 demo.py 從 demo_config.toml [lora] 產生；mlx_lm.lora -c 讀。與 demo 1 的 lora.yaml 相同（rank 16、scale 20，不接 embedding）。",
            "lora_parameters:", f"  rank: {p['rank']}", f"  dropout: {p['dropout']}", f"  scale: {p['scale']}",
            "  keys:  # 所有 block 的七個 Linear（mlx-lm 的預設集合，寫出來是為了跟 demo 1 的 lora_embed.yaml 對照）"]
    body += [f"    - {k}" for k in p["keys"]]
    LORA_YAML.write_text("\n".join(body) + "\n", encoding="utf-8")


def model_path(cfg: dict, offline: bool) -> str:
    from huggingface_hub import snapshot_download
    m = cfg["model"]
    rev = locked_revision(m["repo"], m.get("revision"))
    return snapshot_download(m["repo"], revision=rev, local_files_only=offline, ignore_patterns=NON_MODEL_FILES)


def train_args(cfg: dict, path: str, group: str, lr: float, iters: int, out_dir: Path, probe: bool) -> SimpleNamespace:
    """與 `mlx_lm.lora` 命令列完全相同的參數物件（CONFIG_DEFAULTS 補齊沒給的）。"""
    from mlx_lm.lora import CONFIG_DEFAULTS
    t = cfg["train"]
    a = dict(CONFIG_DEFAULTS)
    a.update(model=path, train=True, data=str(data_dir(cfg)), fine_tune_type=group,
             num_layers=int(cfg[group]["num_layers"]), mask_prompt=bool(t["mask_prompt"]), iters=int(iters),
             batch_size=int(t["batch_size"]), learning_rate=float(lr),
             steps_per_report=int(cfg["probe"]["steps_per_report"] if probe else t["steps_per_report"]),
             steps_per_eval=10 ** 9 if probe else int(t["steps_per_eval"]), val_batches=-1,
             adapter_path=str(out_dir), save_every=10 ** 9, max_seq_length=int(t["max_seq_length"]),
             seed=int(cfg["demo"]["seed"]), optimizer=str(t["optimizer"]),
             lora_parameters=lora_parameters(cfg) if group == "lora" else None)
    return SimpleNamespace(**a)


def cli_line(cfg: dict, a: SimpleNamespace) -> str:
    """等價的命令列。資料與 adapter 路徑都相對這個資料夾。"""
    parts = ["mlx_lm.lora"]
    if a.fine_tune_type == "lora":
        parts.append(f"-c {LORA_YAML.name}")
    rel_data = Path(a.data)
    try:
        rel_data = rel_data.relative_to(HERE)
    except ValueError:
        rel_data = Path(cfg["data"]["dir"])
    parts += [f"--model {cfg['model']['repo']}", "--train", f"--data {rel_data}/", f"--fine-tune-type {a.fine_tune_type}",
              f"--num-layers {a.num_layers}"]
    if a.mask_prompt:
        parts.append("--mask-prompt")
    parts += [f"--iters {a.iters}", f"--batch-size {a.batch_size}", f"--learning-rate {fmt_lr(a.learning_rate)}",
              f"--steps-per-report {a.steps_per_report}", f"--steps-per-eval {a.steps_per_eval}", "--val-batches -1",
              f"--max-seq-length {a.max_seq_length}", f"--seed {a.seed}", f"--adapter-path {Path(a.adapter_path).relative_to(HERE)}",
              f"--save-every {a.save_every}"]
    return " \\\n    ".join(parts)


def bf16_lr(lr: float):
    try:
        import mlx.core as mx
        return float(mx.array(lr, dtype=mx.bfloat16).item())
    except Exception:
        return None


class Recorder:
    """mlx-lm 的 TrainingCallback：每次 report 多印一欄 peak memory（872ae88 的 CLI 畫面沒有），並記下來。"""

    def __init__(self, label: str):
        self.label = label
        self.train, self.val = [], []
        self.t0 = time.time()

    def on_train_loss_report(self, info: dict):
        rec = {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in info.items()}
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        self.train.append(rec)
        print(f"  {GREY}[{self.label}]{RESET} step {info['iteration']:>4}   train loss {BOLD}{info['train_loss']:.3f}{RESET}   "
              f"peak memory {BOLD}{info['peak_memory']:.2f} GB{RESET}   {info['tokens_per_second']:,.0f} tok/s   {rec['elapsed_s']:.0f} s")

    def on_val_loss_report(self, info: dict):
        rec = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}
        # mlx-lm 在第 it 步**之前**量 val、報 iteration = it - 1：第一次是 it = 1（還沒更新，記成 step 0），之後每 steps_per_eval 步一次；
        # 「step k」= 做完 k - 1 次更新之後。訓完 500 步之後沒有再量一次 val，最後的數字是 test.jsonl 上另外量的。
        rec["step"] = int(info["iteration"]) + 1 if int(info["iteration"]) > 0 else 0
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        self.val.append(rec)
        print(f"  {GREY}[{self.label}]{RESET} step {rec['step']:>4}   {AMBER}val loss {info['val_loss']:.3f}{RESET}   ({info['val_time']:.1f} s)")


def heldout_loss(model, tokenizer, cfg: dict) -> dict:
    """mlx-lm 自己的 evaluate：ChatDataset(test, mask_prompt=True) 的平均 loss（nats per response token）。與 demo 1 的 (c) 同一把尺。"""
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
    from mlx_lm.tuner.trainer import evaluate
    rows = read_jsonl(data_dir(cfg) / "test.jsonl")
    ds = CacheDataset(ChatDataset(rows, tokenizer, mask_prompt=True))
    t0 = time.time()
    loss = evaluate(model, ds, batch_size=int(cfg["train"]["batch_size"]), num_batches=-1, max_seq_length=int(cfg["train"]["max_seq_length"]))
    return {"loss": round(float(loss), 4), "n": len(rows), "mask_prompt": True, "seconds": round(time.time() - t0, 1)}


# ── 一個 run（子行程）────────────────────────────────────────────
def cmd_run(args, cfg):
    group, lr = args.group, (float(args.lr) if args.lr is not None else None)
    if group != "base" and lr is None:
        raise SystemExit(f"{RED}run --group {group} 要給 --lr{RESET}")
    iters = int(args.iters or cfg["train"]["iters"])
    name = "base" if group == "base" else run_name(group, lr)
    out_json = (PROBE_DIR if args.probe else RUN_DIR) / f"{name}{'.fake' if args.fake else ''}.json"
    rec = {"name": name, "group": group, "lr": lr, "iters": iters if group != "base" else 0, "probe": bool(args.probe),
           "date": now(), "versions": versions(), "seed": int(cfg["demo"]["seed"]), "model": cfg["model"]["repo"]}
    if args.fake:
        rec.update(fake_run(cfg, group, lr, iters, bool(args.probe)))
        rec["fake"] = True
        write_json(out_json, rec)
        print(f"  {GREEN}{name} done [FAKE]{RESET}: test response loss {rec.get('test', {}).get('loss')}")
        return
    import mlx.core as mx
    from mlx_lm.utils import load
    rec["device"] = device_info()
    path = model_path(cfg, offline=not args.online)
    rec["model_snapshot"] = Path(path).name
    reset_peak_memory()
    t0 = time.time()
    if group == "base":
        banner(f"base: held-out response loss on test.jsonl (the reference line)", AMBER)
        model, tokenizer = load(path)
        rec["load_seconds"] = round(time.time() - t0, 1)
        rec["test"] = heldout_loss(model, tokenizer, cfg)
        rec["peak_memory_gb"] = peak_memory_gb()
        print(f"  base  response loss {BOLD}{rec['test']['loss']:.3f}{RESET} nat/token   ({rec['test']['seconds']} s)")
        write_json(out_json, rec)
        note(f"→ {out_json.relative_to(HERE)}")
        return
    import numpy as np
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx.utils import tree_flatten
    out_dir = WEIGHTS / (f"probe_{name}" if args.probe else name)
    a = train_args(cfg, path, group, lr, iters, out_dir, bool(args.probe))
    rec["lr_bf16"] = bf16_lr(lr)
    banner(f"{'probe ' if args.probe else ''}{group}  lr {fmt_lr(lr)}  (bf16: {rec['lr_bf16']:.4g})  {iters} steps", AMBER)
    print(f"{DIM}{cli_line(cfg, a)}{RESET}\n")
    np.random.seed(a.seed)
    model, tokenizer = load(a.model)
    rec["load_seconds"] = round(time.time() - t0, 1)
    train_set, valid_set, _ = load_dataset(a, tokenizer)
    if args.probe:
        valid_set = []
    r = Recorder(name)
    t1 = time.time()
    train_model(a, model, train_set, valid_set, r)
    rec["train_seconds"] = round(time.time() - t1, 1)
    rec["n_trainable"] = int(sum(x.size for _, x in tree_flatten(model.trainable_parameters())))
    rec["n_train"], rec["n_valid"] = len(train_set), len(valid_set) if valid_set else 0
    rec["train_log"], rec["val_log"] = r.train, r.val
    rec["peak_memory_gb"] = peak_memory_gb()
    args_rec = {k: v for k, v in vars(a).items() if k != "model"}
    args_rec["data"] = cfg["data"]["dir"]                    # 紀錄進版控也上站：不留 Mac 的絕對路徑
    args_rec["adapter_path"] = str(out_dir.relative_to(HERE))
    rec["args"], rec["cli"] = args_rec, cli_line(cfg, a)
    rec["seconds_per_step"] = round(rec["train_seconds"] / iters, 2)
    if args.probe:
        first, last = r.train[0]["train_loss"], r.train[-1]["train_loss"]
        rec["diverged"] = probe_diverged(cfg, first, last)
        rec["weights_dir"] = None
        if out_dir.exists():
            shutil.rmtree(out_dir)   # probe 不留權重（full 的一份 2.8 GB）
    else:
        rec["test"] = heldout_loss(model, tokenizer, cfg)
        rec["weights_dir"] = str(out_dir.relative_to(HERE))
        print(f"\n  {GREEN}{name} done{RESET}: {rec['train_seconds']} s ({rec['seconds_per_step']} s/step), peak {rec['peak_memory_gb']:.2f} GB, "
              f"test response loss {BOLD}{rec['test']['loss']:.3f}{RESET}")
    write_json(out_json, rec)
    note(f"→ {out_json.relative_to(HERE)}")


def probe_diverged(cfg: dict, first: float, last: float) -> bool:
    """發散：NaN、最後比第一次高 diverge_nats 以上，或根本沒下來（最後仍高於 blowup_nats——第一次 report 已在 10 步之後，
    LR 太大時 loss 在那之前就炸了，first ≈ last 反而看起來「沒變」；2026-11-08 probe 的 full 1.7e-4 就是 8.1 → 7.9）。"""
    pb = cfg["probe"]
    return bool(math.isnan(last) or last > first + float(pb["diverge_nats"]) or last > float(pb.get("blowup_nats", 4.0)))


def fake_run(cfg: dict, group: str, lr, iters: int, probe: bool) -> dict:
    """--fake：U 形假資料，只為了演練流程。"""
    import random
    rng = random.Random(hash((group, lr)) & 0xFFFF)
    if group == "base":
        return {"test": {"loss": 2.717, "n": 200, "mask_prompt": True, "seconds": 0.1}, "peak_memory_gb": 3.9, "load_seconds": 0.1}
    c = {"full": -5.0, "lora": -4.5}[group]
    best = {"full": 1.62, "lora": 1.70}[group]
    L = best + 0.35 * (math.log10(lr) - c) ** 2 + rng.uniform(-0.01, 0.01)
    diverged = math.log10(lr) > c + 1.1
    spr = int(cfg["probe"]["steps_per_report"] if probe else cfg["train"]["steps_per_report"])
    log = []
    for it in range(spr, iters + 1, spr):
        frac = it / cfg["train"]["iters"]
        loss = (2.7 - (2.7 - L) * (1 - math.exp(-4 * frac))) if not diverged else 2.7 + 0.1 * it / spr
        log.append({"iteration": it, "train_loss": round(loss - 0.3, 3), "peak_memory": 24.5 if group == "full" else 11.0, "tokens_per_second": 600.0})
    val = [{"iteration": max(it - 1, 0), "step": it, "val_loss": round((2.7 - (2.7 - L) * (1 - math.exp(-4 * it / cfg["train"]["iters"]))) if not diverged else 2.7 + 0.1 * it / 25, 4)}
           for it in [0] + list(range(25, iters + 1, 25))] if not probe else []
    out = {"lr_bf16": lr, "train_log": log, "val_log": val, "train_seconds": 1.0, "seconds_per_step": 1.0 / max(iters, 1),
           "peak_memory_gb": 24.5 if group == "full" else 11.0, "n_trainable": 1_400_000_000 if group == "full" else 17_432_576,
           "cli": f"(fake) mlx_lm.lora ... --fine-tune-type {group} --learning-rate {fmt_lr(lr)}", "weights_dir": None}
    if probe:
        out["diverged"] = diverged
    else:
        out["test"] = {"loss": round(L if not diverged else 2.7 + 1.0, 4), "n": 200, "mask_prompt": True, "seconds": 0.1}
        out["weights_dir"] = str((WEIGHTS / run_name(group, lr)).relative_to(HERE))
    return out


def spawn_run(args, group: str, lr, iters=None, probe=False) -> None:
    """在子行程裡跑一個 run（同一個 interpreter；記憶體各自乾淨、掛掉不拖垮整個掃描）。"""
    cmd = [sys.executable, str(HERE / "demo.py")]
    if args.fake:
        cmd.append("--fake")
    if args.online:
        cmd.append("--online")
    cmd += ["run", "--group", group]
    if lr is not None:
        cmd += ["--lr", repr(float(lr))]
    if iters:
        cmd += ["--iters", str(iters)]
    if probe:
        cmd.append("--probe")
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit(f"{RED}子行程失敗（{group} lr {lr}）：exit {r.returncode}；修好之後重跑同一個指令，已完成的 run 會跳過{RESET}")


def run_json(group: str, lr, fake: bool, probe: bool = False) -> Path:
    name = "base" if group == "base" else run_name(group, lr)
    return (PROBE_DIR if probe else RUN_DIR) / f"{name}{'.fake' if fake else ''}.json"


# ── probe ────────────────────────────────────────────────────────
def cmd_probe(args, cfg):
    pb = cfg["probe"]
    steps = int(args.steps or pb["steps"])
    groups = [args.group] if args.group else list(GROUPS)
    banner(cfg["demo"]["title"] + f"  ·  probe: {steps} steps per learning rate" + ("  [FAKE]" if args.fake else ""))
    for g in groups:
        for lr in pb[f"{g}_lrs"]:
            p = run_json(g, lr, args.fake, probe=True)
            if p.exists() and not args.redo:
                note(f"  skip {p.name}（已有；--redo 重跑）")
                continue
            spawn_run(args, g, lr, iters=steps, probe=True)
    summarize_probe(cfg, args.fake, groups)


def summarize_probe(cfg, fake, groups):
    out = {"date": now(), "steps": None, "rule": f"diverged = NaN, or last > first + {cfg['probe']['diverge_nats']}, or last > {cfg['probe'].get('blowup_nats', 4.0)} nats", "groups": {}}
    print(f"\n  {'group':<6}{'lr':>10}{'bf16':>12}{'first':>9}{'last':>9}{'Δ':>8}{'s/step':>8}{'peak GB':>9}   verdict")
    est = {}
    for g in groups:
        rows = []
        for lr in cfg["probe"][f"{g}_lrs"]:
            p = run_json(g, lr, fake, probe=True)
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            out["steps"] = r["iters"]
            first, last = r["train_log"][0]["train_loss"], r["train_log"][-1]["train_loss"]
            div = probe_diverged(cfg, first, last)   # 用現在的規則重判，不信 json 裡存的
            verdict = f"{RED}diverged{RESET}" if div else (f"{GREEN}ok{RESET}" if last < first - 0.05 else f"{AMBER}flat{RESET}")
            print(f"  {g:<6}{fmt_lr(lr):>10}{(r.get('lr_bf16') or lr):>12.4g}{first:>9.3f}{last:>9.3f}{last - first:>+8.3f}{r.get('seconds_per_step', 0):>8.2f}{(r.get('peak_memory_gb') or 0):>9.2f}   {verdict}")
            rows.append({"lr": lr, "lr_bf16": r.get("lr_bf16"), "first": first, "last": last, "diverged": div,
                         "seconds_per_step": r.get("seconds_per_step"), "peak_memory_gb": r.get("peak_memory_gb")})
        out["groups"][g] = rows
        if rows:
            est[g] = sum(x["seconds_per_step"] or 0 for x in rows) / len(rows)
    n_runs = {g: len(cfg["sweep"][f"{g}_lrs"]) + (1 if cfg["sweep"]["cross"] != "none" else 0) for g in groups}
    total = sum(est.get(g, 0) * int(cfg["train"]["iters"]) * n_runs[g] for g in groups)
    print(f"\n  {DIM}estimate for the full sweep ({', '.join(f'{n_runs[g]} {g} runs' for g in groups)} × {cfg['train']['iters']} steps, "
          f"without the 25-step evals): {total / 60:.0f} min{RESET}")
    print(f"  {DIM}the two grids in demo_config.toml [sweep] are a draft: move them so the five points straddle each minimum and stop before 'diverged'.{RESET}")
    out["estimate_sweep_minutes"] = round(total / 60, 1)
    write_json(REHEARSAL / f"probe{'.fake' if fake else ''}.json", out)
    note(f"→ runs/rehearsal/probe{'.fake' if fake else ''}.json")


# ── rehearse ─────────────────────────────────────────────────────
def cmd_rehearse(args, cfg):
    sw = cfg["sweep"]
    groups = [args.group] if args.group else list(GROUPS)
    banner(cfg["demo"]["title"] + "  ·  rehearse " + " → ".join(groups) + ("  [FAKE]" if args.fake else ""))
    if not (data_dir(cfg) / "train.jsonl").exists():
        raise SystemExit(f"{RED}找不到 {cfg['data']['dir']}/train.jsonl：先在 demo 1 跑 ./present.sh prepare{RESET}")
    write_yaml(cfg)
    t0 = time.time()
    if not run_json("base", None, args.fake).exists() or args.redo:
        spawn_run(args, "base", None)
    for g in groups:
        for lr in sw[f"{g}_lrs"]:
            p = run_json(g, lr, args.fake)
            if p.exists() and not args.redo:
                note(f"  skip {p.name}（已有；--redo 重跑）")
                continue
            spawn_run(args, g, lr)
    # 交叉：LoRA 跑在 full 的最佳 LR（p49 最後一列）；both 時 full 也跑在 LoRA 的最佳 LR
    if sw["cross"] != "none" and not args.no_cross and set(groups) == set(GROUPS):
        best = {g: best_of(cfg, g, args.fake) for g in GROUPS}
        pairs = [("lora", best["full"]["lr"])] + ([("full", best["lora"]["lr"])] if sw["cross"] == "both" else [])
        for g, lr in pairs:
            if run_json(g, lr, args.fake).exists() and not args.redo:
                note(f"  cross {g} at lr {fmt_lr(lr)}: already in the grid")
                continue
            banner(f"cross run: {g} at the other group's best lr {fmt_lr(lr)}", AMBER)
            spawn_run(args, g, lr)
        if sw.get("peft_copy"):
            lr = peft_copy_lr(cfg, best["full"]["lr"])
            if not run_json("lora", lr, args.fake).exists() or args.redo:
                banner(f"peft-copy run: LoRA at full's best lr copied in α = 2r units → mlx lr {fmt_lr(lr)}", AMBER)
                spawn_run(args, "lora", lr)
    aggregate(cfg, args.fake)
    print(f"\n  {GREEN}rehearse done{RESET} in {(time.time() - t0) / 60:.0f} min")


def peft_copy_lr(cfg: dict, lr_full_best: float) -> float:
    """「照抄」：一篇用 α = 2r 的論文把 full 的最佳 LR 直接給 LoRA，換成 mlx 的 lr 就是 × 2 / scale（scale 20 → ÷ 10）。"""
    return float(lr_full_best) * 2.0 / float(cfg["lora"]["scale"])


def all_runs(cfg, fake) -> dict:
    """{group: {lr: rec}}，含交叉 run。"""
    out = {g: {} for g in GROUPS}
    for p in sorted(RUN_DIR.glob(f"*{'.fake' if fake else ''}.json")):
        if (p.name.endswith(".fake.json")) != fake or p.name.startswith("base"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("probe") or r["group"] not in out or "test" not in r:
            continue
        out[r["group"]][float(r["lr"])] = r
    return out


def metric_of(cfg, r: dict) -> float:
    if cfg["sweep"]["metric"] == "min" and r.get("val_log"):
        return min(v["val_loss"] for v in r["val_log"])
    return r["test"]["loss"]


def find_run(runs: dict, lr: float):
    return next((r for k, r in runs.items() if lr_in(k, [lr])), None)


def best_of(cfg, group: str, fake: bool) -> dict:
    """該組網格內（交叉 run 不算）metric 最低的 run。"""
    runs = all_runs(cfg, fake)[group]
    done = [(lr, find_run(runs, lr)) for lr in cfg["sweep"][f"{group}_lrs"]]
    done = [(lr, r) for lr, r in done if r is not None]
    if not done:
        raise SystemExit(f"{RED}{group} 還沒有任何完成的 run{RESET}")
    lr, r = min(done, key=lambda t: metric_of(cfg, t[1]))
    return {"lr": lr, "loss": r["test"]["loss"], "name": r["name"], "weights_dir": r.get("weights_dir")}


def aggregate(cfg, fake: bool) -> dict:
    sw = cfg["sweep"]
    runs = all_runs(cfg, fake)
    base_p = run_json("base", None, fake)
    base = json.loads(base_p.read_text(encoding="utf-8"))["test"]["loss"] if base_p.exists() else None
    scale_ratio = float(cfg["lora"]["scale"]) / 2.0   # PEFT 常見 α/r = 2 → ×10
    curves = {"date": now(), "fake": fake, "model": cfg["model"]["repo"], "data": cfg["data"]["dir"], "seed": int(cfg["demo"]["seed"]),
              "iters": int(cfg["train"]["iters"]), "batch_size": int(cfg["train"]["batch_size"]), "metric": sw["metric"],
              "lora": {"rank": cfg["lora"]["rank"], "scale": cfg["lora"]["scale"], "alpha_hu": cfg["lora"]["rank"] * cfg["lora"]["scale"],
                       "keys": cfg["lora"]["keys"], "embedding": False},
              "base_test_loss": base, "groups": {}, "best": {}, "cross": {}}
    for g in GROUPS:
        pts = []
        for lr in sorted(runs[g]):
            r = runs[g][lr]
            pts.append({"lr": lr, "lr_bf16": r.get("lr_bf16"), "in_grid": lr_in(lr, sw[f"{g}_lrs"]), "test_loss": r["test"]["loss"],
                        "val_min": min((v["val_loss"] for v in r.get("val_log", [])), default=None),
                        "final_train_loss": r["train_log"][-1]["train_loss"] if r.get("train_log") else None,
                        "val_curve": [[v["step"], v["val_loss"]] for v in r.get("val_log", [])],
                        "train_curve": [[t["iteration"], t["train_loss"]] for t in r.get("train_log", [])],
                        "train_seconds": r.get("train_seconds"), "seconds_per_step": r.get("seconds_per_step"),
                        "peak_memory_gb": r.get("peak_memory_gb"), "n_trainable": r.get("n_trainable"), "weights_dir": r.get("weights_dir"), "name": r["name"]})
        curves["groups"][g] = pts
        if any(p["in_grid"] for p in pts):
            curves["best"][g] = best_of(cfg, g, fake)
    if all(g in curves["best"] for g in GROUPS):
        bf, bl = curves["best"]["full"], curves["best"]["lora"]
        ratio = bl["lr"] / bf["lr"]
        curves["ratio"] = {"lr_lora_over_full_mlx_units": round(ratio, 3),
                           "lr_lora_over_full_peft_units_alpha_2r": round(ratio * scale_ratio, 3),
                           "note": f"mlx scale {cfg['lora']['scale']} vs. PEFT α/r = 2: multiply the mlx ratio by {scale_ratio:g} to compare with the blog's 10×"}
        lora_at_full = next((p for p in curves["groups"]["lora"] if lr_in(p["lr"], [bf["lr"]])), None)
        full_at_lora = next((p for p in curves["groups"]["full"] if lr_in(p["lr"], [bl["lr"]])), None)
        curves["cross"] = {"lr_full_best": bf["lr"], "lora_at_full_best": lora_at_full["test_loss"] if lora_at_full else None,
                           "full_at_full_best": bf["loss"],
                           "lr_lora_best": bl["lr"], "full_at_lora_best": full_at_lora["test_loss"] if full_at_lora else None,
                           "lora_at_lora_best": bl["loss"],
                           "gap_each_at_own_best": round(bl["loss"] - bf["loss"], 4),
                           "gap_both_at_full_best": round(lora_at_full["test_loss"] - bf["loss"], 4) if lora_at_full else None}
        lr_pc = peft_copy_lr(cfg, bf["lr"])
        lora_pc = next((p for p in curves["groups"]["lora"] if lr_in(p["lr"], [lr_pc])), None)
        curves["cross"].update({"lr_full_best_peft_copy_mlx": lr_pc,
                                "lora_at_full_best_peft_copy": lora_pc["test_loss"] if lora_pc else None,
                                "gap_peft_copy": round(lora_pc["test_loss"] - bf["loss"], 4) if lora_pc else None})
        curves["edge"] = {g: (curves["best"][g]["lr"] in (min(sw[f"{g}_lrs"]), max(sw[f"{g}_lrs"]))) for g in GROUPS}   # 最佳在網格邊緣 = 網格該移
    write_json(REHEARSAL / f"curves{'.fake' if fake else ''}.json", curves)
    note(f"→ runs/rehearsal/curves{'.fake' if fake else ''}.json")
    # best.json：demo 4 讀這個找「各自最佳 LR」的權重
    if curves["best"]:
        best = {"date": now(), "fake": fake, "model": cfg["model"]["repo"], "seed": curves["seed"],
                "note": "demo 4 reads weights_dir; LoRA needs mlx_lm.fuse before SVD. Paths relative to demos/w08_d3_lr_sweep/.",
                **{g: curves["best"][g] for g in curves["best"]}}
        write_json(REHEARSAL / f"best{'.fake' if fake else ''}.json", best)
        note(f"→ runs/rehearsal/best{'.fake' if fake else ''}.json")
        if cfg["weights"]["keep"] == "best" and not fake:
            prune_weights(curves)
    return curves


def prune_weights(curves: dict) -> None:
    keep = {curves["best"][g]["weights_dir"] for g in curves["best"]}
    for p in curves["groups"]["full"]:
        wd = p.get("weights_dir")
        if wd and wd not in keep and (HERE / wd).exists():
            shutil.rmtree(HERE / wd)
            note(f"  pruned {wd}（full FT 的非最佳 run，約 2.8 GB；demo_config [weights] keep = all 可以留）")


def cmd_aggregate(args, cfg):
    aggregate(cfg, args.fake)


# ── replay ───────────────────────────────────────────────────────
def text_plot(series: dict, width=64, height=12, xlabel="log10 lr", ylabel="loss") -> str:
    """純文字散點圖：series = {label: (marker, [(x, y), ...])}。"""
    xs = [x for _, pts in series.values() for x, _ in pts]
    ys = [y for _, pts in series.values() for _, y in pts if y is not None and not math.isnan(y)]
    if not xs or not ys:
        return "  (nothing to plot)"
    x0, x1 = min(xs) - 0.15, max(xs) + 0.15
    y0, y1 = min(ys) - 0.05, min(max(ys) + 0.05, min(ys) + 1.5)
    grid = [[" "] * width for _ in range(height)]
    for label, (mk, pts) in series.items():
        for x, y in pts:
            if y is None or math.isnan(y):
                continue
            yy = min(y, y1)
            c = int((x - x0) / (x1 - x0) * (width - 1))
            r = int((y1 - yy) / (y1 - y0) * (height - 1))
            grid[r][c] = mk if grid[r][c] == " " else "#"
    lines = []
    for i, row in enumerate(grid):
        yv = y1 - (y1 - y0) * i / (height - 1)
        lines.append(f"  {yv:5.2f} │{''.join(row)}")
    lines.append("        └" + "─" * width)
    ticks = sorted({round(x) for x in xs} | {math.floor(x0), math.ceil(x1)})
    tl = [" "] * (width + 16)
    for t in ticks:
        c = 9 + int((t - x0) / (x1 - x0) * (width - 1))
        s = f"1e{t}"
        if 0 <= c < len(tl) - len(s):
            for k, ch in enumerate(s):
                tl[c + k] = ch
    lines.append("".join(tl) + f"   ({xlabel})")
    return "\n".join(lines)


def cmd_replay(args, cfg):
    p = REHEARSAL / f"curves{'.fake' if args.fake else ''}.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    cv = json.loads(p.read_text(encoding="utf-8"))
    banner(f"REPLAY — rehearsed {cv['date']}, not live" + ("  [FAKE]" if args.fake else ""), AMBER)
    print(f"  {cv['model']}, {cv['data']}, seed {cv['seed']}, {cv['iters']} steps × batch {cv['batch_size']}; "
          f"LoRA r = {cv['lora']['rank']}, mlx scale {cv['lora']['scale']:g} (= Hu's α/r → α = {cv['lora']['alpha_hu']:g}), every Linear of every block, no embedding.")
    print(f"  loss = held-out response loss on test.jsonl (nats/token, mlx-lm evaluate, mask_prompt); base = {cv['base_test_loss']}")
    # 第一張圖：最終 loss 對 LR
    pause("\n⏎ figure 1: final loss vs. learning rate, two sweeps ")
    series = {}
    for g, mk in (("full", "F"), ("lora", "L")):
        pts = [(math.log10(q["lr"]), q["test_loss"]) for q in cv["groups"][g] if q["in_grid"]]
        series[g] = (mk, pts)
    print(text_plot(series))
    print(f"  F = full fine-tuning   L = LoRA   (# = both)\n")
    print(f"  {'':<6}" + "".join(f"{'lr ' + fmt_lr(q['lr']):>14}" for q in cv["groups"]["full"] if q["in_grid"]))
    for g in GROUPS:
        row = [q for q in cv["groups"][g] if q["in_grid"]]
        print(f"  {g:<6}" + "".join(f"{fmt_lr(q['lr']):>14}" for q in row) + "   ← lr")
        print(f"  {'':<6}" + "".join(f"{BOLD if cv['best'].get(g, {}).get('lr') == q['lr'] else ''}{q['test_loss']:>14.3f}{RESET}" for q in row) + "   ← test loss")
    if "ratio" in cv:
        bf, bl, rt = cv["best"]["full"], cv["best"]["lora"], cv["ratio"]
        print(f"\n  best lr: full {BOLD}{fmt_lr(bf['lr'])}{RESET} (loss {bf['loss']:.3f}),  LoRA {BOLD}{fmt_lr(bl['lr'])}{RESET} (loss {bl['loss']:.3f})")
        print(f"  ratio LoRA/full: {BOLD}{rt['lr_lora_over_full_mlx_units']:.2g}×{RESET} in mlx units (scale {cv['lora']['scale']:g}); "
              f"{BOLD}{rt['lr_lora_over_full_peft_units_alpha_2r']:.2g}×{RESET} in PEFT units (α = 2r) — the blog's claim is about 10×")
        for g, e in (cv.get("edge") or {}).items():
            if e:
                print(f"  {AMBER}{g}: the best point is at the edge of its grid — the minimum may lie outside; extend [sweep] {g}_lrs and re-run{RESET}")
    # 第二張圖：固定 LR 的比較
    pause("\n⏎ figure 2: both at full fine-tuning's best lr, vs. each at its own best ")
    cr = cv.get("cross", {})
    curves = {}
    def find(g, lr):
        return next((q for q in cv["groups"][g] if lr_in(q["lr"], [lr])), None)
    if cr:
        for label, g, lr in (("full @ own best", "full", cr["lr_full_best"]), ("LoRA @ own best", "lora", cr["lr_lora_best"]),
                             ("LoRA @ full's best", "lora", cr["lr_full_best"]), ("full @ LoRA's best", "full", cr["lr_lora_best"]),
                             ("LoRA @ full's best ÷10", "lora", cr.get("lr_full_best_peft_copy_mlx") or -1)):
            q = find(g, lr)
            if q and q.get("val_curve"):
                curves[label] = q
        steps = sorted({s for q in curves.values() for s, _ in q["val_curve"]})
        print(f"  val response loss (valid.jsonl, 100 rows) every {cv['iters'] // max(len(steps) - 1, 1)} steps\n")
        print(f"  {'step':>6}" + "".join(f"{lab:>22}" for lab in curves))
        for s in steps:
            print(f"  {s:>6}" + "".join(f"{dict((a, b) for a, b in q['val_curve']).get(s, float('nan')):>22.3f}" for q in curves.values()))
        finals = ["test %.3f" % q["test_loss"] for q in curves.values()]
        print(f"\n  {'':<6}" + "".join(f"{s:>22}" for s in finals) + f"   ← after {cv['iters']} steps, on test.jsonl")
    pause("\n⏎ the table on p49 ")
    if cr:
        print(f"\n  {'':<40}{'full fine-tuning':>18}{'LoRA, r = 16':>16}{'ratio / gap':>14}")
        print(f"  {'best learning rate':<40}{fmt_lr(cr['lr_full_best']):>18}{fmt_lr(cr['lr_lora_best']):>16}{cv['ratio']['lr_lora_over_full_mlx_units']:>13.2g}×")
        print(f"  {'loss, each at its own best':<40}{cr['full_at_full_best']:>18.3f}{cr['lora_at_lora_best']:>16.3f}{cr['gap_each_at_own_best']:>+14.3f}")
        la, gap = cr.get("lora_at_full_best"), cr.get("gap_both_at_full_best")
        la_s = "%.3f" % la if la is not None else "—"
        gap_s = "%+.3f" % gap if gap is not None else "—"
        print(f"  {'loss, both at full fine-tuning’s best':<40}{cr['full_at_full_best']:>18.3f}{la_s:>16}{gap_s:>14}")
        fa = cr.get("full_at_lora_best")
        if fa is not None:
            print(f"  {'(and full at LoRA’s best)':<40}{fa:>18.3f}{cr['lora_at_lora_best']:>16.3f}{fa - cr['lora_at_lora_best']:>+14.3f}")
        pc = cr.get("lora_at_full_best_peft_copy")
        if pc is not None:
            print(f"  {'LoRA at full’s best, copied in α = 2r':<40}{'(mlx lr ' + fmt_lr(cr['lr_full_best_peft_copy_mlx']) + ')':>18}{pc:>16.3f}{cr['gap_peft_copy']:>+14.3f}")
        print(f"\n  {DIM}one model, one dataset, one seed — not a replication. base = {cv['base_test_loss']}. "
              f"PEFT-unit ratio (α = 2r): {cv['ratio']['lr_lora_over_full_peft_units_alpha_2r']:.2g}×.{RESET}")


# ── inspect ──────────────────────────────────────────────────────
def n_lora_formula(cfgj: dict, r: int) -> int:
    d, h, kv, hd, ff, L = cfgj["hidden_size"], cfgj["num_attention_heads"], cfgj["num_key_value_heads"], cfgj["head_dim"], cfgj["intermediate_size"], cfgj["num_hidden_layers"]
    per = (d + h * hd) * r + 2 * (d + kv * hd) * r + (h * hd + d) * r + 3 * (d + ff) * r
    return per * L


def cmd_inspect(args, cfg):
    v = versions()
    banner(cfg["demo"]["title"] + "  ·  inspect" + ("  [FAKE]" if args.fake else ""))
    rec = {"date": now(), "versions": v, "device": device_info() if not args.fake else {}}
    print(f"  python {v['python']}  mlx {v.get('mlx')}  mlx-lm {v.get('mlx-lm')} @ {v.get('mlx_lm_commit', '?')[:7] if v.get('mlx_lm_commit') else '?'}  transformers {v.get('transformers')}")
    dd = data_dir(cfg)
    counts = {n: (len(read_jsonl(dd / f"{n}.jsonl")) if (dd / f"{n}.jsonl").exists() else None) for n in ("train", "valid", "test")}
    rec["data"] = {"dir": cfg["data"]["dir"], **counts}
    print(f"  data {cfg['data']['dir']}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ("" if all(counts.values()) else f"  {RED}← 先在 demo 1 跑 prepare{RESET}"))
    sw, scale_ratio = cfg["sweep"], float(cfg["lora"]["scale"]) / 2.0
    print(f"\n  {'group':<6}{'lr (mlx)':>12}{'as bf16':>12}{'PEFT units (α = 2r)':>22}")
    rec["grids"] = {}
    for g in GROUPS:
        rec["grids"][g] = []
        for lr in sw[f"{g}_lrs"]:
            b = bf16_lr(lr) if not args.fake else None
            peft = lr * scale_ratio if g == "lora" else None
            print(f"  {g:<6}{fmt_lr(lr):>12}{(f'{b:.4g}' if b is not None else '—'):>12}{(fmt_lr(peft) if peft else '—'):>22}")
            rec["grids"][g].append({"lr": lr, "lr_bf16": b, "lr_peft_units": peft})
    print(f"  {DIM}LoRA: mlx scale {cfg['lora']['scale']:g} = Hu's α/r → α = {cfg['lora']['rank'] * cfg['lora']['scale']:g}; "
          f"PEFT-unit lr = mlx lr × {scale_ratio:g}. cross = {sw['cross']}, metric = {sw['metric']}, keep weights = {cfg['weights']['keep']}{RESET}")
    if not args.fake:
        try:
            path = Path(model_path(cfg, offline=not args.online))
            cj = json.loads((path / "config.json").read_text(encoding="utf-8"))
            n_lora = n_lora_formula(cj, int(cfg["lora"]["rank"]))
            rec["model"] = {"snapshot": path.name, "N_LoRA_formula": n_lora, "layers": cj["num_hidden_layers"], "tie_word_embeddings": cj.get("tie_word_embeddings")}
            print(f"\n  model snapshot {path.name}: {cj['num_hidden_layers']} layers, tie_word_embeddings = {cj.get('tie_word_embeddings')}; N_LoRA (r = {cfg['lora']['rank']}, 7 Linear, all layers) = {n_lora:,}")
            dj = HERE / cfg["check"]["data_json"]
            if dj.exists():
                led = json.loads(dj.read_text(encoding="utf-8"))["ledger"][cfg["check"]["ledger_key"]]
                ok = led["N_LoRA"] == n_lora
                print(f"  w08-data.json {cfg['check']['ledger_key']}: N_LoRA {led['N_LoRA']:,} {'=' if ok else '≠'} formula  {GREEN + 'ok' if ok else RED + 'MISMATCH'}{RESET}; "
                      f"M_full {led['M_full_GB']:.1f} GB is the 16-byte ledger — mlx's Adam is pure bf16 (8 bytes/param, no master), so expect less")
                rec["check"] = {"N_LoRA_json": led["N_LoRA"], "ok": ok}
        except Exception as e:   # noqa: BLE001
            print(f"  {AMBER}model not readable offline: {e}{RESET}  （第一次先在 demo 1 跑 ./present.sh fetch）")
    print(f"\n  {DIM}full fine-tuning here = mlx-lm `--fine-tune-type full --num-layers -1`: every block unfrozen; embedding (tied LM head) and the final norm stay frozen "
          f"(lora.py train_model: `for l in model.layers[-max(num_layers, 0):]: l.unfreeze()`). LoRA also leaves them alone, so both sweeps move the same set of matrices.{RESET}")
    write_json(REHEARSAL / f"inspect{'.fake' if args.fake else ''}.json", rec)
    note(f"→ runs/rehearsal/inspect{'.fake' if args.fake else ''}.json")


# ── main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "probe", "rehearse", "run", "aggregate", "replay"])
    ap.add_argument("--fake", action="store_true", help="沒有 mlx 也能演練（輸出是假的）")
    ap.add_argument("--online", action="store_true", help="允許連網（預設 HF_HUB_OFFLINE）")
    ap.add_argument("--group", choices=["lora", "full", "base"], help="run：哪一組；probe／rehearse：只做一組")
    ap.add_argument("--lr", help="run：learning rate")
    ap.add_argument("--iters", type=int, help="run：步數（預設 [train].iters）")
    ap.add_argument("--probe", action="store_true", help="run：probe 模式（不存權重、不量 valid／test）")
    ap.add_argument("--steps", type=int, help="probe：每個 LR 跑幾步（預設 [probe].steps）")
    ap.add_argument("--redo", action="store_true", help="probe／rehearse：已有紀錄的 run 也重跑")
    ap.add_argument("--no-cross", action="store_true", help="rehearse：不做固定 LR 的交叉 run")
    a = ap.parse_args()
    cfg = load_cfg()
    {"inspect": cmd_inspect, "probe": cmd_probe, "rehearse": cmd_rehearse, "run": cmd_run, "aggregate": cmd_aggregate, "replay": cmd_replay}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
