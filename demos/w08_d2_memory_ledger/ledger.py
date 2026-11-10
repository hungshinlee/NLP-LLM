#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 2 (a)：三個數字——full fine-tuning、LoRA、QLoRA 的記憶體帳，從一份 config.json 算出來（投影片 p32 前導頁、p33）。

    ./present.sh count      課前：config.json 從哪裡讀到的、逐矩陣數 N、對 safetensors 的參數總數
    ./present.sh table      課堂：先印 N 與 N_LoRA，Enter 之後印四列（bytes/參數、GB、進不進得了 64 GB 與 16 GB）
    ./present.sh flops      課堂：每個 token 的 FLOPs，6N 對 4N + 6 N_LoRA
    ./present.sh mlx        同一本帳，照這台機器的框架實際存的 dtype 再算一次（demo 2 (b) 的三列 + 沒跑的那一列）
    ./present.sh files      本機已下載的權重檔：只讀 safetensors 的 header，依 dtype 加總 bytes（凍住的那一份在磁碟上多大）
    ./present.sh check      對投影片的數字來源 slides/assets/w08/w08-data.json 逐個核對（那個檔不在就跳過）

純標準函式庫，不載模型、不需要網路。config.json 依序找：Hugging Face 的本機快取 → 這個資料夾的 configs/ → demo_config.toml 的 [models.*.shape]。
GB 一律是 10^9 bytes。所有數字都**不含活化**與框架開銷——那是 (b) 量的東西。

兩本帳：
  教科書的混合精度帳（p25）：一個可訓練參數 16 bytes = bf16 權重 2 + 梯度 2 + fp32 master 4 + 兩個動量各 4；凍住的只剩權重的 2。
  mlx 實際存的（讀 mlx/optimizers 的 Adam 與 mlx-lm 的 LoRALinear）：Adam 的 m、v 用 zeros_like(parameter)，dtype 跟參數；沒有 fp32 master。
      bf16 的可訓練參數 = 2 + 2 + 2 + 2 = 8 bytes；LoRA 的 A、B 是 float32 → 4 + 4 + 4 + 4 = 16 bytes（剛好也是 16，但拆法不同）。
      mlx-lm 的 --fine-tune-type full 只 unfreeze model.layers：embedding、LM head、最後的 norm 維持凍住（2 bytes）。
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10 以下：pip install tomli
    import tomli as tomllib   # type: ignore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, note, now, read_lock, revision_key  # noqa: E402

CONFIG = HERE / "demo_config.toml"
REHEARSAL = HERE / "runs" / "rehearsal"
GB = 1e9
GATES = ("q", "k", "v", "o", "gate", "up", "down")
# 每參數的 bit 數：NF4 + double quantization（QLoRA 論文 §3：4 + 8/64 + 32/(64·256)）；mlx 的 affine 4-bit（group 64，每組一個 16-bit scale 與一個 16-bit bias）
BITS_NF4_DQ = 4 + 8 / 64 + 32 / (64 * 256)
BITS_MLX_4BIT = 4 + 32 / 64


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def pause(prompt: str = "") -> str:
    try:
        return input(f"{DIM}{prompt}{RESET}")
    except EOFError:
        return ""


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


# ── config.json 從哪裡來 ─────────────────────────────────────────
def hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser() / "hub"


def snapshot_dir(repo: str) -> Path | None:
    """本機快取裡這個 repo 的快照資料夾：versions.lock 鎖的 revision 優先，其次 refs/main，再其次任何一個。"""
    root = hf_cache() / ("models--" + repo.replace("/", "--"))
    cands = []
    rev = read_lock().get(revision_key(repo))
    if rev:
        cands.append(root / "snapshots" / rev)
    ref = root / "refs" / "main"
    if ref.exists():
        cands.append(root / "snapshots" / ref.read_text().strip())
    if (root / "snapshots").is_dir():
        cands += sorted(p for p in (root / "snapshots").iterdir() if p.is_dir())
    for c in cands:
        if (c / "config.json").exists():
            return c
    return None


def load_shape(cfg: dict, key: str) -> dict:
    """回傳 {repo, source, d, d_ff, n_layer, n_head, n_kv, d_head, V, tied, safetensors_total}。"""
    m = cfg["models"][key]
    repo = m["shape_repo"]
    snap = snapshot_dir(repo)
    local = HERE / "configs" / (repo.replace("/", "--") + ".json")
    if snap is not None:
        c, source = json.loads((snap / "config.json").read_text(encoding="utf-8")), f"Hugging Face cache · snapshot {snap.name[:10]}"
    elif local.exists():
        c, source = json.loads(local.read_text(encoding="utf-8")), f"configs/{local.name} (a saved copy)"
    else:
        c, source = dict(m["shape"]), "demo_config.toml [models.%s.shape] (a saved copy)" % key
    return dict(repo=repo, source=source, d=int(c["hidden_size"]), d_ff=int(c["intermediate_size"]), n_layer=int(c["num_hidden_layers"]),
                n_head=int(c["num_attention_heads"]), n_kv=int(c["num_key_value_heads"]), d_head=int(c["head_dim"]),
                V=int(c["vocab_size"]), tied=bool(c["tie_word_embeddings"]), safetensors_total=int(m["shape"]["safetensors_total"]),
                safetensors_note=m["shape"].get("note", ""))


# ── 數參數 ───────────────────────────────────────────────────────
def linears(s: dict) -> dict:
    """一個 block 的七個 Linear：(d_in, d_out)。Qwen3：都沒有 bias；k、v 是 n_kv 個 head（GQA）。"""
    d, dff, nh, nkv, dh = s["d"], s["d_ff"], s["n_head"], s["n_kv"], s["d_head"]
    return {"q": (d, nh * dh), "k": (d, nkv * dh), "v": (d, nkv * dh), "o": (nh * dh, d),
            "gate": (d, dff), "up": (d, dff), "down": (dff, d)}


def count(s: dict) -> dict:
    lin = linears(s)
    per_lin = sum(a * b for a, b in lin.values())
    per_norm = 2 * s["d"] + 2 * s["d_head"]            # 兩個 RMSNorm（d）＋ q_norm、k_norm（d_head）
    emb = s["V"] * s["d"]
    head = 0 if s["tied"] else emb
    final_norm = s["d"]
    n_blocks = s["n_layer"] * (per_lin + per_norm)
    N = n_blocks + emb + head + final_norm
    return dict(linears=lin, per_block_linear=per_lin, per_block_norm=per_norm, N_blocks=n_blocks, N_blocks_linear=s["n_layer"] * per_lin,
                N_embedding=emb, N_lm_head=head, N_final_norm=final_norm, N_norms=s["n_layer"] * per_norm + final_norm,
                N=N, matches_safetensors=(N == s["safetensors_total"]))


def n_lora(s: dict, r: int, modules=GATES, layers: int | None = None) -> int:
    lin = linears(s)
    L = s["n_layer"] if layers is None else layers
    return L * sum(r * (lin[k][0] + lin[k][1]) for k in modules)


def ledger(s: dict, r: int = 16, modules=GATES, layers: int | None = None) -> dict:
    """教科書的混合精度帳（p25）：可訓練 16、凍住 2。"""
    c = count(s)
    N, NL = c["N"], n_lora(s, r, modules, layers)
    ad = 16 * NL / GB
    rows = {
        "full": dict(label="full fine-tuning", bpp="16", frozen_GB=0.0, train_GB=16 * N / GB),
        "lora": dict(label=f"LoRA, r = {r}, bf16 base", bpp="2 + 16·N_LoRA/N", frozen_GB=2 * N / GB, train_GB=ad),
        "qlora_nf4": dict(label="QLoRA, NF4 + double quantization", bpp="b_q + 16·N_LoRA/N", train_GB=ad,
                          frozen_GB=(c["N_blocks_linear"] * BITS_NF4_DQ / 8 + (N - c["N_blocks_linear"]) * 2) / GB,
                          b_q=BITS_NF4_DQ / 8, note="block Linears at 4.127 bits; embedding, LM head, norms stay bf16"),
        "qlora_mlx": dict(label="QLoRA, mlx's 4-bit", bpp="b_q + 16·N_LoRA/N", train_GB=ad,
                          frozen_GB=((N - c["N_norms"]) * BITS_MLX_4BIT / 8 + c["N_norms"] * 2) / GB,
                          b_q=BITS_MLX_4BIT / 8, note="everything but the norms at 4.5 bits, embedding and LM head included"),
    }
    for row in rows.values():
        row["GB"] = row["frozen_GB"] + row["train_GB"]
        row["fits_64"], row["fits_16"] = row["GB"] <= 64, row["GB"] <= 16
    return dict(r=r, modules=list(modules), layers=s["n_layer"] if layers is None else layers, N=N, N_LoRA=NL, N_LoRA_pct=100 * NL / N, rows=rows)


def flops(s: dict, r: int = 16, modules=GATES, layers: int | None = None) -> dict:
    N, NL = count(s)["N"], n_lora(s, r, modules, layers)
    full, lora = 6 * N, 2 * N + 2 * N + 6 * NL
    return dict(full_per_token=full, lora_per_token=lora, ratio=lora / full, forward=2 * N, backward_input_grad=2 * N,
                backward_weight_grad_skipped=2 * N, adapter=6 * NL)


def mlx_ledger(s: dict, kind: str, r: int = 16, modules=GATES, layers: int | None = None) -> dict:
    """這台機器的框架實際存的（不含活化）。kind: full | lora | qlora。回傳 bytes 的拆解。"""
    c = count(s)
    N, NL = c["N"], n_lora(s, r, modules, layers)
    if kind == "full":      # mlx-lm 的 full：只有 model.layers 可訓練，bf16 的 權重＋梯度＋m＋v = 8；其餘凍住 2
        frozen, train, n_train = 2 * (N - c["N_blocks"]), 8 * c["N_blocks"], c["N_blocks"]
        how = "8 bytes x the blocks (bf16 weight + gradient + m + v) + 2 x the frozen embedding, LM head, final norm"
    elif kind == "lora":
        frozen, train, n_train = 2 * N, 16 * NL, NL
        how = "2 bytes x N (bf16, frozen) + 16 x N_LoRA (float32 weight + gradient + m + v)"
    elif kind == "qlora":
        frozen, train, n_train = (N - c["N_norms"]) * BITS_MLX_4BIT / 8 + c["N_norms"] * 2, 16 * NL, NL
        how = "4.5 bits x N (mlx affine 4-bit, frozen) + 16 x N_LoRA (float32 weight + gradient + m + v)"
    else:
        raise ValueError(kind)
    return dict(kind=kind, frozen_GB=frozen / GB, train_GB=train / GB, GB=(frozen + train) / GB, n_trainable=n_train, how=how,
                textbook_GB=(16 * N if kind == "full" else (frozen + 16 * NL)) / GB)


# ── safetensors 的 header（只讀 header，不讀權重）─────────────────
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U32": 4, "I32": 4, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "I64": 8, "U64": 8, "BOOL": 1}


def safetensors_summary(folder: Path) -> dict | None:
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        return None
    by, n_tensors, total = {}, 0, 0
    for f in files:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            head = json.loads(fh.read(n))
        for name, t in head.items():
            if name == "__metadata__":
                continue
            nbytes = t["data_offsets"][1] - t["data_offsets"][0]
            kind = name.rsplit(".", 1)[-1] if name.rsplit(".", 1)[-1] in ("scales", "biases") else "weights"
            k = f"{t['dtype']} {kind}"
            e = by.setdefault(k, {"bytes": 0, "elements": 0, "tensors": 0})
            e["bytes"] += nbytes
            e["elements"] += nbytes // DTYPE_BYTES.get(t["dtype"], 1)
            e["tensors"] += 1
            n_tensors += 1
            total += nbytes
    return dict(files=len(files), tensors=n_tensors, bytes=total, GB=total / GB, by_dtype=by)


# ── 畫面 ─────────────────────────────────────────────────────────
def fit(ok: bool) -> str:
    return f"{GREEN}yes{RESET}" if ok else f"{RED}no {RESET}"


def bar(gb: float, lines=(16, 64), top=140.0, width=56) -> str:
    """0 … top GB 的橫條，兩條線的位置用 │ 標出來。"""
    n = min(width, max(1, round(gb / top * width)))
    cells = ["█" if i < n else " " for i in range(width)]
    for x in lines:
        i = min(width - 1, round(x / top * width))
        cells[i] = f"{AMBER}│{RESET}" if cells[i] == " " else f"{AMBER}┃{RESET}"
    return "".join(cells) + (" →" if gb > top else "")


def pick(cfg: dict, args) -> tuple[str, dict, int, tuple, int | None]:
    key = args.model or cfg["ledger"]["model"]
    if key not in cfg["models"]:
        raise SystemExit(f"{RED}demo_config.toml 沒有 [models.{key}]；有的是 {', '.join(cfg['models'])}{RESET}")
    r = int(args.rank or cfg["ledger"]["rank"])
    mods = tuple((args.modules or ",".join(cfg["ledger"]["modules"])).split(","))
    layers = args.layers
    return key, load_shape(cfg, key), r, mods, layers


def head_line(s: dict) -> None:
    print(f"  {BOLD}{s['repo']}{RESET}   {GREY}config.json ← {s['source']}{RESET}")
    print(f"  {GREY}d = {s['d']}, d_ff = {s['d_ff']}, {s['n_layer']} blocks, {s['n_head']} heads ({s['n_kv']} KV heads), d_head = {s['d_head']}, "
          f"V = {s['V']:,}, embedding {'tied to' if s['tied'] else 'separate from'} the LM head{RESET}")


def cmd_count(args, cfg) -> dict:
    key, s, r, mods, layers = pick(cfg, args)
    c = count(s)
    banner(cfg["demo"]["title"] + "  ·  count N, matrix by matrix")
    head_line(s)
    print(f"\n  {'one block':<34}{'d_in × d_out':>18}{'parameters':>18}")
    for k, (a, b) in c["linears"].items():
        print(f"    {k + '_proj':<32}{f'{a:,} × {b:,}':>18}{a * b:>18,}")
    print(f"    {'4 norms (2 × d, 2 × d_head)':<32}{'':>18}{c['per_block_norm']:>18,}")
    print(f"  {'× ' + str(s['n_layer']) + ' blocks':<34}{'':>18}{c['N_blocks']:>18,}")
    vd = f"{s['V']:,} × {s['d']:,}"
    print(f"  {'embedding  V × d':<34}{vd:>18}{c['N_embedding']:>18,}")
    print(f"  {'LM head' + (' (tied: no new parameters)' if s['tied'] else '  d × V'):<34}{'':>18}{c['N_lm_head']:>18,}")
    print(f"  {'final norm':<34}{'':>18}{c['N_final_norm']:>18,}")
    ok = c["matches_safetensors"]
    print(f"  {BOLD}{'N':<34}{'':>18}{c['N']:>18,}{RESET}")
    print(f"  {'safetensors total (Hub API)':<34}{'':>18}{s['safetensors_total']:>18,}   {GREEN + 'equal' if ok else RED + 'DIFFERENT'}{RESET}")
    return dict(model=key, shape=s, count=c)


def cmd_table(args, cfg) -> dict:
    key, s, r, mods, layers = pick(cfg, args)
    L = ledger(s, r, mods, layers)
    banner(cfg["demo"]["title"] + "  ·  three numbers from one config.json")
    head_line(s)
    print(f"\n  N      = {BOLD}{L['N']:>14,}{RESET}   counted matrix by matrix" + ("" if count(s)["matches_safetensors"] else f"   {RED}≠ safetensors total{RESET}"))
    print(f"  N_LoRA = {BOLD}{L['N_LoRA']:>14,}{RESET}   r = {r} on {len(mods)} gate{'s' if len(mods) > 1 else ''} ({', '.join(mods)}) of {L['layers']} blocks"
          f"   = {BOLD}{L['N_LoRA_pct']:.3f} %{RESET} of N")
    if not args.all:
        pause("\n  Enter → the four rows ")
    print(f"\n  {'':<34}{'bytes per parameter':<21}{'frozen + trained':>18}{'GB':>9}   {'64 GB':<6}{'16 GB':<6}")
    for row in L["rows"].values():
        parts = f"{row['frozen_GB']:.2f} + {row['train_GB']:.2f}" if row["frozen_GB"] else ""
        print(f"  {row['label']:<34}{row['bpp']:<21}{parts:>18}{BOLD}{row['GB']:>9.1f}{RESET}   {fit(row['fits_64'])}   {fit(row['fits_16'])}")
    print(f"\n  {GREY}0 … 140 GB; the two marks are 16 GB and 64 GB{RESET}")
    for row in L["rows"].values():
        print(f"  {row['label']:<34}{bar(row['GB'])}")
    print(f"\n  {GREY}b_q: {BITS_NF4_DQ:.3f} bits = {BITS_NF4_DQ / 8:.3f} bytes (NF4 + double quantization; embedding, LM head and norms left in bf16)")
    print(f"       {BITS_MLX_4BIT:.1f} bits = {BITS_MLX_4BIT / 8:.4f} bytes (mlx's affine 4-bit, group 64; embedding and LM head quantized too)")
    print(f"  all bytes before a single activation · GB = 10^9 bytes{RESET}")
    return dict(model=key, shape=s, ledger=L)


def cmd_flops(args, cfg) -> dict:
    key, s, r, mods, layers = pick(cfg, args)
    F = flops(s, r, mods, layers)
    banner(cfg["demo"]["title"] + "  ·  FLOPs per token (no attention term, like 6N)")
    head_line(s)
    print(f"\n  full fine-tuning   2N forward + 4N backward                        = 6N            = {BOLD}{F['full_per_token']:.3e}{RESET}")
    print(f"  LoRA, r = {r:<3}      2N forward + 2N backward (input gradients only) + 6·N_LoRA    = {BOLD}{F['lora_per_token']:.3e}{RESET}")
    print(f"  {GREY}skipped: the 2N for the frozen weights' gradients · the side roads' own cost: {F['adapter']:.2e}{RESET}")
    print(f"\n  ratio  LoRA / full = {BOLD}{F['ratio']:.3f}{RESET}   (two thirds would be 0.667)")
    return dict(model=key, flops=F)


def cmd_mlx(args, cfg) -> dict:
    """(b) 的三個設定 + 沒跑的那一列：教科書的帳與 mlx 實際存的各算一次。"""
    banner(cfg["demo"]["title"] + "  ·  the same ledger, the way mlx stores it (no activations)")
    print(f"  {GREY}textbook (mixed precision): trainable 16 = bf16 weight 2 + gradient 2 + fp32 master 4 + m 4 + v 4; frozen 2")
    print(f"  mlx: Adam's m and v take the parameter's dtype, and there is no fp32 master copy →")
    print(f"       a trainable bf16 weight is 2 + 2 + 2 + 2 = 8; a LoRA factor is float32: 4 + 4 + 4 + 4 = 16")
    print(f"       mlx-lm's full fine-tuning unfreezes the blocks only (embedding, LM head, final norm stay frozen){RESET}\n")
    print(f"  {'run':<20}{'model':<26}{'textbook':>10}{'mlx':>9}   {'= frozen + trained':<22}{'trainable':>16}")
    out = {}
    for name, spec in cfg["measure"]["settings"].items():
        s = load_shape(cfg, spec["model"])
        M = mlx_ledger(s, spec["kind"], int(cfg["ledger"]["rank"]))
        out[name] = dict(M, model=spec["model"], repo=cfg["models"][spec["model"]]["repo"], run=bool(spec.get("run", True)))
        tag = "" if spec.get("run", True) else f"   {AMBER}not run{RESET}"
        print(f"  {spec['label']:<20}{cfg['models'][spec['model']]['short']:<26}{M['textbook_GB']:>8.1f} GB{BOLD}{M['GB']:>7.1f} GB{RESET}   "
              f"{M['frozen_GB']:>6.2f} + {M['train_GB']:<13.2f}{M['n_trainable']:>16,}{tag}")
    print(f"\n  {GREY}GB = 10^9 bytes · these are the 'formula' numbers that part (b) compares its measured peaks with{RESET}")
    return out


def cmd_files(args, cfg) -> dict:
    banner(cfg["demo"]["title"] + "  ·  the weight files on this disk (safetensors headers only)")
    out = {}
    for key, m in cfg["models"].items():
        snap = snapshot_dir(m["repo"])
        summ = safetensors_summary(snap) if snap else None
        if summ is None:
            print(f"  {m['repo']:<36}{GREY}not in the local cache ({hf_cache()}){RESET}")
            continue
        out[key] = dict(summ, repo=m["repo"], snapshot=snap.name)
        print(f"  {BOLD}{m['repo']:<36}{RESET}{summ['GB']:>8.2f} GB   {summ['tensors']:,} tensors in {summ['files']} file{'s' if summ['files'] > 1 else ''}")
        for k, e in sorted(summ["by_dtype"].items(), key=lambda kv: -kv[1]["bytes"]):
            print(f"      {k:<18}{e['bytes'] / GB:>8.3f} GB   {e['elements']:>16,} elements   {e['tensors']:>5} tensors")
        s = load_shape(cfg, key)
        c = count(s)
        pred = (2 * c["N"] if m.get("bits", 16) == 16 else (c["N"] - c["N_norms"]) * BITS_MLX_4BIT / 8 + c["N_norms"] * 2) / GB
        print(f"      {GREY}the ledger's frozen copy: {pred:.2f} GB → files are {summ['GB'] - pred:+.3f} GB off{RESET}")
        out[key]["ledger_frozen_GB"] = pred
    return out


def cmd_check(args, cfg) -> dict:
    p = (HERE / cfg["check"]["data_json"]).resolve()
    banner(cfg["demo"]["title"] + "  ·  check against the slides' numbers")
    if not p.exists():
        note(f"  {cfg['check']['data_json']} is not here (it lives in the course's private repository); nothing to check.")
        return {"skipped": True}
    d = json.loads(p.read_text(encoding="utf-8"))
    pairs = []
    for key, dkey, lkey in (("qwen3_8b", "qwen3_8b", "qwen3_8b_r16_all"), ("qwen3_1p7b_base", "qwen3_1p7b_base", "qwen3_1p7b_base_r16_all")):
        s = load_shape(cfg, key)
        c, L = count(s), ledger(s, 16)
        ref_c, ref_l = d["configs"][dkey], d["ledger"][lkey]
        pairs += [(f"{key} N", c["N"], ref_c["N"]), (f"{key} N_blocks_linear", c["N_blocks_linear"], ref_c["N_blocks_linear"]),
                  (f"{key} N_LoRA", L["N_LoRA"], ref_l["N_LoRA"]), (f"{key} M_full", L["rows"]["full"]["GB"], ref_l["M_full_GB"]),
                  (f"{key} M_LoRA", L["rows"]["lora"]["GB"], ref_l["M_lora_GB"]), (f"{key} M_QLoRA NF4", L["rows"]["qlora_nf4"]["GB"], ref_l["M_qlora_nf4_GB"]),
                  (f"{key} M_QLoRA mlx", L["rows"]["qlora_mlx"]["GB"], ref_l["M_qlora_mlx_GB"])]
    F = flops(load_shape(cfg, "qwen3_8b"), 16)
    ref_f = d["flops"]["qwen3_8b_r16_all"]
    pairs += [("flops full", F["full_per_token"], ref_f["full_per_token"]), ("flops lora", F["lora_per_token"], ref_f["lora_per_token"]), ("flops ratio", F["ratio"], ref_f["ratio"])]
    for name, spec in cfg["measure"]["settings"].items():     # mlx 實際存的那一本帳（舊版的 json 沒有這個鍵就跳過）
        ref_m = d["ledger"].get("as_mlx_stores_it", {}).get(name)
        if ref_m:
            M = mlx_ledger(load_shape(cfg, spec["model"]), spec["kind"], int(cfg["ledger"]["rank"]))
            pairs += [(f"as mlx stores it: {name}", M["GB"], ref_m["GB"]), (f"  trainable: {name}", M["n_trainable"], ref_m["n_trainable"])]
    bad = 0
    for name, a, b in pairs:
        ok = abs(a - b) <= 1e-9 * max(1.0, abs(b))
        bad += not ok
        print(f"  {GREEN + 'ok ' if ok else RED + 'BAD'}{RESET}  {name:<34}{a:>22,.6f}{b:>22,.6f}")
    print(f"\n  {len(pairs) - bad}/{len(pairs)} equal")
    return {"n": len(pairs), "bad": bad}


def cmd_rehearse(args, cfg) -> None:
    args.all = True
    rec = {"date": now(), "python": sys.version.split()[0]}
    rec["count"] = {k: cmd_count(argparse.Namespace(**{**vars(args), "model": k}), cfg) for k in cfg["models"] if cfg["models"][k].get("ledger", True)}
    rec["table"] = cmd_table(args, cfg)
    rec["flops"] = cmd_flops(args, cfg)
    rec["mlx"] = cmd_mlx(args, cfg)
    rec["files"] = cmd_files(args, cfg)
    rec["check"] = cmd_check(args, cfg)
    write_json(REHEARSAL / "ledger.json", rec)
    note(f"\n→ runs/rehearsal/ledger.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="W8 demo 2 (a): the memory ledger from one config.json")
    ap.add_argument("cmd", choices=["count", "table", "flops", "mlx", "files", "check", "rehearse"])
    ap.add_argument("--model", help="demo_config.toml 的 [models.*] 鍵（預設 [ledger] model）")
    ap.add_argument("--rank", type=int, help="LoRA 的 r（預設 [ledger] rank）")
    ap.add_argument("--modules", help="逗號分隔：q,k,v,o,gate,up,down 的子集（預設全部七個）")
    ap.add_argument("--layers", type=int, help="只接最後幾個 block（預設全部；mlx-lm 的預設是 16）")
    ap.add_argument("--all", action="store_true", help="table 不停下來等 Enter")
    args = ap.parse_args()
    cfg = load_cfg()
    {"count": cmd_count, "table": cmd_table, "flops": cmd_flops, "mlx": cmd_mlx, "files": cmd_files, "check": cmd_check, "rehearse": cmd_rehearse}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
