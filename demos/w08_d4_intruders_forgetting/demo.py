#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 4：旁路留下了什麼——intruder dimensions 與遺忘（投影片 p54 前導頁、p55 live、p58 replay）。

    ./present.sh inspect        課前：版本、demo 3 的兩個權重資料夾在不在、矩陣鍵名與形狀對得上、四段 held-out 的 byte 數 → runs/rehearsal/inspect.json
    ./present.sh intruders      課堂 (a)：對 [intruders].matrices 的每個矩陣，數 full 與 LoRA 各有幾個 intruder dimension（p55 的表）。
                                函數在 intruders_live.py（現場寫）；--ref 用 intruders_final.py；--rehearsal 把結果寫進 runs/rehearsal/intruders.json
    ./present.sh scan           彩排前：所有層 × [scan].names 的 intruder 數（用來挑 p55 要看的那一層）→ runs/rehearsal/scan.json
    ./present.sh forget         彩排 (b)：base／LoRA／full 在四段 held-out 上的 bits-per-byte → runs/rehearsal/forgetting.json
    ./present.sh replay         課堂 (b)：讀 forgetting.json，Enter 一列印 p58 的表（不載模型）；--intruders 改印 intruders.json
    ./present.sh rehearse       彩排全套：inspect → intruders --ref --rehearsal → forget（--intervene 加做介入）
    ./present.sh intervene      可選（若時間）：把 LoRA 模型 intruder 方向的奇異值乘 λ，再量 (b)——回升多少 → runs/rehearsal/intervene.json
    ./present.sh fake ...       沒有 mlx 也能演練流程（輸出是假的，畫面會標明）

(a) 不需要 mlx：權重直接從 safetensors 讀（bf16 → float32 → float64），SVD 走 numpy（CPU）。LoRA 的 W' = W0 + scale · (lora_a @ lora_b)^T
    在 float64 裡算——不是 mlx_lm.fuse 存回去的 bf16 版本；兩者的差（bf16 捨掉了多少 Δ）程式另外印。
(b) 走 mlx_lm.load：base、base + full adapters（整層的 bf16 權重）、base + LoRA adapters（不合併，與 demo 3 量 test loss 時同一條路）。
    每段文字切成不重疊的視窗、前面放一個 <|endoftext|>，bits-per-byte = Σ NLL / ln 2 / UTF-8 bytes——W1 demo 2 與 W7 demo 3 的那把尺。

相依：numpy；(b) 另需 mlx、mlx-lm（git commit 釘在 ../versions.lock）、transformers（tokenizer）、huggingface_hub。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10：pip install tomli
    import tomli as tomllib   # type: ignore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info,  # noqa: E402
                    locked_revision, note, now, peak_memory_gb, reset_peak_memory, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
LIVE = RUNS / "live"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]
RUNS_ORDER = ("full", "lora")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def pause(prompt: str = "") -> str:
    try:
        return input(f"{DIM}{prompt}  [Enter] 繼續，q 離開{RESET} ").strip().lower()
    except EOFError:
        return ""


def rel(p: Path) -> str:
    """進紀錄的路徑一律相對 demos/（紀錄會上站，不留 Mac 的絕對路徑）。"""
    try:
        return str(p.resolve().relative_to(HERE.parent))
    except ValueError:
        return p.name


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 權重在哪 ─────────────────────────────────────────────────────
def resolve_weights(cfg: dict) -> dict:
    """demo 3 的 best.json → {"full": {"dir": Path, "lr", "loss", "name"}, "lora": {...}}；[weights].full_dir／lora_dir 可覆蓋。"""
    w = cfg["weights"]
    out, best = {}, {}
    bp = (HERE / w["best_json"]).resolve()
    if bp.exists():
        best = read_json(bp)
    for g in RUNS_ORDER:
        if w.get(f"{g}_dir"):
            d = (HERE / w[f"{g}_dir"]).resolve()
            info = {"dir": d, "lr": None, "loss": None, "name": d.name, "from": "demo_config.toml"}
        elif g in best:
            d = (bp.parent.parent.parent / best[g]["weights_dir"]).resolve()
            info = {"dir": d, "lr": best[g].get("lr"), "loss": best[g].get("loss"), "name": best[g].get("name"), "from": rel(bp)}
        else:
            raise SystemExit(f"{RED}找不到 {g} 的權重：{bp} 沒有這一組，也沒填 [weights].{g}_dir。先在 ../w08_d3_lr_sweep 跑完 rehearse。{RESET}")
        out[g] = info
    out["base_test_loss"] = None
    base_json = bp.parent / "runs" / "base.json"
    if base_json.exists():
        out["base_test_loss"] = read_json(base_json).get("test", {}).get("loss")
    return out


def model_path(cfg: dict, offline: bool = True) -> str:
    m = cfg["model"]
    rev = locked_revision(m["repo"], m.get("revision"))
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(m["repo"], revision=rev, local_files_only=offline, ignore_patterns=NON_MODEL_FILES)
    except ImportError:
        import os
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        p = home / "hub" / ("models--" + m["repo"].replace("/", "--")) / "snapshots" / (rev or "")
        if not p.exists():
            raise SystemExit(f"{RED}沒有 huggingface_hub，也找不到快照 {p}{RESET}")
        return str(p)


# ── safetensors：不靠 mlx 讀 bf16 ──────────────────────────────────
_DT = {"BF16": ("bf16", 2), "F16": ("<f2", 2), "F32": ("<f4", 4), "F64": ("<f8", 8)}


def st_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_read(path: Path, keys: list[str]) -> dict:
    """只讀要的那幾個張量（float64）。bf16 → 左移 16 bit 變 float32，無損。"""
    import numpy as np
    header, base = st_header(path)
    out = {}
    with open(path, "rb") as f:
        for k in keys:
            h = header[k]
            kind, _ = _DT[h["dtype"]]
            a, b = h["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if kind == "bf16":
                x = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                x = np.frombuffer(raw, dtype=kind)
            out[k] = x.astype(np.float64).reshape(h["shape"])
    return out


class BaseWeights:
    """base 模型的 safetensors（一個檔或 index 分片）。只讀要的鍵。"""

    def __init__(self, snapshot: str):
        self.dir = Path(snapshot)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.map = {k: self.dir / v for k, v in read_json(idx)["weight_map"].items()}
        else:
            single = self.dir / "model.safetensors"
            self.map = {k: single for k in st_header(single)[0] if k != "__metadata__"}

    def shape(self, key: str) -> list[int]:
        return st_header(self.map[key])[0][key]["shape"]

    def get(self, key: str):
        return st_read(self.map[key], [key])[key]


def adapter_config(d: Path) -> dict:
    return read_json(d / "adapter_config.json")


def fused_lora(W0, lora_a, lora_b, scale: float):
    """mlx-lm 的 LoRALinear：y = W0 x + scale · (x @ lora_a @ lora_b) → W' = W0 + scale · (lora_a @ lora_b)^T，d_out × d_in。"""
    return W0 + scale * (lora_a @ lora_b).T


def to_bf16(x):
    """numpy 模擬 bf16 的 round-to-nearest-even（看 bf16 合併會把多少 Δ 捨掉）。"""
    import numpy as np
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.view(np.float32).astype(np.float64)


def matrix_key(layer: int, name: str) -> str:
    return f"model.layers.{layer}.{name}"


def load_matrices(cfg: dict, weights: dict, layer: int, name: str, fake: bool) -> dict:
    """{W0, full: W', lora: W', lora_delta, full_delta, scale}，全是 float64 的 d_out × d_in。"""
    import numpy as np
    if fake:
        rng = np.random.default_rng(layer * 13 + len(name))
        d_out, d_in = (96, 64) if "down" in name else (64, 64)
        W0 = rng.standard_normal((d_out, d_in)) / math.sqrt(d_in)
        full = W0 + 0.02 * rng.standard_normal((d_out, d_in)) / math.sqrt(d_in)
        a, b = rng.standard_normal((d_in, 2)), rng.standard_normal((2, d_out))
        delta = 0.15 * (a @ b).T / math.sqrt(d_in)
        return {"W0": W0, "full": full, "lora": W0 + delta, "lora_delta": delta, "full_delta": full - W0, "scale": 20.0, "shape": [d_out, d_in]}
    key = matrix_key(layer, name)
    bw = BaseWeights(model_path(cfg))
    W0 = bw.get(key + ".weight")
    full = st_read(weights["full"]["dir"] / "adapters.safetensors", [key + ".weight"])[key + ".weight"]
    lcfg = adapter_config(weights["lora"]["dir"])
    scale = float(lcfg["lora_parameters"]["scale"])
    lw = st_read(weights["lora"]["dir"] / "adapters.safetensors", [key + ".lora_a", key + ".lora_b"])
    lora = fused_lora(W0, lw[key + ".lora_a"], lw[key + ".lora_b"], scale)
    return {"W0": W0, "full": full, "lora": lora, "lora_delta": lora - W0, "full_delta": full - W0, "scale": scale, "shape": list(W0.shape)}


# ── (a) intruder dimensions ──────────────────────────────────────
def intruder_fn(ref: bool):
    mod = load_module(HERE / ("intruders_final.py" if ref else "intruders_live.py"), "intruders_impl")
    return mod.count_intruders


def analyse_matrix(cfg: dict, mats: dict, fn, eps_grid: list[float], dtype: str) -> dict:
    """對一個矩陣：兩個 run 各數 intruder，加幾個診斷量（Δ 的大小對 W0 的譜、Δ 的有效秩、bf16 合併留下多少）。"""
    import numpy as np
    ic = cfg["intruders"]
    k = int(ic["k"])
    cast = (lambda x: x.astype(np.float32)) if dtype == "float32" else (lambda x: x)
    W0 = cast(mats["W0"])
    S0 = np.linalg.svd(W0, compute_uv=False)
    out = {"shape": mats["shape"], "k": k, "dtype": dtype, "scale": mats["scale"],
           "W0": {"sigma_1": float(S0[0]), "sigma_k": float(S0[k - 1]), "sigma_min": float(S0[-1]), "frob": float(np.linalg.norm(W0))}, "runs": {}}
    for g in RUNS_ORDER:
        t0 = time.time()
        W1 = cast(mats[g])
        r = fn(W0, W1, k, eps_grid)
        D = mats[f"{g}_delta"]
        Sd = np.linalg.svd(cast(D), compute_uv=False)
        rank = int((Sd >= float(ic["delta_rank_ratio"]) * Sd[0]).sum()) if Sd[0] > 0 else 0
        rec = {"counts": r["counts"], "best": [round(float(x), 4) for x in r["best"]],
               "delta": {"frob": float(np.linalg.norm(D)), "frob_over_W0": float(np.linalg.norm(D) / np.linalg.norm(W0)),
                         "sigma_1": float(Sd[0]), "sigma_1_over_W0_sigma_k": float(Sd[0] / S0[k - 1]), "sigma_1_over_W0_sigma_min": float(Sd[0] / S0[-1]),
                         "effective_rank": rank, "rank_ratio": float(ic["delta_rank_ratio"])},
               "seconds": round(time.time() - t0, 1)}
        if "S1_topk" in r:
            rec["S1_topk"] = [round(float(x), 4) for x in r["S1_topk"]]
            rec["argmax"] = r.get("argmax")
        if g == "lora":
            kept = to_bf16(mats["lora"]) - mats["W0"]
            rec["bf16_fuse"] = {"kept_frob_fraction": float(np.linalg.norm(kept) / np.linalg.norm(D)) if np.linalg.norm(D) > 0 else None,
                                "entries_unchanged_fraction": float((kept == 0).mean())}
        out["runs"][g] = rec
    return out


def print_matrix_result(label: str, res: dict, eps_table: list[float]) -> None:
    runs = {"full": "full fine-tuning", "lora": "LoRA, r = 16"}
    head = f"  {'matrix':<26}{'run':<20}" + "".join(f"{'ε = ' + str(e):>10}" for e in eps_table)
    print(head)
    for g in RUNS_ORDER:
        c = res["runs"][g]["counts"]
        cells = "".join(f"{BOLD}{c[str(e)]:>10}{RESET}" for e in eps_table)
        print(f"  {label if g == 'full' else '':<26}{runs[g]:<20}{cells}")
    print(f"  {DIM}top-{res['k']} max|cos| — full: {' '.join(f'{b:.2f}' for b in res['runs']['full']['best'])}{RESET}")
    print(f"  {DIM}               LoRA: {' '.join(f'{b:.2f}' for b in res['runs']['lora']['best'])}{RESET}")
    w = res["W0"]
    for g in RUNS_ORDER:
        d = res["runs"][g]["delta"]
        print(f"  {DIM}Δ({g}): ‖Δ‖_F/‖W0‖_F {d['frob_over_W0']:.2e}; σ₁(Δ)/σ_k(W0) {d['sigma_1_over_W0_sigma_k']:.2e}, σ₁(Δ)/σ_min(W0) {d['sigma_1_over_W0_sigma_min']:.2e}; "
              f"effective rank (≥ {d['rank_ratio']:.0%} of σ₁) {d['effective_rank']}{RESET}")
    bf = res["runs"]["lora"].get("bf16_fuse")
    if bf and bf["kept_frob_fraction"] is not None:
        print(f"  {DIM}if fused in bf16 (mlx_lm.fuse): {bf['kept_frob_fraction']:.0%} of ‖Δ_LoRA‖_F survives; {bf['entries_unchanged_fraction']:.0%} of entries unchanged{RESET}")
    print(f"  {DIM}W0: σ₁ {w['sigma_1']:.3f}, σ_k {w['sigma_k']:.3f}, σ_min {w['sigma_min']:.4f}; shape {res['shape']}{RESET}\n")


def cmd_intruders(args, cfg):
    ic = cfg["intruders"]
    weights = {} if args.fake else resolve_weights(cfg)
    fn = intruder_fn(ref=bool(args.ref or args.fake))
    dtype = "float32" if args.float32 else ic["dtype"]
    eps_table, eps_sweep = [float(e) for e in ic["eps_table"]], sorted({float(e) for e in list(ic["eps_sweep"]) + list(ic["eps_table"])})
    banner(f"demo 4 (a) · intruder dimensions, k = {ic['k']}" + ("  [FAKE]" if args.fake else "") + ("  [reference implementation]" if args.ref else "  [intruders_live.py]"))
    if not args.fake:
        for g in RUNS_ORDER:
            w = weights[g]
            print(f"  {g:<5} {rel(w['dir'])}  (demo 3: lr {w['lr']}, test loss {w['loss']})")
        print()
    rec = {"date": now(), "fake": bool(args.fake), "ref": bool(args.ref or args.fake), "k": int(ic["k"]), "eps_table": eps_table, "dtype": dtype,
           "model": cfg["model"]["repo"], "numpy": __import__("numpy").__version__,
           "weights": {g: {"dir": rel(weights[g]["dir"]), "lr": weights[g]["lr"], "loss": weights[g]["loss"]} for g in RUNS_ORDER} if not args.fake else None,
           "matrices": []}
    for m in ic["matrices"]:
        layer, name = int(m["layer"]), m["name"]
        label = f"layer {layer} {name.split('.')[-1]}"
        t0 = time.time()
        mats = load_matrices(cfg, weights, layer, name, args.fake)
        res = analyse_matrix(cfg, mats, fn, eps_sweep, dtype)
        res.update(layer=layer, name=name, label=label, load_seconds=round(time.time() - t0, 1))
        print_matrix_result(label, res, eps_table)
        rec["matrices"].append(res)
    out = (REHEARSAL / "intruders.json") if args.rehearsal else (LIVE / f"intruders-{now().replace(':', '')}.json")
    if args.fake:
        out = out.with_suffix(".fake.json")
    write_json(out, rec)
    note(f"→ {rel(out)}")
    return rec


def cmd_scan(args, cfg):
    import numpy as np
    sc, ic = cfg["scan"], cfg["intruders"]
    weights = {} if args.fake else resolve_weights(cfg)
    fn = intruder_fn(ref=True)
    dtype = "float32" if args.float32 else ic["dtype"]
    eps = float(sc["eps_scan"])
    eps_grid = sorted({float(e) for e in list(ic["eps_sweep"]) + [eps]})
    n_layers = 28 if args.fake else int(read_json(Path(model_path(cfg)) / "config.json")["num_hidden_layers"])
    layers = [int(x) for x in (args.layers.split(",") if args.layers else (sc["layers"] or range(n_layers)))]
    names = args.names.split(",") if args.names else list(sc["names"])
    banner(f"scan · {len(layers)} layers × {len(names)} matrices × 2 runs, ε = {eps}, k = {ic['k']}, {dtype}" + ("  [FAKE]" if args.fake else ""))
    rec = {"date": now(), "fake": bool(args.fake), "eps": eps, "k": int(ic["k"]), "dtype": dtype, "layers": layers, "names": names, "cells": []}
    out = REHEARSAL / ("scan.fake.json" if args.fake else "scan.json")
    print(f"  {'layer':>5}  " + "".join(f"{n.split('.')[-1] + ' full/lora':>24}" for n in names))
    for layer in layers:
        row = []
        for name in names:
            t0 = time.time()
            mats = load_matrices(cfg, weights, layer, name, args.fake)
            res = analyse_matrix(cfg, mats, fn, eps_grid, dtype)
            res.update(layer=layer, name=name, seconds=round(time.time() - t0, 1))
            rec["cells"].append(res)
            cf, cl = res["runs"]["full"]["counts"][str(eps)], res["runs"]["lora"]["counts"][str(eps)]
            row.append(f"{cf:>10} /{cl:>5}  {DIM}{res['seconds']:5.0f}s{RESET}")
        print(f"  {layer:>5}  " + "".join(f"{c:>24}" for c in row), flush=True)
        write_json(out, rec)   # 每層存一次，中斷也有東西
    tot = {g: sum(c["runs"][g]["counts"][str(eps)] for c in rec["cells"]) for g in RUNS_ORDER}
    rec["totals"] = tot
    write_json(out, rec)
    print(f"\n  totals at ε = {eps}: full {BOLD}{tot['full']}{RESET}, LoRA {BOLD}{tot['lora']}{RESET}  (the paper's Fig. 4 sums over the whole model)")
    note(f"→ {rel(out)}")


# ── (b) 遺忘：bits-per-byte ───────────────────────────────────────
def load_texts(cfg: dict, fake: bool = False) -> list[dict]:
    out = []
    for t in cfg["forgetting"]["texts"]:
        src = t["source"]
        if fake:
            b, info = (t["key"] + " ").encode("utf-8") * 100, {"path": t.get("path", "fake"), "fake": True}
        elif src == "file":
            p = (HERE / t["path"]).resolve()
            b, info = p.read_text(encoding="utf-8").strip().encode("utf-8"), {"path": rel(p)}
        elif src == "wikipedia":
            p = (HERE / t["path"]).resolve()
            if not p.exists():
                raise SystemExit(f"{RED}沒有 {rel(p)}：先在 ../w07_d3_quality_filter 跑 ./present.sh fetch{RESET}")
            arts = [json.loads(line) for line in open(p, encoding="utf-8")]
            meta_p = p.with_name(p.name + ".meta.json")
            meta = read_json(meta_p) if meta_p.exists() else {}
            b = b"\n\n".join(a["text"].encode("utf-8") for a in arts)
            info = {"path": rel(p), "articles": len(arts), **{k: meta.get(k) for k in ("repo", "license", "row_group", "seed")}}
        elif src == "corpus_tail":
            d2 = (HERE / t["demo2"]).resolve()
            mod = load_module(d2 / "train.py", "w07_d2_train")
            with open(d2 / "demo_config.toml", "rb") as f:
                cfg2 = tomllib.load(f)
            c = mod.load_corpus(cfg2, t["corpus"])
            b = c["held"]
            info = {"corpus": t["corpus"], "repo": c["repo"], "license": c["license"], "held_docs": c["held_docs"], "via": rel(d2 / "train.py")}
        else:
            raise ValueError(src)
        mb = int(cfg["forgetting"]["max_bytes"])
        if mb and len(b) > mb:
            b, info["truncated_to"] = b[:mb], mb
        out.append({"key": t["key"], "label": t["label"], "source": src, "bytes": b, "n_bytes": len(b), **info})
    return out


class Scorer:
    """一個 mlx 模型：encode、逐視窗 NLL。prefix 照 W1 demo 2：BOS 若有，否則 <|endoftext|>。"""

    def __init__(self, path: str, adapter_dir: Path | None):
        from mlx_lm import load
        t0 = time.time()
        self.model, self.tokenizer = load(path, adapter_path=str(adapter_dir) if adapter_dir else None)
        self.load_seconds = round(time.time() - t0, 1)
        tok = self.tokenizer
        if getattr(tok, "bos_token_id", None) is not None:
            self.prefix = int(tok.bos_token_id)
        else:
            vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
            self.prefix = int(vocab.get("<|endoftext|>", tok.eos_token_id))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def window_nll(self, ids: list[int]) -> float:
        import mlx.core as mx
        out = self.model(mx.array([self.prefix] + ids)[None])
        logits = (out[0] if isinstance(out, (tuple, list)) else out)[0, :-1].astype(mx.float32)
        tgt = mx.array(ids)
        lse = mx.logsumexp(logits, axis=-1)
        picked = mx.take_along_axis(logits, tgt[:, None], axis=-1).squeeze(-1)
        return float((lse - picked).sum().item())

    def bpb(self, text: dict, window: int) -> dict:
        t0 = time.time()
        ids = self.encode(text["bytes"].decode("utf-8"))
        nll, n_win = 0.0, 0
        for i in range(0, len(ids), window):
            nll += self.window_nll(ids[i:i + window])
            n_win += 1
        B = text["n_bytes"]
        return {"bytes": B, "tokens": len(ids), "windows": n_win, "nll_nats": round(nll, 2),
                "bpb": round(nll / math.log(2) / B, 4), "nats_per_token": round(nll / len(ids), 4), "seconds": round(time.time() - t0, 1)}

    def close(self):
        import mlx.core as mx
        del self.model
        for owner in (mx, getattr(mx, "metal", None)):
            fn = getattr(owner, "clear_cache", None) if owner is not None else None
            if fn:
                try:
                    fn()
                except Exception:
                    pass
                break


def fake_bpb(key: str, model: str, seed: int) -> dict:
    import random
    rnd = random.Random(f"{key}{model}{seed}")
    base = {"en_paragraph": 0.95, "zh_paragraph": 1.35, "wikipedia": 0.88, "edu_tail": 0.82}.get(key, 1.0)
    bump = {"base": 0.0, "lora": 0.02, "full": 0.05}[model] * (3 if "zh" in key else 1)
    v = base + bump + rnd.uniform(-0.01, 0.01)
    return {"bytes": 1000, "tokens": 250, "windows": 1, "nll_nats": round(v * math.log(2) * 1000, 2), "bpb": round(v, 4), "nats_per_token": round(v * math.log(2) * 4, 4), "seconds": 0.0}


def print_forget_table(rec: dict, upto: int | None = None) -> None:
    """p58 的表：列 = 四段 held-out，欄 = base／LoRA／full 的 bits-per-byte（括號是對 base 的差）。"""
    order, labels = rec["models_order"], rec["models_labels"]
    W = 28
    print(f"  {'held-out text (bits per byte)':<42}" + "".join(f"{labels[g][:W - 2]:>{W}}" for g in order))
    rows = rec["texts"][:upto] if upto is not None else rec["texts"]
    for t in rows:
        line = f"  {t['label']:<42}"
        for g in order:
            v = rec["results"][g][t["key"]]["bpb"]
            d = v - rec["results"]["base"][t["key"]]["bpb"]
            cell = f"{v:.4f}" if g == "base" else f"{v:.4f} ({d:+.4f})"
            line += f"{BOLD}{cell:>{W}}{RESET}"
        print(line)


def cmd_forget(args, cfg):
    fc = cfg["forgetting"]
    order, labels = list(fc["models"]["order"]), dict(fc["models"]["labels"])
    weights = {} if args.fake else resolve_weights(cfg)
    banner("demo 4 (b) · bits-per-byte on four held-out texts, three models" + ("  [FAKE]" if args.fake else ""), AMBER)
    texts = load_texts(cfg, fake=bool(args.fake))
    for t in texts:
        print(f"  {t['label']:<42} {t['n_bytes']:>10,} bytes  {DIM}{t.get('path') or t.get('via')}{RESET}")
    print()
    rec = {"date": now(), "fake": bool(args.fake), "model": cfg["model"]["repo"], "window_tokens": int(fc["window_tokens"]), "prefix": "<|endoftext|> (BOS if the tokenizer has one)",
           "ruler": "bits per byte = sum of token NLL (nats) / ln 2 / UTF-8 bytes; non-overlapping windows, each preceded by the prefix token",
           "models_order": order, "models_labels": labels,
           "weights": None if args.fake else {g: {"dir": rel(weights[g]["dir"]), "lr": weights[g]["lr"], "test_loss": weights[g]["loss"]} for g in RUNS_ORDER},
           "base_test_loss": None if args.fake else weights.get("base_test_loss"),
           "texts": [{k: v for k, v in t.items() if k != "bytes"} for t in texts], "results": {}, "versions": versions()}
    if not args.fake:
        rec["device"] = device_info()
        path = model_path(cfg)
    for g in order:
        print(f"  {AMBER}{labels[g]}{RESET}")
        if args.fake:
            rec["results"][g] = {t["key"]: fake_bpb(t["key"], g, 1) for t in texts}
            for t in texts:
                print(f"    {t['label']:<40} {rec['results'][g][t['key']]['bpb']:.4f} bits/byte  [FAKE]")
            continue
        reset_peak_memory()
        sc = Scorer(path, None if g == "base" else weights[g]["dir"])
        print(f"    {DIM}loaded in {sc.load_seconds} s{RESET}")
        rec["results"][g] = {}
        for t in texts:
            r = sc.bpb(t, int(fc["window_tokens"]))
            rec["results"][g][t["key"]] = r
            print(f"    {t['label']:<40} {BOLD}{r['bpb']:.4f}{RESET} bits/byte  {DIM}{r['tokens']:,} tokens, {r['windows']} windows, {r['seconds']} s{RESET}")
        rec["results"][g]["_peak_memory_gb"] = peak_memory_gb()
        rec["results"][g]["_load_seconds"] = sc.load_seconds
        sc.close()
        print()
    out = REHEARSAL / ("forgetting.fake.json" if args.fake else "forgetting.json")
    write_json(out, rec)
    print()
    print_forget_table(rec)
    print_sft_line(rec)
    note(f"→ {rel(out)}")
    return rec


def print_sft_line(rec: dict) -> None:
    w, b = rec.get("weights"), rec.get("base_test_loss")
    if w and b is not None:
        print(f"\n  {DIM}the fine-tuning target itself (demo 3, test.jsonl response loss, nats/token): base {b} → LoRA {w['lora']['test_loss']} → full {w['full']['test_loss']}{RESET}")


def cmd_replay(args, cfg):
    if args.intruders:
        p = REHEARSAL / ("intruders.fake.json" if args.fake else "intruders.json")
        if not p.exists():
            raise SystemExit(f"{RED}沒有 {rel(p)}：先跑 ./present.sh rehearse（或 intruders --ref --rehearsal）{RESET}")
        rec = read_json(p)
        banner(f"replay · demo 4 (a) · intruders.json ({rec['date']})" + ("  [FAKE]" if rec.get("fake") else ""))
        for res in rec["matrices"]:
            print_matrix_result(res["label"], res, rec["eps_table"])
        return
    p = REHEARSAL / ("forgetting.fake.json" if args.fake else "forgetting.json")
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {rel(p)}：先跑 ./present.sh forget{RESET}")
    rec = read_json(p)
    banner(f"replay · demo 4 (b) · bits-per-byte, {rec['date']}" + ("  [FAKE]" if rec.get("fake") else ""), AMBER)
    print(f"  {DIM}{rec['ruler']}; window {rec['window_tokens']} tokens{RESET}\n")
    for i in range(1, len(rec["texts"]) + 1):
        if i > 1 and pause(f"→ row {i}") == "q":
            return
        print()
        print_forget_table(rec, upto=i)
    print_sft_line(rec)
    print(f"\n  {DIM}one model, one dataset, one seed — a direction, not a replication of anyone's number{RESET}")


# ── 可選：v3 的介入 ───────────────────────────────────────────────
def cmd_intervene(args, cfg):
    """LoRA 模型：每個被 LoRA 的矩陣在 float64 合併、SVD、intruder 方向（ε）的奇異值乘 λ、寫回 float32 的 nn.Linear；整個模型轉 float32 再量 (b)。
    對照組 λ = 1（同樣 float32 合併、不介入）。**2026-11-08 建好、未在 Mac 實測**。"""
    import numpy as np
    iv, ic, fc = cfg["intervene"], cfg["intruders"], cfg["forgetting"]
    weights = resolve_weights(cfg)
    eps, k = float(iv["eps"]), int(ic["k"])
    lambdas = [1.0] + [float(x) for x in iv["lambdas"]]
    texts = load_texts(cfg)
    banner(f"intervene · scale intruder singular values (ε = {eps}, k = {k}) by λ ∈ {lambdas[1:]}; λ = 1 is the float32 control", AMBER)
    if args.fake:
        raise SystemExit("intervene 沒有 fake 模式")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm import load
    path = model_path(cfg)
    rec = {"date": now(), "eps": eps, "k": k, "lambdas": lambdas, "lora": rel(weights["lora"]["dir"]), "texts": [{k2: v for k2, v in t.items() if k2 != "bytes"} for t in texts], "results": {}, "per_matrix": []}
    # 1. 一次 SVD，記下每個矩陣的 (U1, S1, V1t, intruder idx)——所有 λ 共用
    model, tokenizer = load(path, adapter_path=str(weights["lora"]["dir"]))
    svds = []
    t0 = time.time()
    for name, m in model.named_modules():
        if not hasattr(m, "fuse") or not hasattr(m, "lora_a"):
            continue
        W0 = np.array(m.linear.weight.astype(mx.float32), dtype=np.float64)
        A, B = np.array(m.lora_a, dtype=np.float64), np.array(m.lora_b, dtype=np.float64)
        W1 = fused_lora(W0, A, B, float(m.scale))
        U0 = np.linalg.svd(W0, full_matrices=False)[0]
        U1, S1, V1t = np.linalg.svd(W1, full_matrices=False)
        best = np.abs(U1[:, :k].T @ U0).max(axis=1)
        idx = [int(i) for i in np.where(best < eps)[0]]
        svds.append((name, U1, S1, V1t, idx))
        rec["per_matrix"].append({"name": name, "intruders": idx, "best": [round(float(b), 3) for b in best]})
        print(f"  {name:<48} intruders {idx}", flush=True)
    rec["svd_seconds"] = round(time.time() - t0, 1)
    rec["n_intruders_total"] = sum(len(s[4]) for s in svds)
    print(f"\n  {rec['n_intruders_total']} intruder directions in {len(svds)} matrices ({rec['svd_seconds']} s)\n")
    del model
    # 2. 每個 λ：重建、寫回、轉 float32、量四段
    for lam in lambdas:
        model, tokenizer = load(path, adapter_path=str(weights["lora"]["dir"]))
        new = []
        for name, U1, S1, V1t, idx in svds:
            S = S1.copy()
            S[idx] *= lam
            W = (U1 * S) @ V1t
            lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
            lin.weight = mx.array(W.astype(np.float32))
            new.append((name, lin))
        model.update_modules(tree_unflatten(new))
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
        sc = Scorer.__new__(Scorer)
        sc.model, sc.tokenizer = model, tokenizer
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        sc.prefix = int(getattr(tokenizer, "bos_token_id", None) or vocab.get("<|endoftext|>", tokenizer.eos_token_id))
        rec["results"][str(lam)] = {}
        print(f"  {AMBER}λ = {lam}{' (control: float32 merge, no intervention)' if lam == 1.0 else ''}{RESET}")
        for t in texts:
            r = sc.bpb(t, int(fc["window_tokens"]))
            rec["results"][str(lam)][t["key"]] = r
            print(f"    {t['label']:<40} {BOLD}{r['bpb']:.4f}{RESET} bits/byte  {DIM}{r['seconds']} s{RESET}")
        sc.close()
        write_json(REHEARSAL / "intervene.json", rec)
        print()
    note(f"→ {rel(REHEARSAL / 'intervene.json')}")


# ── inspect／rehearse ─────────────────────────────────────────────
def cmd_inspect(args, cfg):
    import numpy as np
    banner("inspect · demo 4" + ("  [FAKE]" if args.fake else ""))
    v = versions()
    v["numpy"] = np.__version__
    rec = {"date": now(), "fake": bool(args.fake), "versions": v}
    print(f"  python {v['python']}  numpy {v['numpy']}  mlx {v.get('mlx')}  mlx-lm {v.get('mlx-lm')} ({(v.get('mlx_lm_commit') or '')[:7]})")
    if not args.fake:
        write_lock({"NUMPY_VERSION": np.__version__})
        weights = resolve_weights(cfg)
        bw = BaseWeights(model_path(cfg))
        rec["weights"], rec["check"] = {}, {}
        for g in RUNS_ORDER:
            w, d = weights[g], weights[g]["dir"]
            ok = (d / "adapters.safetensors").exists() and (d / "adapter_config.json").exists()
            size = (d / "adapters.safetensors").stat().st_size if ok else 0
            ac = adapter_config(d) if ok else {}
            rec["weights"][g] = {"dir": rel(d), "exists": ok, "bytes": size, "lr": w["lr"], "test_loss": w["loss"], "from": w["from"],
                                 "fine_tune_type": ac.get("fine_tune_type"), "num_layers": ac.get("num_layers"), "lora_parameters": ac.get("lora_parameters")}
            print(f"  {g:<5} {rel(d)}  {'ok' if ok else RED + 'MISSING' + RESET}  {size / 1e9:.2f} GB  fine_tune_type {ac.get('fine_tune_type')}  lr {w['lr']}  test loss {w['loss']}")
        # 鍵名與形狀：full 的每個鍵都要在 base 裡、形狀相同；LoRA 的 a／b 形狀要能乘出 W0 的形狀
        hdr_full = st_header(weights["full"]["dir"] / "adapters.safetensors")[0]
        hdr_lora = st_header(weights["lora"]["dir"] / "adapters.safetensors")[0]
        bad = [k for k, h in hdr_full.items() if k != "__metadata__" and (k not in bw.map or bw.shape(k) != h["shape"])]
        rec["check"]["full_keys"] = {"n": len(hdr_full) - ("__metadata__" in hdr_full), "mismatch": bad}
        print(f"  full adapters: {rec['check']['full_keys']['n']} tensors, {len(bad)} not matching the base model" + (f" {RED}{bad[:3]}{RESET}" if bad else ""))
        lora_keys = sorted({k.rsplit('.', 1)[0] for k in hdr_lora if k != "__metadata__"})
        badl = []
        for k in lora_keys:
            a, b = hdr_lora[k + ".lora_a"]["shape"], hdr_lora[k + ".lora_b"]["shape"]
            w0 = bw.shape(k + ".weight")
            if not (a[0] == w0[1] and b[1] == w0[0] and a[1] == b[0]):
                badl.append(k)
        rec["check"]["lora_keys"] = {"n": len(lora_keys), "rank": hdr_lora[lora_keys[0] + ".lora_a"]["shape"][1], "dtype": hdr_lora[lora_keys[0] + ".lora_a"]["dtype"], "mismatch": badl}
        print(f"  LoRA adapters: {len(lora_keys)} matrices, rank {rec['check']['lora_keys']['rank']}, stored as {rec['check']['lora_keys']['dtype']}, {len(badl)} shape mismatches")
        rec["matrices"] = []
        for m in cfg["intruders"]["matrices"]:
            key = matrix_key(int(m["layer"]), m["name"])
            sh = bw.shape(key + ".weight")
            rec["matrices"].append({"layer": m["layer"], "name": m["name"], "shape": sh, "in_full": key + ".weight" in hdr_full, "in_lora": key + ".lora_a" in hdr_lora})
            print(f"  p55 matrix: {key}  {sh}  full {'✓' if key + '.weight' in hdr_full else RED + '✗' + RESET}  lora {'✓' if key + '.lora_a' in hdr_lora else RED + '✗' + RESET}")
        rec["base_test_loss"] = weights["base_test_loss"]
    texts = load_texts(cfg, fake=bool(args.fake))
    rec["texts"] = [{k: v for k, v in t.items() if k != "bytes"} for t in texts]
    print()
    for t in texts:
        print(f"  {t['label']:<42} {t['n_bytes']:>10,} bytes  {DIM}{t.get('path') or t.get('via')}{RESET}")
    out = REHEARSAL / ("inspect.fake.json" if args.fake else "inspect.json")
    write_json(out, rec)
    note(f"→ {rel(out)}")


def cmd_rehearse(args, cfg):
    cmd_inspect(args, cfg)
    args.ref, args.rehearsal = True, True
    cmd_intruders(args, cfg)
    cmd_forget(args, cfg)
    if args.intervene:
        cmd_intervene(args, cfg)
    print(f"\n{GREEN}rehearsal done{RESET} → runs/rehearsal/{{inspect,intruders,forgetting}}.json；接著回填 p55 的表頭（ε、層）與 p58 的表")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "intruders", "scan", "forget", "replay", "rehearse", "intervene"])
    ap.add_argument("--fake", action="store_true", help="沒有 mlx 也能演練（輸出是假的）")
    ap.add_argument("--ref", action="store_true", help="intruders：用 intruders_final.py（參考實作）而不是現場寫的 intruders_live.py")
    ap.add_argument("--rehearsal", action="store_true", help="intruders：寫進 runs/rehearsal/intruders.json（預設寫 runs/live/）")
    ap.add_argument("--float32", action="store_true", help="intruders／scan：SVD 用 float32（快約 2 倍）")
    ap.add_argument("--layers", help="scan：只掃這幾層，逗號分隔（預設 [scan].layers，空 = 全部）")
    ap.add_argument("--names", help="scan：只掃這幾個矩陣，逗號分隔（預設 [scan].names）")
    ap.add_argument("--intruders", action="store_true", help="replay：印 (a) 的彩排紀錄而不是 (b)")
    ap.add_argument("--intervene", action="store_true", help="rehearse：最後加做 intervene")
    a = ap.parse_args()
    cfg = load_cfg()
    {"inspect": cmd_inspect, "intruders": cmd_intruders, "scan": cmd_scan, "forget": cmd_forget,
     "replay": cmd_replay, "rehearse": cmd_rehearse, "intervene": cmd_intervene}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
