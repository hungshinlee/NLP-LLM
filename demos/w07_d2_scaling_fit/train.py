#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 2 的語料與訓練：FineWeb-Edu 的抽樣（fetch）、byte-level 的 W4 tiny GPT、沿 D = 20N 射線訓一個尺寸。

demo 3 從檔案路徑載入這個模組（照 W5 demo 2／W6 demo 2 載 W4 的做法，不動 sys.path）：
  fetch_corpus(cfg, which)     串流 HF 的 parquet shard，寫 <corpus.dir>/<file>（JSONL）與 .meta.json；回 meta
  load_corpus(cfg, which)      讀 JSONL → bytes；尾端整篇文件切成 held-out；回 dict（train bytes、held bytes、文件數、dump 分布）
  Demo1(rel)                   W4 demo 1 的 block_final（GPT）與 train（pick_device、param_groups、lr_at、evaluate）
  build_model(d1, size, V, L, dropout, seed)
  count_split(model)           {"total", "emb", "nonemb"}
  train_size(d1, cfg, corpus, size, steps=None, tag="")   訓一個尺寸到 D = tokens_per_param · N（或 --steps），回紀錄 dict

tokenizer：byte-level，V = 256（一個 byte 一個 token），所以 held-out 的 nats/token ÷ ln 2 **就是** bits/byte——W1 p29 那把尺，這裡不需要換分母。
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"
V_BYTES = 256


def load_module(path: Path, name: str):
    """從檔案路徑載入模組，不動 sys.path（各資料夾的 tests.py 會撞名：W4 demo 1 彩排踩過）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Demo1:
    """W4 demo 1 的東西：block_final（GPT、count_params）、train（pick_device、param_groups、lr_at、evaluate）。"""

    def __init__(self, rel: str = "../w04_d1_block_scratch"):
        self.dir = (HERE / rel).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(f"找不到 W4 demo 1 的資料夾 {self.dir}")
        with open(self.dir / "demo_config.toml", "rb") as f:
            self.cfg = tomllib.load(f)
        self.bf = load_module(self.dir / "block_final.py", "w04_block_final")
        self.tr = load_module(self.dir / "train.py", "w04_d1_train")


# ── 語料 ─────────────────────────────────────────────────────────
def corpus_paths(cfg: dict, which: str) -> tuple[Path, Path]:
    cc = cfg["corpus"]
    d = (HERE / cc["dir"]).resolve()
    return d / cc[which]["file"], d / (cc[which]["file"] + ".meta.json")


def fetch_corpus(cfg: dict, which: str, target_mb: float | None = None, quiet: bool = False) -> dict:
    """串流 `repo/path`（第一個 parquet shard）的 row group，直到文字的 UTF-8 bytes 累到 target_mb。只抓需要的 row group，不下載整個 shard（約 2 GB）。
    需要 huggingface_hub（共用 .venv 已有）與 pyarrow（沒有就印出 uv 的安裝指令）。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(f"{RED}沒有 pyarrow（讀 parquet 用）。裝進共用環境：{RESET}\n"
                         f"  cd ~/Course-Hub/nlp_llm/demos && uv pip install --python .venv/bin/python pyarrow && uv pip freeze --python .venv/bin/python > requirements.lock.txt")
    from huggingface_hub import HfFileSystem
    cc = cfg["corpus"][which]
    target = int((target_mb or cfg["corpus"]["target_mb"]) * 1_000_000)
    out, meta_p = corpus_paths(cfg, which)
    out.parent.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    url = f"datasets/{cc['repo']}/{cc['path']}"
    t0 = time.time()
    keep = ("id", "dump", "url", "date", "token_count", "score", "language_score")
    n_docs, n_bytes, dumps = 0, 0, {}
    with fs.open(url, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        if not quiet:
            print(f"{BOLD}fetch {which}{RESET}  {cc['repo']} · {cc['path']} · {md.num_rows:,} rows in {md.num_row_groups} row groups · license {cc['license']}")
            print(f"  target {target / 1e6:.0f} MB of text → {out}")
        cols = [c for c in ("text",) + keep if c in pf.schema_arrow.names]
        # shard 裡的列是依 dump 排的（2026-10-26 試抓：前 40 個 row group 全是 CC-MAIN-2013-20），demo 4 要跨 dump 的文件，
        # 所以不讀「前面」的 row group，而是在整個 shard 上等距挑：先讀第 0 個量它有幾 MB，算出要幾個，再等距取。
        first = pf.read_row_group(0, columns=cols)
        b0 = sum(len(t.encode("utf-8")) for t in first.column("text").to_pylist())
        n_need = min(md.num_row_groups, int(math.ceil(target / max(b0, 1))) + 1)
        picks = sorted(set(round(i * (md.num_row_groups - 1) / max(1, n_need - 1)) for i in range(n_need)))
        if not quiet:
            print(f"  row group 0 holds {b0 / 1e6:.1f} MB → reading {len(picks)} row groups spread evenly over the shard (stride ≈ {picks[1] - picks[0] if len(picks) > 1 else 0}) so that several dumps are represented")
        with open(out, "w", encoding="utf-8") as w:
            for rg in picks:
                tbl = first if rg == 0 else pf.read_row_group(rg, columns=cols)
                rows = tbl.to_pylist()
                for r in rows:
                    text = r["text"]
                    rec = {k: r.get(k) for k in keep if k in r}
                    rec["text"] = text
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_docs += 1
                    n_bytes += len(text.encode("utf-8"))
                    dumps[r.get("dump")] = dumps.get(r.get("dump"), 0) + 1
                    if n_bytes >= target:
                        break
                if not quiet:
                    print(f"  row group {rg:>3}: {n_docs:>7,} docs, {n_bytes / 1e6:7.1f} MB  {time.time() - t0:6.1f} s")
                if n_bytes >= target:
                    break
    meta = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "which": which, "repo": cc["repo"], "path": cc["path"], "license": cc["license"],
            "docs": n_docs, "text_bytes": n_bytes, "row_groups_read": [p for p in picks if p <= rg], "row_groups_total": md.num_row_groups, "rows_in_shard": md.num_rows,
            "dumps": dict(sorted(dumps.items(), key=lambda kv: -kv[1])), "seconds": time.time() - t0}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not quiet:
        print(f"  {GREEN}{n_docs:,} docs, {n_bytes / 1e6:.1f} MB, {len(dumps)} distinct dumps{RESET} (top: " +
              ", ".join(f"{k} {v}" for k, v in list(meta["dumps"].items())[:5]) + f") in {meta['seconds']:.0f} s → {out.name}, {meta_p.name}")
    return meta


def load_corpus(cfg: dict, which: str = "edu") -> dict:
    """JSONL → 一條 bytes（文件以 sep 相接）。held-out 是**尾端**整篇文件，累到 split.held_out_mb；其餘是 train。"""
    p, meta_p = corpus_paths(cfg, which)
    if not p.exists():
        raise FileNotFoundError(f"沒有語料 {p}：先跑 ./present.sh fetch（需要網路；之後都不需要）")
    sp = cfg["split"]
    sep = sp["sep"].encode("utf-8")
    docs, dumps = [], {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append(r["text"].encode("utf-8"))
            dumps[r.get("dump")] = dumps.get(r.get("dump"), 0) + 1
    held_target = int(sp["held_out_mb"] * 1_048_576)
    cut, acc = len(docs), 0
    while cut > 0 and acc < held_target:
        cut -= 1
        acc += len(docs[cut]) + len(sep)
    train_b, held_b = sep.join(docs[:cut]), sep.join(docs[cut:])
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return {"path": str(p), "which": which, "repo": meta.get("repo"), "license": meta.get("license"), "fetched": meta.get("date"),
            "docs": len(docs), "train_docs": cut, "held_docs": len(docs) - cut, "dumps": len(dumps),
            "train_bytes": len(train_b), "held_bytes": len(held_b), "train": train_b, "held": held_b}


# ── 模型 ─────────────────────────────────────────────────────────
def size_name(s: dict) -> str:
    return f"{s['key']}: d {s['d']}, {s['n_layer']} layers, d_ff {s['d_ff']}"


def build_model(d1: Demo1, s: dict, L: int, dropout: float, seed: int, d_head: int):
    import torch
    torch.manual_seed(seed)
    n_head = s["d"] // d_head
    assert n_head * d_head == s["d"], f"d = {s['d']} 不是 d_head = {d_head} 的倍數"
    return d1.bf.GPT(V=V_BYTES, d=s["d"], n_layer=s["n_layer"], n_head=n_head, d_ff=s["d_ff"], L=L, dropout=dropout)


def count_split(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.emb.weight.numel()                       # tied：embedding 與 unembedding 同一份，只算一次
    return {"total": total, "emb": emb, "nonemb": total - emb}


def n_for_ray(split: dict, cfg: dict) -> int:
    return split[cfg["ray"]["count"]]


def plan(d1: Demo1, cfg: dict) -> list[dict]:
    """不訓練：每個尺寸的 N（兩種算法）、D = 20N、步數、lr。estimate 與 replay 的開頭都印這張。"""
    t, ray = cfg["train"], cfg["ray"]
    rows = []
    for s in cfg["sizes"]:
        m = build_model(d1, s, t["L"], 0.0, t["seed"], t["d_head"])
        sp = count_split(m)
        N = sp[ray["count"]]
        D = ray["tokens_per_param"] * N
        steps = max(1, round(D / (t["batch"] * t["L"])))
        rows.append({**s, "n_head": s["d"] // t["d_head"], "params": sp, "N": N, "D": D, "steps": steps,
                     "lr": lr_for(s["d"], t), "warmup": warmup_for(steps, t), "C": 6.0 * N * D})
    return rows


def lr_for(d: int, t: dict) -> float:
    if t["lr_rule"] == "mup_width":
        return t["lr_base"] * t["d_base"] / d
    return t["lr_base"]


def warmup_for(steps: int, t: dict) -> int:
    return max(20, round(t["warmup_frac"] * steps))


# ── 訓練一個尺寸 ──────────────────────────────────────────────────
def train_size(d1: Demo1, cfg: dict, corpus: dict, s: dict, steps: int | None = None, tag: str = "", save_weights: Path | None = None) -> dict:
    import torch
    t, ray, rc, sp = cfg["train"], cfg["ray"], cfg["record"], cfg["split"]
    L, B = t["L"], t["batch"]
    device = d1.tr.pick_device(t["device"])
    model = build_model(d1, s, L, t["dropout"], t["seed"], t["d_head"]).to(device)
    split = count_split(model)
    N = split[ray["count"]]
    D_target = ray["tokens_per_param"] * N
    full_steps = max(1, round(D_target / (B * L)))
    steps = steps or full_steps
    lr, warm = lr_for(s["d"], t), warmup_for(steps, t)
    cfg_lr = {"train": {"lr": lr, "warmup": warm, "steps": steps}}          # W4 的 lr_at 讀這三個鍵：warmup 線性、cosine 到 0.1·lr
    opt = torch.optim.AdamW(d1.tr.param_groups(model, t["weight_decay"]), lr=lr, betas=tuple(t["betas"]), eps=1e-8)
    train_u8 = torch.frombuffer(bytearray(corpus["train"]), dtype=torch.uint8)
    held = torch.frombuffer(bytearray(corpus["held"]), dtype=torch.uint8).long()
    nw = sp["subset_windows"]
    hx = torch.stack([held[i * L:(i + 1) * L] for i in range(nw)]).to(device)
    hy = torch.stack([held[i * L + 1:(i + 1) * L + 1] for i in range(nw)]).to(device)
    n_windows = (train_u8.numel() - 1) // L                                  # 不重疊視窗；一個 epoch = 全部走一遍（打亂順序）
    g = torch.Generator().manual_seed(t["seed"])
    perm, pos, epoch = torch.randperm(n_windows, generator=g), 0, 0
    print(f"\n{BOLD}{tag or 'train'} — {size_name(s)}{RESET}  torch {torch.__version__}, device {device}, {platform.machine()}")
    print(f"  corpus {Path(corpus['path']).name} ({corpus['repo']}, {corpus['license']}): {corpus['docs']:,} docs, train {corpus['train_bytes'] / 1e6:.1f} MB = {n_windows:,} windows of {L}; held-out {corpus['held_docs']} docs, {corpus['held_bytes'] / 1e6:.2f} MB (subset for the curve: first {nw} windows = {nw * L:,} bytes)")
    print(f"  byte-level GPT (V = {V_BYTES}): {s['n_layer']} layers, d {s['d']}, {s['d'] // t['d_head']} heads, d_ff {s['d_ff']}, L {L}, dropout {t['dropout']}; "
          f"params total {split['total']:,} (embedding {split['emb']:,}, non-embedding {split['nonemb']:,}); N for the ray = {ray['count']} = {N:,}")
    print(f"  ray D = {ray['tokens_per_param']}·N = {D_target:,} tokens → {full_steps:,} steps of {B}×{L} = {B * L:,}"
          + (f"  {AMBER}(overridden: {steps} steps){RESET}" if steps != full_steps else "")
          + f"; one epoch = {n_windows:,} steps' worth of windows → {steps * B / n_windows:.2f} epochs")
    print(f"  AdamW lr {lr:.2e} (= {t['lr_base']} × {t['d_base']}/{s['d']}, muP width rule), betas {t['betas']}, wd {t['weight_decay']} (matrices only), warmup {warm}, cosine → 0.1·lr, clip {t['grad_clip']}; seed {t['seed']}")
    curve, t0, tok_seen = [], time.time(), 0

    def record(step: int, train_loss: float, gn: float, lr_now: float):
        model.eval()
        with torch.no_grad():
            logits = model(hx)
            hl = torch.nn.functional.cross_entropy(logits.float().view(-1, V_BYTES), hy.view(-1)).item()
        model.train()
        rec = {"step": step, "tokens_seen": tok_seen, "train_loss": train_loss, "held_subset_loss": hl, "held_subset_bpb": hl / math.log(2),
               "grad_norm": gn, "lr": lr_now, "elapsed_s": time.time() - t0}
        curve.append(rec)
        if step % rc["log_every"] == 0 or step == steps - 1:
            print(f"  step {step:>6}/{steps}  loss {train_loss:.4f}  held {hl:.4f} nats = {hl / math.log(2):.3f} bits/byte  |g| {gn:.2f}  lr {lr_now:.2e}  "
                  f"{rec['elapsed_s']:7.1f} s  {tok_seen / max(rec['elapsed_s'], 1e-9):,.0f} tok/s")

    model.train()
    for step in range(steps):
        if pos + B > n_windows:                                              # 走完一個 epoch：重新打亂
            perm, pos, epoch = torch.randperm(n_windows, generator=g), 0, epoch + 1
        ix = perm[pos:pos + B] * L
        pos += B
        x = torch.stack([train_u8[i:i + L] for i in ix.tolist()]).long().to(device)
        y = torch.stack([train_u8[i + 1:i + L + 1] for i in ix.tolist()]).long().to(device)
        lr_now = d1.tr.lr_at(step, cfg_lr)
        for grp in opt.param_groups:
            grp["lr"] = lr_now
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, V_BYTES), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
        opt.step()
        tok_seen += B * L
        if step % rc["every"] == 0 or step == steps - 1:
            record(step, loss.item(), gn, lr_now)
    seconds = time.time() - t0
    smoke = steps != full_steps
    held_eval = held[: min(held.numel(), 131_072)] if smoke else held         # 煙霧測試只量前 128 KiB（CPU 上整段要好幾分鐘）；正式 run 量整段
    ev = d1.tr.evaluate(model, held_eval, L, t["eval_stride"], device)       # W4 的尺：整段 held-out、視窗 L、stride 128；每個 byte 剛好預測一次
    final = {"step": steps - 1, "tokens_seen": tok_seen, "D_actual": tok_seen, "epochs": tok_seen / (n_windows * L),
             "held_subset_loss": curve[-1]["held_subset_loss"], "held_subset_bpb": curve[-1]["held_subset_bpb"],
             "held_full_nats": ev["nats_per_char"], "held_full_bpb": ev["nats_per_char"] / math.log(2), "held_full_bytes": ev["chars"],
             "train_loss_last": curve[-1]["train_loss"], "seconds": seconds, "tokens_per_s": tok_seen / seconds}
    print(f"  {BOLD}{steps:,} steps in {seconds:.0f} s ({seconds / 60:.1f} min), {final['tokens_per_s']:,.0f} tok/s{RESET}; "
          f"{BLUE}held-out (full, W4's windowed measure, {ev['chars']:,} bytes): {ev['nats_per_char']:.4f} nats/byte = {final['held_full_bpb']:.4f} bits/byte{RESET}; subset {final['held_subset_bpb']:.4f}")
    sample = generate(model, device, t["sample_bytes"], corpus)
    if save_weights:
        save_weights.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_weights / f"{s['key']}.pt")
    return {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": s["key"], "size": {**s, "n_head": s["d"] // t["d_head"], "V": V_BYTES, "L": L},
            "torch": torch.__version__, "device": str(device), "python": platform.python_version(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
            "params": split, "N": N, "D_target": D_target, "steps": steps, "steps_for_ray": full_steps, "lr": lr, "warmup": warm,
            "train": {k: v for k, v in t.items()}, "ray": ray, "record": rc,
            "corpus": {k: v for k, v in corpus.items() if k not in ("train", "held")}, "n_windows": n_windows,
            "curve": curve, "final": final, "sample": sample}


def generate(model, device, n: int, corpus: dict) -> str:
    """從 held-out 開頭 64 個 byte 接著生成 n 個 byte（只是看看；不是評估）。"""
    import torch
    model.eval()
    prompt = torch.frombuffer(bytearray(corpus["held"][:64]), dtype=torch.uint8).long()[None].to(device)
    out = prompt
    with torch.no_grad():
        for _ in range(n):
            logits = model(out[:, -model.L:])[:, -1].float()
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            out = torch.cat([out, nxt], 1)
    model.train()
    return bytes(out[0, 64:].tolist()).decode("utf-8", errors="replace")
