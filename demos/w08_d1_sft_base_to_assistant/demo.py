#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 1：SFT 把 base 變成助理（投影片 p8 前導頁、p9 (a)、p15 (b)(c)、p42 (d)）。

    ./present.sh fetch        第一次：下載 Qwen3-1.7B-Base 與 databricks-dolly-15k，revision 寫進 ../versions.lock（要網路）
    ./present.sh prepare      從鎖定的 Dolly 檔抽 train／valid／test（純字元過濾 + seed，任何機器重跑都相同）→ data/、lora.yaml
    ./present.sh inspect      課前：版本、config、chat template、停止記號、N 與 N_LoRA 對 w08-data.json、資料的 token 長度 → runs/rehearsal/inspect.json
    ./present.sh rehearse     課前彩排（全套）：訓 A／B／C 三個 adapter → (a) 三題五欄 → (b) mask 與探針 → (c) held-out loss → (d) 12 步
                              --part train|generate|mask|heldout|live 只做一段；--adapter A|B|C 只訓一個
    ./present.sh              課堂 (a)：載 base 與 base+C（show_adapter），三題各印 base 原文續寫與 SFT 後的回答，標有沒有自己停
    ./present.sh mask         課堂 (b)：一筆訓練資料逐 token 標 loss；base 在 template 三個位置的探針
    ./present.sh replay-mask  課堂 (c)：兩個 adapter 的 held-out response loss 與三題輸出並排（讀彩排紀錄，不載模型）
    ./present.sh live-train   課堂 (d)：投影片那一行指令，只跑 12 步，每 4 步印 train loss 與 peak memory
    ./present.sh replay       現場失敗：印 (a) 的彩排紀錄（不載模型）
    ./present.sh fake ...     沒有 mlx 也能演練流程（輸出是假的，畫面會標明）

訓練走 mlx-lm 自己的 `mlx_lm.lora`（tuner 的 train_model），不另寫訓練迴圈：投影片 p42 那一行指令與這裡跑的是同一組參數，
程式會把等價的命令列印出來。差別只有一處：這個 commit（872ae88）的 CLI 畫面不印 peak memory（它只進 callback），
所以這裡掛一個 callback 把 `peak_memory` 印成一欄並記進 json。

mask 的規則照 mlx-lm `ChatDataset.process`：tokens = apply_chat_template(messages)；--mask-prompt 時 offset = len(apply_chat_template(messages[:-1],
add_generation_prompt=True))，loss 只算 index ≥ offset 的 target（`default_loss` 的 steps ≥ lengths[:, 0]）。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）、transformers（tokenizer）、huggingface_hub。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
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
                    locked_revision, note, now, peak_memory_gb, read_lock, revision_key, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
DATA = HERE / "data"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
LIVE = RUNS / "live"
WEIGHTS = RUNS / "weights"
LORA_YAML = HERE / "lora.yaml"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]
CITATION_RE = re.compile(r"\[\d{1,3}\]")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def dataset_lock_key(repo: str) -> str:
    return "DATASET_REVISION__" + re.sub(r"\W", "_", repo).upper()


def visible(s: str) -> str:
    """換行與 tab 印成可見的記號，讓 template 的結構看得到。"""
    return s.replace("\n", "⏎\n").replace("\t", "⇥")


def pause(prompt: str = "") -> str:
    try:
        return input(f"{DIM}{prompt}{RESET}")
    except EOFError:
        return ""


# ── 資料 ─────────────────────────────────────────────────────────
def dolly_path(cfg: dict, offline: bool) -> Path:
    from huggingface_hub import hf_hub_download
    d = cfg["data"]
    rev = read_lock().get(dataset_lock_key(d["repo"]))
    p = hf_hub_download(d["repo"], d["file"], repo_type="dataset", revision=rev, local_files_only=offline)
    return Path(p)


def to_messages(row: dict) -> dict:
    """Dolly 的一筆 → mlx-lm 的 chat 格式。context 接在指令後面、空一行；拿掉 [42] 這類引註（dataset card 的建議）。"""
    instruction = row["instruction"].strip()
    context = CITATION_RE.sub("", row.get("context", "") or "").strip()
    response = CITATION_RE.sub("", row["response"]).strip()
    user = instruction + ("\n\n" + context if context else "")
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": response}],
            "category": row["category"], "has_context": bool(context)}


def cmd_prepare(args, cfg):
    d = cfg["data"]
    src = Path(args.dolly) if args.dolly else dolly_path(cfg, offline=not args.online)
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, dropped = [], {}
    for r in rows:
        ctx = CITATION_RE.sub("", r.get("context", "") or "").strip()
        why = None
        if not (d["min_response_chars"] <= len(r["response"].strip()) <= d["max_response_chars"]):
            why = "response length"
        elif not (d["min_instruction_chars"] <= len(r["instruction"].strip()) <= d["max_instruction_chars"]):
            why = "instruction length"
        elif len(ctx) > d["max_context_chars"]:
            why = "context length"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
        else:
            kept.append(r)
    rng = random.Random(int(cfg["demo"]["seed"]))
    rng.shuffle(kept)
    n_tr, n_va, n_te = int(d["n_train"]), int(d["n_valid"]), int(d["n_test"])
    splits = {"train": kept[:n_tr], "valid": kept[n_tr:n_tr + n_va], "test": kept[n_tr + n_va:n_tr + n_va + n_te]}
    DATA.mkdir(exist_ok=True)
    stats = {}
    for name, part in splits.items():
        recs = [to_messages(r) for r in part]
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\n")
        cats = {}
        for rec in recs:
            cats[rec["category"]] = cats.get(rec["category"], 0) + 1
        stats[name] = {"n": len(recs), "with_context": sum(rec["has_context"] for rec in recs), "categories": cats,
                       "mean_user_chars": round(sum(len(rec["messages"][0]["content"]) for rec in recs) / len(recs), 1),
                       "mean_response_chars": round(sum(len(rec["messages"][1]["content"]) for rec in recs) / len(recs), 1)}
    # (b) 要印的那一筆：train 裡沒有 context、最短的一筆
    train_recs = [to_messages(r) for r in splits["train"]]
    cands = [(len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]), i) for i, r in enumerate(train_recs) if not r["has_context"]]
    _, idx = min(cands)
    (DATA / "example.jsonl").write_text(json.dumps({"messages": train_recs[idx]["messages"], "train_index": idx}, ensure_ascii=False) + "\n", encoding="utf-8")
    # mlx_lm.lora 的 lora_parameters 只能從 YAML 設定檔給（CLI 沒有 --rank）；p42 那一行的 -c lora_embed.yaml 就是這個檔
    write_yamls(cfg)
    meta = {"source": {"repo": d["repo"], "file": d["file"], "license": d["license"],   # 不寫日期：重跑要零 diff
                                      "revision": read_lock().get(dataset_lock_key(d["repo"])), "rows": len(rows)},
            "filter": {k: d[k] for k in ("min_response_chars", "max_response_chars", "min_instruction_chars", "max_instruction_chars", "max_context_chars")},
            "kept": len(kept), "dropped": dropped, "seed": int(cfg["demo"]["seed"]), "splits": stats,
            "mask_example_train_index": idx}
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    banner(f"prepare  ·  {d['repo']} ({d['license']})  ·  {len(rows):,} rows → kept {len(kept):,}")
    for k, v in dropped.items():
        print(f"  dropped {v:>5,}  {k}")
    for name, s in stats.items():
        print(f"  {name:<6} {s['n']:>5}  with context {s['with_context']:>4}   mean user {s['mean_user_chars']:>6} chars, response {s['mean_response_chars']:>6} chars")
    print(f"  categories (train): " + ", ".join(f"{k} {v}" for k, v in sorted(stats['train']['categories'].items())))
    print(f"  mask example: train.jsonl line {idx} → data/example.jsonl")
    note(f"→ data/train.jsonl, valid.jsonl, test.jsonl, example.jsonl, meta.json, lora.yaml, lora_embed.yaml")


def lora_parameters(cfg: dict, embedding: bool) -> dict:
    """mlx-lm 的 lora_parameters。embedding=True 時給 keys：七個 Linear 的相對名 + model.embed_tokens（LoRAEmbedding，連 tied head 的 as_linear）。"""
    t = cfg["train"]
    p = {"rank": int(t["rank"]), "dropout": float(t["dropout"]), "scale": float(t["scale"])}
    if embedding:
        p["keys"] = list(t["lora_keys"]) + [t["embedding_key"]]
    return p


def yaml_for(embedding: bool) -> Path:
    return HERE / ("lora_embed.yaml" if embedding else "lora.yaml")


def write_yamls(cfg: dict) -> None:
    for emb in (False, True):
        p = lora_parameters(cfg, emb)
        body = ["# 由 demo.py prepare 從 demo_config.toml [train] 產生；mlx_lm.lora -c 讀。", "lora_parameters:",
                f"  rank: {p['rank']}", f"  dropout: {p['dropout']}", f"  scale: {p['scale']}"]
        if emb:
            body.append("  keys:  # 七個 Linear 照 mlx-lm 預設，再加 embedding（Qwen3 的 LM head 與它綁在一起）")
            body += [f"    - {k}" for k in p["keys"]]
        yaml_for(emb).write_text("\n".join(body) + "\n", encoding="utf-8")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mask_example(cfg: dict) -> dict:
    idx = int(cfg["mask"]["example_index"])
    if idx >= 0:
        return read_jsonl(DATA / "train.jsonl")[idx]
    return read_jsonl(DATA / "example.jsonl")[0]


# ── 後端 ─────────────────────────────────────────────────────────
def model_path(cfg: dict, offline: bool, weights: bool = True) -> str:
    from huggingface_hub import snapshot_download
    m = cfg["model"]
    rev = locked_revision(m["repo"], m.get("revision"))
    kw = {"ignore_patterns": NON_MODEL_FILES} if weights else {"allow_patterns": ["*.json", "*.txt", "tokenizer.model"]}
    return snapshot_download(m["repo"], revision=rev, local_files_only=offline, **kw)


def lora_args(cfg: dict, path: str, mask_prompt: bool, iters: int, steps_per_report: int, adapter_dir: Path,
              steps_per_eval: int | None = None, embedding: bool = False) -> SimpleNamespace:
    """與 `mlx_lm.lora` 命令列完全相同的參數物件（CONFIG_DEFAULTS 補齊沒給的）。"""
    from mlx_lm.lora import CONFIG_DEFAULTS
    t = cfg["train"]
    a = dict(CONFIG_DEFAULTS)
    a.update(model=path, train=True, data=str(DATA), fine_tune_type="lora", num_layers=int(t["num_layers"]),
             mask_prompt=bool(mask_prompt), iters=int(iters), batch_size=int(t["batch_size"]),
             learning_rate=float(t["learning_rate"]), steps_per_report=int(steps_per_report),
             steps_per_eval=int(steps_per_eval if steps_per_eval is not None else t["steps_per_eval"]),
             val_batches=-1, adapter_path=str(adapter_dir), save_every=10 ** 9, max_seq_length=int(t["max_seq_length"]),
             seed=int(cfg["demo"]["seed"]), optimizer="adam", lora_parameters=lora_parameters(cfg, embedding))
    return SimpleNamespace(**a)


def cli_line(cfg: dict, a: SimpleNamespace) -> str:
    """等價的命令列（投影片 p42 那一行；lora_parameters 走 -c lora.yaml 或 lora_embed.yaml）。"""
    emb = "keys" in (a.lora_parameters or {})
    parts = ["mlx_lm.lora", f"-c {yaml_for(emb).name}", f"--model {cfg['model']['repo']}", "--train", "--data data/",
             f"--fine-tune-type {a.fine_tune_type}", f"--num-layers {a.num_layers}"]
    if a.mask_prompt:
        parts.append("--mask-prompt")
    lr = f"{a.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    parts += [f"--iters {a.iters}", f"--batch-size {a.batch_size}", f"--learning-rate {lr}",
              f"--steps-per-report {a.steps_per_report}", f"--steps-per-eval {a.steps_per_eval}", "--val-batches -1",
              f"--max-seq-length {a.max_seq_length}", f"--seed {a.seed}", f"--adapter-path {Path(a.adapter_path).relative_to(HERE)}",
              f"--save-every {a.save_every}"]
    return " \\\n    ".join(parts)


class Recorder:
    """mlx-lm 的 TrainingCallback：每次 report 印一欄 peak memory（CLI 畫面沒有），並記下來。"""

    def __init__(self, label: str, echo: bool = True):
        self.label, self.echo = label, echo
        self.train, self.val = [], []
        self.t0 = time.time()

    def on_train_loss_report(self, info: dict):
        rec = {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in info.items()}
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        self.train.append(rec)
        if self.echo:
            print(f"  {GREY}[{self.label}]{RESET} step {info['iteration']:>4}   train loss {BOLD}{info['train_loss']:.3f}{RESET}   "
                  f"peak memory {BOLD}{info['peak_memory']:.2f} GB{RESET}   {info['tokens_per_second']:,.0f} tok/s   {rec['elapsed_s']:.0f} s")

    def on_val_loss_report(self, info: dict):
        rec = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        self.val.append(rec)
        if self.echo:
            print(f"  {GREY}[{self.label}]{RESET} step {info['iteration'] + 1:>4}   {AMBER}val loss {info['val_loss']:.3f}{RESET}   ({info['val_time']:.1f} s)")


def train_adapter(cfg: dict, path: str, mask_prompt: bool, iters: int, steps_per_report: int, adapter_dir: Path,
                  label: str, with_valid: bool = True, steps_per_eval: int | None = None, embedding: bool = False) -> dict:
    """照 mlx_lm.lora.run 的順序，只是 callback 換成 Recorder（run() 會把傳入的 callback 蓋掉，所以不直接呼叫它）。"""
    import numpy as np
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load
    from common import reset_peak_memory
    a = lora_args(cfg, path, mask_prompt, iters, steps_per_report, adapter_dir, steps_per_eval, embedding)
    reset_peak_memory()   # 同一個行程裡前面載過的模型不算進這個 run 的 peak
    print(f"{DIM}{cli_line(cfg, a)}{RESET}\n")
    np.random.seed(a.seed)
    t0 = time.time()
    model, tokenizer = load(a.model)
    load_s = time.time() - t0
    train_set, valid_set, _ = load_dataset(a, tokenizer)
    if not with_valid:
        valid_set = []
    rec = Recorder(label)
    t1 = time.time()
    train_model(a, model, train_set, valid_set, rec)
    from mlx.utils import tree_flatten
    n_tr = int(sum(x.size for _, x in tree_flatten(model.trainable_parameters())))
    # 紀錄進版控也上站：路徑一律相對、不留 Mac 的絕對路徑（model 只留 HF snapshot 的 sha）
    args_rec = {k: v for k, v in vars(a).items() if k != "model"}
    args_rec["data"] = str(Path(a.data).relative_to(HERE))
    args_rec["adapter_path"] = str(Path(a.adapter_path).relative_to(HERE))
    return {"label": label, "mask_prompt": mask_prompt, "embedding": embedding, "n_trainable": n_tr, "iters": iters,
            "args": args_rec, "model": cfg["model"]["repo"], "model_snapshot": Path(a.model).name,
            "cli": cli_line(cfg, a), "load_seconds": round(load_s, 1), "train_seconds": round(time.time() - t1, 1),
            "n_train": len(train_set), "n_valid": len(valid_set) if valid_set else 0, "train_log": rec.train, "val_log": rec.val,
            "peak_memory_gb": peak_memory_gb(), "adapter_dir": str(adapter_dir.relative_to(HERE))}


class MLXBackend:
    fake = False

    def __init__(self, cfg: dict, offline: bool, adapter_dir: Path | None = None, label: str = ""):
        from mlx_lm import load
        t0 = time.time()
        self.path = model_path(cfg, offline)
        self.model, self.tokenizer = load(self.path, adapter_path=str(adapter_dir) if adapter_dir else None)
        for tok in cfg["model"].get("extra_eos", []):
            self.tokenizer.add_eos_token(tok)
        self.label = label or (cfg["model"]["short"] + (f" + {adapter_dir.name}" if adapter_dir else ""))
        self.adapter = str(adapter_dir.relative_to(HERE)) if adapter_dir else None
        self.load_seconds = time.time() - t0

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def template(self, messages: list[dict], add_generation_prompt: bool, **kw) -> list[int]:
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw)

    def template_str(self, messages: list[dict], add_generation_prompt: bool, **kw) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)

    def piece(self, tid: int) -> str:
        return self.tokenizer.decode([tid])

    def generate(self, prompt_tokens: list[int], cfg: dict, echo: bool = True) -> dict:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=float(cfg["generate"]["temperature"]))
        max_tokens = int(cfg["generate"]["max_tokens"])
        text, last, t0 = "", None, time.time()
        for r in stream_generate(self.model, self.tokenizer, mx.array(prompt_tokens), max_tokens=max_tokens, sampler=sampler):
            if echo:
                print(r.text, end="", flush=True)
            text += r.text
            last = r
        stop_tok = None
        if last is not None and last.finish_reason == "stop":
            stop_tok = self.tokenizer.decode([last.token])
        return {"text": text, "finish_reason": last.finish_reason if last else None, "stop_token": stop_tok,
                "generation_tokens": last.generation_tokens if last else 0, "prompt_tokens": len(prompt_tokens),
                "seconds": round(time.time() - t0, 2), "generation_tps": round(last.generation_tps, 1) if last else None,
                "peak_memory_gb": round(last.peak_memory, 2) if last else None}

    def next_token_dist(self, prompt_tokens: list[int], top_k: int, targets: list[int]) -> dict:
        import mlx.core as mx
        logits = self.model(mx.array(prompt_tokens)[None])[0, -1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        mx.eval(logp)
        order = mx.argsort(-logp)
        top = [(int(order[i]), float(logp[order[i]])) for i in range(top_k)]
        out = {"top": [{"token": tid, "piece": self.piece(tid), "p": round(2.718281828 ** lp, 4)} for tid, lp in top], "targets": {}}
        for t in targets:
            rank = int((logp > logp[t]).sum()) + 1
            out["targets"][str(t)] = {"piece": self.piece(t), "rank": rank, "p": round(2.718281828 ** float(logp[t]), 6)}
        return out

    def heldout_loss(self, cfg: dict, mask_prompt: bool) -> dict:
        """mlx-lm 自己的 evaluate：ChatDataset(test, mask_prompt) 的平均 loss（nats/token）。"""
        from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
        from mlx_lm.tuner.trainer import evaluate
        rows = read_jsonl(DATA / "test.jsonl")
        ds = CacheDataset(ChatDataset(rows, self.tokenizer, mask_prompt=mask_prompt))
        t0 = time.time()
        loss = evaluate(self.model, ds, batch_size=int(cfg["train"]["batch_size"]), num_batches=-1,
                        max_seq_length=int(cfg["train"]["max_seq_length"]))
        return {"loss": round(float(loss), 4), "n": len(rows), "mask_prompt": mask_prompt, "seconds": round(time.time() - t0, 1)}


class FakeBackend:
    """--fake：沒有 Apple silicon 也能演練流程。輸出是假的，畫面會標明。"""
    fake = True
    _IDS = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643, "assistant": 77091, "<think>": 151667}

    def __init__(self, cfg, offline, adapter_dir=None, label=""):
        self.path, self.load_seconds = "(fake)", 0.0
        self.adapter = str(adapter_dir.relative_to(HERE)) if adapter_dir else None
        self.label = (label or cfg["model"]["short"] + (f" + {adapter_dir.name}" if adapter_dir else "")) + " [FAKE]"
        self._words = {}

    def _id(self, w: str) -> int:
        if w in self._IDS:
            return self._IDS[w]
        return self._words.setdefault(w, 1000 + len(self._words))

    def encode(self, s: str) -> list[int]:
        return [self._id(w) for w in re.findall(r"<\|[a-z_]+\|>|<think>|</think>|\n|\S+", s)]

    def template_str(self, messages, add_generation_prompt, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s

    def template(self, messages, add_generation_prompt, **kw):
        return self.encode(self.template_str(messages, add_generation_prompt))

    def piece(self, tid):
        inv = {v: k for k, v in {**self._IDS, **self._words}.items()}
        return inv.get(tid, f"<{tid}>")

    def generate(self, prompt_tokens, cfg, echo=True):
        time.sleep(0.2)
        if self.adapter:
            text, fin, stop, n = "<think>\n\n</think>\n\n(fake answer: a short reply in the trained format)", "stop", "<|im_end|>", 18
        else:
            text, fin, stop, n = "(fake continuation: the base model keeps writing a document...)\n\nQuestion: (another one)\n", "length", None, int(cfg["generate"]["max_tokens"])
        if echo:
            print(text, end="", flush=True)
        return {"text": text, "finish_reason": fin, "stop_token": stop, "generation_tokens": n, "prompt_tokens": len(prompt_tokens),
                "seconds": 0.2, "generation_tps": 90.0, "peak_memory_gb": 3.9, "fake": True}

    def next_token_dist(self, prompt_tokens, top_k, targets):
        return {"top": [{"token": 1000 + i, "piece": f"(fake {i})", "p": round(0.5 / (i + 1), 4)} for i in range(top_k)],
                "targets": {str(t): {"piece": self.piece(t), "rank": 7 if not self.adapter else 1, "p": 0.01 if not self.adapter else 0.98} for t in targets},
                "fake": True}

    def heldout_loss(self, cfg, mask_prompt):
        base = 2.1 if not self.adapter else {"A": 1.55, "B": 1.58, "C": 1.50}.get(self.adapter[-1], 1.6)
        return {"loss": base + (0.3 if not mask_prompt else 0.0), "n": 200, "mask_prompt": mask_prompt, "seconds": 0.1, "fake": True}


def backend(cfg, args, adapter: str | None = None):
    cls = FakeBackend if args.fake else MLXBackend
    adir = WEIGHTS / f"adapters_{adapter}" if adapter else None
    if adir is not None and not args.fake and not (adir / "adapters.safetensors").exists():
        raise SystemExit(f"{RED}沒有 {adir.relative_to(HERE)}/adapters.safetensors：先跑 ./present.sh rehearse --part train{RESET}")
    note(f"載入 {cfg['model']['short']}" + (f" + adapter {adapter}" if adapter else "") + f"  ←  {cfg['model']['repo']}"
         + (f"@{locked_revision(cfg['model']['repo'])[:10]}" if locked_revision(cfg['model']['repo']) else ""))
    b = cls(cfg, offline=not args.online, adapter_dir=adir)
    note(f"  完成：{b.load_seconds:.1f} 秒" + (f"，peak memory {peak_memory_gb():.2f} GB" if not b.fake and peak_memory_gb() else ""))
    return b


# ── (a) 三題 ──────────────────────────────────────────────────────
def question_prompts(b, q: dict, variant: str) -> tuple[list[int], str]:
    """variant: base_raw（原文，沒有 template）／templated（chat template + generation prompt）。"""
    if variant == "base_raw":
        return b.encode(q["text"]), q["text"]
    msgs = [{"role": "user", "content": q["text"]}]
    return b.template(msgs, add_generation_prompt=True), b.template_str(msgs, add_generation_prompt=True)


def stop_line(r: dict) -> str:
    if r["finish_reason"] == "stop":
        return f"{GREEN}stopped by itself{RESET} on {BOLD}{visible(r['stop_token'] or '?').strip()}{RESET} after {r['generation_tokens']} tokens"
    return f"{RED}did not stop{RESET} — hit the limit of {r['generation_tokens']} tokens"


def run_question(b, cfg, q, variant: str, echo: bool = True) -> dict:
    toks, shown = question_prompts(b, q, variant)
    if echo:
        tag = "raw text, no template" if variant == "base_raw" else "chat template"
        print(f"\n  {GREY}{b.label} · {tag} · prompt {len(toks)} tokens{RESET}")
        if variant != "base_raw":
            print(f"  {GREY}{visible(shown)}{RESET}", end="")
        print(f"{BOLD}", end="")
    r = b.generate(toks, cfg, echo=echo)
    if echo:
        print(f"{RESET}\n  → {stop_line(r)}   {GREY}{r['seconds']} s, {r['generation_tps']} tok/s{RESET}")
    r.update({"variant": variant, "label": q["label"], "question": q["text"], "model": b.label, "adapter": b.adapter})
    return r


def cmd_show(args, cfg):
    qs = cfg["generate"]["questions"]
    banner(cfg["demo"]["title"] + "  ·  (a) base vs. fine-tuned, three questions")
    name = args.adapter or cfg["train"]["show_adapter"]
    base = backend(cfg, args)
    sft = backend(cfg, args, adapter=name)
    print(f"\n  {len(qs)} questions. Enter: the base model continues the raw text; Enter again: the fine-tuned model (adapter {name}) answers through the chat template."
          + ("  (--base-templated: a third output, the base model through the template)" if not args.base_templated else ""))
    recs = []
    for i, q in enumerate(qs, 1):
        pause(f"\n[{i}/{len(qs)}] {q['label']}: {q['text']}   ⏎ ")
        recs.append(run_question(base, cfg, q, "base_raw"))
        if args.base_templated:
            pause("  ⏎ the base model, through the chat template ")
            recs.append(run_question(base, cfg, q, "templated"))
        pause(f"  ⏎ after SFT (adapter {name}) ")
        recs.append(run_question(sft, cfg, q, "templated"))
    pause("\n⏎ the table ")
    print_table(recs, cfg)
    LIVE.mkdir(parents=True, exist_ok=True)
    p = LIVE / f"{now().replace(':', '')}_generate{'.fake' if base.fake else ''}.json"
    p.write_text(json.dumps({"date": now(), "records": recs}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {p.relative_to(HERE)}")


def print_table(recs: list[dict], cfg: dict) -> None:
    def cell(r):
        if r is None:
            return "—"
        return ("stop" if r["finish_reason"] == "stop" else "no") + f" ({r['generation_tokens']} tok)"
    names = sorted({adapter_name(r) for r in recs if r.get("adapter")})
    hdr = f"\n  {'question':<24}{'base, raw: stopped?':<22}{'base, template':<20}" + "".join(f"{'after SFT ' + n + ': stopped?':<26}" for n in names)
    print(hdr)
    for q in cfg["generate"]["questions"]:
        rs = {r["variant"] + ("/" + adapter_name(r) if r.get("adapter") else ""): r for r in recs if r["label"] == q["label"]}
        row = f"  {q['label']:<24}{cell(rs.get('base_raw')):<22}{cell(rs.get('templated')):<20}" + "".join(f"{cell(rs.get('templated/' + n)):<26}" for n in names)
        print(row)
    print(f"\n  {DIM}stop = the model emitted a stop token ({', '.join(['<|endoftext|>'] + cfg['model'].get('extra_eos', []))}); "
          f"no = it hit max_tokens = {cfg['generate']['max_tokens']}{RESET}")


def adapter_name(r: dict) -> str:
    return r["adapter"].rsplit("_", 1)[-1]


def cmd_replay(args, cfg):
    p = REHEARSAL / ("generate.fake.json" if args.fake else "generate.json")
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse --part generate{RESET}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    banner(f"REPLAY — rehearsed {rec['date']}, not live" + ("  [FAKE]" if args.fake else ""), AMBER)
    name = args.adapter or cfg["train"]["show_adapter"]
    for q in cfg["generate"]["questions"]:
        pause(f"\n{q['label']}: {q['text']}   ⏎ ")
        for r in [r for r in rec["records"] if r["label"] == q["label"]]:
            if r["variant"] == "templated" and not r.get("adapter") and not args.base_templated:
                continue
            if r.get("adapter") and adapter_name(r) != name and not args.all_adapters:
                continue
            tag = {"base_raw": "base, raw text"}.get(r["variant"], f"adapter {adapter_name(r)}" if r.get("adapter") else "base, chat template")
            print(f"\n  {GREY}{tag} · {r['model']}{RESET}\n{BOLD}{r['text']}{RESET}\n  → {stop_line(r)}")
    print_table(rec["records"], cfg)


# ── (b) mask 與探針 ───────────────────────────────────────────────
def mask_rows(b, messages: list[dict]) -> dict:
    """照 mlx-lm ChatDataset.process：tokens 與 --mask-prompt 的 offset。"""
    tokens = b.template(messages, add_generation_prompt=False)
    offset = len(b.template(messages[:-1], add_generation_prompt=True))
    rows = []
    for i, t in enumerate(tokens):
        rows.append({"i": i, "token": t, "piece": b.piece(t),
                     "loss_default": i >= 1,             # 第 0 個 token 沒有人預測它
                     "loss_mask_prompt": i >= offset})
    return {"tokens": tokens, "offset": offset, "n": len(tokens), "n_loss_default": len(tokens) - 1,
            "n_loss_mask_prompt": len(tokens) - offset, "rows": rows,
            "rendered": b.template_str(messages, add_generation_prompt=False),
            "generation_prompt": b.template_str(messages[:-1], add_generation_prompt=True)}


def print_mask(m: dict, label: str) -> None:
    print(f"\n  one training example, {m['n']} tokens, rendered by the tokenizer's chat template ({label}):")
    print(f"  {GREY}{visible(m['rendered'])}{RESET}")
    print(f"\n  {'#':>4}  {'token':<28}{'default':^9}{'--mask-prompt':^15}")
    for r in m["rows"]:
        piece = visible(r["piece"]).replace("\n", "")
        d = "loss" if r["loss_default"] else "—"
        k = (GREEN + f"{'loss':^15}" + RESET) if r["loss_mask_prompt"] else (GREY + f"{'—':^15}" + RESET)
        mark = "  ◀ offset" if r["i"] == m["offset"] else ""
        print(f"  {r['i']:>4}  {repr(piece)[1:-1]:<28}{d:^9}{k}{mark}")
    print(f"\n  default: loss on {m['n_loss_default']} of {m['n']} tokens (all but the first);  "
          f"--mask-prompt: loss on {m['n_loss_mask_prompt']} (index ≥ offset = {m['offset']} = the length of the prompt with the generation prompt appended)")


def probes(b, cfg, messages: list[dict]) -> list[dict]:
    """三個位置：user 的 <|im_end|>⏎ 之後（template 說接 <|im_start|>）、<|im_start|> 之後（接 assistant）、assistant⏎ 之後（SFT 資料裡接 <think>）。"""
    ids = {name: b.encode(name)[0] for name in ("<|im_start|>", "<|im_end|>", "<think>")}
    ids["assistant"] = b.encode("assistant")[0]
    user_only = b.template(messages[:-1], add_generation_prompt=False)
    gen = b.template(messages[:-1], add_generation_prompt=True)
    probes_ = [("after the user turn (<|im_end|>⏎)", user_only, ids["<|im_start|>"], "<|im_start|>"),
               ("after <|im_start|>", user_only + [ids["<|im_start|>"]], ids["assistant"], "assistant"),
               ("after <|im_start|>assistant⏎ (the generation prompt)", gen, ids["<think>"], "<think>")]
    out = []
    k = int(cfg["mask"]["top_k"])
    for where, toks, target, tname in probes_:
        d = b.next_token_dist(toks, k, [target])
        t = d["targets"][str(target)]
        out.append({"where": where, "prompt_tokens": len(toks), "template_says": tname, "rank": t["rank"], "p": t["p"],
                    "top": d["top"], "model": b.label, "adapter": b.adapter})
    return out


def print_probes(ps: list[dict]) -> None:
    for p in ps:
        tops = "  ".join(f"{repr(visible(t['piece']).replace(chr(10), ''))[1:-1]} {t['p']:.3f}" for t in p["top"])
        color = GREEN if p["rank"] == 1 else (AMBER if p["rank"] <= 5 else RED)
        print(f"  {p['where']:<52} template says {BOLD}{p['template_says']:<13}{RESET} → rank {color}{p['rank']:>5}{RESET}, p = {p['p']:.4f}")
        print(f"  {GREY}{'':52} top-{len(p['top'])}: {tops}{RESET}")


def cmd_mask(args, cfg):
    ex = mask_example(cfg)
    banner(cfg["demo"]["title"] + "  ·  (b) one example, the mask, and three probes")
    b = backend(cfg, args)
    m = mask_rows(b, ex["messages"])
    print_mask(m, cfg["model"]["short"])
    pause("\n⏎ the base model's next-token distribution at three positions of the template ")
    ps = probes(b, cfg, ex["messages"])
    print(f"\n  {b.label}:")
    print_probes(ps)
    p = REHEARSAL / ("mask.fake.json" if b.fake else "mask.json")
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        for name in cfg["train"]["adapters"]:
            if name in rec.get("probes", {}):
                print(f"\n  {GREY}rehearsed {rec['date']}, adapter {name} ({cfg['train']['adapters'][name]['label']}):{RESET}")
                print_probes(rec["probes"][name])


# ── (c) held-out ─────────────────────────────────────────────────
def cmd_replay_mask(args, cfg):
    p = REHEARSAL / ("heldout.fake.json" if args.fake else "heldout.json")
    g = REHEARSAL / ("generate.fake.json" if args.fake else "generate.json")
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse --part heldout{RESET}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    ads = cfg["train"]["adapters"]
    banner(f"REPLAY — (c) the adapters, rehearsed {rec['date']}" + ("  [FAKE]" if args.fake else ""), AMBER)
    tr = rec.get("training", {})
    print(f"\n  same data ({rec['n_test']} held-out examples for the loss), same seed, same steps and learning rate; A/B differ in the mask, C = A plus LoRA on the embedding rows.\n")
    print(f"  {'adapter':<62}{'loss on prompt?':<17}{'embedding rows?':<17}{'held-out response loss':>24}{'all-token loss':>18}{'final train loss':>18}")
    for name in ["base"] + list(ads):
        h = rec["heldout"].get(name)
        if not h:
            continue
        lab = "base (no adapter)" if name == "base" else ads[name]["label"]
        onp = "—" if name == "base" else ("no" if ads[name]["mask_prompt"] else "yes")
        emb = "—" if name == "base" else ("yes" if ads[name].get("embedding") else "no")
        ft = tr.get(name, {}).get("final_train_loss")
        print(f"  {lab:<62}{onp:<17}{emb:<17}{h['response']['loss']:>20.3f} nat{h['all']['loss']:>14.3f} nat{(f'{ft:.3f}' if ft else '—'):>18}")
    if "A" in rec["heldout"] and "B" in rec["heldout"]:
        d = rec["heldout"]["A"]["response"]["loss"] - rec["heldout"]["B"]["response"]["loss"]
        print(f"\n  A − B on held-out responses: {BOLD}{d:+.3f} nat/token{RESET}  ({'A lower' if d < 0 else 'B lower' if d > 0 else 'equal'})")
    if "A" in rec["heldout"] and "C" in rec["heldout"]:
        d = rec["heldout"]["C"]["response"]["loss"] - rec["heldout"]["A"]["response"]["loss"]
        print(f"  C − A on held-out responses: {BOLD}{d:+.3f} nat/token{RESET}  (what the embedding rows buy)")
    print(f"  {DIM}response loss = mlx-lm's own evaluate() with mask_prompt=True on data/test.jsonl (nats per response token, incl. the <think> block and <|im_end|>); "
          f"all-token = mask_prompt=False. Note B's train loss is over prompt + response, A's and C's over responses only — not comparable across the mask.{RESET}")
    if g.exists():
        gen = json.loads(g.read_text(encoding="utf-8"))
        pause("\n⏎ the three answers, adapter by adapter ")
        for q in cfg["generate"]["questions"]:
            rs = {adapter_name(r): r for r in gen["records"] if r["label"] == q["label"] and r.get("adapter")}
            print(f"\n  {BOLD}{q['label']}{RESET}: {q['text']}")
            for name in ads:
                if name in rs:
                    r = rs[name]
                    print(f"  {GREY}[{name}]{RESET} {visible(r['text'].strip()[:500])}\n      → {stop_line(r)}")


# ── (d) 現場 12 步 ────────────────────────────────────────────────
def cmd_live_train(args, cfg, rehearsal: bool = False):
    lt = cfg["live_train"]
    banner(cfg["demo"]["title"] + f"  ·  (d) the slide's command, {lt['iters']} steps")
    out_dir = WEIGHTS / "adapters_live"   # 不放 runs/rehearsal/：那裡整個會上站（80 MB 的 safetensors 2026-11-08 差一點進公開 repo）
    spec = cfg["train"]["adapters"][args.adapter or cfg["train"]["show_adapter"]]
    emb = bool(spec.get("embedding"))
    if args.fake:
        a = SimpleNamespace(fine_tune_type="lora", num_layers=-1, mask_prompt=bool(spec["mask_prompt"]), iters=int(lt["iters"]), batch_size=4,
                            learning_rate=float(cfg["train"]["learning_rate"]), steps_per_report=int(lt["steps_per_report"]),
                            steps_per_eval=10 ** 9, max_seq_length=1024, seed=0, adapter_path=str(out_dir), save_every=10 ** 9,
                            lora_parameters=lora_parameters(cfg, emb))
        print(f"{DIM}{cli_line(cfg, a)}{RESET}\n")
        log = []
        for it in range(int(lt["steps_per_report"]), int(lt["iters"]) + 1, int(lt["steps_per_report"])):
            time.sleep(0.3)
            rec = {"iteration": it, "train_loss": round(2.4 - 0.05 * it, 3), "peak_memory": 4.6, "tokens_per_second": 900.0}
            print(f"  {GREY}[live FAKE]{RESET} step {it:>4}   train loss {BOLD}{rec['train_loss']:.3f}{RESET}   peak memory {BOLD}{rec['peak_memory']:.2f} GB{RESET}")
            log.append(rec)
        res = {"fake": True, "train_log": log, "cli": cli_line(cfg, a), "iters": int(lt["iters"])}
    else:
        path = model_path(cfg, offline=not args.online)
        res = train_adapter(cfg, path, bool(spec["mask_prompt"]), iters=int(lt["iters"]), steps_per_report=int(lt["steps_per_report"]),
                            adapter_dir=out_dir, label="live", with_valid=False, steps_per_eval=10 ** 9, embedding=emb)
    res["date"] = now()
    name = args.adapter or cfg["train"]["show_adapter"]
    print(f"\n  {DIM}stopped after {lt['iters']} steps on purpose; the adapter used in (a) is the rehearsed one (runs/weights/adapters_{name}, "
          f"{cfg['train']['iters']} steps of the same command).{RESET}")
    target = REHEARSAL if rehearsal else LIVE
    target.mkdir(parents=True, exist_ok=True)
    p = target / ("live_train.fake.json" if args.fake else ("live_train.json" if rehearsal else f"{now().replace(':', '')}_live_train.json"))
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    note(f"→ {p.relative_to(HERE)}")


# ── inspect ──────────────────────────────────────────────────────
def safetensors_param_count(model_dir: Path) -> int | None:
    import struct
    total = 0
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            c = 1
            for s in v["shape"]:
                c *= s
            total += c
    return total


def n_lora_formula(cfgj: dict, r: int) -> dict:
    d, h, kv, hd, ff, L = cfgj["hidden_size"], cfgj["num_attention_heads"], cfgj["num_key_value_heads"], cfgj["head_dim"], cfgj["intermediate_size"], cfgj["num_hidden_layers"]
    per = {"q": (d + h * hd) * r, "k": (d + kv * hd) * r, "v": (d + kv * hd) * r, "o": (h * hd + d) * r,
           "gate": (d + ff) * r, "up": (d + ff) * r, "down": (ff + d) * r}
    return {"per_layer": per, "per_layer_total": sum(per.values()), "layers": L, "N_LoRA": sum(per.values()) * L}


def cmd_inspect(args, cfg):
    v = versions()
    banner(cfg["demo"]["title"] + "  ·  inspect")
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    di = device_info() if not args.fake else {}
    if di:
        print(f"  device           {di.get('device_name')}  memory {int(di.get('memory_size', 0)) / 2**30:.0f} GiB  "
              f"max recommended working set {int(di.get('max_recommended_working_set_size', 0)) / 2**30:.0f} GiB")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"  {RED}這台不是 Apple silicon：只能用 --fake 演練{RESET}")
    rec = {"date": now(), "versions": v, "device": di, "model": cfg["model"]["repo"]}
    # model files
    try:
        mp = Path(model_path(cfg, offline=not args.online, weights=not args.fake))
    except Exception as e:   # noqa: BLE001
        print(f"  {RED}模型檔不在本機（{e.__class__.__name__}）：先跑 ./present.sh fetch{RESET}")
        mp = None
    if mp is not None:
        cfgj = json.loads((mp / "config.json").read_text(encoding="utf-8"))
        tc = json.loads((mp / "tokenizer_config.json").read_text(encoding="utf-8"))
        gc = json.loads((mp / "generation_config.json").read_text(encoding="utf-8")) if (mp / "generation_config.json").exists() else {}
        print(f"\n  config.json: {cfgj['num_hidden_layers']} layers, d = {cfgj['hidden_size']}, d_ff = {cfgj['intermediate_size']}, "
              f"{cfgj['num_attention_heads']} heads / {cfgj['num_key_value_heads']} KV heads, head_dim {cfgj['head_dim']}, "
              f"V = {cfgj['vocab_size']:,}, tie_word_embeddings = {cfgj.get('tie_word_embeddings')}, dtype {cfgj.get('torch_dtype')}")
        has_tpl = bool(tc.get("chat_template")) or (mp / "chat_template.jinja").exists()
        print(f"  tokenizer_config.json: chat_template {'present' if has_tpl else 'ABSENT'}; eos_token {tc.get('eos_token')!r}; pad_token {tc.get('pad_token')!r}; bos_token {tc.get('bos_token')!r}")
        print(f"  generation_config.json: eos_token_id {gc.get('eos_token_id')}  → mlx-lm stops on these plus the tokenizer's eos; "
              f"this demo adds {cfg['model'].get('extra_eos')} (p9: 'stopped by itself' is judged against that set)")
        N = safetensors_param_count(mp)
        if N:
            print(f"  safetensors: N = {N:,} parameters → 2N = {2 * N / 1e9:.2f} GB frozen bf16, 16N = {16 * N / 1e9:.2f} GB full fine-tuning")
        nl = n_lora_formula(cfgj, int(cfg["train"]["rank"]))
        print(f"  LoRA r = {cfg['train']['rank']} on q,k,v,o,gate,up,down of every block: {nl['per_layer_total']:,} per layer × {nl['layers']} = N_LoRA = {nl['N_LoRA']:,}"
              f" ({100 * nl['N_LoRA'] / N:.2f}% of N; 16 N_LoRA = {16 * nl['N_LoRA'] / 1e9:.3f} GB)" if N else "")
        n_emb = (cfgj["vocab_size"] + cfgj["hidden_size"]) * int(cfg["train"]["rank"])
        print(f"  adapter C adds LoRA on the (tied) embedding: (V + d) r = ({cfgj['vocab_size']:,} + {cfgj['hidden_size']}) × {cfg['train']['rank']} = {n_emb:,} more → {nl['N_LoRA'] + n_emb:,} trainable")
        nl["N_LoRA_embedding_extra"] = n_emb
        rec.update({"config": cfgj, "chat_template_present": has_tpl, "eos_token": tc.get("eos_token"), "generation_eos_token_id": gc.get("eos_token_id"),
                    "N": N, "N_LoRA_formula": nl})
        # against the slides' json
        dj = (HERE / cfg["check"]["data_json"]).resolve()
        if dj.exists():
            led = json.loads(dj.read_text(encoding="utf-8"))["ledger"][cfg["check"]["ledger_key"]]
            ok_n = led["N_LoRA"] == nl["N_LoRA"]
            ok_m = N is not None and abs(led["M_full_GB"] - 16 * N / 1e9) < 1e-6
            print(f"  w08-data.json [{cfg['check']['ledger_key']}]: N_LoRA {led['N_LoRA']:,} {'✓' if ok_n else '✗'};  M_full {led['M_full_GB']:.3f} GB {'✓' if ok_m else '✗'};  "
                  f"M_lora_weights {led['M_lora_weights_GB']:.3f} GB;  M_lora {led['M_lora_GB']:.3f} GB (weights + 16 bytes per LoRA parameter; no activations)")
            rec["check"] = {"N_LoRA_ok": ok_n, "M_full_ok": ok_m, "ledger": led}
        # trainable count from mlx itself
        if not args.fake:
            try:
                from mlx_lm.tuner.utils import linear_to_lora_layers
                from mlx_lm.utils import load
                from mlx.utils import tree_flatten
                model, tokenizer = load(str(mp))
                model.freeze()
                linear_to_lora_layers(model, int(cfg["train"]["num_layers"]), {"rank": int(cfg["train"]["rank"]), "dropout": 0.0, "scale": float(cfg["train"]["scale"])})
                n_tr = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
                n_all = sum(x.size for _, x in tree_flatten(model.parameters()))
                print(f"  mlx linear_to_lora_layers(num_layers={cfg['train']['num_layers']}): trainable {n_tr:,} {'✓ = formula' if n_tr == nl['N_LoRA'] else '✗ ≠ formula'}; "
                      f"all parameters incl. LoRA {n_all:,}; eos_token_ids after load {sorted(tokenizer.eos_token_ids)}")
                rec["mlx_trainable"] = n_tr
                rec["mlx_eos_token_ids"] = sorted(tokenizer.eos_token_ids)
                del model
            except Exception as e:   # noqa: BLE001
                print(f"  {AMBER}mlx 計數略過：{e.__class__.__name__}: {e}{RESET}")
    # data
    if (DATA / "train.jsonl").exists():
        meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8")) if (DATA / "meta.json").exists() else {}
        print(f"\n  data/: " + ", ".join(f"{k} {s['n']}" for k, s in meta.get("splits", {}).items()) + f"  ← {meta.get('source', {}).get('repo')} @ {str(meta.get('source', {}).get('revision'))[:10]} ({meta.get('source', {}).get('license')}), seed {meta.get('seed')}")
        tok = None
        if mp is not None:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(mp))
            except Exception as e:   # noqa: BLE001
                print(f"  {AMBER}tokenizer 載不起來（{e.__class__.__name__}），token 長度略過{RESET}")
        if tok is not None:
            lens = {}
            for name in ("train", "valid", "test"):
                rows = read_jsonl(DATA / f"{name}.jsonl")
                full = [len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=False)) for r in rows]
                resp = [len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=False)) - len(tok.apply_chat_template(r["messages"][:-1], tokenize=True, add_generation_prompt=True, return_dict=False)) for r in rows]
                full_s, resp_s = sorted(full), sorted(resp)
                lens[name] = {"n": len(rows), "tokens_mean": round(sum(full) / len(full), 1), "tokens_median": full_s[len(full) // 2], "tokens_p90": full_s[int(0.9 * len(full))],
                              "tokens_max": full_s[-1], "response_tokens_mean": round(sum(resp) / len(resp), 1), "response_share": round(sum(resp) / sum(full), 3),
                              "over_max_seq_length": sum(1 for x in full if x > int(cfg["train"]["max_seq_length"]))}
                s = lens[name]
                print(f"  {name:<6} tokens/example mean {s['tokens_mean']:>6}  median {s['tokens_median']:>4}  p90 {s['tokens_p90']:>4}  max {s['tokens_max']:>4}   "
                      f"response share {100 * s['response_share']:.0f}%   over {cfg['train']['max_seq_length']}: {s['over_max_seq_length']}")
            t = cfg["train"]
            steps_per_epoch = lens["train"]["n"] / int(t["batch_size"])
            print(f"  {t['iters']} iters × batch {t['batch_size']} = {t['iters'] * int(t['batch_size']):,} examples = {t['iters'] / steps_per_epoch:.1f} epochs; "
                  f"≈ {t['iters'] * int(t['batch_size']) * lens['train']['tokens_mean'] / 1e6:.2f} M tokens seen (incl. padding to 32)")
            rec["data"] = lens
            ex = mask_example(cfg)
            s = tok.apply_chat_template(ex["messages"], tokenize=False)
            print(f"\n  the (b) example (data/example.jsonl), rendered: {len(tok.apply_chat_template(ex['messages'], tokenize=True, return_dict=False))} tokens")
            print(f"  {GREY}{visible(s)}{RESET}")
    else:
        print(f"\n  {AMBER}data/ 還沒建：先跑 ./present.sh prepare{RESET}")
    t = cfg["train"]
    print(f"\n  LoRA scale {t['scale']} with rank {t['rank']}: mlx writes y = Wx + scale·BAx, so scale = α/r → α = {float(t['scale']) * int(t['rank']):g} in Hu et al.'s notation (p27)")
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    p = REHEARSAL / ("inspect.fake.json" if args.fake else "inspect.json")
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    note(f"→ {p.relative_to(HERE)}")


# ── fetch ────────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    m = cfg["model"]
    info = api.model_info(m["repo"], revision=m.get("revision") or None)
    banner(f"下載 {m['repo']} @ {info.sha}")
    snapshot_download(m["repo"], revision=info.sha, ignore_patterns=NON_MODEL_FILES)
    write_lock({revision_key(m["repo"]): info.sha})
    d = cfg["data"]
    dinfo = api.dataset_info(d["repo"])
    banner(f"下載 {d['repo']} @ {dinfo.sha}  ({d['license']})")
    hf_hub_download(d["repo"], d["file"], repo_type="dataset", revision=dinfo.sha)
    write_lock({dataset_lock_key(d["repo"]): dinfo.sha})
    print(f"{GREEN}完成，兩個 revision 已寫進 versions.lock；接著 ./present.sh prepare{RESET}")


# ── rehearse ─────────────────────────────────────────────────────
def cmd_rehearse(args, cfg):
    parts = args.part or ["train", "generate", "mask", "heldout", "live"]
    out = REHEARSAL
    out.mkdir(parents=True, exist_ok=True)
    sfx = ".fake" if args.fake else ""
    v = versions()
    banner(cfg["demo"]["title"] + "  ·  rehearse " + " → ".join(parts) + ("  [FAKE]" if args.fake else ""))
    if "train" in parts:
        names = [args.adapter] if args.adapter else list(cfg["train"]["adapters"].keys())
        p = out / f"adapters{sfx}.json"
        rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"adapters": {}}
        rec.update({"date": now(), "versions": v, "device": device_info() if not args.fake else {}})
        for name in names:
            spec = cfg["train"]["adapters"][name]
            adir = WEIGHTS / f"adapters_{name}"
            banner(f"train adapter {name}: {spec['label']}", AMBER)
            if args.fake:
                log = [{"iteration": it, "train_loss": round(2.5 - 1.0 * (1 - 0.97 ** it), 3), "peak_memory": 4.6, "tokens_per_second": 900.0}
                       for it in range(10, int(cfg["train"]["iters"]) + 1, 10)]
                res = {"label": spec["label"], "mask_prompt": spec["mask_prompt"], "iters": int(cfg["train"]["iters"]), "train_log": log,
                       "val_log": [{"iteration": 0, "val_loss": 2.6}, {"iteration": int(cfg["train"]["iters"]) - 1, "val_loss": 1.6}], "train_seconds": 1.0, "fake": True}
                adir.mkdir(parents=True, exist_ok=True)
            else:
                path = model_path(cfg, offline=not args.online)
                res = train_adapter(cfg, path, bool(spec["mask_prompt"]), int(cfg["train"]["iters"]), int(cfg["train"]["steps_per_report"]), adir, label=name,
                                    embedding=bool(spec.get("embedding")))
            res["final_train_loss"] = res["train_log"][-1]["train_loss"] if res["train_log"] else None
            res["final_val_loss"] = res["val_log"][-1]["val_loss"] if res["val_log"] else None
            res["date"] = now()
            rec["adapters"][name] = res
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
            print(f"\n  {GREEN}adapter {name} done{RESET}: {res['train_seconds']} s, final train loss {res['final_train_loss']}, final val loss {res['final_val_loss']}"
                  + (f", peak {res.get('peak_memory_gb'):.2f} GB" if res.get("peak_memory_gb") else ""))
        note(f"→ {p.relative_to(HERE)}")
    if "generate" in parts:
        banner("(a) three questions × {base raw, base templated, A, B}", AMBER)
        recs = []
        base = backend(cfg, args)
        for q in cfg["generate"]["questions"]:
            print(f"\n{BOLD}{q['label']}{RESET}: {q['text']}")
            recs.append(run_question(base, cfg, q, "base_raw"))
            recs.append(run_question(base, cfg, q, "templated"))
        del base
        for name in cfg["train"]["adapters"]:
            b = backend(cfg, args, adapter=name)
            for q in cfg["generate"]["questions"]:
                print(f"\n{BOLD}{q['label']}{RESET}: {q['text']}")
                recs.append(run_question(b, cfg, q, "templated"))
            del b
        p = out / f"generate{sfx}.json"
        p.write_text(json.dumps({"date": now(), "versions": v, "max_tokens": cfg["generate"]["max_tokens"], "temperature": cfg["generate"]["temperature"],
                                 "records": recs}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print_table(recs, cfg)
        note(f"→ {p.relative_to(HERE)}")
    if "mask" in parts:
        banner("(b) the mask and the probes: base, A, B", AMBER)
        ex = mask_example(cfg)
        rec = {"date": now(), "example": ex, "probes": {}}
        for name in [None] + list(cfg["train"]["adapters"]):
            b = backend(cfg, args, adapter=name)
            if name is None:
                m = mask_rows(b, ex["messages"])
                print_mask(m, cfg["model"]["short"])
                rec["mask"] = {k: v for k, v in m.items()}
                (out / f"mask{sfx}.txt").write_text(m["rendered"], encoding="utf-8")
            ps = probes(b, cfg, ex["messages"])
            print(f"\n  {b.label}:")
            print_probes(ps)
            rec["probes"][name or "base"] = ps
            del b
        p = out / f"mask{sfx}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        note(f"→ {p.relative_to(HERE)}, mask{sfx}.txt")
    if "heldout" in parts:
        banner("(c) held-out loss on data/test.jsonl: base, A, B", AMBER)
        rec = {"date": now(), "n_test": len(read_jsonl(DATA / "test.jsonl")), "heldout": {}, "training": {}}
        ap = out / f"adapters{sfx}.json"
        if ap.exists():
            tr = json.loads(ap.read_text(encoding="utf-8"))["adapters"]
            rec["training"] = {k: {"final_train_loss": v.get("final_train_loss"), "final_val_loss": v.get("final_val_loss"), "mask_prompt": v.get("mask_prompt")} for k, v in tr.items()}
        for name in [None] + list(cfg["train"]["adapters"]):
            b = backend(cfg, args, adapter=name)
            r = {"response": b.heldout_loss(cfg, mask_prompt=True), "all": b.heldout_loss(cfg, mask_prompt=False)}
            print(f"  {b.label:<44} response-only {BOLD}{r['response']['loss']:.3f}{RESET} nat/token   all tokens {r['all']['loss']:.3f}   ({r['response']['seconds']} + {r['all']['seconds']} s)")
            rec["heldout"][name or "base"] = r
            del b
        p = out / f"heldout{sfx}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        note(f"→ {p.relative_to(HERE)}")
    if "live" in parts:
        cmd_live_train(args, cfg, rehearsal=True)


# ── main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "prepare", "inspect", "rehearse", "show", "mask", "replay-mask", "live-train", "replay"])
    ap.add_argument("--fake", action="store_true", help="沒有 mlx 也能演練（輸出是假的）")
    ap.add_argument("--online", action="store_true", help="允許連網（預設 HF_HUB_OFFLINE）")
    ap.add_argument("--part", action="append", choices=["train", "generate", "mask", "heldout", "live"], help="rehearse：只做某幾段")
    ap.add_argument("--adapter", choices=["A", "B", "C"], help="rehearse --part train：只訓一個；show／replay／live-train：用哪一個（預設 show_adapter）")
    ap.add_argument("--all-adapters", action="store_true", help="replay：三題每個 adapter 的輸出都印")
    ap.add_argument("--base-templated", action="store_true", help="show／replay：多印 base 套 chat template 的那一欄")
    ap.add_argument("--dolly", help="prepare：直接指定 Dolly 的 jsonl（不經 HF cache）")
    a = ap.parse_args()
    cfg = load_cfg()
    {"fetch": cmd_fetch, "prepare": cmd_prepare, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "show": cmd_show, "mask": cmd_mask,
     "replay-mask": cmd_replay_mask, "live-train": cmd_live_train, "replay": cmd_replay}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
