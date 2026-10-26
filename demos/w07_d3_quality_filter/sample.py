#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 3 的 (a) 半邊：從兩份語料抽頁、打亂、用 FineWeb-Edu 的分類器打分；另外抓 Wikipedia 的 held-out。

  read_pool(cfg, cfg2, which, n)        讀 JSONL 前 n 篇（id、dump、url、language_score、score、text）
  pick_pages(cfg, cfg2, seed, exclude)  每份語料各抽 per_corpus 篇（長度與 language_score 的條件），合在一起用 seed 打亂；回 list
  Classifier(cfg, offline)              載 HuggingFaceFW/fineweb-edu-classifier（BERT + 迴歸頭）；.score(text) → float，.int_score → 0–5
  fetch_classifier(cfg)                 下載並把 revision 寫進 ../versions.lock
  fetch_wikipedia(cfg, cfg2)            讀 wikimedia/wikipedia 的一個 row group，用 seed 挑整篇到 target_mb，寫 ../w07_corpus/wikipedia_heldout.jsonl
  load_wikipedia(cfg, cfg2)             → bytes（文章以空行相接）與 meta

分類器的用法照 model card（logits 一個數，int_score = round(clip(score, 0, 5))）；FineWeb-Edu 留的是 int_score ≥ 3。
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


COMMON = load_module(HERE.parent / "common.py", "nlp_demos_common")      # versions.lock 的讀寫（revision 以 repo 為鍵，demo 之間共用）


# ── 抽頁 ─────────────────────────────────────────────────────────
def corpus_dir(cfg: dict, cfg2: dict) -> Path:
    """demo 2 的 [corpus].dir 是相對 demo 2 資料夾寫的，所以先走到 demo 2 再解析。"""
    return (HERE / cfg["shared"]["demo2"] / cfg2["corpus"]["dir"]).resolve()


def corpus_file(cfg: dict, cfg2: dict, which: str) -> Path:
    return corpus_dir(cfg, cfg2) / cfg2["corpus"][which]["file"]


def read_pool(cfg: dict, cfg2: dict, which: str, n: int | None = None) -> list[dict]:
    p = corpus_file(cfg, cfg2, which)
    if not p.exists():
        raise FileNotFoundError(f"沒有語料 {p}：先到 ../w07_d2_scaling_fit 跑 ./present.sh fetch --corpus both")
    n = n or cfg["sample"]["pool"]
    meta_p = p.with_name(p.name + ".meta.json")
    total = json.loads(meta_p.read_text(encoding="utf-8")).get("docs") if meta_p.exists() else None
    stride = max(1, (total or n) // n)                      # 等距取：shard 裡的列依 dump 排，只讀「前面」會全是同一個 dump
    out = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride == 0:
                out.append(json.loads(line))
                if len(out) >= n:
                    break
    return out


def eligible(r: dict, sc: dict) -> bool:
    t = r["text"]
    return sc["min_chars"] <= len(t) <= sc["max_chars"] and (r.get("language_score") or 0.0) >= sc["min_language_score"] \
        and r["id"] not in sc["exclude"]


def pick_pages(cfg: dict, cfg2: dict, seed: int | None = None, exclude: list[str] | None = None) -> list[dict]:
    sc = dict(cfg["sample"])
    sc["exclude"] = list(sc["exclude"]) + list(exclude or [])
    seed = cfg["sample"]["seed"] if seed is None else seed
    pages = []
    for which in ("web", "edu"):
        pool = [r for r in read_pool(cfg, cfg2, which) if eligible(r, sc)]
        if len(pool) < sc["per_corpus"]:
            raise SystemExit(f"{RED}{which} 符合條件的頁面只有 {len(pool)} 篇，放寬 [sample] 的 min/max_chars 或加大 pool{RESET}")
        rng = random.Random(f"{seed}-{which}")
        for r in rng.sample(pool, sc["per_corpus"]):
            pages.append({"id": r["id"], "source": which, "dump": r.get("dump"), "url": r.get("url"), "language_score": r.get("language_score"),
                          "dataset_score": r.get("score"), "chars": len(r["text"]), "text": r["text"]})
    random.Random(f"{seed}-shuffle").shuffle(pages)
    for i, p in enumerate(pages, 1):
        p["page"] = i
    return pages


def show_page(p: dict, cfg: dict, reveal: bool = False) -> None:
    n = cfg["sample"]["show_chars"]
    t = p["text"]
    body = t if len(t) <= n else t[:n].rstrip() + " …"
    head = f"{BOLD}page {p['page']}{RESET}  {DIM}{p['chars']} chars{RESET}"
    if reveal:
        lab = cfg["corpora"][p["source"]]
        head += f"   {GREEN if p['source'] == 'edu' else AMBER}{lab}{RESET}  {DIM}{p['dump']}  {p['url']}{RESET}"
    print(f"\n{head}\n{GREY}{'─' * 72}{RESET}")
    print(body)
    print(f"{GREY}{'─' * 72}{RESET}")


# ── 分類器 ───────────────────────────────────────────────────────
IGNORE = ["src/*", "*.bin", "*.slurm", "utils/*"]          # 不抓訓練腳本與 training_args.bin；prompt.txt 是標註用的 prompt，README 有引


def pick_device(name: str):
    import torch
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name != "cpu":
        print(f"{AMBER}{name} 不可用，分類器改用 CPU{RESET}")
    return torch.device("cpu")


def fetch_classifier(cfg: dict) -> str:
    from huggingface_hub import HfApi, snapshot_download
    c = cfg["classifier"]
    api = HfApi()
    info = api.model_info(c["repo"], revision=c["revision"] or None)
    t0 = time.time()
    path = snapshot_download(c["repo"], revision=info.sha, ignore_patterns=IGNORE)
    COMMON.write_lock({COMMON.revision_key(c["repo"]): info.sha})
    print(f"{GREEN}{c['repo']} @ {info.sha[:10]} → {path}{RESET}  ({time.time() - t0:.0f} s; revision 寫進 ../versions.lock)")
    return info.sha


class Classifier:
    """HuggingFaceFW/fineweb-edu-classifier：BertForSequenceClassification、num_labels = 1（regression）。score 是連續值，int_score 是 0–5。"""

    def __init__(self, cfg: dict, offline: bool = True):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        c = cfg["classifier"]
        self.repo = c["repo"]
        self.revision = COMMON.locked_revision(c["repo"], c["revision"] or None)
        t0 = time.time()
        self.path = snapshot_download(c["repo"], revision=self.revision, local_files_only=offline, ignore_patterns=IGNORE)
        self.device = pick_device(c["device"])
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path).to(self.device).eval()
        self.max_length = c["max_length"]
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.load_s = time.time() - t0
        import transformers
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def score(self, text: str) -> dict:
        import torch
        t0 = time.time()
        x = self.tok(text, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            s = self.model(**{k: v.to(self.device) for k, v in x.items()}).logits.squeeze(-1).float().item()
        return {"score": s, "int_score": int(round(max(0.0, min(s, 5.0)))), "tokens": int(x["input_ids"].shape[1]),
                "truncated": bool(x["input_ids"].shape[1] >= self.max_length), "seconds": time.time() - t0}

    def describe(self) -> str:
        return (f"{self.repo}" + (f"@{self.revision[:10]}" if self.revision else "") + f": {type(self.model).__name__}, {self.n_params / 1e6:.1f} M params, "
                f"device {self.device}, max_length {self.max_length}; loaded in {self.load_s:.1f} s; torch {self.versions['torch']}, transformers {self.versions['transformers']}")


def score_distribution(clf: Classifier, cfg: dict, cfg2: dict, n: int, seed: int) -> dict:
    """每份語料隨機 n 篇的分數分布（彩排用；課堂不做）。回 {which: {"scores": [...], "hist": [6], "mean": ...}}。"""
    out = {}
    for which in ("web", "edu"):
        pool = read_pool(cfg, cfg2, which, n=max(cfg["sample"]["pool"], 4 * n))
        rng = random.Random(f"{seed}-dist-{which}")
        docs = rng.sample(pool, min(n, len(pool)))
        t0 = time.time()
        scores = [clf.score(d["text"]) for d in docs]
        hist = [0] * 6
        for s in scores:
            hist[s["int_score"]] += 1
        ds = [d.get("score") for d in docs if d.get("score") is not None]
        out[which] = {"n": len(docs), "mean": sum(s["score"] for s in scores) / len(scores), "hist": hist,
                      "frac_ge3": sum(1 for s in scores if s["int_score"] >= 3) / len(scores), "truncated": sum(1 for s in scores if s["truncated"]),
                      "seconds": time.time() - t0, "dataset_score_mean": (sum(ds) / len(ds)) if ds else None,
                      "max_abs_diff_vs_dataset": max((abs(s["score"] - d["score"]) for s, d in zip(scores, docs) if d.get("score") is not None), default=None)}
    return out


# ── Wikipedia held-out ────────────────────────────────────────────
def wiki_paths(cfg: dict, cfg2: dict) -> tuple[Path, Path]:
    d = corpus_dir(cfg, cfg2)
    return d / cfg["wikipedia"]["file"], d / (cfg["wikipedia"]["file"] + ".meta.json")


def fetch_wikipedia(cfg: dict, cfg2: dict, quiet: bool = False) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(f"{RED}沒有 pyarrow（讀 parquet 用）：cd ~/Course-Hub/nlp_llm/demos && uv pip install --python .venv/bin/python pyarrow{RESET}")
    from huggingface_hub import HfFileSystem
    w = cfg["wikipedia"]
    out, meta_p = wiki_paths(cfg, cfg2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    t0 = time.time()
    with fs.open(f"datasets/{w['repo']}/{w['path']}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        rg = min(w["row_group"], md.num_row_groups - 1)
        if not quiet:
            print(f"{BOLD}fetch wikipedia{RESET}  {w['repo']} · {w['path']} · {md.num_rows:,} rows in {md.num_row_groups} row groups · license {w['license']}; reading row group {rg}")
        rows = pf.read_row_group(rg, columns=["id", "title", "text"]).to_pylist()
    rng = random.Random(w["seed"])
    rng.shuffle(rows)
    target = int(w["target_mb"] * 1_048_576)
    picked, n_bytes = [], 0
    for r in rows:
        if len(r["text"]) > w["max_article_chars"]:
            continue
        picked.append(r)
        n_bytes += len(r["text"].encode("utf-8")) + 2
        if n_bytes >= target:
            break
    with open(out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({"id": r["id"], "title": r["title"], "text": r["text"]}, ensure_ascii=False) + "\n")
    meta = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "repo": w["repo"], "path": w["path"], "license": w["license"], "row_group": rg, "rows_in_row_group": len(rows),
            "seed": w["seed"], "articles": len(picked), "text_bytes": n_bytes, "titles": [r["title"] for r in picked], "seconds": time.time() - t0}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not quiet:
        print(f"  {GREEN}{len(picked)} articles, {n_bytes / 1e6:.2f} MB{RESET} in {meta['seconds']:.0f} s → {out.name}  (first titles: " + "; ".join(meta["titles"][:5]) + ")")
    return meta


def load_wikipedia(cfg: dict, cfg2: dict) -> tuple[bytes, dict]:
    out, meta_p = wiki_paths(cfg, cfg2)
    if not out.exists():
        raise FileNotFoundError(f"沒有 {out}：先跑 ./present.sh fetch（需要網路）")
    arts = [json.loads(line) for line in open(out, encoding="utf-8")]
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return b"\n\n".join(a["text"].encode("utf-8") for a in arts), {**meta, "articles": len(arts)}
