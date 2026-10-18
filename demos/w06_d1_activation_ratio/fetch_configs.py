#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 1 的課前抓取：三個模型各抓 config.json、model.safetensors.index.json，以及每個 shard 的 safetensors header。

    ./present.sh fetch                 # demo_config.toml 裡 local = false 的模型全抓
    ./present.sh fetch qwen30          # 只抓一個
    python3 fetch_configs.py --force   # 已有也重抓

**不抓權重。** safetensors 的檔頭是「8 bytes 的 little-endian 長度 N，接著 N bytes 的 JSON（每個張量的 dtype、shape、offset）」，
所以兩個 HTTP range 請求就拿得到整份形狀表——一個 30B 的 bf16 checkpoint 有十幾個 shard，每個 header 幾十 KB，全部幾 MB、壓縮後不到 1 MB。
純標準函式庫（urllib），不需要 huggingface_hub；huggingface.co 的 /resolve/ 會 302 到 CDN，urllib 跟著跳並保留 Range header。

結果放 configs/<org>--<name>/：
    config.json                     原樣
    model.safetensors.index.json    原樣（只有一個 shard 的模型沒有這個檔，會寫一個只含 weight_map 的替代品）
    tensors.json.gz                 {name: {"dtype", "shape", "file"}} 加 "__meta__"（total_size、shard 數、抓取時間、HF 回的 commit）；gzip，
                                    因為 Qwen3-Next 有七萬多個張量名、未壓縮 8 MB，壓縮後三個模型合計不到 1 MB
"""
from __future__ import annotations

import gzip
import json
import struct
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "demo_config.toml"
OUT = HERE / "configs"
BOLD, DIM, RESET, GREEN, RED, AMBER = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m"
UA = "course-hub-w06-demo1/1.0 (python-urllib; fetches config + safetensors headers only)"


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def repo_dir(repo: str) -> Path:
    return OUT / repo.replace("/", "--")


def _url(endpoint: str, repo: str, rev: str, filename: str) -> str:
    return f"{endpoint}/{repo}/resolve/{rev}/{filename}"


def _get(url: str, rng: tuple[int, int] | None = None) -> tuple[bytes, dict]:
    headers = {"User-Agent": UA}
    if rng is not None:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), dict(r.headers)


def read_remote_header(url: str) -> dict:
    """safetensors：前 8 bytes 是 header 長度 N（<Q），接著 N bytes 的 JSON。兩個 range 請求。"""
    head, hdrs = _get(url, (0, 7))
    if len(head) != 8:
        raise RuntimeError(f"range 請求沒有被尊重（回了 {len(head)} bytes）：{url}")
    (n,) = struct.unpack("<Q", head)
    body, _ = _get(url, (8, 8 + n - 1))
    if len(body) != n:
        raise RuntimeError(f"header 長度不符：要 {n}，拿到 {len(body)}：{url}")
    return json.loads(body.decode("utf-8"))


def fetch_one(repo: str, endpoint: str, rev: str, force: bool = False) -> Path:
    d = repo_dir(repo)
    if d.joinpath("tensors.json.gz").exists() and not force:
        print(f"{DIM}{repo}: configs/{d.name}/ 已有，跳過（--force 重抓）{RESET}")
        return d
    d.mkdir(parents=True, exist_ok=True)
    print(f"{BOLD}{repo}{RESET}")
    # 1. config.json
    raw, hdrs = _get(_url(endpoint, repo, rev, "config.json"))
    commit = hdrs.get("X-Repo-Commit") or hdrs.get("x-repo-commit")
    (d / "config.json").write_bytes(raw)
    print(f"  config.json  {len(raw):,} bytes   commit {commit or '?'}")
    # 2. index（沒有就是單一 shard）
    try:
        raw_idx, _ = _get(_url(endpoint, repo, rev, "model.safetensors.index.json"))
        index = json.loads(raw_idx)
        (d / "model.safetensors.index.json").write_bytes(raw_idx)
        shards = sorted(set(index["weight_map"].values()))
        total_size = (index.get("metadata") or {}).get("total_size")
        print(f"  index        {len(shards)} shards   metadata.total_size = {total_size:,} bytes" if total_size else f"  index        {len(shards)} shards")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        shards, total_size, index = ["model.safetensors"], None, None
        print("  index        沒有 index：單一 model.safetensors")
    # 3. 每個 shard 的 header
    tensors: dict = {}
    for i, fn in enumerate(shards, 1):
        h = read_remote_header(_url(endpoint, repo, rev, fn))
        meta = h.pop("__metadata__", None)
        for k, v in h.items():
            tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": fn}
        print(f"\r  headers      {i}/{len(shards)}  {fn}  ({len(h)} tensors)", end="", flush=True)
    print()
    if index is None:
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"note": "synthesised locally: the repo has a single shard and no index"},
                        "weight_map": {k: "model.safetensors" for k in tensors}}, indent=1), encoding="utf-8")
    tensors["__meta__"] = {"repo": repo, "revision": rev, "commit": commit, "fetched": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                           "n_shards": len(shards), "n_tensors": len(tensors), "index_total_size_bytes": total_size,
                           "how": "config.json + model.safetensors.index.json + the 8-byte-length-prefixed JSON header of every shard, via HTTP range requests; no weights"}
    with gzip.open(d / "tensors.json.gz", "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(tensors, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  tensors.json.gz  {len(tensors) - 1:,} tensors, {(d / 'tensors.json.gz').stat().st_size / 1024:,.0f} KB → configs/{d.name}/")
    return d


def main(argv: list[str]) -> int:
    cfg = load_cfg()
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("--")]
    models = cfg["models"]
    if not names:
        names = [n for n, m in models.items() if not m.get("local")]
    bad = [n for n in names if n not in models]
    if bad:
        print(f"{RED}不認得 {bad}：demo_config.toml 裡的名字是 {', '.join(models)}{RESET}")
        return 2
    rc = 0
    for n in names:
        m = models[n]
        if m.get("local"):
            print(f"{DIM}{n}: local = true，讀本機 HF cache，不抓{RESET}")
            continue
        try:
            fetch_one(m["repo"], cfg["fetch"]["hf_endpoint"], cfg["fetch"].get("revision", "main"), force)
        except Exception as exc:                                  # noqa: BLE001
            print(f"{RED}{n}: {type(exc).__name__}: {exc}{RESET}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
