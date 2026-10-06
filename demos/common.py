# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""nlp_llm/demos 底下所有 demo 共用的小工具。

- 共用環境：demos/.venv（setup.sh 建），版本鎖在 demos/versions.lock
- 模型 revision 以 repo 為鍵（MODEL_REVISION__<repo>），不同 demo 用到同一個模型時共用同一個鎖
- 終端機樣式、版本資訊、MLX 記憶體查詢
只用標準函式庫；mlx 相關的東西都在函式裡才 import，--fake 模式不需要 mlx。
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import re
from pathlib import Path

DEMOS = Path(__file__).resolve().parent
LOCK = DEMOS / "versions.lock"
LOCK_HEADER = "# 由 demos/setup.sh 與各 demo 的 fetch 產生。要換版本就刪掉對應那一行再重跑。"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
BLUE, AMBER, RED, GREEN, GREY = "\033[34m", "\033[33m", "\033[31m", "\033[32m", "\033[90m"


def banner(text: str, color: str = BLUE) -> None:
    line = "─" * 72
    print(f"\n{color}{line}\n{BOLD}{text}{RESET}\n{color}{line}{RESET}")


def note(text: str) -> None:
    print(f"{GREY}{text}{RESET}")


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ── 鎖定檔（KEY=value，setup.sh 也讀寫同一個格式）────────────────
def read_lock() -> dict:
    out = {}
    if LOCK.exists():
        for line in LOCK.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def write_lock(updates: dict) -> None:
    lock = read_lock()
    if all(lock.get(k) == v for k, v in updates.items()):
        return  # 內容沒變就不動檔案，避免假的 git 變更
    lock.update(updates)
    body = [LOCK_HEADER] + [f"{k}={v}" for k, v in sorted(lock.items())]
    LOCK.write_text("\n".join(body) + "\n", encoding="utf-8")


def revision_key(repo: str) -> str:
    return "MODEL_REVISION__" + re.sub(r"\W", "_", repo).upper()


def locked_revision(repo: str, configured: str | None = None) -> str | None:
    return configured or read_lock().get(revision_key(repo)) or None


# ── 版本與機器 ───────────────────────────────────────────────────
def versions() -> dict:
    from importlib import metadata
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("mlx", "mlx-lm", "mlx-metal", "transformers", "huggingface-hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    v["mlx_lm_commit"] = read_lock().get("MLX_LM_COMMIT")
    return v


def _mx_fn(name: str):
    import mlx.core as mx
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, name, None) if owner is not None else None
        if fn is not None:
            return fn
    return None


def device_info() -> dict:
    fn = _mx_fn("device_info")
    if fn is None:
        return {}
    try:
        return {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                for k, v in dict(fn()).items()}
    except Exception:
        return {}


def peak_memory_gb() -> float | None:
    fn = _mx_fn("get_peak_memory")
    try:
        return fn() / 1e9 if fn else None
    except Exception:
        return None


def reset_peak_memory() -> None:
    fn = _mx_fn("reset_peak_memory")
    try:
        if fn:
            fn()
    except Exception:
        pass


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
