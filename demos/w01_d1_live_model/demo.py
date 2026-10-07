#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W1 demo 1（投影片 p9「Same prompt. A 2026 model. Live.」）的現場 demo。

子命令（用 nlp_llm/demos/setup.sh 建好的共用 .venv 執行；通常經由 present.sh）：

    fetch     下載 demo_config.toml 裡的模型，並把實際的 revision 記進 demos/versions.lock
    check     課前體檢：套件版本、機器、模型是否已在本機、試載入並暖機
    probe     課前挑題：每題跑 N 次，粗判對錯，結果存 runs/probe-*.jsonl
    rehearse  課前彩排：兩個步驟各跑一次，輸出存成 runs/rehearsal/（現場失敗時的退路）
    show      課堂用：載入一次模型，按 Enter 逐步執行兩個步驟
    replay    現場失敗時：把彩排錄下的輸出重播出來（畫面會標明「非現場」）

所有執行紀錄（含 prompt、參數、套件版本、tok/s、峰值記憶體）都寫進 runs/，
這也是 M5 Max 上的第一批實測數字（見 nlp_llm/HANDOVER.md 6b）。

相依：mlx、mlx-lm（git commit 釘在 demos/versions.lock）、transformers。Python 3.11+（用 tomllib）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "demo_config.toml"
sys.path.insert(0, str(HERE.parent))
from common import LOCK, read_lock, write_lock, revision_key  # noqa: E402  共用的鎖定檔
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"

# ── 終端機樣式：投影到教室時要大、要清楚，不依賴任何套件 ──────────
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
BLUE, AMBER, RED, GREEN, GREY = "\033[34m", "\033[33m", "\033[31m", "\033[32m", "\033[90m"


def banner(text: str, color: str = BLUE) -> None:
    line = "─" * 72
    print(f"\n{color}{line}\n{BOLD}{text}{RESET}\n{color}{line}{RESET}")


def note(text: str) -> None:
    print(f"{GREY}{text}{RESET}")


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ── 設定與鎖定檔 ─────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    ids = [q["id"] for q in cfg["questions"]]
    if len(ids) != len(set(ids)):
        sys.exit("demo_config.toml：questions 的 id 重複")
    if cfg["step2"]["chosen"] not in ids:
        sys.exit(f"demo_config.toml：step2.chosen = {cfg['step2']['chosen']!r} 不在 questions 裡")
    return cfg


def step1_prompt(cfg: dict) -> str:
    src = (HERE / cfg["step1"]["prompt_source"]).resolve()
    data = json.loads(src.read_text(encoding="utf-8"))
    return data[cfg["step1"]["prompt_key"]]


def chosen_question(cfg: dict, qid: str | None = None) -> dict:
    qid = qid or cfg["step2"]["chosen"]
    for q in cfg["questions"]:
        if q["id"] == qid:
            return q
    sys.exit(f"找不到問題 {qid!r}；可用的有：{', '.join(q['id'] for q in cfg['questions'])}")


# ── 版本與機器資訊 ───────────────────────────────────────────────
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


def device_info() -> dict:
    """mx.device_info 與 mx.metal.device_info 在不同版本都出現過，兩個都試。"""
    import mlx.core as mx
    for fn in (getattr(mx, "device_info", None),
               getattr(getattr(mx, "metal", None), "device_info", None)):
        if fn is None:
            continue
        try:
            info = fn()
            return {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                    for k, v in dict(info).items()}
        except Exception:
            continue
    return {}


def peak_memory_gb() -> float | None:
    import mlx.core as mx
    for fn in (getattr(mx, "get_peak_memory", None),
               getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
        if fn is not None:
            try:
                return fn() / 1e9
            except Exception:
                pass
    return None


def reset_peak_memory() -> None:
    import mlx.core as mx
    for fn in (getattr(mx, "reset_peak_memory", None),
               getattr(getattr(mx, "metal", None), "reset_peak_memory", None)):
        if fn is not None:
            try:
                fn()
                return
            except Exception:
                pass


# ── 後端 ─────────────────────────────────────────────────────────
class MLXBackend:
    """真正的後端。只在 Apple silicon + mlx 可用時能跑。"""

    def __init__(self, repo: str, revision: str | None, offline: bool):
        from huggingface_hub import snapshot_download
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.repo = repo
        t0 = time.time()
        # ignore_patterns：離線時 huggingface_hub ≥1.3 會核對 snapshot 完整性，用 mlx_lm.load 抓的模型
        # 沒有 README.md / .gitattributes，不排除會被判成 IncompleteSnapshotError（2026-09-20 踩到）
        path = snapshot_download(repo, revision=revision or None, local_files_only=offline,
                                 ignore_patterns=["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"])
        self.path = path
        self.model, self.tokenizer = load(path)
        self.load_seconds = time.time() - t0

    def render_chat(self, user: str, system: str = "") -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        # enable_thinking=False：Qwen 系列預設會先「想」，課堂上不要那段；
        # 其他模型的 chat template 不認得這個變數就會忽略它。
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float, seed: int):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx.random.seed(seed)
        reset_peak_memory()
        sampler = make_sampler(temp=temperature, top_p=top_p)
        last = None
        for r in stream_generate(self.model, self.tokenizer, prompt,
                                 max_tokens=max_tokens, sampler=sampler):
            last = r
            yield r.text, None
        stats = {}
        if last is not None:
            stats = {"prompt_tokens": last.prompt_tokens,
                     "prompt_tps": round(last.prompt_tps, 1),
                     "generation_tokens": last.generation_tokens,
                     "generation_tps": round(last.generation_tps, 1),
                     "peak_memory_gb": round(last.peak_memory, 2),
                     "finish_reason": last.finish_reason}
        yield "", stats


class FakeBackend:
    """--fake：沒有 Apple silicon 也能演練流程。畫面會清楚標示這不是模型輸出。"""

    def __init__(self, repo: str, revision: str | None, offline: bool):
        self.repo, self.path, self.load_seconds = repo, "(fake)", 0.0

    def render_chat(self, user: str, system: str = "") -> str:
        return f"<user>{user}</user><assistant>"

    def stream(self, prompt, max_tokens, temperature, top_p, seed):
        text = "［FAKE 後端：這不是模型輸出，只用來演練流程］ " + \
               "台灣高鐵目前共有 8 個車站：台北、板橋、桃園、新竹、台中、嘉義、台南、左營。"
        for ch in text[:max_tokens]:
            time.sleep(0.004)
            yield ch, None
        yield "", {"prompt_tokens": len(prompt), "prompt_tps": 0.0,
                   "generation_tokens": len(text), "generation_tps": 0.0,
                   "peak_memory_gb": 0.0, "finish_reason": "fake"}


def make_backend(args, cfg: dict):
    which = args.model
    m = cfg["models"][which]
    revision = m.get("revision") or read_lock().get(revision_key(m["repo"])) or None
    cls = FakeBackend if args.fake else MLXBackend
    note(f"載入 {m['label']}  ←  {m['repo']}" + (f"@{revision[:10]}" if revision else ""))
    b = cls(m["repo"], revision, offline=not args.online)
    note(f"載入完成：{b.load_seconds:.1f} 秒")
    b.label = m["label"]
    return b


# ── 生成、顯示、記錄 ─────────────────────────────────────────────
def run_step(backend, cfg: dict, step: str, qid: str | None = None, show_prompt: bool = True) -> dict:
    d = cfg["demo"]
    if step == "1":
        seed_text = step1_prompt(cfg)
        if cfg["step1"]["mode"] == "chat":
            prompt = backend.render_chat(cfg["step1"]["chat_instruction"] + seed_text)
            # 畫面上連指令一起顯示：學生要看得到模型拿到的不只是那八個詞
            shown = cfg["step1"]["chat_instruction"].strip() + "  " + seed_text
        else:
            prompt = seed_text
            shown = seed_text
        max_tokens, q = d["max_tokens_continuation"], None
        heading = "Step 1 · 同樣的八個詞，交給一個 2026 年的模型"
    else:
        q = chosen_question(cfg, qid)
        prompt = backend.render_chat(q["text"], cfg["step2"].get("system", ""))
        shown, max_tokens = q["text"], d["max_tokens_answer"]
        heading = "Step 2 · 一個具體、可以查證的問題"

    banner(heading)
    note(f"模型：{backend.label}　temperature={d['temperature']}　max_tokens={max_tokens}")
    if show_prompt:
        print(f"\n{BOLD}prompt{RESET}  {AMBER}{shown}{RESET}\n")
    print(f"{BOLD}model{RESET}   ", end="", flush=True)
    if step == "1" and cfg["step1"]["mode"] == "raw":
        print(f"{GREY}{shown}{RESET}", end="", flush=True)

    out, stats, t0 = [], {}, time.time()
    try:
        for piece, st in backend.stream(prompt, max_tokens, d["temperature"], d["top_p"], d["seed"]):
            if st is not None:
                stats = st
                break
            out.append(piece)
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        stats["interrupted"] = True
        print(f"{RED}  [中斷]{RESET}")
    wall = time.time() - t0
    print()

    text = "".join(out)
    rec = {"time": now(), "step": step, "model": backend.repo, "prompt": prompt,
           "shown_prompt": shown, "output": text, "params": {
               "temperature": d["temperature"], "top_p": d["top_p"],
               "seed": d["seed"], "max_tokens": max_tokens},
           "stats": stats, "wall_seconds": round(wall, 2),
           "fake": isinstance(backend, FakeBackend), "versions": versions()}
    if q is not None:
        rec["question_id"] = q["id"]
        rec["judgement"] = judge(text, q)

    if stats:
        note(f"\n{stats.get('generation_tokens')} tokens · prompt {stats.get('prompt_tps')} tok/s · "
             f"decode {stats.get('generation_tps')} tok/s · 峰值記憶體 {stats.get('peak_memory_gb')} GB")
    return rec


def judge(text: str, q: dict) -> dict:
    """粗篩而已：accept 的每一條都要命中、reject 任何一條命中就算錯。一定要人眼再看。"""
    missing = [p for p in q.get("accept", []) if not re.search(p, text)]
    hit_reject = [p for p in q.get("reject", []) if re.search(p, text)]
    return {"looks_correct": not missing and not hit_reject,
            "missing": missing, "rejected_by": hit_reject}


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def reveal_answer(q: dict) -> None:
    print(f"\n{BOLD}查證過的答案{RESET}  {GREEN}{q['answer']}{RESET}")
    note(f"出處：{q['source']}")


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for name, m in cfg["models"].items():
        if args.only and name != args.only:
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {name}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha)
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已寫進 versions.lock{RESET}")


def cmd_check(args, cfg):
    banner("課前體檢")
    v = versions()
    for k, val in v.items():
        flag = "" if val else f"  {RED}← 缺{RESET}"
        print(f"  {k:<16} {val}{flag}")
    if not v.get("mlx_lm_commit"):
        print(f"  {AMBER}demos/versions.lock 沒有 MLX_LM_COMMIT：請先跑 demos/setup.sh{RESET}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"  {RED}這台不是 Apple silicon：只能用 --fake 演練{RESET}")
    if not args.fake:
        info = device_info()
        for k in ("architecture", "memory_size", "max_recommended_working_set_size", "device_name"):
            if k in info:
                val = info[k]
                if isinstance(val, (int, float)) and "size" in k:
                    val = f"{val / 2**30:.1f} GiB"
                print(f"  {k:<16} {val}")
    print(f"  {'HF_HUB_OFFLINE':<16} {os.environ.get('HF_HUB_OFFLINE', '(未設)')}")

    for name in cfg["models"]:
        if args.only and name != args.only:
            continue
        a = argparse.Namespace(**{**vars(args), "model": name})
        try:
            b = make_backend(a, cfg)
        except Exception as e:  # 模型沒下載、版本不合……都在這裡現形
            print(f"  {RED}{name} 載入失敗：{type(e).__name__}: {e}{RESET}")
            continue
        rec = run_step(b, cfg, "1", show_prompt=False) if not args.no_warmup else None
        if rec:
            append_jsonl(RUNS / "check.jsonl", rec)
        del b


def cmd_probe(args, cfg):
    a = argparse.Namespace(**{**vars(args)})
    b = make_backend(a, cfg)
    qs = [q for q in cfg["questions"] if not args.question or q["id"] in args.question]
    log = RUNS / f"probe-{args.model}-{dt.datetime.now():%Y%m%d-%H%M}.jsonl"
    summary = []
    for q in qs:
        wrong = 0
        for i in range(args.repeat):
            rec = run_step(b, cfg, "2", qid=q["id"])
            rec["repeat_index"] = i
            append_jsonl(log, rec)
            ok = rec["judgement"]["looks_correct"]
            wrong += not ok
            print(f"  {'粗判：對' if ok else '粗判：錯'}  {GREY}{rec['judgement']}{RESET}")
        summary.append((q["id"], wrong, args.repeat))
        reveal_answer(q)
    banner("probe 結果（粗判，要人眼確認）", AMBER)
    for qid, wrong, n in summary:
        print(f"  {qid:<16} 錯 {wrong}/{n}")
    note(f"完整輸出在 {log.relative_to(HERE)}。挑一題「每次都錯、語氣篤定」的，填進 step2.chosen。")


def cmd_rehearse(args, cfg):
    b = make_backend(args, cfg)
    for step in ("1", "2"):
        rec = run_step(b, cfg, step)
        rec["rehearsal"] = True
        path = REHEARSAL / f"step{step}-{args.model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        if step == "2":
            print(f"  粗判：{rec['judgement']}")
            reveal_answer(chosen_question(cfg))
        note(f"→ {path.relative_to(HERE)}")


def cmd_show(args, cfg):
    os.system("clear")
    banner(cfg["demo"]["title"], AMBER)
    b = make_backend(args, cfg)
    log = RUNS / "live" / f"{dt.datetime.now():%Y%m%d}.jsonl"
    steps = ["1", "2"]
    i = 0
    while True:
        label = {"1": "Step 1（同一個 prompt）", "2": "Step 2（繁中問題）"}
        hint = f"Enter = {label[steps[i]]}" if i < len(steps) else "Enter = 結束"
        try:
            key = input(f"\n{GREY}[{hint} · r = 重跑上一步 · a = 顯示答案 · q = 離開]{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key == "q":
            break
        if key == "a":
            reveal_answer(chosen_question(cfg))
            continue
        if key == "r" and i > 0:
            i -= 1
        if i >= len(steps):
            break
        rec = run_step(b, cfg, steps[i])
        append_jsonl(log, rec)
        i += 1
    note(f"本場紀錄：{log.relative_to(HERE)}")


def cmd_replay(args, cfg):
    for step in ("1", "2"):
        path = REHEARSAL / f"step{step}-{args.model}.json"
        if not path.exists():
            print(f"{RED}沒有彩排紀錄 {path.relative_to(HERE)}；課前先跑 rehearse{RESET}")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        banner(f"課前錄下的輸出（{rec['time']}，非現場）· Step {step}", RED)
        note(f"模型：{rec['model']}　versions：mlx {rec['versions'].get('mlx')} / "
             f"mlx-lm {rec['versions'].get('mlx_lm_commit') or rec['versions'].get('mlx-lm')}")
        print(f"\n{BOLD}prompt{RESET}  {AMBER}{rec['shown_prompt']}{RESET}\n")
        print(f"{BOLD}model{RESET}   ", end="")
        for ch in rec["output"]:
            print(ch, end="", flush=True)
            time.sleep(args.delay)
        print()
        if not args.no_pause and step == "1":
            try:
                input(f"\n{GREY}[Enter = Step 2]{RESET} ")
            except (EOFError, KeyboardInterrupt):
                return
        if step == "2" and "question_id" in rec:
            reveal_answer(chosen_question(cfg, rec["question_id"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="primary", help="demo_config.toml 的 [models.X]，預設 primary")
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（畫面會標示 FAKE）")
    p.add_argument("--online", action="store_true", help="允許連網下載（預設只用本機快取）")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--only")
    c = sub.add_parser("check"); c.add_argument("--only"); c.add_argument("--no-warmup", action="store_true")
    pr = sub.add_parser("probe"); pr.add_argument("--repeat", type=int, default=3)
    pr.add_argument("--question", nargs="*")
    sub.add_parser("rehearse")
    sub.add_parser("show")
    r = sub.add_parser("replay"); r.add_argument("--delay", type=float, default=0.012)
    r.add_argument("--no-pause", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        args.online = True
    cfg = load_config()
    {"fetch": cmd_fetch, "check": cmd_check, "probe": cmd_probe, "rehearse": cmd_rehearse,
     "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
