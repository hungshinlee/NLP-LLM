#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 3（投影片 p43「Live: one model, one set of sums, three ways of writing the numbers」）。

同一個模型、同一批隨機多位數加法，三種寫法——(A) 原樣、(B) 千分位逗號、(C) 逐位加空白——依位數報正確率。
主模型的 pre-tokenizer 是 \\p{N}{1,3}（由左至右三位一組，與進位方向相反），Qwen（逐位）當對照組。

子命令（用 nlp_llm/demos/setup.sh 建好的共用 .venv；通常經由 present.sh）：

    fetch      下載 demo_config.toml 裡的候選模型，revision 記進 demos/versions.lock（--only <key> 只抓一個）
    inspect    只載 tokenizer：三種寫法各怎麼切（主模型必須是 1–3 位一組，對照組必須逐位）；chat template 能不能關 thinking
    scan       課前選模型：每個候選 × 三種位數 × 三種寫法各 n_scan 題，印正確率格子，找「中段」的那個
    rehearse   課前彩排：main 與 control 各跑一張完整的表（n_rehearse 題／格），存 runs/rehearsal/<key>.json
    show       課堂：按 Enter 逐列跑主模型（n_live 題／格）；q = 對照組跑 7 位數那列；f = 顯示彩排的完整表；x = 離開
    replay     現場失敗時：顯示彩排的表（畫面標明「非現場」）

正確率的判定：有 = 就取最後一個 = 之後、否則取最後一段「數字（可夾逗號或空白）」，拿掉逗號與空白，與正確答案比。
thinking 用 apply_chat_template(enable_thinking=False) 關；輸出裡若仍出現 <think>，計數並剝掉，畫面會標。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, GREEN, GREY, RED, RESET, append_jsonl,  # noqa: E402
                    banner, locked_revision, note, now, reset_peak_memory, peak_memory_gb,
                    revision_key, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "config.json",
                   "generation_config.json"]
FORMATS = [("A", "as written"), ("B", "with commas"), ("C", "spaced")]


# ── 設定與題目 ───────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for k in (cfg["demo"]["main"], cfg["demo"]["control"]):
        if k not in cfg["models"]:
            sys.exit(f"demo_config.toml：{k!r} 不在 [models]")
    return cfg


def make_problems(nd: int, n: int, seed: int) -> list[dict]:
    """nd 位數的加法 n 題。同一個 seed、同一個 nd 得到同一批題，三種寫法共用。"""
    rng = random.Random(seed * 1000 + nd)
    out = []
    for i in range(n):
        a = rng.randint(10 ** (nd - 1), 10 ** nd - 1)
        b = rng.randint(10 ** (nd - 1), 10 ** nd - 1)
        out.append({"i": i, "a": a, "b": b, "sum": a + b})
    return out


def fmt_num(x: int, f: str) -> str:
    if f == "A":
        return str(x)
    if f == "B":
        return f"{x:,}"
    return " ".join(str(x))


def render_user(cfg, p: dict, f: str) -> str:
    return cfg["prompt"]["user"].format(a=fmt_num(p["a"], f), b=fmt_num(p["b"], f))


NUM_RUN = re.compile(r"\d[\d,\s]*")


def parse_answer(text: str) -> tuple[str | None, bool]:
    """回傳 (抽出的數字, 是否出現 <think>)。"""
    thought = "<think>" in text
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    t = t.split("</think>")[-1] if "</think>" in t else t
    # 模型常重述題目（"9585 + 7535 = 17120"）：有 = 就取最後一個 = 之後的數字，否則取最後一段數字。
    # 2026-09-20 scan 發現取「第一段」會把題目當答案。
    if "=" in t:
        t = t.rsplit("=", 1)[1]
    # 沒有 = 時取第一個含數字的行的第一段數字（避免把後面的說明文字裡的數字當答案）
    line = next((ln for ln in t.splitlines() if re.search(r"\d", ln)), "")
    m = NUM_RUN.search(line)
    if not m:
        return None, thought
    digits = re.sub(r"[^\d]", "", m.group(0))
    return digits or None, thought


# ── 後端 ─────────────────────────────────────────────────────────
class MLXModel:
    def __init__(self, key: str, spec: dict, offline: bool, weights: bool = True):
        from huggingface_hub import snapshot_download
        t0 = time.time()
        rev = locked_revision(spec["repo"], spec.get("revision"))
        kw = {"ignore_patterns": NON_MODEL_FILES} if weights else {"allow_patterns": TOKENIZER_FILES}
        self.path = snapshot_download(spec["repo"], revision=rev, local_files_only=offline, **kw)
        self.key, self.label, self.repo, self.kind = key, spec["label"], spec["repo"], spec["kind"]
        if weights:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.path)
        else:
            from transformers import AutoTokenizer
            self.model, self.tokenizer = None, AutoTokenizer.from_pretrained(self.path)
        self.load_seconds = time.time() - t0
        self.think_supported = None

    def render(self, cfg, user: str) -> str:
        msgs = [{"role": "system", "content": cfg["prompt"]["system"]}, {"role": "user", "content": user}]
        try:
            s = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.think_supported = True
        except TypeError:
            s = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            self.think_supported = False
        return s

    def pieces(self, s: str) -> list[str]:
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        conv = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if conv:
            return [conv(i) for i in ids]
        return [self.tokenizer.decode([i]) for i in ids]

    def answer(self, cfg, prompt: str) -> tuple[str, dict]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=cfg["demo"]["temperature"])
        t0 = time.time()
        text = generate(self.model, self.tokenizer, prompt, max_tokens=cfg["demo"]["max_tokens"],
                        sampler=sampler, verbose=False)
        return text, {"seconds": round(time.time() - t0, 2)}


class FakeModel:
    """--fake：沒有 Apple silicon 也能演練。正確率是假的：主模型 (A) 差、(B) 好；對照組三欄接近。"""

    def __init__(self, key, spec, offline, weights=True):
        self.key, self.label, self.repo, self.kind = key, spec["label"], "(fake)", spec["kind"]
        self.path, self.load_seconds, self.think_supported = "(fake)", 0.0, True
        self.rng = random.Random(7)

    def render(self, cfg, user):
        return f"<user>{user}</user><assistant>"

    def pieces(self, s):
        if self.kind == "main":
            return re.findall(r"\d{1,3}|[^\d]", s)
        return list(s)

    def answer(self, cfg, prompt):
        m = re.search(r"(\d[\d,\s]*)\s*\+\s*(\d[\d,\s]*)", prompt)
        a, b = (int(re.sub(r"\D", "", x)) for x in m.groups())
        nd = len(str(a))
        f = "C" if " " in m.group(1).strip() else ("B" if "," in m.group(1) else "A")
        base = {4: 0.95, 7: 0.6, 10: 0.3}.get(nd, 0.5)
        p = {"A": base - 0.25, "B": base + 0.1, "C": base}[f] if self.kind == "main" else base
        ok = self.rng.random() < p
        s = a + b if ok else a + b + self.rng.choice([1000, -1000, 100, 10])
        return (f"{s:,}" if f == "B" else str(s)), {"seconds": 0.0}


def make_model(key, cfg, args, weights=True):
    spec = cfg["models"][key]
    cls = FakeModel if args.fake else MLXModel
    note(f"載入 {spec['label']}{'（只載 tokenizer）' if not weights else ''}  ←  {spec['repo']}")
    m = cls(key, spec, offline=not args.online, weights=weights)
    note(f"  完成：{m.load_seconds:.1f} 秒")
    return m


# ── 一格 ─────────────────────────────────────────────────────────
def run_cell(model, cfg, nd: int, f: str, n: int, live: bool = False) -> dict:
    probs = make_problems(nd, n, cfg["demo"]["seed"])
    correct, thoughts, unparsed, recs = 0, 0, 0, []
    t0 = time.time()
    for p in probs:
        user = render_user(cfg, p, f)
        text, st = model.answer(cfg, model.render(cfg, user))
        ans, thought = parse_answer(text)
        ok = ans == str(p["sum"])
        correct += ok; thoughts += thought; unparsed += ans is None
        recs.append({"a": p["a"], "b": p["b"], "sum": p["sum"], "prompt": user, "raw": text[:200], "parsed": ans, "ok": ok})
        if live:
            mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
            print(f"    {mark} {user:<40} → {text.strip()[:40]!r}")
    return {"digits": nd, "format": f, "n": n, "correct": correct, "acc": round(correct / n, 3),
            "thought_outputs": thoughts, "unparsed": unparsed, "seconds": round(time.time() - t0, 1), "problems": recs}


# ── 表格 ─────────────────────────────────────────────────────────
def print_table(label: str, cells: dict, digits: list[int], n: int | None = None, tag: str = ""):
    """cells[(nd, f)] = cell dict。"""
    w = 16
    print(f"\n{BOLD}{label}{RESET}{('  ' + GREY + tag + RESET) if tag else ''}")
    print(f"{'digits':<8}" + "".join(f"{BOLD}{('(' + f + ') ' + name):>{w}}{RESET}" for f, name in FORMATS) + f"{GREY}{'B − A':>10}{RESET}")
    print("─" * (8 + w * 3 + 10))
    for nd in digits:
        row = f"{nd:<8}"
        vals = {}
        for f, _ in FORMATS:
            c = cells.get((nd, f))
            if c is None:
                row += f"{'·':>{w}}"; continue
            vals[f] = c["acc"]
            extra = f" ({c['correct']}/{c['n']})"
            flag = f"{AMBER}!{RESET}" if c.get("thought_outputs") else " "
            row += f"{100*c['acc']:>{w-len(extra)-1}.0f}%{extra}{flag}"
        d = f"{100*(vals['B']-vals['A']):+.0f} pt" if "A" in vals and "B" in vals else ""
        print(row + f"{GREY}{d:>10}{RESET}")
    if any(c.get("thought_outputs") for c in cells.values()):
        note("  ! = 該格有輸出夾了 <think>（thinking 沒關乾淨），已剝掉再判分")


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for key, m in cfg["models"].items():
        if args.only and key != args.only:
            continue
        try:
            info = api.model_info(m["repo"], revision=m.get("revision") or None)
            banner(f"下載 {key}: {m['repo']} @ {info.sha}")
            snapshot_download(m["repo"], revision=info.sha, ignore_patterns=NON_MODEL_FILES)
            write_lock({revision_key(m["repo"]): info.sha})
            print(f"{GREEN}完成，revision 已寫進 demos/versions.lock{RESET}")
        except Exception as e:
            print(f"{RED}{key}：{type(e).__name__}: {e}\n  → repo 名稱是憑印象寫的，到 https://huggingface.co/mlx-community 對一次再改 demo_config.toml{RESET}")


def cmd_inspect(args, cfg):
    ex = 1234567
    for key in cfg["models"]:
        try:
            m = make_model(key, cfg, args, weights=False)
        except Exception as e:
            print(f"  {RED}{key}：{type(e).__name__}: {e}{RESET}"); continue
        banner(f"{m.label}  [{m.kind}]", AMBER)
        for f, name in FORMATS:
            s = fmt_num(ex, f)
            print(f"  ({f}) {s:<24} → {m.pieces(s)}")
        pa = m.pieces(str(ex))
        groups3 = all(len(re.sub(r'\D', '', t)) <= 3 for t in pa) and any(len(re.sub(r'\D', '', t)) == 3 for t in pa)
        per_digit = all(len(re.sub(r'\D', '', t)) <= 1 for t in pa)
        verdict = ("1–3 位一組 → 可當主模型" if groups3 else "逐位 → 只能當對照組" if per_digit else "其他切法（整串或依頻率）→ 主模型與對照組都不適合")
        want_ok = (m.kind == "main" and groups3) or (m.kind == "control" and per_digit)
        print(f"  {GREEN if want_ok else RED}{verdict}；設定裡是 {m.kind}{'' if want_ok else '——對不上，改 demo_config.toml'}{RESET}")
        try:
            m.render(cfg, "1 + 1 =")
            print(f"  chat template 接受 enable_thinking=False：{m.think_supported}")
        except Exception as e:
            print(f"  {RED}chat template 失敗：{type(e).__name__}: {e}{RESET}")


def cmd_scan(args, cfg):
    n = cfg["demo"]["n_scan"]
    digits = cfg["demo"]["digits"]
    keys = [args.only] if args.only else list(cfg["models"])
    summary = {}
    for key in keys:
        try:
            m = make_model(key, cfg, args)
        except Exception as e:
            print(f"  {RED}{key}：{type(e).__name__}: {e}{RESET}"); continue
        cells = {}
        for nd in digits:
            for f, _ in FORMATS:
                cells[(nd, f)] = run_cell(m, cfg, nd, f, n)
        print_table(f"{m.label}  [{m.kind}]", cells, digits, tag=f"scan · {n} 題／格 · {now()}")
        summary[key] = {f"{nd}{f}": c["acc"] for (nd, f), c in cells.items()}
        del m
    banner("怎麼選", AMBER)
    print("  主模型：找 (A) 在 20–80% 之間、而且 (B) 明顯高於 (A) 的那一列——那就是課堂上要看的位數。")
    print("  三欄全對 → 模型太大，換小的；三欄全錯 → floor effect，換大的或減位數。")
    print("  對照組：(A)(B)(C) 差距小的那個 Qwen；差距若不小，講稿裡要照實說。")
    print("  選定後改 demo_config.toml 的 main／control，再跑 rehearse。")
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    (REHEARSAL / ("scan%s.json" % ("-FAKE" if args.fake else ""))).write_text(
        json.dumps({"time": now(), "n": n, "fake": args.fake, "summary": summary, "versions": versions()},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def record(cfg, m, cells, n, fake) -> dict:
    return {"time": now(), "model": {"key": m.key, "label": m.label, "repo": m.repo, "kind": m.kind,
                                     "think_supported": m.think_supported},
            "n": n, "seed": cfg["demo"]["seed"], "digits": cfg["demo"]["digits"], "prompt": cfg["prompt"],
            "cells": [c for c in cells.values()], "fake": fake, "versions": versions()}


def cmd_rehearse(args, cfg):
    n = cfg["demo"]["n_rehearse"]
    digits = cfg["demo"]["digits"]
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    for role in ("main", "control"):
        key = cfg["demo"][role]
        m = make_model(key, cfg, args)
        if not args.fake:
            reset_peak_memory()
        cells = {}
        for nd in digits:
            for f, _ in FORMATS:
                cells[(nd, f)] = run_cell(m, cfg, nd, f, n)
                note(f"  {nd} 位 ({f})：{cells[(nd, f)]['correct']}/{n}，{cells[(nd, f)]['seconds']} 秒")
        pm = peak_memory_gb() if not args.fake else None
        print_table(f"{role}: {m.label}", cells, digits, tag=f"彩排 · {n} 題／格 · {now()}" + (f" · 峰值 {pm:.1f} GB" if pm else ""))
        rec = record(cfg, m, cells, n, args.fake)
        rec["peak_memory_gb"] = pm
        p = REHEARSAL / f"{role}-{key}{'-FAKE' if args.fake else ''}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        note(f"→ {p.relative_to(HERE)}")
        del m


def load_rehearsal(cfg, role, fake) -> dict | None:
    p = REHEARSAL / f"{role}-{cfg['demo'][role]}{'-FAKE' if fake else ''}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cells_of(rec) -> dict:
    return {(c["digits"], c["format"]): c for c in rec["cells"]}


def wait(msg: str) -> str:
    try:
        return input(f"\n{GREY}[{msg} · q = 對照組 · f = 彩排完整表 · x = 離開]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "x"


def cmd_show(args, cfg):
    os.system("clear")
    banner(cfg["demo"]["title"] + ("（FAKE）" if args.fake else ""), AMBER)
    n = cfg["demo"]["n_live"]
    digits = cfg["demo"]["digits"]
    main = make_model(cfg["demo"]["main"], cfg, args)
    control = None
    cells, ccells = {}, {}
    i = 0
    while True:
        nd = digits[i] if i < len(digits) else None
        key = wait(f"Enter = {nd} 位數，三種寫法各 {n} 題" if nd else "Enter = 結束")
        if key == "x":
            break
        if key == "f":
            for role in ("main", "control"):
                rec = load_rehearsal(cfg, role, args.fake)
                if rec:
                    print_table(f"{role}: {rec['model']['label']}", cells_of(rec), rec["digits"],
                                tag=f"彩排 {rec['time']} · {rec['n']} 題／格（非現場）")
                else:
                    print(f"{RED}沒有 {role} 的彩排紀錄{RESET}")
            continue
        if key == "q":
            if control is None:
                control = make_model(cfg["demo"]["control"], cfg, args)
            nd_c = 7 if 7 in digits else digits[len(digits) // 2]
            banner(f"對照組 {control.label}：{nd_c} 位數", AMBER)
            for f, _ in FORMATS:
                ccells[(nd_c, f)] = run_cell(control, cfg, nd_c, f, n, live=False)
                note(f"  ({f})：{ccells[(nd_c, f)]['correct']}/{n}")
            print_table(f"對照組 {control.label}", ccells, [nd_c], tag=f"現場 · {n} 題／格")
            continue
        if nd is None:
            break
        banner(f"{main.label}：{nd} 位數", AMBER)
        for f, name in FORMATS:
            print(f"\n  ({f}) {name}")
            cells[(nd, f)] = run_cell(main, cfg, nd, f, n, live=True)
        print_table(main.label, cells, digits, tag=f"現場 · {n} 題／格")
        i += 1
    if cells:
        append_jsonl(RUNS / "live" / f"{dt.datetime.now():%Y%m%d}.jsonl", record(cfg, main, cells, n, args.fake))


def cmd_replay(args, cfg):
    any_ = False
    for role in ("main", "control"):
        rec = load_rehearsal(cfg, role, args.fake)
        if not rec:
            print(f"{RED}沒有 {role} 的彩排紀錄（runs/rehearsal/{role}-{cfg['demo'][role]}.json）；課前先跑 rehearse{RESET}"); continue
        any_ = True
        banner(f"課前錄下的結果（{rec['time']}，非現場）· {role}", RED)
        print_table(rec["model"]["label"], cells_of(rec), rec["digits"], tag=f"{rec['n']} 題／格")
        if role == "main" and wait("Enter = 對照組") == "x":
            return
    if not any_:
        return


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（正確率是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "scan"):
        sp = sub.add_parser(c); sp.add_argument("--only")
    for c in ("inspect", "rehearse", "show", "replay"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    if not hasattr(args, "only"):
        args.only = None
    if args.cmd == "fetch":
        args.online = True
    cfg = load_config()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "scan": cmd_scan, "rehearse": cmd_rehearse,
     "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
