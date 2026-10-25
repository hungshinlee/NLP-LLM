#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 3：lost-in-the-middle 的縮小復現（投影片 p39「Replay + live: demo 3 — key-value retrieval, answer at five positions」）。

    ./present.sh inspect    課前：只載 tokenizer，校準兩檔的 K、印一題的 prompt 長相與 token 數 → runs/rehearsal/inspect.json
    ./present.sh rehearse   課前彩排：兩檔 × 五個位置 × Q 題，每題一次生成 → runs/rehearsal/{rehearsal,summary}.json
    ./present.sh            課堂：載模型，現場三題（答案在開頭／中間／結尾），每題 Enter；最後 r 印彩排的表
    ./present.sh replay     現場失敗：只印彩排的表（不載模型）
    ./present.sh fetch      模型已在 W2 demo 3 下載並鎖定，通常不必；換模型時才跑
    ./present.sh fake       沒有 mlx 也能演練流程（回答是假的，畫面會標明）

任務照 Liu et al.（Lost in the Middle, TACL 2024）§4 的合成 key-value retrieval：
  K 組隨機 UUID 鍵值對排成一個 JSON 物件，問其中一個鍵的值。同一題的 K 組不變，只把含答案的那一組
  搬到第 1、⌊K/4⌋、⌊K/2⌋、⌊3K/4⌋、K 個位置——五個 prompt 之間只有答案的位置在動。
  K 由目標 token 數校準（約 4k 與 16k，含 chat template），inspect 會印出來。

判定：回覆去掉引號與空白後逐字等於答案 = exact match（表格用的數）。另記較寬的「回覆裡第一個 UUID 是答案」，
  以及答錯的種類：拿了 context 裡另一組的值（記它離答案幾格）、拿了鍵、UUID 不在 context 裡、沒有 UUID。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）。Python 3.11+（tomllib）。
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
import tomllib
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, append_jsonl,  # noqa: E402
                    banner, device_info, locked_revision, note, now, peak_memory_gb, revision_key,
                    versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
LIVE = RUNS / "live"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "config.json",
                   "generation_config.json"]
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# 五個位置，依投影片 p39 右表的列序。index 是 1-indexed。
POSITIONS = [("first", "1", lambda K: 1),
             ("quarter", "⌊K/4⌋", lambda K: max(1, K // 4)),
             ("middle", "⌊K/2⌋", lambda K: max(1, K // 2)),
             ("three_quarter", "⌊3K/4⌋", lambda K: max(1, 3 * K // 4)),
             ("last", "K", lambda K: K)]
POS_INDEX = {name: i for i, (name, _, _) in enumerate(POSITIONS)}


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


# ── 題目 ─────────────────────────────────────────────────────────
def make_pairs(K: int, rng: random.Random) -> list[tuple[str, str]]:
    """K 組 UUID 鍵值對（版本 4 的格式，由 rng 產生，可重現）。"""
    def u() -> str:
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    return [(u(), u()) for _ in range(K)]


def make_question(K: int, seed: int) -> dict:
    """一題 = K 組鍵值對 + 答案那一組。pairs[0] 是答案，其餘 K-1 組是干擾項（順序固定）。"""
    rng = random.Random(seed)
    pairs = make_pairs(K, rng)
    return {"seed": seed, "K": K, "answer": pairs[0], "others": pairs[1:]}


def arrange(q: dict, pos: int) -> list[tuple[str, str]]:
    """把答案那一組放到第 pos 個位置（1-indexed），其餘不動。"""
    return q["others"][:pos - 1] + [q["answer"]] + q["others"][pos - 1:]


def render_user(cfg: dict, pairs: list[tuple[str, str]], key: str) -> str:
    body = json.dumps(dict(pairs), indent=1)
    return cfg["prompt"]["instruction"] + "\n\nJSON data:\n" + body + "\n\n" + cfg["prompt"]["question"].format(key=key)


def question_seeds(cfg: dict, size_label: str, n: int, live: bool = False) -> list[int]:
    """每檔各自一段 seed；live 另一段，與彩排不重疊。"""
    base = int(cfg["demo"]["seed"]) + (int(cfg["demo"]["live_offset"]) if live else 0)
    offset = 100_000 * (1 + [s["label"] for s in cfg["sizes"]].index(size_label))
    return [base + offset + i for i in range(n)]


# ── 判定 ─────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return text.strip().strip("\"'`.。,;: \n\t").strip().lower()


def judge(reply: str, q: dict, pos: int) -> dict:
    """exact match 與答錯的種類。pos 是答案的位置（1-indexed）。"""
    answer = q["answer"][1]
    norm = normalize(reply)
    found = [m.lower() for m in UUID_RE.findall(reply)]
    first = found[0] if found else None
    em = norm == answer
    lenient = first == answer
    pairs = arrange(q, pos)
    values = {v: i + 1 for i, (_, v) in enumerate(pairs)}
    keys = {k: i + 1 for i, (k, _) in enumerate(pairs)}
    if em:
        kind, offset = "correct", 0
    elif lenient:
        kind, offset = "correct_with_extra_words", 0
    elif first is None:
        kind, offset = "no_uuid", None
    elif first in values:
        kind, offset = "other_value", values[first] - pos      # context 裡另一組的值，離答案幾格
    elif first in keys:
        kind, offset = "key", keys[first] - pos
    else:
        kind, offset = "not_in_context", None
    return {"em": em, "lenient": lenient, "kind": kind, "offset": offset, "first_uuid": first, "n_uuids": len(found)}


# ── 後端 ─────────────────────────────────────────────────────────
class MLXModel:
    fake = False

    def __init__(self, key: str, spec: dict, offline: bool, weights: bool = True):
        from huggingface_hub import snapshot_download
        t0 = time.time()
        rev = locked_revision(spec["repo"], spec.get("revision"))
        # ignore_patterns：離線時 huggingface_hub ≥1.3 會核對 snapshot 完整性（W2 demo 1 踩過）
        kw = {"ignore_patterns": NON_MODEL_FILES} if weights else {"allow_patterns": TOKENIZER_FILES}
        self.path = snapshot_download(spec["repo"], revision=rev, local_files_only=offline, **kw)
        self.key, self.label, self.repo, self.revision = key, spec["label"], spec["repo"], rev
        if weights:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.path)
        else:
            from transformers import AutoTokenizer
            self.model, self.tokenizer = None, AutoTokenizer.from_pretrained(self.path)
        self.load_seconds = time.time() - t0
        self.think_supported = None
        cfgj = json.loads((Path(self.path) / "config.json").read_text(encoding="utf-8"))
        self.max_pos = cfgj.get("max_position_embeddings")
        self.rope_scaling = cfgj.get("rope_scaling")

    def render(self, cfg: dict, user: str) -> str:
        msgs = [{"role": "system", "content": cfg["prompt"]["system"]}, {"role": "user", "content": user}]
        try:   # Qwen3 的 template 認 enable_thinking；Llama 3.1 的 jinja 會忽略多餘的 kwarg
            s = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.think_supported = True
        except TypeError:
            s = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            self.think_supported = False
        return s

    def count(self, s: str) -> int:
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def answer(self, cfg: dict, prompt: str) -> tuple[str, dict]:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=float(cfg["demo"]["temperature"]))
        max_tokens = int(cfg["demo"]["max_tokens"])
        t0 = time.time()
        try:
            from mlx_lm import stream_generate
            text, last = "", None
            for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
                text += r.text
                last = r
            st = {"seconds": round(time.time() - t0, 2)}
            for k in ("prompt_tokens", "prompt_tps", "generation_tokens", "generation_tps", "peak_memory", "finish_reason"):
                v = getattr(last, k, None)
                if v is not None:
                    st[k] = round(v, 2) if isinstance(v, float) else v
        except ImportError:   # 舊版 mlx-lm 沒有 stream_generate
            from mlx_lm import generate
            text = generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
            st = {"seconds": round(time.time() - t0, 2)}
        if "<think>" in text:   # 備用的 Qwen3 若仍 think，剝掉並標記
            st["thought"] = True
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return text, st


class FakeModel:
    """--fake：沒有 Apple silicon 也能演練。回答是假的：答對的機率依位置呈 U 型、16k 比 4k 深；畫面會標明。"""
    fake = True

    def __init__(self, key, spec, offline, weights=True):
        self.key, self.label, self.repo, self.revision = key, spec["label"] + " [FAKE]", spec["repo"], None
        self.path, self.load_seconds, self.think_supported = "(fake)", 0.0, None
        self.max_pos, self.rope_scaling = 131072, {"rope_type": "llama3", "factor": 8.0}
        self.rng = random.Random(0)

    def render(self, cfg, user):
        return "<system>" + cfg["prompt"]["system"] + "</system>\n" + user

    def count(self, s: str) -> int:
        return int(len(s) / 2.9)   # UUID 字串的粗估；真值由 inspect 印

    def answer(self, cfg, prompt):
        m = re.search(r'Key: "([0-9a-f-]{36})"', prompt)
        pairs = re.findall(r'"([0-9a-f-]{36})": "([0-9a-f-]{36})"', prompt)
        d = dict(pairs)
        idx = [k for k, _ in pairs].index(m.group(1)) if m and m.group(1) in d else 0
        K = max(1, len(pairs))
        x = idx / max(1, K - 1)                       # 0 = 開頭、1 = 結尾
        depth = 0.6 if K > 200 else 0.25
        p_ok = 1.0 - depth * (1 - (2 * x - 1) ** 2)   # U 型
        time.sleep(0.05)
        st = {"seconds": 0.05, "prompt_tokens": self.count(prompt), "prompt_tps": 3000.0, "generation_tokens": 24, "fake": True}
        if self.rng.random() < p_ok:
            return '"' + d[m.group(1)] + '"' if m else "?", st
        j = min(K - 1, max(0, idx + self.rng.choice([-3, -2, -1, 1, 2, 3])))
        return pairs[j][1], st


def make_model(key: str, cfg: dict, args, weights: bool = True):
    spec = cfg["models"][key]
    cls = FakeModel if args.fake else MLXModel
    rev = locked_revision(spec["repo"], spec.get("revision"))
    note(f"載入 {spec['label']}{'（只載 tokenizer）' if not weights else ''}  ←  {spec['repo']}" + (f"@{rev[:10]}" if rev else ""))
    m = cls(key, spec, offline=not args.online, weights=weights)
    note(f"  完成：{m.load_seconds:.1f} 秒")
    return m


# ── 校準 K ────────────────────────────────────────────────────────
def calibrate(model, cfg: dict, size: dict) -> dict:
    """K = 0 時：量 prompt token 數對 K 的斜率與截距，解出最接近 target_tokens 的 K。"""
    if int(size.get("k") or 0) > 0:
        K = int(size["k"])
        q = make_question(K, int(cfg["demo"]["seed"]))
        tok = model.count(model.render(cfg, render_user(cfg, arrange(q, 1), q["answer"][0])))
        return {"label": size["label"], "K": K, "target_tokens": size["target_tokens"], "tokens_at_K": tok, "fixed": True}
    seed = int(cfg["demo"]["seed"])
    k1, k2 = 40, 80
    t = {}
    for k in (k1, k2):
        q = make_question(k, seed)
        t[k] = model.count(model.render(cfg, render_user(cfg, arrange(q, 1), q["answer"][0])))
    per_pair = (t[k2] - t[k1]) / (k2 - k1)
    base = t[k1] - k1 * per_pair
    K = max(8, round((int(size["target_tokens"]) - base) / per_pair))
    q = make_question(K, seed)
    tok = model.count(model.render(cfg, render_user(cfg, arrange(q, 1), q["answer"][0])))
    return {"label": size["label"], "K": K, "target_tokens": size["target_tokens"], "tokens_at_K": tok,
            "tokens_per_pair": round(per_pair, 2), "tokens_base": round(base, 1), "fixed": False}


def positions_for(K: int) -> list[tuple[str, str, int]]:
    return [(name, label, fn(K)) for name, label, fn in POSITIONS]


# ── 一題 ─────────────────────────────────────────────────────────
def ask(model, cfg: dict, q: dict, pos: int) -> dict:
    pairs = arrange(q, pos)
    user = render_user(cfg, pairs, q["answer"][0])
    prompt = model.render(cfg, user)
    text, st = model.answer(cfg, prompt)
    verdict = judge(text, q, pos)
    if "prompt_tokens" not in st:
        st["prompt_tokens"] = model.count(prompt)
    return {"seed": q["seed"], "K": q["K"], "pos": pos, "key": q["answer"][0], "answer": q["answer"][1],
            "reply": text.strip()[:300], **verdict, **st}


def mark(rec: dict) -> str:
    if rec["em"]:
        return f"{GREEN}✓{RESET}"
    if rec["lenient"]:
        return f"{AMBER}✓{RESET}{GREY}(extra words){RESET}"
    return f"{RED}✗{RESET}"


def describe_wrong(rec: dict) -> str:
    k = rec["kind"]
    if k == "other_value":
        return f"another pair's value, {rec['offset']:+d} from the answer"
    if k == "key":
        return f"a key, not a value ({rec['offset']:+d} from the answer)"
    if k == "not_in_context":
        return "a UUID that is not in the context"
    if k == "no_uuid":
        return "no UUID in the reply"
    return ""


# ── 表 ───────────────────────────────────────────────────────────
def summarize(cells: list[dict]) -> dict:
    """cells: 每個（檔次 × 位置）一筆 {size, pos_name, records}。"""
    out = []
    for c in cells:
        recs = c["records"]
        n = len(recs)
        kinds = {}
        for r in recs:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        offs = [r["offset"] for r in recs if r["kind"] == "other_value"]
        out.append({"size": c["size"], "K": c["K"], "pos_name": c["pos_name"], "pos_label": c["pos_label"], "pos": c["pos"],
                    "n": n, "em": sum(r["em"] for r in recs), "lenient": sum(r["lenient"] for r in recs),
                    "acc": round(sum(r["em"] for r in recs) / n, 3) if n else None,
                    "acc_lenient": round(sum(r["lenient"] for r in recs) / n, 3) if n else None,
                    "kinds": kinds, "mean_offset": round(sum(offs) / len(offs), 1) if offs else None,
                    "mean_prompt_tokens": round(sum(r.get("prompt_tokens") or 0 for r in recs) / n) if n else None,
                    "mean_seconds": round(sum(r["seconds"] for r in recs) / n, 2) if n else None,
                    "mean_prompt_tps": round(sum(r.get("prompt_tps") or 0 for r in recs) / n) if n and all(r.get("prompt_tps") for r in recs) else None})
    return {"cells": out}


def print_table(summary: dict, sizes: list[str], tag: str = "") -> None:
    cells = {(c["size"], c["pos_name"]): c for c in summary["cells"]}
    Ks = {s: next((c["K"] for c in summary["cells"] if c["size"] == s), None) for s in sizes}
    toks = {s: next((c["mean_prompt_tokens"] for c in summary["cells"] if c["size"] == s), None) for s in sizes}
    hdr = f"  {'answer position':<18}"
    for s in sizes:
        tok = f"{toks[s]:,} tok" if toks[s] else "? tok"
        hdr += f"{'≈' + s + ' (K=' + str(Ks[s]) + ', ' + tok + ')':>30}"
    print(hdr + ("   " + tag if tag else ""))
    for name, label, _ in POSITIONS:
        row = f"  {label:<18}"
        for s in sizes:
            c = cells.get((s, name))
            if not c:
                row += f"{'—':>30}"
                continue
            cell = f"{c['em']}/{c['n']} = {100 * c['acc']:3.0f}%"
            if c["lenient"] != c["em"]:
                cell += f" ({c['lenient']} lenient)"
            cell += f"  {c['mean_seconds']:.1f} s"
            row += f"{cell:>30}"
        print(row)
    print(f"\n  {DIM}exact match = reply equals the value after stripping quotes and whitespace; "
          f"lenient = the first UUID in the reply is the value; time per question incl. prefill{RESET}")
    # 答錯的種類
    lines = []
    for s in sizes:
        for name, label, _ in POSITIONS:
            c = cells.get((s, name))
            if not c:
                continue
            wrong = {k: v for k, v in c["kinds"].items() if k not in ("correct", "correct_with_extra_words")}
            if wrong:
                parts = []
                for k, v in sorted(wrong.items()):
                    if k == "other_value" and c["mean_offset"] is not None:
                        parts.append(f"{v} other value (mean offset {c['mean_offset']:+.1f})")
                    else:
                        parts.append(f"{v} {k.replace('_', ' ')}")
                lines.append(f"    {s:<5} {label:<8} wrong: " + ", ".join(parts))
    if lines:
        print(f"  {DIM}what the wrong answers were:{RESET}")
        print("\n".join(lines))
    print()


def load_summary(fake: bool) -> dict | None:
    p = REHEARSAL / ("summary.fake.json" if fake else "summary.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for name, m in cfg["models"].items():
        if args.only and name != args.only:
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {name}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha, ignore_patterns=NON_MODEL_FILES)
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已寫進 versions.lock{RESET}")


def cmd_inspect(args, cfg):
    v = versions()
    for k, val in v.items():
        print(f"  {k:<16} {val}" + ("" if val else f"  {RED}← 缺{RESET}"))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"  {RED}這台不是 Apple silicon：只能用 --fake 演練{RESET}")
    m = make_model(args.model, cfg, args, weights=False)
    banner(f"{cfg['demo']['title']}  ·  {m.label}")
    print(f"  config.json: max_position_embeddings = {m.max_pos}"
          + (f"   rope_scaling = {json.dumps(m.rope_scaling)}" if m.rope_scaling else ""))
    cal = [calibrate(m, cfg, s) for s in cfg["sizes"]]
    print(f"\n  {'size':<6}{'target tok':>12}{'K':>8}{'tokens at K':>14}{'tok/pair':>10}   positions 1, ⌊K/4⌋, ⌊K/2⌋, ⌊3K/4⌋, K")
    for c in cal:
        pos = ", ".join(str(p) for _, _, p in positions_for(c["K"]))
        print(f"  {c['label']:<6}{c['target_tokens']:>12,}{c['K']:>8}{c['tokens_at_K']:>14,}{str(c.get('tokens_per_pair', '—')):>10}   {pos}"
              + (f"   {GREY}(k fixed in demo_config.toml){RESET}" if c["fixed"] else ""))
    q = make_question(cal[0]["K"], question_seeds(cfg, cal[0]["label"], 1)[0])
    user = render_user(cfg, arrange(q, cal[0]["K"] // 2), q["answer"][0])
    prompt = m.render(cfg, user)
    print(f"\n  one prompt, {cal[0]['label']}, answer at ⌊K/2⌋ (first 6 and last 4 lines of the user turn):")
    lines = user.splitlines()
    for ln in lines[:6] + ["   …"] + lines[-4:]:
        print(f"    {GREY}{ln[:100]}{RESET}")
    print(f"  chat template applied: {len(prompt)} chars, {m.count(prompt):,} tokens"
          + (f"; enable_thinking accepted: {m.think_supported}" if m.think_supported is not None else ""))
    print(f"  each question is asked {len(POSITIONS)} times (one per answer position); rehearse = "
          f"{len(cfg['sizes'])} sizes × {len(POSITIONS)} × Q = {cfg['demo']['q']} → {len(cfg['sizes']) * len(POSITIONS) * int(cfg['demo']['q'])} generations")
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    rec = {"date": now(), "model": m.repo, "revision": m.revision, "versions": v, "max_position_embeddings": m.max_pos,
           "rope_scaling": m.rope_scaling, "calibration": cal, "example_prompt_tokens": m.count(prompt),
           "system": cfg["prompt"]["system"], "instruction": cfg["prompt"]["instruction"]}
    path = REHEARSAL / ("inspect.fake.json" if m.fake else "inspect.json")
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def run_cells(m, cfg, sizes: list[dict], Q: int, live_print: bool = False) -> list[dict]:
    cells = []
    for s in sizes:
        cal = calibrate(m, cfg, s)
        K = cal["K"]
        seeds = question_seeds(cfg, s["label"], Q)
        banner(f"{s['label']}: K = {K} pairs (≈ {cal['tokens_at_K']:,} tokens), {Q} questions × {len(POSITIONS)} positions")
        t_size = time.time()
        per_pos = {name: [] for name, _, _ in POSITIONS}
        for qi, seed in enumerate(seeds):
            q = make_question(K, seed)
            for name, label, pos in positions_for(K):
                r = ask(m, cfg, q, pos)
                r["size"], r["pos_name"] = s["label"], name
                per_pos[name].append(r)
                if live_print:
                    print(f"    q{qi + 1:02d} {label:<8} {mark(r)} {r['seconds']:5.1f} s  {GREY}{r['reply'][:60]!r}{RESET}")
            done = [x for recs in per_pos.values() for x in recs]
            acc = sum(x["em"] for x in done) / len(done)
            print(f"  q{qi + 1:02d}/{Q}  " + "  ".join(f"{label} {mark(per_pos[name][-1])}" for name, label, _ in POSITIONS)
                  + f"   {GREY}running exact match {100 * acc:.0f}%, {time.time() - t_size:.0f} s so far{RESET}")
        for name, label, pos in positions_for(K):
            cells.append({"size": s["label"], "K": K, "pos_name": name, "pos_label": label, "pos": pos, "records": per_pos[name]})
        note(f"  {s['label']} done in {time.time() - t_size:.0f} s")
    return cells


def cmd_rehearse(args, cfg):
    Q = int(args.q or cfg["demo"]["q"])
    smoke = args.q is not None and Q < int(cfg["demo"]["q"])
    out_dir = RUNS / "smoke" if smoke else REHEARSAL
    sizes = [s for s in cfg["sizes"] if not args.size or s["label"] == args.size]
    m = make_model(args.model, cfg, args)
    banner(f"{cfg['demo']['title']}  ·  {m.label}" + (f"  {AMBER}[smoke test, Q = {Q}]{RESET}" if smoke else ""))
    t0 = time.time()
    cells = run_cells(m, cfg, sizes, Q)
    summary = summarize(cells)
    summary.update({"date": now(), "model": m.repo, "revision": m.revision, "versions": versions(),
                    "device": device_info() if not m.fake else {}, "fake": m.fake, "q": Q, "seconds_total": round(time.time() - t0),
                    "peak_memory_gb": peak_memory_gb() if not m.fake else None,
                    "max_position_embeddings": m.max_pos, "rope_scaling": m.rope_scaling,
                    "system": cfg["prompt"]["system"], "instruction": cfg["prompt"]["instruction"], "temperature": cfg["demo"]["temperature"]})
    print_table(summary, [s["label"] for s in sizes], tag=f"{AMBER}彩排 {summary['date'][:16]}{RESET}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = ".fake.json" if m.fake else ".json"
    (out_dir / ("summary" + sfx)).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    full = {k: v for k, v in summary.items() if k != "cells"}
    full["cells"] = cells
    (out_dir / ("rehearsal" + sfx)).write_text(json.dumps(full, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    note(f"→ {out_dir.relative_to(HERE)}/summary{sfx}, rehearsal{sfx}   ({summary['seconds_total']} s total)")


def wait(msg: str, extra: str = " · r = 彩排的表") -> str:
    try:
        key = input(f"\n{GREY}[{msg}{extra} · x = 離開]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        key = "x"
    print()
    return key


def cmd_show(args, cfg):
    size_label = args.size or cfg["demo"]["live_size"]
    size = next(s for s in cfg["sizes"] if s["label"] == size_label)
    m = make_model(args.model, cfg, args)
    cal = calibrate(m, cfg, size)
    K = cal["K"]
    banner(f"{cfg['demo']['title']}  ·  {m.label}" + (f"  {RED}[FAKE]{RESET}" if m.fake else ""))
    print(f"  {size_label}: K = {K} UUID pairs as one JSON object (≈ {cal['tokens_at_K']:,} tokens with the chat template);"
          f" nominal length {m.max_pos:,}" + (f", rope_scaling factor {m.rope_scaling.get('factor')}" if m.rope_scaling else ""))
    summary = load_summary(m.fake)
    if summary and summary.get("model") != m.repo:
        note(f"彩排紀錄是另一個模型（{summary.get('model')}），r 不拿來對照")
        summary = None
    LIVE.mkdir(parents=True, exist_ok=True)
    log = LIVE / f"{now()[:10].replace('-', '')}.jsonl"
    live_names = list(cfg["demo"]["live_positions"])
    seeds = question_seeds(cfg, size_label, len(live_names), live=True)
    pos_of = dict((name, (label, pos)) for name, label, pos in positions_for(K))
    print(f"  three live questions, answer at " + " / ".join(pos_of[n][0] for n in live_names) + f" of {K}; these are not the rehearsed questions.")
    i = 0
    while i < len(live_names):
        key = wait(f"Enter = question {i + 1}: answer at {pos_of[live_names[i]][0]} (pair {pos_of[live_names[i]][1]} of {K})")
        if key == "x":
            break
        if key == "r":
            if summary:
                print_table(summary, [s["label"] for s in cfg["sizes"]], tag=f"{AMBER}彩排 {summary['date'][:16]}，非現場{RESET}")
            else:
                print(f"  {RED}沒有 runs/rehearsal/summary.json：先跑 ./present.sh rehearse{RESET}")
            continue
        q = make_question(K, seeds[i])
        label, pos = pos_of[live_names[i]]
        pairs = arrange(q, pos)
        head = json.dumps(dict(pairs[:1]))[:-1] + ", … (" + str(K) + " pairs) … }"
        print(f"  {BOLD}JSON data:{RESET}  {GREY}{head}{RESET}")
        print(f"  {BOLD}Key:{RESET} {json.dumps(q['answer'][0])}   {GREY}(pair {pos} of {K} — {label}){RESET}")
        r = ask(m, cfg, q, pos)
        r.update({"size": size_label, "pos_name": live_names[i], "date": now()})
        tps = f", prefill {r['prompt_tps']:,.0f} tok/s" if r.get("prompt_tps") else ""
        print(f"  {BOLD}Reply:{RESET} {r['reply']!r}")
        print(f"  {mark(r)}  expected {q['answer'][1]}   {GREY}{r['seconds']:.1f} s, {r.get('prompt_tokens') or 0:,} prompt tokens{tps}{RESET}"
              + (f"   {AMBER}{describe_wrong(r)}{RESET}" if not r["em"] and describe_wrong(r) else ""))
        append_jsonl(log, {"model": m.repo, **r})
        i += 1
    if i >= len(live_names):
        key = wait("Enter = 彩排的表", extra="")
        if key != "x" and summary:
            print_table(summary, [s["label"] for s in cfg["sizes"]], tag=f"{AMBER}彩排 {summary['date'][:16]}，非現場{RESET}")
        elif key != "x":
            print(f"  {RED}沒有 runs/rehearsal/summary.json：先跑 ./present.sh rehearse{RESET}")
    print()


def cmd_replay(args, cfg):
    summary = load_summary(args.fake)
    if not summary:
        print(f"{RED}沒有 runs/rehearsal/summary.json：先跑 ./present.sh rehearse{RESET}")
        return
    banner(f"{AMBER}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {summary['date']}  {summary['model']}")
    v = summary.get("versions") or {}
    print(f"  Q = {summary['q']} per cell, temperature {summary.get('temperature')}, {summary.get('seconds_total')} s total"
          + (f", peak memory {summary['peak_memory_gb']:.1f} GB" if summary.get("peak_memory_gb") else "")
          + f"; mlx-lm {v.get('mlx-lm')}, mlx {v.get('mlx')}")
    print(f"  system: {summary.get('system')!r}")
    sizes = []
    for c in summary["cells"]:
        if c["size"] not in sizes:
            sizes.append(c["size"])
    print_table(summary, sizes, tag=f"{AMBER}彩排 {summary['date'][:16]}{RESET}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（回答是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--model", default="primary", help="demo_config.toml 的 [models.*] 名稱（primary / backup）")
    p.add_argument("--only", help="fetch：只抓一個模型")
    p.add_argument("--size", help="rehearse／show：只做一檔（4k / 16k）")
    p.add_argument("--q", type=int, help="rehearse：每格幾題；小於 demo_config 的 q 就是煙霧測試，寫到 runs/smoke/")
    sub = p.add_subparsers(dest="cmd", required=True)   # present.sh 把選項放在子命令前面，所以選項全掛在主 parser
    for c in ("fetch", "inspect", "rehearse", "show", "replay"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
