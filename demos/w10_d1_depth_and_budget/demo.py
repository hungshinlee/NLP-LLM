#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 1：不想／點點／想 × 鏈長，再掃推理的預算（投影片 p17 前導頁、p19 (a)、p28 (b)）。

    ./present.sh test            不載模型：狀態機（上限、不准停、回答的上限）、切 token、接 prompt、q 的擬合、翻轉的數法
    ./present.sh inspect         課前：版本、兩種模板的結尾、<think>／</think>／"." 的 id、一段真的推理長什麼樣子、兩個動作各做一次
    ./present.sh probe           彩排前：tokens/s 對 batch size（--part speed）、每個 m 上三種答法的小樣本（--part levels）、估時
    ./present.sh rehearse        彩排全套：(a) 鏈長 → (b) 預算 → 現場那一題；做過的跳過（--force 重跑、--part chain|budget|live 只做一段）
    ./present.sh live            課堂 (a)：一題三種答法，Enter 一種
    ./present.sh replay          課堂 (a)：讀 runs/rehearsal/chain.json，Enter 一欄（不載模型）
    ./present.sh replay-budget   課堂 (b)：讀 runs/rehearsal/budget.json，Enter 一段（不載模型）
    ./present.sh replay-live     現場那一題失敗：印彩排的 live.json（不載模型）
    ./present.sh tiny            亂數權重的小模型走一遍真的流程（不下載任何東西；寫到 runs/smoke/）

題目與 verifier 是上週 demo 2 的（../w09_d2_grpo_scratch/grpo.py，從檔案路徑載入）。budget forcing 在 forcing.py。
同一題的三種答法：不想（模板的 enable_thinking=False）、點點（推理換成等長的「.」）、想（enable_thinking=True）。
(b) 的每一欄都從 (a)「想」的那一段推理出發：上限 c ＝ 只留前 c 個 token 再接 </think>；不准停 k 次 ＝ 在它想停的地方接「Wait」繼續。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）、transformers（tokenizer）、huggingface_hub。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10
    import tomli as tomllib   # type: ignore

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("demos_common", HERE.parent / "common.py")
G = _load("w09_d2_grpo", HERE.parent / "w09_d2_grpo_scratch" / "grpo.py")     # 出題器與 verifier
F = _load("w10_d1_forcing", HERE / "forcing.py")
AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, BLUE = C.AMBER, C.BOLD, C.DIM, C.GREEN, C.GREY, C.RED, C.RESET, C.BLUE
banner, note, now = C.banner, C.note, C.now

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def out_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "rehearsal")


def cache_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "") / "cache"


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def pause(prompt: str = "") -> str:
    try:
        return input(f"{DIM}{prompt}{RESET}")
    except EOFError:
        print()
        return ""


def r4(x):
    return None if x is None else float(f"{x:.4g}")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / max(n, 1))


def settings(cfg: dict, args) -> dict:
    """[task]、[chain]、[budget]、[sampler] 合起來；--tiny 與命令列的覆寫在這裡套。"""
    s = {**cfg["task"], "ms": list(cfg["chain"]["ms"]), "n": int(cfg["chain"]["n"]), "caps": list(cfg["budget"]["caps"]),
         "waits": sorted(int(w) for w in cfg["budget"]["waits"]), "easy_m": int(cfg["budget"]["easy_m"]), "medium_m": int(cfg["budget"]["medium_m"]),
         "batch_size": int(cfg["sampler"]["batch_size"]), "seed": int(cfg["demo"]["seed"])}
    if args.tiny:
        t = cfg["tiny"]
        s.update(n=int(t["n"]), answer_max_tokens=int(t["answer_max_tokens"]), think_max_tokens=int(t["think_max_tokens"]),
                 caps=list(t["caps"]), batch_size=int(t["batch_size"]), prompt="{expr}=")
    if args.n:
        s["n"] = int(args.n)
    if args.batch_size:
        s["batch_size"] = int(args.batch_size)
    return s


def level_for(s: dict, m: int) -> dict:
    return {"terms": int(m) + 1, "lo": s["lo"], "hi": s["hi"], "ops": list(s["ops"]), "mul_lo": s["mul_lo"], "mul_hi": s["mul_hi"]}


def problems_for(s: dict, m: int, n: int | None = None) -> list[dict]:
    return G.make_problems(random.Random(s["seed"] * 1000 + int(m)), level_for(s, m), int(n or s["n"]))


# ── 模型 ────────────────────────────────────────────────────────
class TinyTok:
    """--tiny 用的字元級 tokenizer。只為了不下載也能走一遍流程。"""
    VOCAB = list("0123456789+-×= .") + ["\n", "<think>", "</think>", "Wait", "<eos>"]

    def __init__(self):
        self.eos_token_ids = {len(self.VOCAB) - 1}
        self._by_len = sorted(range(len(self.VOCAB)), key=lambda i: -len(self.VOCAB[i]))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        return messages[0]["content"] + "\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            for j in self._by_len:
                if s.startswith(self.VOCAB[j], i):
                    out.append(j)
                    i += len(self.VOCAB[j])
                    break
            else:
                i += 1
        return out

    def decode(self, ids):
        return "".join(self.VOCAB[i] for i in ids if i < len(self.VOCAB) - 1)


class Runner:
    """模型、tokenizer、sampler，與三種答法。"""

    def __init__(self, cfg: dict, args):
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        t0 = time.time()
        self.cfg, self.tiny, self.s = cfg, bool(args.tiny), settings(cfg, args)
        if self.tiny:
            from mlx_lm.models import qwen3
            self.tok = TinyTok()
            mx.random.seed(self.s["seed"])
            a = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
                                rms_norm_eps=1e-6, vocab_size=len(TinyTok.VOCAB), num_key_value_heads=2, max_position_embeddings=512,
                                rope_theta=1e6, head_dim=16, tie_word_embeddings=True)
            self.model, self.path = qwen3.Model(a), None
            mx.eval(self.model.parameters())
        else:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            m = cfg["model"]
            self.path = snapshot_download(m["repo"], revision=C.locked_revision(m["repo"], m.get("revision")), local_files_only=not args.online,
                                          ignore_patterns=NON_MODEL_FILES)
            self.model, self.tok = load(self.path)
        sc = cfg["sampler"]
        self.sampler = make_sampler(temp=float(sc["temperature"]), top_p=float(sc["top_p"]), top_k=int(sc["top_k"]))
        fc = dict(cfg["forcing"])
        if self.tiny:
            fc["answer_prefix"] = cfg["tiny"]["answer_prefix"]
        self.ids = F.think_ids(self.tok, fc)
        self.load_seconds = time.time() - t0
        self.tokens, self.seconds = 0, 0.0

    # prompt
    def prompt_ids(self, problem: dict, thinking: bool) -> list[int]:
        q = self.s["prompt"].format(expr=problem["expr"])
        text = self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return list(self.tok.encode(text, add_special_tokens=False))

    def forcer(self, waits: int = 0, n_think0: int = 0, think_cap: int | None = None):
        return F.Forcer(self.ids, self.sampler, think_cap if think_cap is not None else self.s["think_max_tokens"],
                        self.s["answer_max_tokens"], waits=waits, n_think0=n_think0)

    def gen(self, items: list[dict], tag: str, echo: bool = True, batch_size: int | None = None) -> list[dict]:
        import mlx.core as mx
        if not items:
            return []
        mx.random.seed(self.s["seed"] + sum(ord(c) * (i + 1) for i, c in enumerate(tag)))

        def progress(done, total):
            if echo and sys.stdout.isatty():
                print(f"\r  {GREY}{tag}: {done}/{total}{RESET}", end="", flush=True)

        res, info = F.generate(self.model, self.ids.eos, items, self.sampler, int(batch_size or self.s["batch_size"]), progress)
        self.tokens += info["generation_tokens"]
        self.seconds += info["seconds"]
        if echo:
            rate = info["generation_tokens"] / info["seconds"] if info["seconds"] > 0 else 0.0
            print(f"\r  {GREY}{tag}: {len(items)} sequences, {info['generation_tokens']:,} tokens, {info['seconds']:.1f} s ({rate:.0f} tok/s){RESET}")
        self.last_info = info
        return res

    # 判定
    def judge(self, answer_tokens: list[int], answer: int) -> dict:
        text = self.tok.decode(list(answer_tokens)).strip()
        return {"text": text, "right": int(G.reward_strict(text, answer) > 0), "numbers": len(G.numbers_in(text))}

    # 三種答法
    def answer_after(self, prefixes: list[list[int]], problems: list[dict], tag: str, echo: bool = True) -> list[dict]:
        """prefix 已經含 </think>：只生成回答。"""
        res = self.gen([{"prompt": p, "max_tokens": self.s["answer_max_tokens"]} for p in prefixes], tag, echo)
        out = []
        for r, p in zip(res, problems):
            toks = [t for t in r["tokens"] if t not in self.ids.eos]
            j = self.judge(toks, p["answer"])
            j["cut"] = int(r["finish"] != "stop")
            out.append(j)
        return out

    def direct(self, problems: list[dict], tag: str = "no thinking", echo: bool = True) -> list[dict]:
        return self.answer_after([self.prompt_ids(p, False) + self.ids.answer_prefix for p in problems], problems, tag, echo)

    def parse_think(self, r: dict, problem: dict) -> dict:
        sp = F.split_think(r["tokens"], self.ids)
        j = self.judge(sp["answer"], problem["answer"]) if sp["closed"] else {"text": "", "right": 0, "numbers": 0}
        ev = [e[0] for e in r["events"]]
        j.update(tokens=r["tokens"], n_think=len(sp["think"]), closed=int(sp["closed"]), opened=int(sp["open_at"] is not None),
                 capped=int("cap" in ev), cut=int("answer_cut" in ev or r["finish"] != "stop"))
        return j

    def think(self, problems: list[dict], tag: str = "thinking", echo: bool = True, batch_size: int | None = None) -> list[dict]:
        items = []
        for p in problems:
            f = self.forcer()
            items.append({"prompt": self.prompt_ids(p, True), "max_tokens": f.max_tokens(), "forcer": f})
        return [self.parse_think(r, p) for r, p in zip(self.gen(items, tag, echo, batch_size), problems)]

    def dots(self, problems: list[dict], natural: list[dict], tag: str = "dots", echo: bool = True) -> list[dict]:
        pre = [F.prefix_dots(self.prompt_ids(p, True), self.ids, nat["n_think"]) for p, nat in zip(problems, natural)]
        return self.answer_after(pre, problems, tag, echo)

    def think_text(self, nat: dict) -> str:
        return self.tok.decode(F.split_think(nat["tokens"], self.ids)["think"]).strip()


def load_runner(cfg, args) -> Runner:
    what = "tiny random Qwen3 (no download)" if args.tiny else f"{cfg['model']['short']}  ←  {cfg['model']['repo']}"
    note(f"載入 {what}")
    r = Runner(cfg, args)
    pk = C.peak_memory_gb()
    note(f"  完成：{r.load_seconds:.1f} 秒" + (f"；peak memory {pk:.2f} GB" if pk else ""))
    return r


def header(run: Runner, args) -> dict:
    sc = run.cfg["sampler"]
    return {"date": now(), "versions": C.versions(), "device": C.device_info(), "tiny": bool(args.tiny),
            "model": None if args.tiny else run.cfg["model"]["repo"],
            "sampler": {k: sc[k] for k in ("temperature", "top_p", "top_k")}, "batch_size": run.s["batch_size"], "seed": run.s["seed"],
            "task": {k: run.s[k] for k in ("prompt", "lo", "hi", "ops", "mul_lo", "mul_hi", "answer_max_tokens", "think_max_tokens")},
            "forcing": {**run.cfg["forcing"], "answer_prefix": run.tok.decode(run.ids.answer_prefix)}}


def stale(rec: dict, run: Runner, args) -> bool:
    """存檔的設定（模型、sampler、題目、記號）與現在的不同：那份紀錄不能接著用。"""
    h = header(run, args)
    return bool(rec) and any(rec.get(k) != h[k] for k in ("model", "sampler", "task", "forcing", "seed"))


# ── (a) 鏈長 ─────────────────────────────────────────────────────
def mode_stats(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["right"] for r in rows)
    return {"n": n, "k": k, "acc": r4(k / n), "cut_share": r4(mean([r["cut"] for r in rows])),
            "multi_number_share": r4(mean([1.0 if r["numbers"] > 1 else 0.0 for r in rows]))}


def chain_one(run: Runner, m: int, args) -> tuple[dict, list[dict]]:
    """一個 m：不想、想、點點。回傳（統計, 每題的自然推理）。"""
    problems = problems_for(run.s, m)
    t0, tok0 = time.time(), run.tokens
    d = run.direct(problems, f"m={m} no thinking")
    t = run.think(problems, f"m={m} thinking")
    o = run.dots(problems, t, f"m={m} dots")
    lens = [x["n_think"] for x in t]
    st = {"m": m, "direct": mode_stats(d), "dots": mode_stats(o), "think": mode_stats(t)}
    st["think"].update(think_len_mean=r4(mean(lens)), think_len_median=r4(statistics.median(lens)), think_len_max=max(lens),
                       capped_share=r4(mean([x["capped"] for x in t])), no_answer_share=r4(mean([1 - x["closed"] for x in t])),
                       no_open_share=r4(mean([1 - x["opened"] for x in t])))
    st["rows"] = [{"expr": p["expr"], "answer": p["answer"], "d": a["right"], "o": b["right"], "t": c["right"], "L": c["n_think"],
                   "d_text": a["text"][:40], "t_text": c["text"][:40]} for p, a, b, c in zip(problems, d, o, t)]
    st["seconds"] = round(time.time() - t0, 1)
    st["generation_tokens"] = run.tokens - tok0
    return st, t


def save_natural(args, m: int, natural: list[dict]) -> None:
    write_json(cache_dir(args) / f"natural_m{m}.json", [{"tokens": x["tokens"]} for x in natural])


def load_natural(run: Runner, args, m: int) -> list[dict] | None:
    p = cache_dir(args) / f"natural_m{m}.json"
    if not p.exists():
        return None
    problems = problems_for(run.s, m)
    data = read_json(p)
    if len(data) != len(problems):
        return None
    out = []
    for x, pr in zip(data, problems):
        sp = F.split_think(x["tokens"], run.ids)
        j = run.judge(sp["answer"], pr["answer"]) if sp["closed"] else {"text": "", "right": 0, "numbers": 0}
        j.update(tokens=x["tokens"], n_think=len(sp["think"]), closed=int(sp["closed"]))
        out.append(j)
    return out


def chain_fit(per_m: dict) -> dict:
    counts = {int(m): (v["think"]["k"], v["think"]["n"]) for m, v in per_m.items()}
    f = F.fit_q(counts)
    return {"q": r4(f["q"]), "half_life_steps": r4(f["half_life_steps"]), "per_m_q": {str(m): r4(v) for m, v in f["per_m_q"].items()},
            "predicted": {str(m): r4(v) for m, v in f["predicted"].items()}}


def rehearse_chain(run: Runner, args) -> None:
    out = out_dir(args) / "chain.json"
    rec = read_json(out) if out.exists() and not args.force else {}
    if stale(rec, run, args):
        note("chain.json 是用不同的設定跑的：全部重跑")
        rec = {}
    rec.setdefault("per_m", {})
    banner(f"(a) chain length · n = {run.s['n']} per m · m ∈ {run.s['ms']}", AMBER)
    for m in run.s["ms"]:
        have = rec["per_m"].get(str(m))
        if have and have["think"]["n"] == run.s["n"] and (cache_dir(args) / f"natural_m{m}.json").exists():
            note(f"m={m}: 已有，跳過（--force 重跑）")
            continue
        st, natural = chain_one(run, m, args)
        save_natural(args, m, natural)
        rec["per_m"][str(m)] = st
        rec.update(header(run, args))
        rec["fit"] = chain_fit(rec["per_m"])
        write_json(out, rec)
        print(f"  {GREEN}m={m}{RESET}: no thinking {st['direct']['acc']:.2f} · dots {st['dots']['acc']:.2f} · thinking {BOLD}{st['think']['acc']:.2f}{RESET} "
              f"({st['think']['think_len_mean']:.0f} thinking tokens on average, {st['think']['capped_share']:.0%} hit the cap) · {st['seconds'] / 60:.1f} min")
        neg = [r for r in st["rows"] if r["answer"] < 0]
        pos = [r for r in st["rows"] if r["answer"] >= 0]
        by = "  ".join(f"{name} {mean([r[k] for r in pos]) or 0:.2f} / {mean([r[k] for r in neg]) or 0:.2f}" for k, name in (("d", "no thinking"), ("o", "dots"), ("t", "thinking")))
        print(f"  {GREY}  result ≥ 0 ({len(pos)}) / result < 0 ({len(neg)}):  {by}{RESET}")
        cuts = {k: st[k]["cut_share"] for k in ("direct", "dots", "think")}
        flag = RED if max(cuts.values()) > 0.05 else GREY
        print(f"  {flag}  answers that ran into the {run.s['answer_max_tokens']}-token limit: no thinking {cuts['direct']:.0%} · dots {cuts['dots']:.0%} · thinking {cuts['think']:.0%}"
              + ("  ← above 5%: those answers are judged wrong for the wrong reason; stop and look at t_text in chain.json" if flag == RED else "") + RESET)
    rec["peak_gb"] = r4(C.peak_memory_gb())
    write_json(out, rec)


# ── (b) 預算 ─────────────────────────────────────────────────────
def column(name: str, kind: str, right: list[int], lens: list, changed: list[int] | None = None) -> dict:
    n = len(right)
    return {"name": name, "kind": kind, "n": n, "k": sum(right), "acc": r4(sum(right) / n), "think_len_mean": r4(mean(lens)),
            "changed_share": None if changed is None else r4(mean(changed)), "right": right, "len": lens}


def budget_level(run: Runner, args, m: int, chain_rows: list[dict], label: str) -> dict:
    """一個難度的全部欄位。每一欄都從同一段自然推理出發。"""
    s, ids = run.s, run.ids
    problems = problems_for(s, m)
    natural = load_natural(run, args, m)
    if natural is None:
        raise SystemExit(f"{RED}沒有 m={m} 的自然推理（runs/cache/natural_m{m}.json）：先跑 ./present.sh rehearse --part chain{RESET}")
    prompts = [run.prompt_ids(p, True) for p in problems]
    n = len(problems)
    nat_len = [x["n_think"] for x in natural]
    nat_right = [x["right"] for x in natural]
    cols = [column("0", "direct", [r["d"] for r in chain_rows], [0] * n)]
    t0, tok0 = time.time(), run.tokens

    # 上限：只有推理比上限長的題目要重新作答
    for c in s["caps"]:
        idx = [i for i in range(n) if nat_len[i] > c]
        ans = run.answer_after([F.prefix_capped(prompts[i], natural[i]["tokens"], ids, c) for i in idx], [problems[i] for i in idx], f"{label} cap {c}")
        right = list(nat_right)
        for i, a in zip(idx, ans):
            right[i] = a["right"]
        cols.append(column(str(c), "cap", right, [min(L, c) for L in nat_len], [1 if i in set(idx) else 0 for i in range(n)]))
    cols.append(column("none", "natural", nat_right, nat_len))

    # 對照：推理不動，只把回答重抽一次
    idx = [i for i in range(n) if natural[i]["closed"]]
    ans = run.answer_after([F.prefix_capped(prompts[i], natural[i]["tokens"], ids, nat_len[i]) for i in idx], [problems[i] for i in idx], f"{label} redraw")
    redraw = list(nat_right)
    for i, a in zip(idx, ans):
        redraw[i] = a["right"]

    # 不准停：只跑最大的 K；較小的 k 從同一段的第 k+1 次想停的地方截
    K = max(s["waits"])
    items, ext_idx = [], []
    for i in range(n):
        pre = F.prefix_wait(prompts[i], natural[i]["tokens"], ids)
        if pre is None or nat_len[i] + len(ids.wait_seq) >= s["think_max_tokens"]:
            continue
        f = run.forcer(waits=K - 1, n_think0=nat_len[i] + len(ids.wait_seq))
        items.append({"prompt": pre, "max_tokens": f.max_tokens(), "forcer": f})
        ext_idx.append(i)
    ext = run.gen(items, f"{label} +{K} Wait")
    full, attempts, final = {}, {}, {}
    for i, r in zip(ext_idx, ext):
        sp = F.split_think(r["tokens"], ids)
        nat_think = F.split_think(natural[i]["tokens"], ids)["think"]
        full[i] = nat_think + ids.wait_seq + sp["think"]
        attempts[i] = [len(nat_think)] + [e[1] for e in r["events"] if e[0] == "wait"]
        j = run.judge(sp["answer"], problems[i]["answer"]) if sp["closed"] else {"text": "", "right": 0, "numbers": 0}
        final[i] = {"right": j["right"], "text": j["text"], "len": len(full[i]), "capped": int(any(e[0] == "cap" for e in r["events"]))}
    wait_cols, examples = {}, []
    for k in s["waits"]:
        right, lens, todo = list(nat_right), list(nat_len), []
        for i in ext_idx:
            if k < K and len(attempts[i]) > k:
                todo.append(i)
                lens[i] = attempts[i][k]
            else:
                right[i], lens[i] = final[i]["right"], final[i]["len"]
        head = {i: F._head(natural[i]["tokens"], ids) for i in todo}
        ans = run.answer_after([prompts[i] + head[i] + full[i][:attempts[i][k]] + ids.close_seq + ids.answer_prefix for i in todo],
                               [problems[i] for i in todo], f"{label} +{k} Wait (answers)")
        for i, a in zip(todo, ans):
            right[i] = a["right"]
        wait_cols[k] = column(f"+{k}", "wait", right, lens, [1 if i in set(ext_idx) else 0 for i in range(n)])
        cols.append(wait_cols[k])

    for a, b in zip(cols, cols[1:]):
        b["flips_vs_left"] = F.flips(a["right"], b["right"])
    cols[0]["flips_vs_left"] = None
    for c in cols:
        c["flips_vs_none"] = F.flips(nat_right, c["right"]) if c["kind"] == "wait" else None

    # 被拉長之後由對變錯的題目，留兩題的文字
    last = wait_cols[K]["right"]
    for i in ext_idx:
        if nat_right[i] and not last[i] and len(examples) < 2:
            nat_think = F.split_think(natural[i]["tokens"], ids)["think"]
            examples.append({"expr": problems[i]["expr"], "answer": problems[i]["answer"], "natural_answer": natural[i]["text"][:60],
                             "natural_thinking_tokens": len(nat_think), "natural_thinking_tail": run.tok.decode(nat_think).strip()[-400:],
                             "added_thinking_tokens": final[i]["len"] - len(nat_think), "added_thinking_head": run.tok.decode(full[i][len(nat_think):]).strip()[:600],
                             "final_answer": final[i]["text"][:60]})
    rows = [{"expr": p["expr"], "answer": p["answer"], "right": {c["name"]: c["right"][i] for c in cols}, "len": {c["name"]: c["len"][i] for c in cols}}
            for i, p in enumerate(problems)]
    for c in cols:
        del c["right"], c["len"]
    return {"m": m, "n": n, "columns": cols, "redraw": {"n_redrawn": len(idx), **F.flips(nat_right, redraw), "acc": r4(sum(redraw) / n)},
            "extended": {"n": len(ext_idx), "hit_cap_share": r4(mean([final[i]["capped"] for i in ext_idx])),
                         "mean_stop_attempts": r4(mean([len(attempts[i]) for i in ext_idx])), "K": K},
            "examples": examples, "rows": rows, "seconds": round(time.time() - t0, 1), "generation_tokens": run.tokens - tok0}


def rehearse_budget(run: Runner, args) -> None:
    out, chain = out_dir(args) / "budget.json", out_dir(args) / "chain.json"
    if not chain.exists():
        raise SystemExit(f"{RED}先跑 ./present.sh rehearse --part chain{RESET}")
    ch = read_json(chain)
    rec = read_json(out) if out.exists() and not args.force else {}
    if stale(ch, run, args):
        raise SystemExit(f"{RED}chain.json 是用不同的設定跑的：先重跑 ./present.sh rehearse --part chain{RESET}")
    if stale(rec, run, args) or rec.get("chain_date") != ch["date"]:
        rec = {}
    rec.setdefault("levels", {})
    banner(f"(b) thinking budget · caps {run.s['caps']} · Wait × {run.s['waits']}", AMBER)
    for label, m in (("easy", run.s["easy_m"]), ("medium", run.s["medium_m"])):
        if str(m) not in ch["per_m"]:
            raise SystemExit(f"{RED}chain.json 沒有 m={m}：[budget] 的 easy_m／medium_m 要在 [chain] ms 裡{RESET}")
        have = rec["levels"].get(label)
        if have and have["m"] == m and have["n"] == ch["per_m"][str(m)]["think"]["n"]:
            note(f"{label} (m={m}): 已有，跳過（--force 重跑）")
            continue
        lv = budget_level(run, args, m, ch["per_m"][str(m)]["rows"], label)
        rec["levels"][label] = lv
        rec.update(header(run, args))
        rec["chain_date"] = ch["date"]
        write_json(out, rec)
        print(f"  {GREEN}{label} (m={m}){RESET}: " + " · ".join(f"{c['name']} {c['acc']:.2f}" for c in lv["columns"]) + f" · {lv['seconds'] / 60:.1f} min")
    rec["peak_gb"] = r4(C.peak_memory_gb())
    write_json(out, rec)


# ── 現場那一題 ───────────────────────────────────────────────────
def live_problem(run: Runner, args) -> tuple[int, dict]:
    lc = run.cfg["live"]
    m = run.s["ms"][0] if args.tiny else int(args.m or lc["m"])
    return m, problems_for(run.s, m, max(run.s["n"], int(lc["index"]) + 1))[int(lc["index"])]


def live_record(run: Runner, args, echo: bool = True) -> dict:
    m, p = live_problem(run, args)
    t0 = time.time()
    t = run.think([p], "thinking (shown in the third step)", echo, batch_size=1)[0]
    t_sec = time.time() - t0
    d = run.direct([p], "no thinking", echo)[0]
    o = run.dots([p], [t], "dots", echo)[0]
    return {"date": now(), "m": m, "expr": p["expr"], "answer": p["answer"],
            "direct": {"text": d["text"], "right": d["right"], "cut": d["cut"]},
            "dots": {"n": t["n_think"], "text": o["text"], "right": o["right"], "cut": o["cut"]},
            "think": {"n": t["n_think"], "thinking": run.think_text(t), "text": t["text"], "right": t["right"], "capped": t["capped"],
                      "closed": t["closed"], "seconds": round(t_sec, 1), "finished_at": time.time()}}


def mark(right) -> str:
    return f"{GREEN}✓ right{RESET}" if right else f"{RED}✗ wrong{RESET}"


def show_problem(expr: str, m: int) -> None:
    print(f"\n  {BOLD}Compute {expr}{RESET}      {GREY}{m} steps in a row{RESET}")


def show_live(rec: dict, cfg: dict, interactive: bool = True, problem_shown: bool = False) -> None:
    wait = pause if interactive else (lambda *_: None)
    if not problem_shown:
        show_problem(rec["expr"], rec["m"])
    wait("  [Enter] the correct result ")
    print(f"  correct result: {BOLD}{rec['answer']}{RESET}")
    wait("  [Enter] 1 · no thinking ")
    print(f"\n  {BLUE}1 · no thinking{RESET}        answer: {BOLD}{rec['direct']['text'] or '(nothing)'}{RESET}   {mark(rec['direct']['right'])}")
    wait("  [Enter] 2 · dots ")
    n = rec["dots"]["n"]
    print(f"\n  {BLUE}2 · dots{RESET}               its own reasoning was {BOLD}{n}{RESET} tokens long → replaced by {n} × \".\"")
    print(f"  {GREY}<think>{RESET} {'.' * min(n, 60)}{' …' if n > 60 else ''} {GREY}</think>{RESET}")
    print(f"                         answer: {BOLD}{rec['dots']['text'] or '(nothing)'}{RESET}   {mark(rec['dots']['right'])}")
    wait("  [Enter] 3 · thinking ")
    ago = time.time() - rec["think"].get("finished_at", 0)
    when = f"generated {ago:.0f} s ago, before the first Enter" if ago < 3600 else "generated in rehearsal"
    print(f"\n  {BLUE}3 · thinking{RESET}           {GREY}{rec['think']['n']} tokens in {rec['think']['seconds']} s; {when}{RESET}")
    delay = float(cfg["live"]["line_delay"]) if interactive else 0.0
    for line in rec["think"]["thinking"].splitlines():
        print(f"  {GREY}│{RESET} {line}")
        if delay:
            time.sleep(delay)
    if rec["think"]["capped"]:
        print(f"  {AMBER}(reached the safety cap; </think> was forced){RESET}")
    print(f"                         answer: {BOLD}{rec['think']['text'] or '(nothing)'}{RESET}   {mark(rec['think']['right'])}")
    note("\n  One problem proves nothing: ./present.sh replay for the whole table.")


def cmd_live(args, cfg):
    banner(cfg["demo"]["title"].split(" — ")[0] + " (a) — one problem, answered three ways")
    run = load_runner(cfg, args)
    m, p = live_problem(run, args)
    show_problem(p["expr"], m)
    note("  (the model is answering it three ways now; the results appear one Enter at a time)")
    rec = live_record(run, args, echo=False)
    if not args.tiny:
        C.append_jsonl(RUNS / "live" / "live.jsonl", rec)
    show_live(rec, cfg, interactive=not args.no_pause, problem_shown=True)


def cmd_replay_live(args, cfg):
    p = out_dir(args) / "live.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse --part live{RESET}")
    rec = read_json(p)
    banner(f"replay of the rehearsal ({rec['date']}) — one problem, answered three ways")
    show_live(rec, cfg, interactive=not args.no_pause)


# ── rehearse ────────────────────────────────────────────────────
def cmd_rehearse(args, cfg):
    parts = args.part or ["chain", "budget", "live"]
    t0 = time.time()
    run = load_runner(cfg, args)
    if "chain" in parts:
        rehearse_chain(run, args)
    if "budget" in parts:
        rehearse_budget(run, args)
    if "live" in parts:
        banner("the live problem, once, for replay-live", AMBER)
        rec = live_record(run, args)
        rec.update(versions=C.versions(), model=None if args.tiny else cfg["model"]["repo"])
        write_json(out_dir(args) / "live.json", rec)
        show_live(rec, cfg, interactive=False)
    print(f"\n{GREEN}rehearsal finished in {(time.time() - t0) / 60:.1f} min{RESET} · {run.tokens:,} tokens generated · "
          f"./present.sh replay · ./present.sh replay-budget")


# ── replay ──────────────────────────────────────────────────────
def cmd_replay(args, cfg):
    p = out_dir(args) / "chain.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    rec = read_json(p)
    wait = (lambda *_: None) if args.no_pause else pause
    ms = sorted(rec["per_m"], key=int)
    n = rec["per_m"][ms[0]]["think"]["n"]
    banner(f"demo 1 (a) · rehearsal of {rec['date']} · {rec.get('model') or 'tiny random model'} · {n} problems per row")
    order = [("direct", "no thinking"), ("think", "thinking"), ("dots", "dots")]
    cols = [("direct", "no thinking"), ("dots", "dots"), ("think", "thinking")]

    def table(shown: set) -> None:
        print(f"\n  {'steps m':<9}" + "".join(f"{name:>14}" for _, name in cols))
        for m in ms:
            v = rec["per_m"][m]
            print(f"  {m:<9}" + "".join((f"{BOLD}{v[key]['acc']:>14.2f}{RESET}" if key in shown else f"{'·':>14}") for key, _ in cols))

    shown: set = set()
    table(shown)
    for key, name in order:
        wait(f"  [Enter] {name} ")
        shown.add(key)
        table(shown)
    worst = max(se(rec["per_m"][m][k]["acc"], n) for m in ms for k, _ in cols)
    note(f"  each cell is {n} problems, one sampled answer each: standard error up to ±{worst:.2f}")
    wait("  [Enter] how long it thought, and q^m ")
    fit = rec["fit"]
    print(f"\n  {'steps m':<9}{'thinking tokens':>17}{'hit the cap':>13}{'thinking':>10}{'q^m, one q':>12}{'acc^(1/m)':>11}")
    for m in ms:
        t = rec["per_m"][m]["think"]
        pq = fit["per_m_q"].get(m)
        print(f"  {m:<9}{t['think_len_mean']:>17.0f}{t['capped_share']:>13.0%}{t['acc']:>10.2f}{fit['predicted'][m]:>12.2f}" + (f"{pq:>11.3f}" if pq else f"{'—':>11}"))
    hl = fit["half_life_steps"]
    print(f"\n  one q fitted to the thinking column (maximum likelihood, steps assumed independent): {BOLD}q = {fit['q']:.3f}{RESET}"
          + (f"   → half of the problems wrong at about {hl:.0f} steps" if hl else ""))
    multi = max(rec["per_m"][m]["direct"]["multi_number_share"] for m in ms)
    cut = max(rec["per_m"][m]["direct"]["cut_share"] for m in ms)
    note(f"  no thinking: at most {multi:.0%} of answers contained more than one number, at most {cut:.0%} ran into the {rec['task']['answer_max_tokens']}-token limit")
    note(f"  sampler: temperature {rec['sampler']['temperature']}, top-p {rec['sampler']['top_p']}, top-k {rec['sampler']['top_k']} · one model, one kind of problem, one seed")


def cmd_replay_budget(args, cfg):
    p = out_dir(args) / "budget.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    rec = read_json(p)
    wait = (lambda *_: None) if args.no_pause else pause
    banner(f"demo 1 (b) · rehearsal of {rec['date']} · {rec.get('model') or 'tiny random model'} · budget forcing")
    levels = [(k, rec["levels"][k]) for k in ("easy", "medium") if k in rec["levels"]]
    names = [c["name"] for c in levels[0][1]["columns"]]
    shown_names = ["+" + n[1:] + " Wait" if n.startswith("+") else n for n in names]
    w = max(7, max(len(x) for x in shown_names) + 2)

    def row(label: str, cells: list[str], bold: bool = False) -> None:
        print(f"  {label:<28}" + "".join((BOLD if bold else "") + f"{c:>{w}}" + RESET for c in cells))

    def part(kinds: set) -> None:
        print()
        row("limit on thinking tokens", [n if c["kind"] in kinds else "" for n, c in zip(shown_names, levels[0][1]["columns"])])
        for label, lv in levels:
            row(f"{label} (m = {lv['m']}): accuracy", [f"{c['acc']:.2f}" if c["kind"] in kinds else "" for c in lv["columns"]], bold=True)
        for label, lv in levels:
            row(f"{label}: thinking tokens used", [f"{c['think_len_mean']:.0f}" if c["kind"] in kinds else "" for c in lv["columns"]])

    wait("  [Enter] forcing it to stop: limits from 0 to none ")
    part({"direct", "cap", "natural"})
    for label, lv in levels:
        bound = [c for c in lv["columns"] if c["kind"] == "cap"]
        note(f"  {label}: the limit cut the reasoning short in " + ", ".join(f"{c['changed_share']:.0%} ({c['name']})" for c in bound) + " of problems")
    wait("  [Enter] not letting it stop: Wait × 1, 2, 4 ")
    part({"direct", "cap", "natural", "wait"})
    for label, lv in levels:
        e = lv["extended"]
        note(f"  {label}: {e['n']} of {lv['n']} could be extended; {e['hit_cap_share']:.0%} of those ran into the {rec['task']['think_max_tokens']}-token safety cap")
    wait("  [Enter] flips: right in the column to the left, wrong here ")
    print()
    row("limit on thinking tokens", shown_names)
    for label, lv in levels:
        row(f"{label}: right → wrong", ["—" if c["flips_vs_left"] is None else str(c["flips_vs_left"]["right_to_wrong"]) for c in lv["columns"]], bold=True)
        row(f"{label}: wrong → right", ["—" if c["flips_vs_left"] is None else str(c["flips_vs_left"]["wrong_to_right"]) for c in lv["columns"]])
    for label, lv in levels:
        r = lv["redraw"]
        vs = " · ".join(f"{c['name']}: {c['flips_vs_none']['right_to_wrong']} / {c['flips_vs_none']['wrong_to_right']}" for c in lv["columns"] if c["flips_vs_none"])
        note(f"  {label}: against \"none\" (right → wrong / wrong → right)  {vs}")
        note(f"  {label}: control — same reasoning, only the answer drawn again: {r['right_to_wrong']} right → wrong, {r['wrong_to_right']} wrong → right (of {lv['n']})")
    n = levels[0][1]["n"]
    note(f"  each accuracy is {n} problems: standard error up to ±{max(se(c['acc'], n) for _, lv in levels for c in lv['columns']):.2f}. "
         f"Every column continues or truncates the same reasoning; only the answer is drawn again.")
    if any(lv["examples"] for _, lv in levels):
        wait("  [Enter] one problem that flipped ")
        for label, lv in levels:
            for ex in lv["examples"][:1]:
                print(f"\n  {BOLD}{ex['expr']}{RESET}   {GREY}correct: {ex['answer']} · {label}{RESET}")
                print(f"  after {ex['natural_thinking_tokens']} thinking tokens it answered {GREEN}{ex['natural_answer']}{RESET}")
                print(f"  {GREY}│ … {' '.join(ex['natural_thinking_tail'].split())[-160:]}{RESET}")
                print(f"  not allowed to stop, {ex['added_thinking_tokens']} more tokens:")
                print(f"  {GREY}│ {' '.join(ex['added_thinking_head'].split())[:300]} …{RESET}")
                print(f"  then it answered {RED}{ex['final_answer']}{RESET}")


# ── test：不載模型 ───────────────────────────────────────────────
def cmd_test(args, cfg):
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler
    banner("demo 1 · test — the state machine, the prefixes, the fit (no model)")
    fails = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {(GREEN + 'ok  ') if cond else (RED + 'FAIL')}{RESET} {name}" + (f"  {GREY}{detail}{RESET}" if detail else ""))
        if not cond:
            fails.append(name)

    V, OPEN, CLOSE, DOT, EOS, NL, WAIT = 12, 8, 9, 7, 11, 6, 10
    ids = F.ThinkIds(open=OPEN, close=CLOSE, dot=DOT, eos={EOS}, open_seq=[OPEN, NL], close_seq=[NL, CLOSE, NL], wait_seq=[WAIT])
    sampler = make_sampler(temp=1.0)

    def simulate(forcer, seed: int, p_close: float = 0.15, p_eos: float = 0.25, max_steps: int = 400) -> list[int]:
        """假的模型：推理時以 p_close 的機率想寫 </think>；回答時以 p_eos 的機率結束。照 mlx-lm 的方式呼叫 processor。"""
        mx.random.seed(seed)
        toks, ctx, closed = [], [0, OPEN], False
        for _ in range(max_steps):
            base = [1.0] * V
            base[OPEN] = base[EOS] = base[CLOSE] = 0.0
            if closed:
                base[EOS] = p_eos * 6 / (1 - p_eos)
            else:
                base[CLOSE] = p_close * 9 / (1 - p_close)
            logits = mx.log(mx.array(base) + 1e-30)[None]
            logits = forcer(mx.array(ctx), logits)
            t = int(sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item())
            toks.append(t)
            ctx.append(t)
            closed = closed or t == CLOSE
            if t == EOS:
                break
        return toks

    # 上限
    ok_cap, ok_ans, ok_seq = True, True, True
    for seed in range(40):
        f = F.Forcer(ids, sampler, think_cap=5, answer_max=4)
        sp = F.split_think(simulate(f, seed, p_close=0.05), ids)
        capped = any(e[0] == "cap" for e in f.events)
        ok_cap &= sp["closed"] and len(sp["think"]) <= 5 + 1 and (not capped or len(sp["think"]) == 6)
        ok_seq &= (not capped) or sp["think"][-1] == NL
        ok_ans &= len(sp["answer"]) <= 4 + 1
    check("cap: thinking never exceeds the limit; the closing sequence is sent whole", ok_cap and ok_seq)
    check("answer limit: at most answer_max tokens after the closing sequence", ok_ans)
    # 回答之前固定接的那幾個 token：自己結束與被逼停都一樣
    idp = F.ThinkIds(open=OPEN, close=CLOSE, dot=DOT, eos={EOS}, open_seq=[OPEN, NL], close_seq=[NL, CLOSE, NL], wait_seq=[WAIT], answer_prefix=[5, 5])
    ok_pre, n_nat, n_cap = True, 0, 0
    for seed in range(60):
        f = F.Forcer(idp, sampler, think_cap=6, answer_max=3)
        sp = F.split_think(simulate(f, 300 + seed, p_close=0.15), idp)
        capped = any(e[0] == "cap" for e in f.events)
        n_nat, n_cap = n_nat + (not capped), n_cap + capped
        ok_pre &= sp["answer"][:3] == [NL, 5, 5] and len(sp["answer"]) <= 3 + 3
    check("answer prefix: the same tokens follow </think> whether the model stopped by itself or was stopped", ok_pre and n_nat > 5 and n_cap > 5,
          f"{n_nat} natural, {n_cap} forced")
    # 不准停
    ok_wait, ok_none, n_wait = True, True, 0
    for seed in range(40):
        f = F.Forcer(ids, sampler, think_cap=10_000, answer_max=4, waits=3)
        toks = simulate(f, 100 + seed, p_close=0.3)
        sp = F.split_think(toks, ids)
        waits = [e for e in f.events if e[0] == "wait"]
        n_wait += len(waits)
        ok_wait &= sp["closed"] and sp["think"].count(WAIT) >= 3 and len(waits) == 3
        ok_wait &= all(sp["think"][e[1]] == WAIT for e in waits)
        ok_none &= toks.count(CLOSE) == 1
    check("no stopping: exactly `waits` stop attempts are replaced by Wait, at the recorded positions", ok_wait, f"{n_wait} in 40 runs")
    check("no stopping: </think> appears once, after the last Wait", ok_none)
    # 同分布：擋 0 次時不該改變分布——推理長度的平均對 1/p − 1
    lens = []
    for seed in range(300):
        f = F.Forcer(ids, sampler, think_cap=10_000, answer_max=2, waits=0)
        lens.append(len(F.split_think(simulate(f, 1000 + seed, p_close=0.2), ids)["think"]))
    lens_w = []
    for seed in range(300):
        f = F.Forcer(ids, sampler, think_cap=10_000, answer_max=2, waits=1)
        toks = simulate(f, 5000 + seed, p_close=0.2)
        lens_w.append([e[1] for e in f.events if e[0] == "wait"][0])
    m0, m1 = mean(lens), mean(lens_w)
    check("no stopping: the first stop attempt comes where the model would have stopped (mean length 1/p − 1 = 4)", abs(m0 - 4) < 0.8 and abs(m1 - 4) < 0.8,
          f"plain {m0:.2f}, first attempt under forcing {m1:.2f}")
    # 切與接
    nat = [OPEN, NL, 1, 2, 3, NL, CLOSE, NL, 4, 2, EOS]
    sp = F.split_think(nat, ids)
    check("split: thinking and answer", sp["think"] == [NL, 1, 2, 3, NL] and sp["answer"] == [NL, 4, 2] and sp["closed"])
    check("split: no </think> means no answer", F.split_think([OPEN, 1, 2], ids)["answer"] == [] and not F.split_think([OPEN, 1, 2], ids)["closed"])
    check("prefix: dots", F.prefix_dots([0], ids, 3) == [0, OPEN, NL, DOT, DOT, DOT, NL, CLOSE, NL])
    check("prefix: capped keeps the first tokens of the same reasoning", F.prefix_capped([0], nat, ids, 2) == [0, OPEN, NL, 1, NL, CLOSE, NL])
    check("prefix: a cap longer than the reasoning keeps all of it", F.prefix_capped([0], nat, ids, 99) == [0, OPEN, NL, 1, 2, 3, NL, NL, CLOSE, NL])
    check("prefix: Wait replaces </think>", F.prefix_wait([0], nat, ids) == [0, OPEN, NL, 1, 2, 3, NL, WAIT] and F.prefix_wait([0], [OPEN, 1], ids) is None)
    # 擬合與翻轉
    rng = random.Random(0)
    counts = {m: (sum(rng.random() < 0.97 ** m for _ in range(4000)), 4000) for m in (2, 4, 6, 8, 12)}
    fq = F.fit_q(counts)
    check("fit: recovers q = 0.97 from simulated counts", abs(fq["q"] - 0.97) < 0.003, f"q = {fq['q']:.4f}, half-life {fq['half_life_steps']:.1f} steps")
    check("flips", F.flips([1, 1, 0, 0], [1, 0, 1, 0]) == {"right_to_wrong": 1, "wrong_to_right": 1})
    # 題目
    s = settings(cfg, args)
    for m in s["ms"]:
        ps = problems_for(s, m, 5)
        ok = all(len(G.numbers_in(p["expr"])) == m + 1 for p in ps) and ps == problems_for(s, m, 5)
        check(f"problems: m = {m} has {m} operators; same seed, same problems", ok, ps[0]["expr"] + f" = {ps[0]['answer']}")
    tt = TinyTok()
    check("tiny tokenizer round trip", tt.decode(tt.encode("<think>\n12+3.\n</think>\n\nWait")) == "<think>\n12+3.\n</think>\n\nWait")
    print(f"\n  {GREEN}all passed{RESET}" if not fails else f"\n  {RED}{len(fails)} failed{RESET}")
    if fails:
        raise SystemExit(1)


# ── inspect ─────────────────────────────────────────────────────
def cmd_inspect(args, cfg):
    banner("demo 1 · inspect — templates, token ids, one real piece of reasoning, the two moves")
    run = load_runner(cfg, args)
    ids, tok, s = run.ids, run.tok, run.s
    v = C.versions()
    print(f"  mlx {v.get('mlx')} · mlx-lm {v.get('mlx-lm')} @ {str(v.get('mlx_lm_commit'))[:7]} · python {v['python']}")
    p = problems_for(s, s["ms"][0], 1)[0]
    rec = {"date": now(), "versions": v, "device": C.device_info(), "tiny": bool(args.tiny), "model": None if args.tiny else cfg["model"]["repo"],
           "path": None if run.path is None else Path(run.path).name}
    for thinking in (True, False):
        pid = run.prompt_ids(p, thinking)
        tail = tok.decode(pid)[-70:]
        rec[f"template_tail_thinking_{thinking}"] = tail
        print(f"  enable_thinking={thinking!s:<5} prompt: {len(pid)} tokens, ends with {tail!r}")
    rec["ids"] = {"open": ids.open, "close": ids.close, "dot": ids.dot, "eos": sorted(ids.eos), "open_seq": ids.open_seq, "close_seq": ids.close_seq,
                  "wait_seq": ids.wait_seq, "answer_prefix": ids.answer_prefix}
    print(f"  <think> = {ids.open} · </think> = {ids.close} · \".\" = {ids.dot} · stop tokens {sorted(ids.eos)}")
    print(f"  open_seq {ids.open_seq} · close_seq {ids.close_seq} · wait_seq {ids.wait_seq} ({[tok.decode([t]) for t in ids.wait_seq]})")
    print(f"  answer_prefix {ids.answer_prefix} ({tok.decode(ids.answer_prefix)!r}) — forced after </think> in all three ways of answering")
    print(f"\n  {BOLD}{p['expr']}{RESET}  {GREY}(correct: {p['answer']}){RESET}")
    nat = run.think([p], "natural", batch_size=1)[0]
    pieces = [tok.decode([t]) for t in nat["tokens"][:8]]
    sp = F.split_think(nat["tokens"], ids)
    after = [tok.decode([t]) for t in nat["tokens"][sp["close_at"]:sp["close_at"] + 4]] if sp["closed"] else []
    print(f"  natural: first tokens {pieces}; {nat['n_think']} thinking tokens; after the reasoning {after}; answer {nat['text']!r} {mark(nat['right'])}")
    rec["natural"] = {"first_pieces": pieces, "n_think": nat["n_think"], "opened": nat["opened"], "closed": nat["closed"], "around_close": after,
                      "answer": nat["text"], "right": nat["right"]}
    if not nat["opened"]:
        print(f"  {AMBER}the model did not write <think> itself: check the template tail above{RESET}")
    d = run.direct([p])[0]
    print(f"  no thinking: {d['text']!r} {mark(d['right'])}" + (f" {AMBER}[cut]{RESET}" if d["cut"] else ""))
    o = run.dots([p], [nat])[0]
    print(f"  dots × {nat['n_think']}: {o['text']!r} {mark(o['right'])}")
    # 動作一：上限 8
    f = run.forcer(think_cap=8)
    r = run.gen([{"prompt": run.prompt_ids(p, True), "max_tokens": f.max_tokens(), "forcer": f}], "cap 8", batch_size=1)[0]
    sp8 = F.split_think(r["tokens"], ids)
    tail8 = sp8["think"][-(len(ids.close_seq) - len(ids.close_seq[ids.close_seq.index(ids.close):])):] if ids.close_seq.index(ids.close) else []
    ok1 = sp8["closed"] and len(sp8["think"]) <= 8 + ids.close_seq.index(ids.close) and (not f.events or tail8 == ids.close_seq[:ids.close_seq.index(ids.close)])
    print(f"  forced stop at 8: {len(sp8['think'])} thinking tokens, events {f.events}, answer {tok.decode(sp8['answer']).strip()!r}  {(GREEN + 'ok') if ok1 else (RED + 'CHECK')}{RESET}")
    ev8 = [list(e) for e in f.events]
    # 動作二：不准停一次
    f = run.forcer(waits=1)
    r = run.gen([{"prompt": run.prompt_ids(p, True), "max_tokens": f.max_tokens(), "forcer": f}], "Wait × 1", batch_size=1)[0]
    spw = F.split_think(r["tokens"], ids)
    waits = [e for e in f.events if e[0] == "wait"]
    ok2 = (len(waits) == 1 and spw["think"][waits[0][1]:waits[0][1] + len(ids.wait_seq)] == ids.wait_seq) or (not waits and any(e[0] == "cap" for e in f.events))
    around = tok.decode(spw["think"][max(0, waits[0][1] - 12):waits[0][1] + 24]) if waits else ""
    print(f"  not allowed to stop once: {len(spw['think'])} thinking tokens, events {f.events}  {(GREEN + 'ok') if ok2 else (RED + 'CHECK')}{RESET}")
    if around:
        print(f"  {GREY}… {' '.join(around.split())} …{RESET}")
    rec["moves"] = {"cap8": {"n_think": len(sp8["think"]), "events": ev8, "ok": bool(ok1)},
                    "wait1": {"n_think": len(spw["think"]), "events": [list(e) for e in f.events], "ok": bool(ok2), "around": around}}
    rec["peak_gb"] = r4(C.peak_memory_gb())
    write_json(out_dir(args) / "inspect.json", rec)
    note(f"\n  → {(out_dir(args) / 'inspect.json').relative_to(HERE)}")


# ── probe ───────────────────────────────────────────────────────
def cmd_probe(args, cfg):
    pc = cfg["probe"]
    parts = args.part or ["speed", "levels"]
    out = out_dir(args) / "probe.json"
    rec = read_json(out) if out.exists() else {}
    banner("demo 1 · probe — tokens per second, and how the three ways of answering look on a few problems")
    run = load_runner(cfg, args)
    rec.update(date=now(), versions=C.versions(), device=C.device_info(), tiny=bool(args.tiny), model=None if args.tiny else cfg["model"]["repo"])
    s = run.s
    if "speed" in parts:
        m = s["ms"][0] if args.tiny else int(pc["speed_m"])
        problems = problems_for(s, m, int(pc["speed_n"]))
        print(f"\n  thinking on m = {m} problems; tokens per second by batch size")
        rec["speed"] = []
        for bs in ([int(args.batch_size)] if args.batch_size else pc["batch_sizes"]):
            ps = problems[:max(2, min(len(problems), bs))]
            run.think(ps, f"batch {bs}", echo=False, batch_size=bs)
            i = run.last_info
            rate = i["generation_tokens"] / i["seconds"]
            rec["speed"].append({"batch_size": bs, "sequences": len(ps), "generation_tokens": i["generation_tokens"], "seconds": round(i["seconds"], 1),
                                 "tokens_per_second": r4(rate), "peak_gb": r4(C.peak_memory_gb())})
            print(f"  batch {bs:<3} {len(ps):>3} sequences {i['generation_tokens']:>8,} tokens {i['seconds']:>7.1f} s   {BOLD}{rate:>6.0f} tok/s{RESET}")
            write_json(out, rec)
        # 同樣的生成、不掛 processor：狀態機本身要多少時間
        bs = s["batch_size"]
        ps = problems[:max(2, min(len(problems), bs))]
        items = [{"prompt": run.prompt_ids(p, True), "max_tokens": s["think_max_tokens"] + s["answer_max_tokens"]} for p in ps]
        run.gen(items, "no processor", echo=False, batch_size=bs)
        i = run.last_info
        rec["speed_no_processor"] = {"batch_size": bs, "tokens_per_second": r4(i["generation_tokens"] / i["seconds"])}
        print(f"  batch {bs:<3} without the logits processor: {i['generation_tokens'] / i['seconds']:.0f} tok/s")
        write_json(out, rec)
        note(f"  把 demo_config.toml [sampler] batch_size 設成 tok/s 不再明顯上升的那一個（現在是 {s['batch_size']}）。")
    if "levels" in parts:
        n = s["n"] if args.tiny else int(args.n or pc["n"])
        print(f"\n  {n} problems per m, one sampled answer each (standard error up to ±{0.5 / math.sqrt(n):.2f})")
        print(f"  {'m':<4}{'no thinking':>12}{'dots':>8}{'thinking':>10}{'think tokens: mean':>20}{'max':>7}{'hit cap':>9}{'>1 number':>11}   example")
        rec["levels"] = {}
        total = 0.0
        for m in s["ms"]:
            problems = problems_for(s, m, n)
            tok0 = run.tokens
            d = run.direct(problems, echo=False)
            t = run.think(problems, echo=False)
            o = run.dots(problems, t, echo=False)
            lens = [x["n_think"] for x in t]
            st = {"n": n, "direct": r4(mean([x["right"] for x in d])), "dots": r4(mean([x["right"] for x in o])), "think": r4(mean([x["right"] for x in t])),
                  "think_len_mean": r4(mean(lens)), "think_len_max": max(lens), "capped_share": r4(mean([x["capped"] for x in t])),
                  "direct_multi_number_share": r4(mean([1.0 if x["numbers"] > 1 else 0.0 for x in d])),
                  "direct_cut_share": r4(mean([x["cut"] for x in d])), "think_answer_cut_share": r4(mean([x["cut"] for x in t])), "tokens_per_problem": r4((run.tokens - tok0) / n),
                  "example": {"expr": problems[0]["expr"], "answer": problems[0]["answer"], "direct": d[0]["text"][:40], "think": t[0]["text"][:40]}}
            rec["levels"][str(m)] = st
            total += st["tokens_per_problem"] * s["n"]
            print(f"  {m:<4}{st['direct']:>12.2f}{st['dots']:>8.2f}{st['think']:>10.2f}{st['think_len_mean']:>20.0f}{st['think_len_max']:>7}{st['capped_share']:>9.0%}"
                  f"{st['direct_multi_number_share']:>11.0%}   {GREY}{problems[0]['expr']} = {problems[0]['answer']} → {d[0]['text'][:14]!r} / {t[0]['text'][:14]!r}{RESET}")
            write_json(out, rec)
        rate = run.tokens / run.seconds if run.seconds else None
        if rate:
            est = total / rate / 60
            rec["estimate"] = {"chain_tokens": int(total), "tokens_per_second": r4(rate), "chain_minutes": r4(est), "n": s["n"]}
            print(f"\n  (a) at n = {s['n']} per m: about {total:,.0f} tokens → {BOLD}{est:.0f} min{RESET} at the {rate:.0f} tok/s measured here; "
                  f"(b) adds the extended reasoning of two values of m — not estimated")
        note("  看三件事：不想那一欄有沒有從高掉到低（都高或都低就調 [task] 的 lo／hi／ops 或 [chain] ms）；"
             "想的那一欄有沒有撞到 think_max_tokens；[budget] 的 easy_m／medium_m 各挑「想」接近全對與明顯不到全對的 m。")
        write_json(out, rec)
    note(f"\n  → {out.relative_to(HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    ap.add_argument("--online", action="store_true", help="允許連 Hugging Face（預設離線）")
    ap.add_argument("--force", action="store_true", help="rehearse 重跑已有的部分")
    ap.add_argument("--part", action="append", choices=["chain", "budget", "live", "speed", "levels"], help="rehearse／probe 只做這一段（可重複）")
    ap.add_argument("--n", type=int, help="暫時換每個 m 的題數")
    ap.add_argument("--m", type=int, help="live 暫時換步數")
    ap.add_argument("--batch-size", type=int, help="暫時換 batch size")
    ap.add_argument("--no-pause", action="store_true", help="replay／live 不等 Enter")
    ap.add_argument("cmd", choices=["test", "inspect", "probe", "rehearse", "live", "replay", "replay-budget", "replay-live"])
    args = ap.parse_args()
    cfg = load_cfg()
    {"test": cmd_test, "inspect": cmd_inspect, "probe": cmd_probe, "rehearse": cmd_rehearse, "live": cmd_live, "replay": cmd_replay,
     "replay-budget": cmd_replay_budget, "replay-live": cmd_replay_live}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
