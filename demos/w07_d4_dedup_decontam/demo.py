# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 4：去重與 decontamination（投影片 p46 前導頁、p47「Live: demo 4 — near-duplicates across dumps; MMLU vs. the corpus」）。

  ./present.sh fetch                 一次、要網路：MMLU test split 與 MMLU-Redux 2.0 Virology → data/（幾 MB）
  ./present.sh test | live           三個測試對 dedup_final.py | dedup_live.py（tests.py）
  ./present.sh run [--impl live|final] [--corpus edu|web] [--dumps all|D1,D2] [--bands B --rows R] [--also 32x4,8x16]
                                     (a)：簽章 → LSH 候選對 → Ĵ 分布、精確 J、同 dump／跨 dump、同 URL → 兩對並排；Enter 換 (b, r)
  ./present.sh decontam [--corpus edu|web|both] [--n 13]
                                     (b)：MMLU 14,042 題對語料掃 13-gram 重疊 → 命中的題 → MMLU-Redux 標錯的兩題
  ./present.sh rehearse              課前：tests --save → run（參考實作，edu 與 web，四組 (b, r)；不讀快取，量簽章秒數）→ decontam both → runs/rehearsal/
  ./present.sh replay | show         現場失敗時：讀 runs/rehearsal/ 的 json（replay 一步一個 Enter；show 不停）

(a) 的三個函數由 --impl 指定的模組提供（預設 final；課堂寫 live）。簽章用 multiprocessing 算、存 runs/cache/（.gitignore），
換 (b, r) 不重算。純標準函式庫；fetch 另需 pyarrow 與 huggingface_hub（共用 .venv 已有，demo 2 的 fetch 也用）。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
import pickle
import platform
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                     # Python < 3.11（建置時的 VM）；Mac 的 .venv 是 3.12
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, note, now  # noqa: E402

RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
CACHE = RUNS / "cache"
DATA = HERE / "data"
_WORD = re.compile(r"\w+")
MAX_LIST = 16                 # 候選對清單最多印幾對（全部都在 json 裡）


def load_cfg() -> dict:
    with open(HERE / "demo_config.toml", "rb") as f:
        return tomllib.load(f)


def pause(enabled: bool, msg: str = "Enter 繼續") -> None:
    if enabled:
        try:
            input(f"{GREY}  ⏎ {msg}{RESET}")
        except EOFError:
            print()


# ── 語料 ────────────────────────────────────────────────────────────────
def load_corpus(cfg: dict, which: str, dumps: str) -> tuple[list[dict], dict]:
    path = (HERE / cfg["corpus"][which]).resolve()
    if not path.exists():
        raise SystemExit(f"{RED}找不到 {path}：先 cd ../w07_d2_scaling_fit && ./present.sh fetch --corpus both{RESET}")
    want = None if dumps == "all" else set(dumps.split(","))
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if want is None or d["dump"] in want:
                docs.append({"id": d["id"], "dump": d["dump"], "url": d["url"], "text": d["text"]})
    if not docs:
        raise SystemExit(f"{RED}{path.name} 裡沒有 dump {dumps} 的文件{RESET}")
    counts = collections.Counter(d["dump"] for d in docs)
    info = {"corpus": which, "file": path.name, "dumps_requested": dumps, "docs": len(docs), "dumps": dict(sorted(counts.items())),
            "text_bytes": sum(len(d["text"].encode("utf-8")) for d in docs)}
    return docs, info


# ── (a) 簽章：multiprocessing，impl 由名稱載入（spawn 下子行程自己 import）──
_IMPL = None
_ARGS = None


def _init_worker(impl_name: str, n: int, k_h: int, seed: int) -> None:
    global _IMPL, _ARGS
    sys.path.insert(0, str(HERE))
    _IMPL = importlib.import_module(impl_name)
    _ARGS = (n, k_h, seed)


def _sig_worker(text: str):
    n, k_h, seed = _ARGS
    sh = _IMPL.shingles(text, n)
    if not sh:
        return None
    try:
        return _IMPL.minhash(sh, k_h, seed)
    except TypeError:                           # live 版若沒有 seed 參數
        return _IMPL.minhash(sh, k_h)


def signatures(docs: list[dict], impl_name: str, cfg_m: dict, info: dict, quiet: bool = False, fresh: bool = False) -> tuple[list, float]:
    """所有文件的簽章。快取在 runs/cache/，鍵是（impl 檔案的 md5、語料、dump、n、k_h、seed）；fresh=True 不讀快取（rehearse 要量秒數）。"""
    impl_src = (HERE / f"{impl_name}.py").read_bytes()
    key = hashlib.md5(impl_src + json.dumps([info["file"], info["dumps_requested"], cfg_m["n"], cfg_m["k_h"], cfg_m["seed"]]).encode()).hexdigest()[:12]
    cp = CACHE / f"sigs-{impl_name}-{info['corpus']}-{key}.pkl"
    if cp.exists() and not fresh:
        sigs = pickle.loads(cp.read_bytes())
        if len(sigs) == len(docs):
            if not quiet:
                note(f"  簽章從快取讀（{cp.name}）；要重算就刪掉 runs/cache/")
            return sigs, None
    procs = max(1, (os.cpu_count() or 2) - 1)
    if not quiet:
        print(f"  {len(docs):,} documents × {cfg_m['k_h']} signatures with {impl_name}.py on {procs} processes …", end="", flush=True)
    t0 = time.time()
    with Pool(procs, initializer=_init_worker, initargs=(impl_name, cfg_m["n"], cfg_m["k_h"], cfg_m["seed"])) as pool:
        sigs = pool.map(_sig_worker, [d["text"] for d in docs], chunksize=64)
    dt = time.time() - t0
    if not quiet:
        print(f" {dt:.1f} s")
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps(sigs))
    return sigs, dt


def parse_br(s: str) -> tuple[int, int]:
    b, r = s.lower().split("x")
    return int(b), int(r)


def _estimate(x, y) -> float:
    return sum(p == q for p, q in zip(x, y)) / len(x)


def _jaccard(A, B) -> float:
    u = len(A | B)
    return len(A & B) / u if u else 0.0


def _kind(a: dict, b: dict, J: float) -> str:
    """給人看的一句分類（只看 URL 與 dump）。"""
    ua, ub = a["url"], b["url"]
    strip = lambda u: re.sub(r"^https?://(www\.)?", "", u).rstrip("/").lower()      # noqa: E731
    if ua == ub:
        return "same URL, re-crawled in another dump"
    if strip(ua) == strip(ub):
        return "same page, URL differs only in scheme / www / case"
    ha, hb = strip(ua).split("/")[0], strip(ub).split("/")[0]
    if ha == hb:
        return "same site, different path" + (" (template?)" if J < 0.8 else " (moved page?)")
    return "different sites" + (" (copy / mirror)" if J >= 0.7 else " (shared boilerplate?)")


def analyse_pairs(docs, sigs, pairs, cfg_m, k_used, shingle_cache) -> dict:
    rows = []
    for i, j in sorted(pairs):
        a, b = docs[i], docs[j]
        for k in (i, j):
            if k not in shingle_cache:
                shingle_cache[k] = _IMPL_MAIN.shingles(docs[k]["text"], cfg_m["n"])
        J = _jaccard(shingle_cache[i], shingle_cache[j])
        est = _estimate(sigs[i][:k_used], sigs[j][:k_used])
        rows.append({"i": i, "j": j, "estimate": est, "J": J, "same_dump": a["dump"] == b["dump"], "dumps": [a["dump"], b["dump"]],
                     "same_url": a["url"] == b["url"], "urls": [a["url"], b["url"]], "chars": [len(a["text"]), len(b["text"])], "kind": _kind(a, b, J)})
    edges = cfg_m["hist_edges"]
    hist = collections.OrderedDict()
    for lo, hi in zip(edges[:-1], edges[1:]):
        hist[f"[{lo:.1f}, {hi:.1f})"] = sum(lo <= r["estimate"] < hi for r in rows)
    hist["= 1.0"] = sum(r["estimate"] == 1.0 for r in rows)
    hist["< %.1f" % edges[0]] = sum(r["estimate"] < edges[0] for r in rows)
    return {"candidates": len(rows), "cross_dump": sum(not r["same_dump"] for r in rows), "same_dump": sum(r["same_dump"] for r in rows),
            "same_url": sum(r["same_url"] for r in rows), "est_ge_0_9": sum(r["estimate"] >= 0.9 for r in rows), "J_ge_0_9": sum(r["J"] >= 0.9 for r in rows),
            "J_lt_0_75": sum(r["J"] < 0.75 for r in rows), "exact_1": sum(r["J"] == 1.0 for r in rows), "hist": hist, "pairs": rows}


_IMPL_MAIN = None


def print_pairs(res: dict, b: int, r: int, k_used: int, docs, cfg_m, show_text: bool) -> None:
    thr = (1 / b) ** (1 / r)
    print(f"\n  {BOLD}(b, r) = ({b}, {r}){RESET}  {GREY}k_h used = {k_used}; threshold (1/b)^(1/r) ≈ {thr:.2f}{RESET}")
    print(f"  candidate pairs: {BOLD}{res['candidates']}{RESET}   cross-dump {res['cross_dump']} · same-dump {res['same_dump']} · same URL {res['same_url']}")
    print(f"  Ĵ histogram: " + "  ".join(f"{k}: {v}" for k, v in res["hist"].items() if v or k.startswith("[")))
    print(f"  exact J ≥ 0.9: {res['J_ge_0_9']} (of which J = 1: {res['exact_1']})   exact J < 0.75 (below FineWeb's threshold): {res['J_lt_0_75']}")
    if res["pairs"]:
        kinds = collections.Counter(p["kind"] for p in res["pairs"])
        print("  what they are: " + "; ".join(f"{v} × {k}" for k, v in kinds.most_common()))
        shown = sorted(res["pairs"], key=lambda p: -p["estimate"])[:MAX_LIST]
        print(f"  {GREY}{'Ĵ':>6} {'J':>6}  dumps                              kind{RESET}")
        for p in shown:
            d = f"{p['dumps'][0][8:]} · {p['dumps'][1][8:]}" + ("  (same)" if p["same_dump"] else "")
            print(f"  {p['estimate']:6.3f} {p['J']:6.3f}  {d:<34} {p['kind']}")
            print(f"{GREY}                {p['urls'][0][:88]}\n                {p['urls'][1][:88]}{RESET}")
        if len(res["pairs"]) > MAX_LIST:
            print(f"  {GREY}… {len(res['pairs']) - MAX_LIST} more (all of them are in the json){RESET}")
    if show_text and res["pairs"]:
        picks = _pick_pairs(res["pairs"], cfg_m["show_pairs"])
        for p in picks:
            print(f"\n  {BOLD}── side by side: Ĵ = {p['estimate']:.3f}, exact J = {p['J']:.3f} — {p['kind']}{RESET}")
            for k, idx in enumerate((p["i"], p["j"])):
                d = docs[idx]
                print(f"  {AMBER}[{'AB'[k]}] {d['dump']}  {d['url'][:100]}  ({len(d['text']):,} chars){RESET}")
                txt = re.sub(r"\s+", " ", d["text"][:cfg_m["show_chars"]])
                print(f"      {txt}{'…' if len(d['text']) > cfg_m['show_chars'] else ''}")


def _pick_pairs(rows: list[dict], n: int) -> list[dict]:
    """並排用：先一對跨 dump、不同站、J 最高的（真的抄或鏡像；沒有就退成同站不同 URL），再一對同 URL 的重爬（J = 1）；不夠就補 J 最高的。"""
    picks = []
    diff = [p for p in rows if not p["same_dump"] and not p["same_url"] and p["J"] < 0.999]
    copies = [p for p in diff if p["kind"].startswith("different sites")]
    if copies or diff:
        picks.append(max(copies or diff, key=lambda p: p["J"]))
    same = [p for p in rows if p["same_url"]]
    if same and len(picks) < n:
        picks.append(max(same, key=lambda p: p["J"]))
    for p in sorted(rows, key=lambda p: -p["J"]):
        if len(picks) >= n:
            break
        if p not in picks:
            picks.append(p)
    return picks[:n]


def cmd_run(a, cfg, record: dict | None = None, interactive: bool = True) -> dict:
    global _IMPL_MAIN
    cfg_m = cfg["minhash"]
    impl_name = {"live": "dedup_live", "final": "dedup_final"}[a.impl]
    _IMPL_MAIN = importlib.import_module(impl_name)
    for fn in ("shingles", "minhash", "lsh_candidates"):
        if not callable(getattr(_IMPL_MAIN, fn, None)):
            raise SystemExit(f"{RED}{impl_name}.py 還沒有 {fn}：先寫完（./present.sh live 看測試），或 --impl final{RESET}")
    docs, info = load_corpus(cfg, a.corpus, a.dumps)
    banner(f"W7 demo 4 (a) · near-duplicates in {info['file']} · {impl_name}.py")
    print(f"  {info['docs']:,} documents, {info['text_bytes'] / 1e6:.0f} MB, {len(info['dumps'])} dumps: " + ", ".join(f"{k[8:]} {v:,}" for k, v in info["dumps"].items()))
    print(f"  word {cfg_m['n']}-gram shingles · k_h = {cfg_m['k_h']} · seed {cfg_m['seed']}")
    sigs, dt = signatures(docs, impl_name, cfg_m, info, fresh=getattr(a, "fresh", False))
    n_none = sum(s is None for s in sigs)
    if n_none:
        note(f"  {n_none} documents shorter than {cfg_m['n']} words have no signature and are skipped")
    settings = [(a.bands, a.rows)] + [parse_br(s) for s in (a.also.split(",") if a.also else [])]
    out = {"date": now(), "impl": impl_name, "python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}", "cpu_count": os.cpu_count(),
           "corpus": info, "minhash": {k: cfg_m[k] for k in ("n", "k_h", "seed")}, "signature_seconds": dt, "no_signature": n_none, "settings": []}
    shingle_cache = {}
    for idx, (b, r) in enumerate(settings):
        k_used = b * r
        if k_used > cfg_m["k_h"]:
            print(f"  {RED}(b, r) = ({b}, {r}) needs {k_used} > k_h = {cfg_m['k_h']} signatures — skipped{RESET}")
            continue
        if idx:
            pause(interactive, f"Enter → (b, r) = ({b}, {r})")
        t0 = time.time()
        pairs = _IMPL_MAIN.lsh_candidates([None if s is None else s[:k_used] for s in sigs], b, r)
        pairs = {tuple(sorted(p)) for p in pairs}
        lsh_dt = time.time() - t0
        res = analyse_pairs(docs, sigs, pairs, cfg_m, k_used, shingle_cache)
        res.update({"b": b, "r": r, "k_used": k_used, "threshold": (1 / b) ** (1 / r), "lsh_seconds": lsh_dt})
        print_pairs(res, b, r, k_used, docs, cfg_m, show_text=(idx == 0))
        out["settings"].append(res)
    if record is not None:
        record.update(out)
    return out


# ── (b) decontamination：13-gram 對 MMLU ─────────────────────────────────
def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _question_text(q: dict) -> str:
    return q["question"] + " " + " ".join(q["choices"])


def build_index(questions: list[dict], n: int) -> tuple[dict, int]:
    """13-gram（空白接起來的字串）→ 題號集合；回傳 (index, 太短的題數)。"""
    index, short = {}, 0
    for qi, q in enumerate(questions):
        toks = _tokens(_question_text(q))
        if len(toks) < n:
            short += 1
            continue
        for s in range(len(toks) - n + 1):
            index.setdefault(" ".join(toks[s:s + n]), set()).add(qi)
    return index, short


WHERE_STIMULUS = "quoted source text (a 'This question refers to…' stimulus)"
WHERE_QUESTION = "the question sentence itself"
WHERE_CHOICES = "a choice"
WHERE_SPAN = "spans question and choices"


def _where(q: dict, grams: list[str], n: int) -> str:
    """共用的 13-gram 落在題目的哪裡：引文 stimulus、題目句本身、選項、還是跨兩者。"""
    qt, ct = " ".join(_tokens(q["question"])), " ".join(_tokens(" ".join(q["choices"])))
    in_q = any(g in qt for g in grams)
    in_c = any(g in ct for g in grams)
    if in_q:
        return WHERE_STIMULUS if q["question"].lower().startswith("this question refers to the following information") else WHERE_QUESTION
    if in_c:
        return WHERE_CHOICES
    return WHERE_SPAN


_INDEX = None
_N = None


def _init_scan(mmlu_path: str, n: int) -> None:
    global _INDEX, _N
    qs = [json.loads(l) for l in open(mmlu_path, encoding="utf-8")]
    _INDEX, _ = build_index(qs, n)
    _N = n


def _scan_worker(item):
    di, text = item
    toks = _tokens(text)
    hits = []
    seen = set()
    for s in range(len(toks) - _N + 1):
        g = " ".join(toks[s:s + _N])
        if g in _INDEX:
            for qi in _INDEX[g]:
                if (qi, g) not in seen:
                    seen.add((qi, g))
                    hits.append((di, qi, g))
    return hits


def mmlu_paths(cfg) -> tuple[Path, Path]:
    return DATA / "mmlu_test.jsonl", DATA / "mmlu_redux_virology.jsonl"


def cmd_decontam(a, cfg, record: dict | None = None, interactive: bool = True) -> dict:
    cfg_d = cfg["decontam"]
    n = a.n or cfg_d["n"]
    mmlu_p, redux_p = mmlu_paths(cfg)
    if not mmlu_p.exists():
        raise SystemExit(f"{RED}沒有 {mmlu_p}：先跑 ./present.sh fetch（需要網路）{RESET}")
    questions = [json.loads(l) for l in open(mmlu_p, encoding="utf-8")]
    index, short = build_index(questions, n)
    banner(f"W7 demo 4 (b) · {n}-gram overlap: MMLU test vs. the corpus")
    print(f"  MMLU test: {len(questions):,} questions, {len({q['subject'] for q in questions})} subjects (question + choices, lower-cased, \\w+ tokens)")
    print(f"  {len(index):,} distinct {n}-grams; {short} questions have fewer than {n} tokens and cannot be caught by this rule")
    out = {"date": now(), "n": n, "questions": len(questions), "too_short": short, "distinct_ngrams": len(index), "corpora": []}
    corpora = ["edu", "web"] if a.corpus == "both" else [a.corpus]
    for which in corpora:
        docs, info = load_corpus(cfg, which, "all")
        procs = max(1, (os.cpu_count() or 2) - 1)
        print(f"\n  scanning {info['file']}: {info['docs']:,} documents, {info['text_bytes'] / 1e6:.0f} MB, on {procs} processes …", end="", flush=True)
        t0 = time.time()
        with Pool(procs, initializer=_init_scan, initargs=(str(mmlu_p), n)) as pool:
            parts = pool.map(_scan_worker, list(enumerate(d["text"] for d in docs)), chunksize=128)
        dt = time.time() - t0
        hits = [h for part in parts for h in part]
        q_hit = sorted({qi for _, qi, _ in hits})
        d_hit = sorted({di for di, _, _ in hits})
        print(f" {dt:.1f} s")
        print(f"  {BOLD}{len(q_hit)} of {len(questions):,} questions share a {n}-gram with the corpus{RESET}  ({len(hits)} (document, question, {n}-gram) hits in {len(d_hit)} documents)")
        rows = []
        by_q = collections.defaultdict(list)
        for di, qi, g in hits:
            by_q[qi].append((di, g))
        for qi in q_hit:
            q = questions[qi]
            ex = by_q[qi]
            grams = sorted({g for _, g in ex})
            urls = sorted({docs[di]["url"] for di, _ in ex})
            rows.append({"qi": qi, "subject": q["subject"], "question": q["question"], "choices": q["choices"], "answer": q["answer"], "where": _where(q, grams, n),
                         "ngrams": grams, "urls": urls, "n_docs": len({di for di, _ in ex})})
        where = collections.Counter(r["where"] for r in rows)
        print("  where the shared " + f"{n}-gram sits: " + "; ".join(f"{v} × {k}" for k, v in where.most_common()))
        # 同一段引文被好幾題共用（MMLU 的歷史題一段 stimulus 配多題）：印一次
        groups = collections.OrderedDict()
        for r in rows:
            key = (r["where"], tuple(r["ngrams"][:2])) if r["where"] == WHERE_STIMULUS else ("q", r["qi"])
            groups.setdefault(key, []).append(r)
        for key, rs in groups.items():
            r = rs[0]
            head = f"{AMBER}[{r['subject']}]{RESET} "
            if len(rs) > 1:
                head += f"{BOLD}{len(rs)} questions share this passage{RESET} — "
            print(f"\n  {head}{r['question'][:200].replace(chr(10), ' ')}{'…' if len(r['question']) > 200 else ''}")
            if len(rs) == 1:
                print(f"      choices: {' | '.join(c[:60] for c in r['choices'])}")
            print(f"      {GREY}{r['where']}{RESET}")
            for g in r["ngrams"][:2]:
                print(f"      {n}-gram: {GREEN}“{g}”{RESET}")
            for u in r["urls"][:2]:
                print(f"      {GREY}in {u[:100]}{RESET}" + (f"  (+{len(r['urls']) - 2} more)" if len(r["urls"]) > 2 and u == r["urls"][1] else ""))
        out["corpora"].append({"corpus": which, "file": info["file"], "docs": info["docs"], "text_bytes": info["text_bytes"], "seconds": dt, "hits": len(hits),
                               "questions_hit": len(q_hit), "documents_hit": len(d_hit), "where": dict(where.most_common()), "passages": len(groups), "rows": rows})
    # MMLU-Redux：第一次提 MMLU 的那一刀
    if redux_p.exists():
        pause(interactive, "Enter → MMLU-Redux (Virology)")
        redux = [json.loads(l) for l in open(redux_p, encoding="utf-8")]
        types = collections.Counter(r["error_type"] for r in redux)
        n_bad = sum(v for k, v in types.items() if k != "ok")
        print(f"\n  {BOLD}MMLU-Redux 2.0, Virology: {n_bad} of {len(redux)} questions flagged{RESET}  " + ", ".join(f"{k} {v}" for k, v in types.most_common()))
        wrong = [r for r in redux if r["error_type"] == "wrong_groundtruth"]
        for r in wrong[:cfg_d["redux_show"]]:
            print(f"\n  {AMBER}{r['question'][:220]}{RESET}")
            for k, c in enumerate(r["choices"]):
                mark = "MMLU key" if k == r["answer"] else ""
                print(f"      {'ABCD'[k]}. {c[:90]}  {GREY}{mark}{RESET}")
            print(f"      Redux: {RED}wrong_groundtruth{RESET} — correct answer {r['correct_answer']!s}; source: {r.get('source') or '—'}")
        out["redux"] = {"subset": "virology", "questions": len(redux), "flagged": n_bad, "error_types": dict(types.most_common()),
                        "shown": [{k: r[k] for k in ("question", "choices", "answer", "error_type", "correct_answer", "source")} for r in wrong[:cfg_d["redux_show"]]]}
    else:
        note("  （沒有 data/mmlu_redux_virology.jsonl：fetch 時沒抓到 Redux，略過）")
    if record is not None:
        record.update(out)
    return out


# ── fetch ────────────────────────────────────────────────────────────────
def cmd_fetch(a, cfg) -> None:
    cfg_d = cfg["decontam"]
    try:
        import pyarrow.parquet as pq
        import pyarrow.ipc as ipc
        from huggingface_hub import HfApi, HfFileSystem
    except ImportError as e:
        raise SystemExit(f"{RED}fetch 需要 pyarrow 與 huggingface_hub（{e}）：cd ~/Course-Hub/nlp_llm/demos && uv pip install --python .venv/bin/python pyarrow huggingface_hub{RESET}")
    import io
    DATA.mkdir(exist_ok=True)
    fs, api = HfFileSystem(), HfApi()
    mmlu_p, redux_p = mmlu_paths(cfg)
    banner("W7 demo 4 · fetch")
    t0 = time.time()
    sha = api.dataset_info(cfg_d["mmlu_repo"]).sha
    with fs.open(f"datasets/{cfg_d['mmlu_repo']}/{cfg_d['mmlu_path']}", "rb") as fh:
        rows = pq.read_table(fh, columns=["question", "subject", "choices", "answer"]).to_pylist()
    with open(mmlu_p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"date": now(), "repo": cfg_d["mmlu_repo"], "path": cfg_d["mmlu_path"], "revision": sha, "license": cfg_d["mmlu_license"], "rows": len(rows),
            "subjects": len({r["subject"] for r in rows}), "seconds": time.time() - t0}
    (DATA / "mmlu_test.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"  {GREEN}{cfg_d['mmlu_repo']}{RESET} {cfg_d['mmlu_path']} @ {sha[:12]} · {len(rows):,} questions, {meta['subjects']} subjects · {cfg_d['mmlu_license']} → {mmlu_p.relative_to(HERE)}  ({meta['seconds']:.0f} s)")
    t0 = time.time()
    sha = api.dataset_info(cfg_d["redux_repo"]).sha
    with fs.open(f"datasets/{cfg_d['redux_repo']}/{cfg_d['redux_path']}", "rb") as fh:
        data = fh.read()
    try:
        tb = ipc.open_stream(io.BytesIO(data)).read_all()
    except Exception:
        tb = ipc.open_file(io.BytesIO(data)).read_all()
    rows = tb.to_pylist()
    with open(redux_p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    types = collections.Counter(r["error_type"] for r in rows)
    meta = {"date": now(), "repo": cfg_d["redux_repo"], "path": cfg_d["redux_path"], "revision": sha, "license": cfg_d["redux_license"], "rows": len(rows),
            "error_types": dict(types.most_common()), "seconds": time.time() - t0}
    (DATA / "mmlu_redux_virology.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"  {GREEN}{cfg_d['redux_repo']}{RESET} {cfg_d['redux_path']} @ {sha[:12]} · {len(rows)} questions, {sum(v for k, v in types.items() if k != 'ok')} flagged · {cfg_d['redux_license']} → {redux_p.relative_to(HERE)}")


# ── rehearse / replay ────────────────────────────────────────────────────
def cmd_rehearse(a, cfg) -> None:
    import subprocess
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    banner("W7 demo 4 · rehearse")
    rc = subprocess.call([sys.executable, str(HERE / "tests.py"), "dedup_final", "--save"])
    if rc:
        raise SystemExit(f"{RED}tests 沒全過，先修 dedup_final.py{RESET}")
    summary = {"date": now(), "python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}", "cpu_count": os.cpu_count(), "dedup": {}, "decontam": {}}
    for which in ("edu", "web"):
        ns = argparse.Namespace(impl="final", corpus=which, dumps=cfg["corpus"]["dumps"], bands=cfg["minhash"]["bands"], rows=cfg["minhash"]["rows"], also=",".join(cfg["minhash"]["also"]), fresh=True)
        rec = {}
        cmd_run(ns, cfg, record=rec, interactive=False)
        (REHEARSAL / f"dedup_{which}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        summary["dedup"][which] = {"docs": rec["corpus"]["docs"], "dumps": len(rec["corpus"]["dumps"]), "text_bytes": rec["corpus"]["text_bytes"], "signature_seconds": rec["signature_seconds"],
                                   "settings": [{k: s[k] for k in ("b", "r", "threshold", "candidates", "cross_dump", "same_dump", "same_url", "J_ge_0_9", "exact_1", "J_lt_0_75", "lsh_seconds")} for s in rec["settings"]],
                                   "kinds": dict(collections.Counter(p["kind"] for p in rec["settings"][0]["pairs"]).most_common())}
    ns = argparse.Namespace(corpus="both", n=None)
    rec = {}
    cmd_decontam(ns, cfg, record=rec, interactive=False)
    (REHEARSAL / "decontam.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary["decontam"] = {"n": rec["n"], "questions": rec["questions"], "too_short": rec["too_short"],
                           "corpora": [{k: c[k] for k in ("corpus", "docs", "text_bytes", "seconds", "hits", "questions_hit", "documents_hit", "where", "passages")} | {"subjects": sorted({r["subject"] for r in c["rows"]})} for c in rec["corpora"]],
                           "redux": {k: v for k, v in rec.get("redux", {}).items() if k != "shown"}}
    (REHEARSAL / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    banner("summary → runs/rehearsal/summary.json", GREEN)
    print_summary(summary)


def _secs(x) -> str:
    return "from cache" if x is None else f"{x:.0f} s"


def print_summary(s: dict) -> None:
    print(f"  {GREY}{s['date']} · Python {s['python']} · {s['machine']} · {s['cpu_count']} CPUs{RESET}")
    for which, d in s["dedup"].items():
        print(f"  {BOLD}{which}{RESET}: {d['docs']:,} docs, {d['dumps']} dumps, {d['text_bytes'] / 1e6:.0f} MB; signatures {_secs(d['signature_seconds'])}")
        for st in d["settings"]:
            print(f"      (b, r) = ({st['b']:>2}, {st['r']:>2})  thr {st['threshold']:.2f}  candidates {st['candidates']:>4}  cross-dump {st['cross_dump']:>3}  same-dump {st['same_dump']:>3}  same URL {st['same_url']:>3}  J ≥ 0.9: {st['J_ge_0_9']:>3} (J = 1: {st['exact_1']})  J < 0.75: {st['J_lt_0_75']:>3}")
        print("      kinds at the first setting: " + "; ".join(f"{k} {v}" for k, v in d["kinds"].items()))
    dc = s["decontam"]
    print(f"  {BOLD}decontam{RESET}: {dc['n']}-gram, {dc['questions']:,} questions ({dc['too_short']} too short)")
    for c in dc["corpora"]:
        print(f"      {c['corpus']}: {c['questions_hit']} questions hit, {c['passages']} distinct passages ({c['hits']} hits in {c['documents_hit']} docs) in {c['seconds']:.0f} s")
        print("          where: " + "; ".join(f"{v} × {k}" for k, v in c["where"].items()) + f"; subjects {', '.join(c['subjects']) or '—'}")
    if dc.get("redux"):
        print(f"      MMLU-Redux {dc['redux']['subset']}: {dc['redux']['flagged']}/{dc['redux']['questions']} flagged")


def cmd_replay(a, cfg, interactive: bool) -> None:
    sp = REHEARSAL / "summary.json"
    if not sp.exists():
        raise SystemExit(f"{RED}沒有 {sp}：先跑 ./present.sh rehearse{RESET}")
    s = json.loads(sp.read_text(encoding="utf-8"))
    banner(f"W7 demo 4 · 彩排紀錄（非現場）· {s['date']}", AMBER)
    which = cfg["corpus"]["default"]
    rec = json.loads((REHEARSAL / f"dedup_{which}.json").read_text(encoding="utf-8"))
    docs, _ = load_corpus(cfg, which, rec["corpus"]["dumps_requested"]) if (HERE / cfg["corpus"][which]).resolve().exists() else ([], None)
    info = rec["corpus"]
    print(f"  (a) {info['file']}: {info['docs']:,} documents, {info['text_bytes'] / 1e6:.0f} MB, {len(info['dumps'])} dumps; signatures {_secs(rec['signature_seconds'])} on {rec['cpu_count']} CPUs ({rec['machine']})")
    for idx, st in enumerate(rec["settings"]):
        if idx:
            pause(interactive, f"Enter → (b, r) = ({st['b']}, {st['r']})")
        print_pairs(st, st["b"], st["r"], st["k_used"], docs, cfg["minhash"], show_text=(idx == 0 and bool(docs)))
    pause(interactive, "Enter → (b) decontamination")
    dc = json.loads((REHEARSAL / "decontam.json").read_text(encoding="utf-8"))
    print(f"\n  (b) {dc['n']}-gram · MMLU test {dc['questions']:,} questions ({dc['too_short']} shorter than {dc['n']} tokens)")
    for c in dc["corpora"]:
        print(f"  {BOLD}{c['file']}{RESET}: {c['questions_hit']} questions share a {dc['n']}-gram with the corpus  ({c['hits']} hits in {c['documents_hit']} documents; {c['seconds']:.0f} s)")
        print("    where: " + "; ".join(f"{v} × {k}" for k, v in c["where"].items()))
        seen = set()
        for r in c["rows"]:
            key = tuple(r["ngrams"][:2]) if r["where"] == WHERE_STIMULUS else r["qi"]
            if key in seen:
                continue
            seen.add(key)
            n_share = sum(1 for x in c["rows"] if x["where"] == WHERE_STIMULUS and tuple(x["ngrams"][:2]) == key) if r["where"] == WHERE_STIMULUS else 1
            tag = f" ({n_share} questions share this passage)" if n_share > 1 else ""
            print(f"    {AMBER}[{r['subject']}]{RESET}{tag} {r['question'][:140].replace(chr(10), ' ')}{'…' if len(r['question']) > 140 else ''}  {GREY}— {r['where']}{RESET}")
            for g in r["ngrams"][:1]:
                print(f"        {GREEN}“{g}”{RESET}  {GREY}in {r['urls'][0][:80]}{RESET}")
    if dc.get("redux"):
        pause(interactive, "Enter → MMLU-Redux (Virology)")
        rd = dc["redux"]
        print(f"\n  {BOLD}MMLU-Redux 2.0, Virology: {rd['flagged']} of {rd['questions']} flagged{RESET}  " + ", ".join(f"{k} {v}" for k, v in rd["error_types"].items()))
        for r in rd["shown"]:
            print(f"\n  {AMBER}{r['question'][:220]}{RESET}")
            for k, c in enumerate(r["choices"]):
                print(f"      {'ABCD'[k]}. {c[:90]}  {GREY}{'MMLU key' if k == r['answer'] else ''}{RESET}")
            print(f"      Redux: {RED}{r['error_type']}{RESET} — correct answer {r['correct_answer']!s}; source: {r.get('source') or '—'}")
    pause(interactive, "Enter → summary")
    print()
    print_summary(s)


def main() -> None:
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p = sub.add_parser("run")
    p.add_argument("--impl", choices=["live", "final"], default="final")
    p.add_argument("--corpus", choices=["edu", "web"], default=cfg["corpus"]["default"])
    p.add_argument("--dumps", default=cfg["corpus"]["dumps"], help="all 或逗號分開的 dump 名")
    p.add_argument("--bands", type=int, default=cfg["minhash"]["bands"])
    p.add_argument("--rows", type=int, default=cfg["minhash"]["rows"])
    p.add_argument("--also", default=",".join(cfg["minhash"]["also"]), help="之後再跑的 (b, r)，如 32x4,8x16；空字串就只跑一組")
    p.add_argument("--no-pause", action="store_true")
    p.add_argument("--fresh", action="store_true", help="不讀 runs/cache/，重算簽章（量秒數用；rehearse 一律如此）")
    p = sub.add_parser("decontam")
    p.add_argument("--corpus", choices=["edu", "web", "both"], default=cfg["corpus"]["default"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--no-pause", action="store_true")
    sub.add_parser("rehearse")
    sub.add_parser("replay")
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "fetch":
        cmd_fetch(a, cfg)
    elif a.cmd == "run":
        cmd_run(a, cfg, interactive=not a.no_pause)
    elif a.cmd == "decontam":
        cmd_decontam(a, cfg, interactive=not a.no_pause)
    elif a.cmd == "rehearse":
        cmd_rehearse(a, cfg)
    elif a.cmd in ("replay", "show"):
        cmd_replay(a, cfg, interactive=(a.cmd == "replay"))


if __name__ == "__main__":
    main()
