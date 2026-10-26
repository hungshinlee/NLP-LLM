#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 3：品質過濾的效果對照（現場猜來源與打分 ＋ 預錄的兩個 run）。投影片 p42（p41 前導頁）。

(a) 現場：FineWeb 與 FineWeb-Edu 的抽樣各隨機三頁（固定 seed）、打亂；學生猜哪頁來自哪份；再用 FineWeb-Edu 的分類器
    （HuggingFaceFW/fineweb-edu-classifier，FineWeb §4 的那一個）給同六頁打 0–5 分，然後揭曉來源。
(b) 預錄：demo 2 的一個尺寸（預設 s4，3.15 M）在兩份**等 byte** 的語料上各訓一次（同步數、同 seed、同超參數），
    在幾個 held-out 上分開報 bits/byte：W1 那段英文、Wikipedia 的抽樣、兩份語料各自的尾端。課堂 replay。

  ./present.sh fetch               一次、要網路：分類器（寫 ../versions.lock）＋ Wikipedia held-out（../w07_corpus/wikipedia_heldout.jsonl）。語料本身由 demo 2 的 fetch --corpus both 抓
  ./present.sh sample [--seed 7] [--exclude id,id]   抽六頁、打亂、存 runs/rehearsal/pages.json，印出來（**含來源**，課前自己看、踩雷就換 seed 或 exclude）
  ./present.sh pages               課堂：Enter 一頁，印六頁，不印來源
  ./present.sh score               課堂：載分類器，六頁各一個分數（Enter 一頁），然後揭曉來源；最後印 held-out 文字的分數
  ./present.sh rehearse [--steps N]  課前彩排：sample（若還沒有）→ score → 分數分布 → 兩個 run → runs/rehearsal/{edu,web}.json、summary.json、curves.svg；權重 → runs/weights/
  ./present.sh train edu|web       只訓一邊 → runs/rehearsal/<which>.json（之後 ./present.sh summarize）
  ./present.sh summarize           只從兩個 json 重做 summary.json 與 curves.svg
  ./present.sh replay              課堂：Enter 一步——兩個 run 的設定 → 每個 held-out 一格的曲線 → p42 的表
  ./present.sh show                同 replay，不等 Enter（加 --smoke 讀 runs/smoke/）

誠實的限定（每次 replay 都印）：三百萬參數的 byte-level 模型、兩份各 200 MB 的語料、一個 seed、六頁；
看到的是「品質是相對 held-out 定義的」這個機制，不是 FineWeb 論文的數字；六頁與分類器的分數是那六頁，不是分布。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "demo_config.toml"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"
LIMIT = ("這是一個三百萬參數的 byte-level 模型、兩份各約 200 MB 的語料、一個 seed；六頁是六頁，分數是那六頁的分數。"
         "看到的是「品質是相對 held-out 定義的」這個機制的形狀，不是 FineWeb 論文的數字。")

sys.path.insert(0, str(HERE))
import sample as S                   # noqa: E402
import train as TR                   # noqa: E402


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def out_dir(cfg: dict, smoke: bool) -> Path:
    return (HERE / "runs" / "smoke") if smoke else (HERE / cfg["rehearse"]["out"])


def pages_path(cfg: dict, outd: Path | None = None) -> Path:
    return (outd or HERE / cfg["rehearse"]["out"]) / "pages.json"


def label(cfg: dict, which: str) -> str:
    return cfg["corpora"][which]


# ── (a) 抽頁、看頁、打分 ──────────────────────────────────────────
def cmd_sample(cfg: dict, cfg2: dict, seed: int | None, exclude: list[str], outd: Path | None = None) -> list[dict]:
    pages = S.pick_pages(cfg, cfg2, seed, exclude)
    p = pages_path(cfg, outd)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": cfg["sample"]["seed"] if seed is None else seed, "exclude": list(cfg["sample"]["exclude"]) + exclude,
           "rule": {k: cfg["sample"][k] for k in ("per_corpus", "min_chars", "max_chars", "min_language_score", "pool")}, "pages": pages}
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{BOLD}six pages, seed {doc['seed']}{RESET}  ({cfg['sample']['per_corpus']} from each corpus; {cfg['sample']['min_chars']}–{cfg['sample']['max_chars']} chars, language_score ≥ {cfg['sample']['min_language_score']}; shuffled)")
    for pg in pages:
        S.show_page(pg, cfg, reveal=True)
    print(f"\n{DIM}→ {p.relative_to(HERE)}  (踩雷就 ./present.sh sample --seed <別的數> 或 --exclude <id>；課堂上 ./present.sh pages 不印來源){RESET}")
    return pages


def load_pages(cfg: dict, outd: Path | None = None) -> list[dict]:
    p = pages_path(cfg, outd)
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh sample{RESET}")
    return json.loads(p.read_text(encoding="utf-8"))["pages"]


def cmd_pages(cfg: dict, pause: bool) -> None:
    pages = load_pages(cfg)
    print(f"{BOLD}{len(pages)} pages — which corpus is each from?{RESET}  {DIM}({label(cfg, 'web')} or {label(cfg, 'edu')}; sources hidden){RESET}")
    for pg in pages:
        if pause:
            input(f"\n  {DIM}Enter → page {pg['page']}{RESET}")
        S.show_page(pg, cfg, reveal=False)


def cmd_score(cfg: dict, cfg2: dict, pause: bool, offline: bool = True, write: bool = True, outd: Path | None = None) -> dict:
    outd = outd or HERE / cfg["rehearse"]["out"]
    pages = load_pages(cfg, outd)
    print(f"{DIM}loading the classifier …{RESET}")
    clf = S.Classifier(cfg, offline=offline)
    print(f"  {clf.describe()}")
    print(f"\n{BOLD}FineWeb-Edu classifier, 0–5{RESET}  {DIM}(the dataset keeps int_score ≥ 3; a page longer than {clf.max_length} tokens is scored on its first {clf.max_length}){RESET}")
    scores = {}
    for pg in pages:
        if pause:
            input(f"\n  {DIM}Enter → score page {pg['page']}{RESET}")
        r = clf.score(pg["text"])
        scores[pg["page"]] = r
        col = GREEN if r["int_score"] >= 3 else AMBER
        print(f"  page {pg['page']}: {col}{BOLD}{r['score']:5.2f}  → {r['int_score']}{RESET}  {DIM}({r['tokens']} tokens{', truncated' if r['truncated'] else ''}, {r['seconds'] * 1000:.0f} ms){RESET}")
    if pause:
        input(f"\n  {DIM}Enter → reveal the sources{RESET}")
    print(f"\n{BOLD}the sources{RESET}")
    print(f"  {'page':<6}{'score':>7}{'int':>5}  {'actually from':<14}{'dataset score':>14}  {'dump':<16} url")
    for pg in pages:
        r = scores[pg["page"]]
        lab = label(cfg, pg["source"])
        ds = f"{pg['dataset_score']:.2f}" if pg.get("dataset_score") is not None else "—"
        print(f"  {pg['page']:<6}{r['score']:>7.2f}{r['int_score']:>5}  {(GREEN if pg['source'] == 'edu' else AMBER)}{lab:<14}{RESET}{ds:>14}  {pg['dump'] or '':<16} {DIM}{pg['url']}{RESET}")
    print(f"  {DIM}dataset score = the score stored with the FineWeb-Edu row (the same classifier, run by the dataset's authors); FineWeb rows have none{RESET}")
    # held-out 文字的分數：Wikipedia 像不像教科書、W1 那段像不像
    held = {}
    try:
        d2 = TR.Demo2(cfg)
        for h in cfg["heldouts"]:
            if h["source"] == "file":
                txt = (HERE / h["path"]).read_text(encoding="utf-8")
                held[h["key"]] = {"label": h["label"], **clf.score(txt), "n": 1}
            elif h["source"] == "wikipedia":
                p, _ = S.wiki_paths(cfg, d2.cfg2)
                if p.exists():
                    arts = [json.loads(l) for l in open(p, encoding="utf-8")][:20]
                    rs = [clf.score(a["text"]) for a in arts]
                    held[h["key"]] = {"label": h["label"], "score": sum(r["score"] for r in rs) / len(rs), "int_score": None,
                                      "min": min(r["score"] for r in rs), "max": max(r["score"] for r in rs), "n": len(rs)}
    except FileNotFoundError:
        pass
    if held:
        if pause:
            input(f"\n  {DIM}Enter → the held-out texts, scored by the same classifier{RESET}")
        print(f"\n{BOLD}the held-out texts of part (b), scored by the same classifier{RESET}")
        for k, h in held.items():
            if h.get("n", 1) > 1:
                print(f"  {h['label']:<40} mean {h['score']:5.2f} over {h['n']} articles (min {h['min']:.2f}, max {h['max']:.2f})")
            else:
                print(f"  {h['label']:<40} {h['score']:5.2f}  → {h['int_score']}")
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "classifier": {"repo": clf.repo, "revision": clf.revision, "device": str(clf.device), "params": clf.n_params,
                                                                     "max_length": clf.max_length, "load_s": clf.load_s, **clf.versions},
           "pages": [{**{k: v for k, v in pg.items() if k != "text"}, **scores[pg["page"]]} for pg in pages], "heldouts": held}
    if write:
        p = outd / "scores.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\n{DIM}→ {p.relative_to(HERE)}{RESET}")
    return rec


# ── (b) 兩個 run ─────────────────────────────────────────────────
def run_both(cfg: dict, d2: TR.Demo2, which_list: list[str], steps: int | None, outd: Path, smoke: bool) -> list[dict]:
    corpora = {w: d2.load_corpus(w) for w in ("edu", "web")}
    for w, c in corpora.items():
        print(f"  {label(cfg, w):<12} {c['docs']:>7,} docs  train {c['train_bytes'] / 1e6:7.1f} MB  held-out tail {c['held_docs']} docs {c['held_bytes'] / 1e6:.2f} MB  ({c['repo']}, {c['license']})")
    eb, wb = corpora["edu"]["train_bytes"], corpora["web"]["train_bytes"]
    print(f"  equal bytes: the two training sets differ by {abs(eb - wb) / 1e6:.2f} MB ({abs(eb - wb) / max(eb, wb) * 100:.2f} %); both runs take the same number of steps, so they read the same number of bytes")
    heldouts = TR.load_heldouts(cfg, d2, corpora, S)
    print(f"  held-outs: " + ", ".join(f"{h['label']} {h['n_bytes']:,} B" for h in heldouts.values()))
    weights = None if smoke else (HERE / cfg["rehearse"]["weights"])
    recs = []
    for w in which_list:
        rec = TR.train_run(cfg, d2, corpora[w], heldouts, steps, tag=f"run — {label(cfg, w)}", save_weights=weights)
        (outd / f"{w}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        recs.append(rec)
    return recs


def load_runs(outd: Path) -> dict:
    return {w: json.loads((outd / f"{w}.json").read_text(encoding="utf-8")) for w in ("edu", "web") if (outd / f"{w}.json").exists()}


def summarize(cfg: dict, outd: Path) -> dict:
    runs = load_runs(outd)
    if len(runs) < 2:
        raise SystemExit(f"{RED}{outd.relative_to(HERE)} 裡要有 edu.json 與 web.json 才能做 summary（有：{', '.join(runs) or '無'}）{RESET}")
    e, w = runs["edu"], runs["web"]
    keys = list(e["final"])
    table = {k: {"label": e["heldouts"][k]["label"], "bytes": e["heldouts"][k]["n_bytes"], "edu_bpb": e["final"][k]["bpb"], "web_bpb": w["final"][k]["bpb"],
                 "diff_bpb": e["final"][k]["bpb"] - w["final"][k]["bpb"], "winner": "edu" if e["final"][k]["bpb"] < w["final"][k]["bpb"] else "web"} for k in keys}
    # demo 2 的 s4 當對照：同語料、同 seed、同步數，edu run 在自己尾端的 bpb 應該與它接近（MPS 非決定性之內）
    d2_ref = None
    p = (HERE / cfg["shared"]["demo2"] / "runs" / "rehearsal" / f"{cfg['train']['size']}.json").resolve()
    if p.exists() and e["steps"] == e["steps_for_ray"]:
        r = json.loads(p.read_text(encoding="utf-8"))
        d2_ref = {"bpb": r["final"]["held_full_bpb"], "date": r["date"], "diff_vs_edu_run": e["final"].get("edu_tail", {}).get("bpb", float("nan")) - r["final"]["held_full_bpb"]}
    scores = json.loads((outd / "scores.json").read_text(encoding="utf-8")) if (outd / "scores.json").exists() else None
    dist_p = outd / "distribution.json"
    dist = json.loads(dist_p.read_text(encoding="utf-8")) if dist_p.exists() else None
    summary = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": e["machine"], "macos": e["macos"], "python": e["python"], "torch": e["torch"], "device": e["device"],
               "size": e["size"], "params": e["params"], "N": e["N"], "steps": e["steps"], "steps_for_ray": e["steps_for_ray"], "tokens": e["tokens"], "lr": e["lr"], "smoke": e["steps"] != e["steps_for_ray"],
               "runs": {k: {"trained_on": r["trained_on"], "corpus": r["corpus"], "epochs": r["epochs"], "seconds": r["seconds"], "tokens_per_s": r["tokens_per_s"], "date": r["date"]} for k, r in runs.items()},
               "heldouts": e["heldouts"], "table": table, "demo2_reference": d2_ref,
               "pages": [{k: v for k, v in pg.items() if k != "text"} for pg in scores["pages"]] if scores else None, "classifier": scores["classifier"] if scores else None,
               "heldout_scores": scores.get("heldouts") if scores else None, "distribution": dist, "limit": LIMIT}
    (outd / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    svg_curves(runs, summary, outd / "curves.svg")
    print(f"{DIM}→ {outd.relative_to(HERE)}/summary.json, curves.svg{RESET}")
    return summary


# ── replay ───────────────────────────────────────────────────────
BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(vals: list[float], lo: float, hi: float) -> str:
    out = []
    for v in vals:
        x = 0.0 if hi <= lo else (min(max(v, lo), hi) - lo) / (hi - lo)
        out.append(BLOCKS[int(round(x * (len(BLOCKS) - 1)))])
    return "".join(out)


def replay(cfg: dict, pause: bool, smoke: bool = False) -> None:
    outd = out_dir(cfg, smoke)
    if not (outd / "summary.json").exists():
        print(f"{RED}沒有 {outd.relative_to(HERE)}/summary.json：先跑 ./present.sh rehearse{RESET}")
        return
    sm = json.loads((outd / "summary.json").read_text(encoding="utf-8"))
    runs = load_runs(outd)
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {sm['date']}, torch {sm['torch']}, {sm['device']}, {sm['machine']}"
          + (f"  {RED}(smoke run: steps overridden){RESET}" if sm.get("smoke") else ""))
    s = sm["size"]
    print(f"  one model, byte-level GPT {s['key']} (W4's block, V = 256, L = {s['L']}; d {s['d']}, {s['n_layer']} layers, {sm['params']['total']:,} params), trained twice with the same seed and the same {sm['steps']:,} steps = {sm['tokens'] / 1e6:.1f} M bytes each:")
    for k, r in sm["runs"].items():
        c = r["corpus"]
        print(f"    on {label(cfg, k):<12} {c['docs']:>7,} docs, train {c['train_bytes'] / 1e6:.1f} MB ({c['repo']}), {r['epochs']:.2f} epochs, {r['seconds'] / 60:.1f} min, {r['tokens_per_s']:,.0f} tok/s")
    ln2 = math.log(2)
    keys = list(sm["table"])
    # 曲線：每個 held-out 一格
    if pause:
        input(f"\n  {DIM}Enter → the curves, one held-out at a time{RESET}")
    e, w = runs["edu"], runs["web"]
    n_pts = len(e["curve"])
    idx = [round(i * (n_pts - 1) / 47) for i in range(48)] if n_pts > 48 else list(range(n_pts))
    for k in keys:
        t = sm["table"][k]
        ev = [e["curve"][i]["held"][k] for i in idx]
        wv = [w["curve"][i]["held"][k] for i in idx]
        lo = min(ev + wv)
        hi = min(max(ev + wv), lo + 2.0)
        print(f"\n  {BOLD}{t['label']}{RESET}  {DIM}({t['bytes']:,} bytes; subset curve, bits/byte, {lo:.2f}–{hi:.2f} on the bars; final = whole held-out, W4's measure){RESET}")
        print(f"    trained on {label(cfg, 'edu'):<12} {sparkline(ev, lo, hi)}  final {GREEN if t['winner'] == 'edu' else ''}{t['edu_bpb']:.4f}{RESET}")
        print(f"    trained on {label(cfg, 'web'):<12} {sparkline(wv, lo, hi)}  final {GREEN if t['winner'] == 'web' else ''}{t['web_bpb']:.4f}{RESET}")
        if pause and k != keys[-1]:
            input(f"  {DIM}Enter → next held-out{RESET}")
    if pause:
        input(f"\n  {DIM}Enter → p42's table{RESET}")
    print_table(cfg, sm)
    print(f"\n  {DIM}curves.svg: open {outd / 'curves.svg'}{RESET}")


def print_table(cfg: dict, sm: dict) -> None:
    print(f"\n  {BOLD}p42's table{RESET}  (bits/byte on each held-out; Δ = FineWeb-Edu-trained − FineWeb-trained, negative = the Edu model is better)")
    print(f"    {'held-out':<42}{'bytes':>10}{'trained on Edu':>16}{'trained on Web':>16}{'Δ':>9}{'  winner'}")
    for k, t in sm["table"].items():
        col = GREEN if t["diff_bpb"] < 0 else AMBER
        print(f"    {t['label']:<42}{t['bytes']:>10,}{t['edu_bpb']:>16.4f}{t['web_bpb']:>16.4f}{col}{t['diff_bpb']:>+9.4f}{RESET}  {label(cfg, t['winner'])}")
    if sm.get("demo2_reference"):
        r = sm["demo2_reference"]
        print(f"    {DIM}reproducibility: demo 2's {sm['size']['key']} on the same Edu corpus and seed measured {r['bpb']:.4f} on the Edu tail ({r['date'][:10]}); this run {r['diff_vs_edu_run']:+.4f}{RESET}")
    if sm.get("heldout_scores"):
        hs = sm["heldout_scores"]
        print(f"    {DIM}classifier on the held-out texts: " + "; ".join(f"{h['label']} {h['score']:.2f}" + (f" (mean of {h['n']})" if h.get('n', 1) > 1 else "") for h in hs.values()) + RESET)
    if sm.get("distribution"):
        d = sm["distribution"]
        print(f"    {DIM}classifier score distribution (rehearsal, {d['web']['n']} random pages each): "
              + "; ".join(f"{label(cfg, k)} mean {v['mean']:.2f}, int 0–5 = {v['hist']}, ≥3 {v['frac_ge3'] * 100:.0f}%" for k, v in d.items() if k in ("web", "edu")) + RESET)
    print(f"  {GREY}{sm['limit']}{RESET}")


# ── SVG（README 與課前看；投影片的圖另由 make_figs_w07.py 的 quality_fig() 讀 json 畫）──
def svg_curves(runs: dict, sm: dict, path: Path) -> None:
    keys = list(sm["table"])
    n = len(keys)
    W, H, l, r, tp, bt, gap = 1000, 330, 56, 16, 48, 56, 36
    pw = (W - l - r - gap * (n - 1)) / n
    e, w = runs["edu"], runs["web"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{l}" y="24" font-size="14" font-weight="600">One byte-level GPT ({sm["size"]["key"]}, {sm["params"]["total"] / 1e6:.2f}M), trained twice on equal bytes — FineWeb-Edu (blue), FineWeb (orange): held-out bits/byte during training, one panel per held-out</text>']
    allT = [c["tokens_seen"] for c in e["curve"] if c["tokens_seen"] > 0]
    tx0, tx1 = math.log10(min(allT)), math.log10(max(allT))
    for i, k in enumerate(keys):
        px = l + i * (pw + gap)
        ev = [c["held"][k] for c in e["curve"] if c["tokens_seen"] > 0]
        wv = [c["held"][k] for c in w["curve"] if c["tokens_seen"] > 0]
        y0 = min(ev + wv) - 0.05
        y1 = min(max(ev + wv), y0 + 2.5) + 0.05
        X = lambda t: px + (math.log10(t) - tx0) / (tx1 - tx0) * pw
        Y = lambda v: tp + (y1 - min(v, y1)) / (y1 - y0) * (H - tp - bt)
        for j in range(6):
            v = y0 + (y1 - y0) * j / 5
            out.append(f'<line x1="{px}" y1="{Y(v):.1f}" x2="{px + pw:.1f}" y2="{Y(v):.1f}" stroke="#eee"/>')
            out.append(f'<text x="{px - 4}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555" font-size="10">{v:.1f}</text>')
        for col, run, vals in (("#0b5cad", e, ev), ("#ea580c", w, wv)):
            pts = " ".join(f"{X(c['tokens_seen']):.1f},{Y(v):.1f}" for c, v in zip([c for c in run["curve"] if c["tokens_seen"] > 0], vals))
            out.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{pts}"/>')
        t = sm["table"][k]
        out.append(f'<text x="{px + pw / 2:.1f}" y="{tp - 8}" text-anchor="middle" fill="#333" font-weight="600">{t["label"]}</text>')
        out.append(f'<text x="{px + 4}" y="{H - bt + 16}" fill="#0b5cad" font-size="11">Edu-trained {t["edu_bpb"]:.3f}</text>')
        out.append(f'<text x="{px + pw - 4:.1f}" y="{H - bt + 16}" text-anchor="end" fill="#ea580c" font-size="11">Web-trained {t["web_bpb"]:.3f}</text>')
        out.append(f'<text x="{px + pw / 2:.1f}" y="{H - bt + 32}" text-anchor="middle" fill="#333" font-size="11">Δ {t["diff_bpb"]:+.3f} bits/byte · {t["bytes"]:,} B</text>')
    out.append(f'<text x="{l}" y="{H - 8}" fill="#555" font-size="11">x: bytes seen (log, {allT[0] / 1e6:.1f}M–{allT[-1] / 1e6:.1f}M); y: bits/byte on a fixed subset; finals = whole held-out</text>')
    out.append(f'<text x="{W - r}" y="{H - 8}" text-anchor="end" fill="#888" font-size="11">rehearsal {sm["date"][:10]} · {sm["machine"]} · torch {sm["torch"]} · one seed</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "sample", "pages", "score", "rehearse", "train", "summarize", "replay", "show"])
    ap.add_argument("which", nargs="?", help="train 時：edu 或 web")
    ap.add_argument("--seed", type=int, default=None, help="sample 時：覆蓋 [sample].seed")
    ap.add_argument("--exclude", default="", help="sample 時：逗號分隔的頁面 id，加到 [sample].exclude 之後")
    ap.add_argument("--steps", type=int, default=None, help="rehearse／train 時：覆蓋步數（煙霧測試）；有這個就寫到 runs/smoke/")
    ap.add_argument("--smoke", action="store_true", help="replay／show／summarize 時：讀 runs/smoke/")
    ap.add_argument("--online", action="store_true", help="score 時：允許分類器上網（預設只讀本機快取）")
    ap.add_argument("--no-dist", action="store_true", help="rehearse 時：跳過分數分布那一步")
    a = ap.parse_args()
    cfg = load_cfg()
    smoke = a.steps is not None or a.smoke
    outd = out_dir(cfg, smoke)
    if a.cmd == "fetch":
        S.fetch_classifier(cfg)
        d2 = TR.Demo2(cfg)
        S.fetch_wikipedia(cfg, d2.cfg2)
        return 0
    if a.cmd in ("replay", "show"):
        replay(cfg, pause=(a.cmd == "replay"), smoke=a.smoke); return 0
    if a.cmd == "summarize":
        sm = summarize(cfg, outd); print_table(cfg, sm); return 0
    d2 = TR.Demo2(cfg)
    exclude = [x.strip() for x in a.exclude.split(",") if x.strip()]
    if a.cmd == "sample":
        cmd_sample(cfg, d2.cfg2, a.seed, exclude); return 0
    if a.cmd == "pages":
        cmd_pages(cfg, pause=True); return 0
    if a.cmd == "score":
        cmd_score(cfg, d2.cfg2, pause=True, offline=not a.online); return 0
    outd.mkdir(parents=True, exist_ok=True)
    if smoke:
        print(f"{AMBER}--steps {a.steps}: 煙霧測試，輸出到 runs/smoke/，不碰 runs/rehearsal/，不存權重{RESET}")
    if a.cmd == "train":
        if a.which not in ("edu", "web"):
            ap.error("train 要指定 edu 或 web")
        run_both(cfg, d2, [a.which], a.steps, outd, smoke)
        print(f"{DIM}→ {outd.relative_to(HERE)}/{a.which}.json（兩邊都有之後 ./present.sh summarize）{RESET}")
        return 0
    # rehearse
    if not pages_path(cfg, outd).exists():
        cmd_sample(cfg, d2.cfg2, a.seed, exclude, outd)
    else:
        print(f"{DIM}六頁沿用 {pages_path(cfg, outd).relative_to(HERE)}（要重抽：./present.sh sample --seed N）{RESET}")
    rec = cmd_score(cfg, d2.cfg2, pause=False, offline=not a.online, outd=outd)
    if not a.no_dist:
        clf = S.Classifier(cfg, offline=not a.online)
        n = cfg["classifier"]["distribution_n"] if not smoke else 20
        print(f"\n{BOLD}classifier score distribution{RESET}  {DIM}({n} random pages from each corpus){RESET}")
        dist = S.score_distribution(clf, cfg, d2.cfg2, n, cfg["sample"]["seed"])
        for k, v in dist.items():
            print(f"  {label(cfg, k):<12} mean {v['mean']:.2f}  int 0–5: {v['hist']}  ≥3 {v['frac_ge3'] * 100:.0f}%  ({v['truncated']} truncated; {v['seconds']:.0f} s)"
                  + (f"  dataset's own score mean {v['dataset_score_mean']:.2f}, max |ours − theirs| {v['max_abs_diff_vs_dataset']:.3f}" if v.get("dataset_score_mean") is not None else ""))
        (outd / "distribution.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": n, **dist}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n{BOLD}(b) two runs{RESET}")
    run_both(cfg, d2, ["edu", "web"], a.steps, outd, smoke)
    sm = summarize(cfg, outd)
    print_table(cfg, sm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
