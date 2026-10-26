#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 2：小規模 scaling law 的擬合與外插（彩排預錄、課堂 replay）。投影片 p33（p32 前導頁）。

W4 demo 1 的 tiny GPT 改成 byte-level（V = 256），五個尺寸（demo_config.toml [[sizes]]），同一份 FineWeb-Edu 的抽樣、同一個 seed，
每一個都訓到 D = 20N（Chinchilla 的射線），learning rate 照 muP 的寬度規則縮；每 50 步在固定的 held-out 子集上記 loss。
然後：用最小的三個點擬合 L(N) = E + A·N^-α、外插到最大的兩個、報誤差；再用五個點全擬合對照；最後把五條 loss 對 token 的曲線
切成 iso-FLOPs 的 envelope（Approach 1）。

  ./present.sh fetch [--corpus edu|web|both] [--mb 200]   串流 FineWeb-Edu（demo 3 另要 FineWeb）的 parquet shard，寫 ../w07_corpus/*.jsonl（一次；需要網路）
  ./present.sh estimate            不訓練：五個尺寸的 N、D、步數、lr，與用 W4 probe.json 的吞吐估的時間（真正的估計是下一行）
  ./present.sh rehearse --steps 30 煙霧測試：五個尺寸各 30 步 + 擬合 + 總表，寫到 runs/smoke/；每個尺寸印 s/step 與換算成正式 run 的分鐘數
  ./present.sh rehearse            課前彩排：五個尺寸各訓到 D = 20N → runs/rehearsal/{s1..s5}.json + fit.json + summary.json + curves.svg；權重 → runs/weights/
  ./present.sh train s3            只訓一個尺寸 → runs/rehearsal/s3.json（之後 ./present.sh fit）
  ./present.sh fit                 只從五個 json 重做擬合、envelope、summary.json 與 curves.svg
  ./present.sh replay              課堂（p33）：Enter 一步——五個點 → 三點擬合與外插誤差 → 五點擬合 → envelope → p33 的表
  ./present.sh show                同 replay，不等 Enter（加 --smoke 讀 runs/smoke/）

誠實的限定（每次 replay 都印）：幾百萬參數的 byte-level 模型、兩百 MB 的語料、一個 seed、五個點、一條射線；
看到的是機制的形狀——點少、共線、外插——不是任何真實 scaling law 的數字。誤差小的正確結論是「兩個數量級內冪律管用」，不是「外插沒問題」。
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
LIMIT = ("這是幾百萬參數的 byte-level 模型、約 200 MB 的語料、一個 seed、五個點、一條射線 D = 20N。"
         "看到的是機制的形狀——點少、共線、外插——不是任何真實 scaling law 的數字；誤差小的正確結論是「兩個數量級內冪律管用」，不是「外插沒問題」。")

sys.path.insert(0, str(HERE))        # 只為了 import 同資料夾的 train.py 與 fit.py（這裡沒有 tests.py 要撞）
import fit as F                      # noqa: E402
import train as T                    # noqa: E402


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def out_dir(cfg: dict, smoke: bool) -> Path:
    return (HERE / "runs" / "smoke") if smoke else (HERE / cfg["rehearse"]["out"])


def fmt_n(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n < 1e8 else f"{n / 1e9:.2f}B"


def fmt_tok(n: float) -> str:
    return f"{n / 1e9:g}B" if n >= 1e9 else (f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}k")


def fmt_c(c: float) -> str:
    e = int(math.floor(math.log10(c)))
    return f"{c / 10 ** e:.1f}e{e}"


# ── estimate ─────────────────────────────────────────────────────
def print_plan(rows: list[dict], cfg: dict) -> None:
    t, ray = cfg["train"], cfg["ray"]
    print(f"\n  {BOLD}the five sizes (byte-level, V = 256, L = {t['L']}; N = {ray['count']} params; D = {ray['tokens_per_param']}·N; batch {t['batch']} × {t['L']} = {t['batch'] * t['L']:,} tokens/step){RESET}")
    print(f"    {'':<4}{'d':>5}{'layers':>8}{'heads':>7}{'d_ff':>6}{'N total':>12}{'non-emb':>12}{'D = 20N':>12}{'steps':>9}{'C = 6ND':>10}{'lr':>10}{'warmup':>8}")
    for r in rows:
        print(f"    {r['key']:<4}{r['d']:>5}{r['n_layer']:>8}{r['n_head']:>7}{r['d_ff']:>6}{r['params']['total']:>12,}{r['params']['nonemb']:>12,}{r['D']:>12,}{r['steps']:>9,}{fmt_c(r['C']):>10}{r['lr']:>10.2e}{r['warmup']:>8}")
    tot = sum(r["D"] for r in rows)
    print(f"    {'':<4}{'':>5}{'':>8}{'':>7}{'':>6}{'':>12}{'':>12}{tot:>12,}{sum(r['steps'] for r in rows):>9,}")
    ratios = [rows[i + 1]["N"] / rows[i]["N"] for i in range(len(rows) - 1)]
    print(f"    N ratios between neighbours: " + ", ".join(f"{x:.2f}" for x in ratios) + f"; span {math.log10(rows[-1]['N'] / rows[0]['N']):.2f} decades")


def estimate(d1: T.Demo1, cfg: dict) -> None:
    rows = T.plan(d1, cfg)
    print_plan(rows, cfg)
    p = d1.dir / "runs" / "rehearsal" / "train.json"
    tps = json.loads(p.read_text(encoding="utf-8")).get("tokens_per_s") if p.exists() else None
    if tps:
        tot = sum(r["D"] for r in rows)
        print(f"\n  W4 demo 1's rehearsal ran {tps:,.0f} tok/s (4.76M params, batch 64 × 256, MPS). At that rate: " +
              ", ".join(f"{r['key']} {r['D'] / tps / 60:.0f} min" for r in rows) + f"; total ≈ {tot / tps / 60:.0f} min.")
        print(f"  {DIM}Small models are overhead-bound (slower per token than that), the 9.9M one is heavier per token; batch here is 32, not 64. "
              f"Run ./present.sh rehearse --steps 30 first — it times each size and converts to minutes for the full run.{RESET}")
    else:
        print(f"  {AMBER}W4 demo 1 沒有 runs/rehearsal/train.json，無法估吞吐{RESET}")


# ── fit + summary ─────────────────────────────────────────────────
def load_runs(outd: Path, cfg: dict) -> list[dict]:
    runs = []
    for s in cfg["sizes"]:
        p = outd / f"{s['key']}.json"
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


def do_fit(cfg: dict, runs: list[dict]) -> dict:
    fc = cfg["fit"]
    by = {r["key"]: r for r in runs}
    loss_of = lambda r: r["final"]["held_full_nats"]                         # 擬合用整段 held-out（W4 的尺），不是子集
    pts_fit = [(k, by[k]["N"], loss_of(by[k])) for k in fc["fit_with"] if k in by]
    pts_test = [(k, by[k]["N"], loss_of(by[k])) for k in fc["test_on"] if k in by]
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "loss_used": "held_full_nats (W4's windowed measure over the whole held-out, nats/byte)",
           "points": [{"key": r["key"], "N": r["N"], "D": r["final"]["D_actual"], "steps": r["steps"], "loss": loss_of(r), "bpb": loss_of(r) / math.log(2),
                       "subset_loss": r["final"]["held_subset_loss"], "seconds": r["final"]["seconds"], "epochs": r["final"]["epochs"]} for r in runs]}
    if len(pts_fit) >= 3:
        f3 = F.fit_power([p[1] for p in pts_fit], [p[2] for p in pts_fit], fc["alpha_lo"], fc["alpha_hi"])
        f3["keys"] = [p[0] for p in pts_fit]
        f3["test"] = F.evaluate_fit(f3, pts_test)
        out["fit3"] = f3
    allp = pts_fit + pts_test
    if len(allp) >= 4:
        f5 = F.fit_power([p[1] for p in allp], [p[2] for p in allp], fc["alpha_lo"], fc["alpha_hi"])
        f5["keys"] = [p[0] for p in allp]
        f5["residuals"] = [{"key": k, "N": N, "measured": L, "fitted": F.predict(f5, N), "error": F.predict(f5, N) - L} for k, N, L in allp]
        out["fit5"] = f5
    if len(runs) >= 2:
        out["envelope"] = F.envelope(runs, fc["envelope_points"])
    return out


def summarize(cfg: dict, outd: Path, runs: list[dict] | None = None) -> dict:
    runs = runs or load_runs(outd, cfg)
    if not runs:
        raise FileNotFoundError(f"{outd.relative_to(HERE)} 裡沒有任何尺寸的 json：先跑 ./present.sh rehearse")
    fit = do_fit(cfg, runs)
    (outd / "fit.json").write_text(json.dumps(fit, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    r0 = runs[0]
    summary = {"date": fit["date"], "machine": r0["machine"], "macos": r0["macos"], "python": r0["python"], "torch": r0["torch"], "device": r0["device"],
               "corpus": r0["corpus"], "ray": r0["ray"], "train": r0["train"], "n_runs": len(runs), "smoke": any(r["steps"] != r["steps_for_ray"] for r in runs),
               "runs": {r["key"]: {"size": r["size"], "params": r["params"], "N": r["N"], "steps": r["steps"], "steps_for_ray": r["steps_for_ray"], "lr": r["lr"], "final": r["final"], "date": r["date"]} for r in runs},
               "total_seconds": sum(r["final"]["seconds"] for r in runs),
               "table": table_rows(fit), "limit": LIMIT}
    (outd / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    svg_curves(runs, fit, summary, outd / "curves.svg")
    print(f"{DIM}→ {outd.relative_to(HERE)}/{{{','.join(r['key'] for r in runs)}}}.json, fit.json, summary.json, curves.svg{RESET}")
    return summary


def table_rows(fit: dict) -> dict:
    """p33 表格的兩列：fit | E | A | α | predicted at N4 / measured | predicted at N5 / measured。bits/byte。"""
    rows = {}
    if "fit3" in fit:
        f = fit["fit3"]
        rows["three"] = {"E_bpb": f["E"] / math.log(2), "A": f["A"], "alpha": f["alpha"], "rss": f["rss"], "alpha_at_edge": f.get("alpha_at_edge"),
                         "test": {t["key"]: {"predicted_bpb": t["predicted_bpb"], "measured_bpb": t["measured_bpb"], "error_bpb": t["error_bpb"], "error_pct": t["error_pct"]} for t in f["test"]}}
    if "fit5" in fit:
        f = fit["fit5"]
        rows["five"] = {"E_bpb": f["E"] / math.log(2), "A": f["A"], "alpha": f["alpha"], "rss": f["rss"], "alpha_at_edge": f.get("alpha_at_edge"),
                        "residuals_bpb": {r["key"]: r["error"] / math.log(2) for r in f["residuals"]}}
    return rows


# ── replay ───────────────────────────────────────────────────────
def replay(cfg: dict, pause: bool, smoke: bool = False) -> None:
    outd = (HERE / "runs" / "smoke") if smoke else (HERE / cfg["rehearse"]["out"])
    if not (outd / "summary.json").exists():
        print(f"{RED}沒有 {outd.relative_to(HERE)}/summary.json：先跑 ./present.sh rehearse{RESET}")
        return
    summary = json.loads((outd / "summary.json").read_text(encoding="utf-8"))
    fit = json.loads((outd / "fit.json").read_text(encoding="utf-8"))
    runs = load_runs(outd, cfg)
    ray = summary["ray"]
    cp = summary["corpus"]
    print(f"{AMBER}{BOLD}REPLAY — 課前彩排的紀錄，不是現場{RESET}  {summary['date']}, torch {summary['torch']}, {summary['device']}, {summary['machine']}"
          + (f"  {RED}(smoke run: steps overridden){RESET}" if summary.get("smoke") else ""))
    print(f"  byte-level GPT (W4's block, V = 256, L = {summary['train']['L']}); corpus {Path(cp['path']).name} ({cp['repo']}, {cp['license']}): "
          f"{cp['docs']:,} docs, train {cp['train_bytes'] / 1e6:.0f} MB, held-out {cp['held_bytes'] / 1e6:.2f} MB; every run on the ray D = {ray['tokens_per_param']}·N (N = {ray['count']} params); "
          f"{summary['n_runs']} runs, {summary['total_seconds'] / 60:.0f} min in total")
    ln2 = math.log(2)
    # 1. 五個點
    if pause:
        input(f"\n  {DIM}Enter → the five points{RESET}")
    print(f"\n  {BOLD}1. five points on the ray{RESET}  (held-out = W4's windowed measure over the whole {cp['held_bytes'] / 1e6:.2f} MB; bits/byte = nats ÷ ln 2 because a token is a byte)")
    print(f"    {'':<4}{'d':>5}{'layers':>8}{'N':>12}{'D':>14}{'steps':>8}{'epochs':>8}{'min':>7}{'held-out nats':>15}{'bits/byte':>11}{'subset':>9}")
    for p in fit["points"]:
        r = summary["runs"][p["key"]]
        print(f"    {p['key']:<4}{r['size']['d']:>5}{r['size']['n_layer']:>8}{p['N']:>12,}{p['D']:>14,}{p['steps']:>8,}{p['epochs']:>8.2f}{p['seconds'] / 60:>7.1f}{p['loss']:>15.4f}{p['bpb']:>11.4f}{p['subset_loss'] / ln2:>9.4f}")
    # 2. 三點擬合
    if "fit3" in fit:
        f3 = fit["fit3"]
        if pause:
            input(f"\n  {DIM}Enter → fit E + A·N^-α to the three smallest ({', '.join(f3['keys'])}) and extrapolate{RESET}")
        print(f"\n  {BOLD}2. three points, three parameters{RESET}  L(N) = E + A·N^-α along D = 20N (so B·D^-β is folded into A)")
        print(f"    E = {f3['E']:.4f} nats ({f3['E'] / ln2:.4f} bits/byte)   A = {f3['A']:.4g}   α = {f3['alpha']:.4f}   residual sum of squares {f3['rss']:.2e}"
              + (f"  {AMBER}(α at the edge of the search range [{f3['search']['lo']}, {f3['search']['hi']}]){RESET}" if f3.get("alpha_at_edge") else ""))
        print(f"    {'':<6}{'N':>12}{'predicted':>12}{'measured':>12}{'error':>10}{'%':>8}   (bits/byte)")
        for t in f3["test"]:
            col = GREEN if abs(t["error_pct"]) < 2 else (AMBER if abs(t["error_pct"]) < 5 else RED)
            print(f"    {t['key']:<6}{t['N']:>12,}{t['predicted_bpb']:>12.4f}{t['measured_bpb']:>12.4f}{col}{t['error_bpb']:>+10.4f}{t['error_pct']:>+8.2f}{RESET}")
        if f3["test"]:
            sign = "above" if f3["test"][-1]["error"] > 0 else "below"
            print(f"    the extrapolated curve passes {sign} the largest measured point — prediction {'too pessimistic' if sign == 'above' else 'too optimistic'} at {f3['test'][-1]['key']}")
    # 3. 五點擬合
    if "fit5" in fit:
        f5 = fit["fit5"]
        if pause:
            input(f"\n  {DIM}Enter → fit all five points{RESET}")
        print(f"\n  {BOLD}3. all five points{RESET}  (least squares; two degrees of freedom left)")
        print(f"    E = {f5['E']:.4f} nats ({f5['E'] / ln2:.4f} bits/byte)   A = {f5['A']:.4g}   α = {f5['alpha']:.4f}   rss {f5['rss']:.2e}"
              + (f"  {AMBER}(α at the edge of the search range){RESET}" if f5.get("alpha_at_edge") else ""))
        if "fit3" in fit:
            f3 = fit["fit3"]
            print(f"    moved from the three-point fit: E {f3['E'] / ln2:.4f} → {f5['E'] / ln2:.4f} bits/byte ({(f5['E'] - f3['E']) / ln2:+.4f}), α {f3['alpha']:.4f} → {f5['alpha']:.4f} ({f5['alpha'] - f3['alpha']:+.4f}), A {f3['A']:.4g} → {f5['A']:.4g}")
        print(f"    residuals (fitted − measured, bits/byte): " + "  ".join(f"{r['key']} {r['error'] / ln2:+.4f}" for r in f5["residuals"]))
    # 4. envelope
    if "envelope" in fit:
        env = fit["envelope"]
        if pause:
            input(f"\n  {DIM}Enter → iso-FLOPs envelope (Approach 1){RESET}")
        print(f"\n  {BOLD}4. iso-FLOPs envelope{RESET}  each run's curve as (C = 6·N·tokens, held-out subset loss); at each C, which run is lowest?")
        keys = [r["key"] for r in runs]
        print(f"    {'C':>9}" + "".join(f"{k:>9}" for k in keys) + f"{'best':>7}{'N best':>12}")
        for row in env["rows"]:
            cells = "".join(f"{(row['per_run'][k] / ln2):>9.3f}" if row["per_run"].get(k) is not None else f"{'·':>9}" for k in keys)
            print(f"    {fmt_c(row['C']):>9}{cells}{row['best']:>7}{row['best_N']:>12,}")
        if env.get("slope_logN_logC") is not None:
            print(f"    best N grows with C: slope of log N_best against log C ≈ {env['slope_logN_logC']:.2f} (five discrete N, so an order of magnitude only; Hoffmann Approach 1: a = 0.50)")
        print(f"    {DIM}· = that run never reached this C (too small) or started past it (too large): the envelope is only where runs overlap{RESET}")
    if pause:
        input(f"\n  {DIM}Enter → p33's table{RESET}")
    print_table(summary)
    print(f"\n  {DIM}curves.svg: open {outd / 'curves.svg'}{RESET}")


def print_table(summary: dict) -> None:
    tb = summary["table"]
    ray = summary["ray"]
    keys = list(summary["runs"])
    print(f"\n  {BOLD}p33's table{RESET}  (bits/byte; E in bits/byte, A and α as fitted in nats)")
    print(f"    {'fit':<22}{'E':>9}{'A':>12}{'α':>9}" + "".join(f"{'pred / meas at ' + k:>24}" for k in keys[3:]))
    if "three" in tb:
        t = tb["three"]
        print(f"    {'three smallest points':<22}{t['E_bpb']:>9.4f}{t['A']:>12.4g}{t['alpha']:>9.4f}{'*' if t.get('alpha_at_edge') else ' '}" + "".join(f"{t['test'][k]['predicted_bpb']:>12.4f} / {t['test'][k]['measured_bpb']:<9.4f}" if k in t["test"] else f"{'':>24}" for k in keys[3:]))
        print(f"    {'':<22}{'':>9}{'':>12}{'':>9}" + "".join(f"{'error ' + format(t['test'][k]['error_bpb'], '+.4f') + ' (' + format(t['test'][k]['error_pct'], '+.1f') + '%)':>24}" if k in t["test"] else f"{'':>24}" for k in keys[3:]))
    if "five" in tb:
        f = tb["five"]
        print(f"    {'all five points':<22}{f['E_bpb']:>9.4f}{f['A']:>12.4g}{f['alpha']:>9.4f}{'*' if f.get('alpha_at_edge') else ' '}" + "".join(f"{'—':>24}" for _ in keys[3:]))
    if any(v.get("alpha_at_edge") for v in tb.values()):
        print(f"    {AMBER}* α stopped at the edge of its search range — the three points are not power-law shaped (smoke runs, or runs that have not converged); read the fit as a failure, not a number{RESET}")
    print(f"  {GREY}every run on D = {ray['tokens_per_param']}·N; N counts {ray['count']} parameters. {summary['limit']}{RESET}")


# ── SVG（README 與課前看；投影片的圖另由 make_figs_w07.py 的 scaling_fit_fig() 讀 json 畫）──
def svg_curves(runs: list[dict], fit: dict, summary: dict, path: Path) -> None:
    W, H, l, r, tp, bt, gap = 1000, 420, 64, 20, 44, 60, 70
    pw = (W - l - r - gap) / 2
    ln2 = math.log(2)
    colors = ["#dc2626", "#ea580c", "#ca8a04", "#0b5cad", "#15803d"]
    col = {rr["key"]: colors[i % len(colors)] for i, rr in enumerate(runs)}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{l}" y="24" font-size="14" font-weight="600">Five byte-level GPTs on the ray D = {summary["ray"]["tokens_per_param"]}N (FineWeb-Edu sample): held-out bits/byte against N, and the five training curves</text>']
    # 左：loss 對 N（log），點 + 兩條擬合線
    px = l
    Ns = [p["N"] for p in fit["points"]]
    Ls = [p["bpb"] for p in fit["points"]]
    x0, x1 = math.log10(min(Ns)) - 0.15, math.log10(max(Ns)) + 0.15
    allL = list(Ls)
    for key in ("fit3", "fit5"):
        if key in fit:
            allL += [F.predict(fit[key], 10 ** x) / ln2 for x in (x0, x1)]
    y0, y1 = min(allL) - 0.05, max(allL) + 0.05
    X = lambda n: px + (math.log10(n) - x0) / (x1 - x0) * pw
    Y = lambda v: tp + (y1 - v) / (y1 - y0) * (H - tp - bt)
    for k in range(6):
        v = y0 + (y1 - y0) * k / 5
        out.append(f'<line x1="{px}" y1="{Y(v):.1f}" x2="{px + pw:.1f}" y2="{Y(v):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{px - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555">{v:.2f}</text>')
    for e in range(int(math.floor(x0)), int(math.ceil(x1)) + 1):
        for m in (1, 2, 5):
            n = m * 10 ** e
            if x0 <= math.log10(n) <= x1:
                out.append(f'<text x="{X(n):.1f}" y="{H - bt + 16}" text-anchor="middle" fill="#555">{fmt_n(n)}</text>')
    out.append(f'<text x="{px + pw / 2:.1f}" y="{H - 12}" text-anchor="middle" fill="#555">N (parameters, log)</text>')
    out.append(f'<text x="{px + 4}" y="{tp + 14}" fill="#333" font-weight="600">held-out bits/byte (whole held-out, W4\'s measure)</text>')
    for key, dash, lab in (("fit3", "", "fit to the three smallest"), ("fit5", "5 4", "fit to all five")):
        if key in fit:
            f = fit[key]
            pts = []
            for i in range(121):
                x = x0 + (x1 - x0) * i / 120
                pts.append(f"{X(10 ** x):.1f},{Y(F.predict(f, 10 ** x) / ln2):.1f}")
            out.append(f'<polyline fill="none" stroke="#333" stroke-width="1.5" stroke-dasharray="{dash}" points="{" ".join(pts)}"/>')
            yy = tp + 50 + (0 if key == "fit3" else 16)
            out.append(f'<line x1="{px + pw - 230}" y1="{yy}" x2="{px + pw - 206}" y2="{yy}" stroke="#333" stroke-width="1.5" stroke-dasharray="{dash}"/>')
            out.append(f'<text x="{px + pw - 200}" y="{yy + 4}" fill="#333">{lab}: E {f["E"] / ln2:.3f}, α {f["alpha"]:.3f}</text>')
    fit_keys = set(fit.get("fit3", {}).get("keys", []))
    for p in fit["points"]:
        filled = p["key"] in fit_keys
        out.append(f'<circle cx="{X(p["N"]):.1f}" cy="{Y(p["bpb"]):.1f}" r="5" fill="{col[p["key"]] if filled else "white"}" stroke="{col[p["key"]]}" stroke-width="2"/>')
        out.append(f'<text x="{X(p["N"]) + 8:.1f}" y="{Y(p["bpb"]) - 8:.1f}" fill="{col[p["key"]]}">{p["key"]}</text>')
    out.append(f'<text x="{px + 4}" y="{tp + 30}" fill="#666" font-size="11">filled = used for the three-point fit; hollow = extrapolation targets</text>')
    # 右：loss 對 token（log），五條
    px = l + pw + gap
    allT = [c["tokens_seen"] for rr in runs for c in rr["curve"] if c["tokens_seen"] > 0]
    allV = [c["held_subset_bpb"] for rr in runs for c in rr["curve"] if c["tokens_seen"] > 0]
    tx0, tx1 = math.log10(min(allT)), math.log10(max(allT))
    vy0, vy1 = min(allV) - 0.05, min(max(allV), min(allV) + 3.0) + 0.05           # 開頭幾步的 loss 很高，縱軸截在最低點 + 3 bits
    XT = lambda n: px + (math.log10(n) - tx0) / (tx1 - tx0) * pw
    YT = lambda v: tp + (vy1 - min(v, vy1)) / (vy1 - vy0) * (H - tp - bt)
    for k in range(6):
        v = vy0 + (vy1 - vy0) * k / 5
        out.append(f'<line x1="{px}" y1="{YT(v):.1f}" x2="{px + pw:.1f}" y2="{YT(v):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{px - 6}" y="{YT(v) + 4:.1f}" text-anchor="end" fill="#555">{v:.2f}</text>')
    for e in range(int(math.floor(tx0)), int(math.ceil(tx1)) + 1):
        n = 10 ** e
        if tx0 <= e <= tx1:
            out.append(f'<text x="{XT(n):.1f}" y="{H - bt + 16}" text-anchor="middle" fill="#555">{fmt_tok(n)}</text>')
    out.append(f'<text x="{px + pw / 2:.1f}" y="{H - 12}" text-anchor="middle" fill="#555">tokens seen (log)</text>')
    out.append(f'<text x="{px + 4}" y="{tp + 14}" fill="#333" font-weight="600">held-out subset bits/byte during training</text>')
    for i, rr in enumerate(runs):
        pts = " ".join(f"{XT(c['tokens_seen']):.1f},{YT(c['held_subset_bpb']):.1f}" for c in rr["curve"] if c["tokens_seen"] > 0)
        out.append(f'<polyline fill="none" stroke="{col[rr["key"]]}" stroke-width="1.8" points="{pts}"/>')
        yy = tp + 34 + 15 * i
        out.append(f'<line x1="{px + pw - 150}" y1="{yy}" x2="{px + pw - 126}" y2="{yy}" stroke="{col[rr["key"]]}" stroke-width="2"/>')
        out.append(f'<text x="{px + pw - 120}" y="{yy + 4}" fill="#333">{rr["key"]} · {fmt_n(rr["N"])}</text>')
    out.append(f'<text x="{W - r}" y="{H - bt + 36}" text-anchor="end" fill="#888" font-size="11">rehearsal {summary["date"][:10]} · {summary["machine"]} · torch {summary["torch"]} · one seed · not a measurement of any large model</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "estimate", "rehearse", "train", "fit", "replay", "show"])
    ap.add_argument("size", nargs="?", help="train 時：哪個尺寸（s1–s5）")
    ap.add_argument("--steps", type=int, default=None, help="覆蓋步數（煙霧測試）；有這個就寫到 runs/smoke/")
    ap.add_argument("--corpus", default="edu", choices=["edu", "web", "both"], help="fetch 時：抓哪一份")
    ap.add_argument("--mb", type=float, default=None, help="fetch 時：覆蓋 target_mb")
    ap.add_argument("--smoke", action="store_true", help="replay／show／fit 時：讀 runs/smoke/ 而不是 runs/rehearsal/")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "fetch":
        for w in (["edu", "web"] if a.corpus == "both" else [a.corpus]):
            T.fetch_corpus(cfg, w, a.mb)
        return 0
    if a.cmd in ("replay", "show"):
        replay(cfg, pause=(a.cmd == "replay"), smoke=a.smoke); return 0
    smoke = a.steps is not None or a.smoke
    outd = out_dir(cfg, smoke)
    if a.cmd == "fit":
        summary = summarize(cfg, outd); print_table(summary); return 0
    d1 = T.Demo1(cfg["shared"]["demo1"])
    if a.cmd == "estimate":
        estimate(d1, cfg); return 0
    if smoke:
        print(f"{AMBER}--steps {a.steps}: 煙霧測試，輸出到 runs/smoke/，不碰 runs/rehearsal/，不存權重{RESET}")
    outd.mkdir(parents=True, exist_ok=True)
    corpus = T.load_corpus(cfg, "edu")
    weights = None if smoke else (HERE / cfg["rehearse"]["weights"])
    sizes = {s["key"]: s for s in cfg["sizes"]}
    if a.cmd == "train":
        if a.size not in sizes:
            ap.error(f"train 要指定尺寸：{', '.join(sizes)}")
        rec = T.train_size(d1, cfg, corpus, sizes[a.size], a.steps, tag=f"train — {a.size}", save_weights=weights)
        (outd / f"{a.size}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{DIM}→ {outd.relative_to(HERE)}/{a.size}.json（之後 ./present.sh fit 重做擬合與 summary）{RESET}")
        return 0
    runs, timing = [], []
    for s in cfg["sizes"]:
        rec = T.train_size(d1, cfg, corpus, s, a.steps, tag=f"rehearse — {s['key']}", save_weights=weights)
        (outd / f"{s['key']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        runs.append(rec)
        per_step = rec["final"]["seconds"] / rec["steps"]
        timing.append((s["key"], per_step, per_step * rec["steps_for_ray"] / 60))
    if smoke:
        print(f"\n  {BOLD}smoke timing → projected full rehearsal{RESET}  (s/step measured over {a.steps} steps incl. warm-up overhead; full run = steps for D = 20N)")
        for k, ps, mins in timing:
            print(f"    {k}: {ps:.3f} s/step → {mins:.0f} min for {sizes[k]['d']}-wide run")
        print(f"    total ≈ {sum(t[2] for t in timing):.0f} min  {DIM}(first steps are slower than steady state on MPS; the real total is usually below this){RESET}")
    summary = summarize(cfg, outd, runs)
    print_table(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
