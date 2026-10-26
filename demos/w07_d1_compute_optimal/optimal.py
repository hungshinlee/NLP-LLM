#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 1 — compute-optimal training, numerically (slides p8–p9 and p22).

Everything here is arithmetic on one fitted equation, Hoffmann et al. 2022 Eq. (10):

    L(N, D) = E + A / N^alpha + B / D^beta,     with  C = 6 N D.

Sub-commands (the slide's 【操作】 block says when each one runs):

  table      (a) N*, D*, D*/N*, L* for C = 10^18 … 10^26 — the closed form, Eq. (4)
  points     (a) three real (N, D) pairs: their C, the optimum at that C, and the loss gap
  inference  (b) hold the loss fixed, charge 2 N D_inf for inference, re-optimise — one row per Enter
  tiny       (c) W4 demo 1's tiny GPT as a point on the same table
  check      compare every number above with slides/assets/w07/w07-data.json (if present)
  rehearse   run all of the above and write runs/rehearsal/*.json
  replay     print the rehearsal records (clearly marked as not live)

Standard library only (tomllib on Python ≥ 3.11, with a small fallback reader for older interpreters).
No model, no network, no third-party package.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # ../common.py (shared terminal styling)
try:
    from common import BOLD, DIM, RESET, GREEN, AMBER, RED, GREY, BLUE, banner, note
except Exception:                                   # run from anywhere, even without common.py
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    BLUE, AMBER, RED, GREEN, GREY = "\033[34m", "\033[33m", "\033[31m", "\033[32m", "\033[90m"

    def banner(text, color=BLUE):
        line = "─" * 72
        print(f"\n{color}{line}\n{BOLD}{text}{RESET}\n{color}{line}{RESET}")

    def note(text):
        print(f"{GREY}{text}{RESET}")


# ── config ─────────────────────────────────────────────────────────────────
def _toml_fallback(text: str) -> dict:
    """Minimal TOML reader for this demo's config (tables, arrays of tables, numbers, strings, flat arrays)."""
    import ast
    out: dict = {}
    cur = out
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
        if not line:
            continue
        if line.startswith("[["):
            name = line.strip("[]").strip()
            out.setdefault(name, []).append({})
            cur = out[name][-1]
        elif line.startswith("["):
            name = line.strip("[]").strip()
            cur = out.setdefault(name, {})
        else:
            k, v = line.split("=", 1)
            v = v.strip()
            if "#" in v and not v.startswith('"'):
                v = v.split("#", 1)[0].strip()
            cur[k.strip()] = ast.literal_eval(v)
    return out


def load_config(path: str = os.path.join(HERE, "demo_config.toml")) -> dict:
    text = open(path, encoding="utf-8").read()
    try:
        import tomllib
        return tomllib.loads(text)
    except ModuleNotFoundError:
        return _toml_fallback(text)


# ── the equation ───────────────────────────────────────────────────────────
def loss(N: float, D: float, c: dict) -> float:
    return c["E"] + c["A"] / N ** c["alpha"] + c["B"] / D ** c["beta"]


def closed_form(c: dict) -> tuple[float, float, float]:
    """Eq. (4): a = beta/(alpha+beta), b = alpha/(alpha+beta), G = (alpha A / beta B)^(1/(alpha+beta))."""
    al, be = c["alpha"], c["beta"]
    a = be / (al + be)
    b = al / (al + be)
    G = (al * c["A"] / (be * c["B"])) ** (1.0 / (al + be))
    return a, b, G


def optimum(C: float, c: dict) -> tuple[float, float]:
    """N* = G (C/6)^a,  D* = G^-1 (C/6)^b  (so that 6 N* D* = C)."""
    a, b, G = closed_form(c)
    return G * (C / 6.0) ** a, (C / 6.0) ** b / G


def D_for_loss(N: float, ell: float, c: dict) -> float:
    """The D that reaches loss ell at this N; inf if no D can (A/N^alpha already exceeds ell - E)."""
    rest = ell - c["E"] - c["A"] / N ** c["alpha"]
    if rest <= 0:
        return float("inf")
    return (c["B"] / rest) ** (1.0 / c["beta"])


def min_cost(ell: float, Dinf: float, c: dict, iters: int = 300, decades: float = 6.0) -> dict:
    """Minimise 6 N D + 2 N D_inf subject to L(N, D) = ell, by golden-section search on log N."""
    lo = math.log((c["A"] / (ell - c["E"])) ** (1.0 / c["alpha"])) + 1e-6   # below this N no D reaches ell
    hi = lo + math.log(10.0 ** decades)

    def cost(lnN: float) -> float:
        N = math.exp(lnN)
        return 6.0 * N * D_for_loss(N, ell, c) + 2.0 * N * Dinf

    g = (math.sqrt(5) - 1) / 2
    x1 = hi - g * (hi - lo)
    x2 = lo + g * (hi - lo)
    f1, f2 = cost(x1), cost(x2)
    for _ in range(iters):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = cost(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = cost(x2)
    N = math.exp((lo + hi) / 2)
    D = D_for_loss(N, ell, c)
    return dict(N=N, D=D, ratio=D / N, train=6.0 * N * D, infer=2.0 * N * Dinf,
                total=6.0 * N * D + 2.0 * N * Dinf)


# ── formatting ─────────────────────────────────────────────────────────────
def human(x: float) -> str:
    """81.4 M, 4.06 B, 2.41 T — three significant figures with a count suffix."""
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            v = x / div
            return (f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}") + " " + suf
    return f"{x:.3g}"


def sci(x: float) -> str:
    """4.06e9 → '4.06×10^9' (plain text; the slide's KaTeX does the typesetting)."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"10^{e}" if abs(m - 1.0) < 5e-4 else f"{m:.2f}×10^{e}"


def hdr(cols: list[str], widths: list[int]) -> None:
    print(BOLD + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths)) + RESET)
    print(GREY + "  ".join("─" * w for w in widths) + RESET)


def row(cells: list[str], widths: list[int], color: str = "") -> None:
    print(color + "  ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + (RESET if color else ""))


def meta() -> dict:
    return dict(date=dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), python=platform.python_version(),
                machine=platform.machine(), system=platform.system(), release=platform.release(), cwd_demo=os.path.basename(HERE))


def wait_enter(label: str, interactive: bool) -> None:
    if interactive and sys.stdin.isatty():
        try:
            input(f"{DIM}  ↵ Enter → {label}{RESET}")
        except EOFError:
            pass


# ── (a) table ──────────────────────────────────────────────────────────────
def compute_table(cfg: dict) -> dict:
    c = cfg["chinchilla"]
    a, b, G = closed_form(c)
    rows = []
    for e in cfg["table"]["log10_C"]:
        C = 10.0 ** e
        N, D = optimum(C, c)
        rows.append(dict(log10_C=e, C=C, N=N, D=D, ratio=D / N, loss=loss(N, D, c), on_slide=e in cfg["table"]["slide_rows"]))
    return dict(constants=c, a=a, b=b, G=G, G_inv2=G ** -2,
                ratio_exponent=(c["alpha"] - c["beta"]) / (c["alpha"] + c["beta"]), rows=rows)


def cmd_table(cfg: dict, res: dict | None = None, title: str = "demo 1 (a) · N*, D* against C — Hoffmann Eq. (10) → Eq. (4)") -> dict:
    res = res or compute_table(cfg)
    c = res["constants"]
    banner(title)
    print(f"  L(N, D) = E + A/N^α + B/D^β   with   E = {c['E']}, A = {c['A']}, B = {c['B']}, α = {c['alpha']}, β = {c['beta']}")
    print(f"  closed form (Eq. 4):  N* = G (C/6)^a,  D* = G⁻¹ (C/6)^b")
    print(f"    a = β/(α+β) = {res['a']:.4f}    b = α/(α+β) = {res['b']:.4f}    G = (αA/βB)^(1/(α+β)) = {res['G']:.4f}    G⁻² = {res['G_inv2']:.3f}")
    print(f"    D*/N* = G⁻² (C/6)^((α−β)/(α+β)),  exponent (α−β)/(α+β) = {res['ratio_exponent']:+.4f}"
          f"  → per decade of C the ratio is multiplied by 10^{res['ratio_exponent']:.3f} = {10 ** res['ratio_exponent']:.3f}")
    print()
    W = [10, 22, 22, 8, 7]
    hdr(["C (FLOPs)", "N*", "D*", "D*/N*", "L*"], W)
    for r in res["rows"]:
        star = "★ " if r["on_slide"] else "  "
        row([f"{star}10^{r['log10_C']}", f"{sci(r['N'])}  ({human(r['N'])})", f"{sci(r['D'])}  ({human(r['D'])})",
             f"{r['ratio']:.1f}", f"{r['loss']:.3f}"], W, BOLD if r["on_slide"] else "")
    note("  ★ = the five rows on slide p9.  N in parameters, D in tokens; L* in nats per token under this fit.")
    return res


# ── (a) points ─────────────────────────────────────────────────────────────
def compute_points(cfg: dict) -> dict:
    c = cfg["chinchilla"]
    pts = []
    for p in cfg["points"]:
        N, D = float(p["N"]), float(p["D"])
        C = 6.0 * N * D
        No, Do = optimum(C, c)
        pts.append(dict(key=p["key"], label=p["label"], note=p.get("note", ""), N=N, D=D, ratio=D / N, C=C,
                        loss=loss(N, D, c), N_opt=No, D_opt=Do, ratio_opt=Do / No, loss_opt=loss(No, Do, c),
                        loss_gap=loss(N, D, c) - loss(No, Do, c)))
    return dict(points=pts)


def cmd_points(cfg: dict, res: dict | None = None) -> dict:
    res = res or compute_points(cfg)
    banner("demo 1 (a) · three real models on the same equation")
    W = [24, 8, 8, 7, 11, 28, 7, 8, 8]
    hdr(["model", "N", "D", "D/N", "C = 6ND", "(N*, D*) at that C", "D*/N*", "L", "L − L*"], W)
    for p in res["points"]:
        row([p["label"], human(p["N"]), human(p["D"]), f"{p['ratio']:.0f}", sci(p["C"]),
             f"{human(p['N_opt'])} / {human(p['D_opt'])}", f"{p['ratio_opt']:.1f}", f"{p['loss']:.4f}", f"{p['loss_gap']:+.4f}"], W)
    note("  L and L* are both evaluated with Eq. (10); the gap is arithmetic on the fit, not a measurement.")
    for p in res["points"]:
        dn = p["N_opt"] / p["N"]
        dd = p["D_opt"] / p["D"]
        print(f"  {DIM}{p['label']}: the optimum at the same C has N ×{dn:.2f}, D ×{dd:.2f}; the fit says the gap is {p['loss_gap']:.4f} nats.{RESET}")
    return res


# ── (b) inference ──────────────────────────────────────────────────────────
def compute_inference(cfg: dict) -> dict:
    c = cfg["chinchilla"]
    inf = cfg["inference"]
    ref = next(p for p in cfg["points"] if p["key"] == inf["reference_point"])
    ell = loss(float(ref["N"]), float(ref["D"]), c)
    iters, dec = int(inf.get("golden_iters", 300)), float(inf.get("search_decades", 6))
    base = min_cost(ell, 0.0, c, iters, dec)
    rows = []
    for Dinf in inf["D_inf"]:
        Dinf = float(Dinf)
        r = min_cost(ell, Dinf, c, iters, dec)
        naive_total = 6.0 * base["N"] * base["D"] + 2.0 * base["N"] * Dinf      # keep the D_inf = 0 shape, pay inference anyway
        r.update(Dinf=Dinf, N_shrink=r["N"] / base["N"], D_grow=r["D"] / base["D"], naive_total=naive_total,
                 saving_vs_base_shape=1.0 - r["total"] / naive_total, guess_first=(Dinf == float(inf.get("guess_first", -1))))
        rows.append(r)
    thr = dict(rule="2·N·D_inf > 6·N·D  ⇔  D_inf > 3·D", D_inf_threshold_at_reference=3.0 * float(ref["D"]),
               D_inf_threshold_at_base_optimum=3.0 * base["D"])
    return dict(reference=dict(key=ref["key"], label=ref["label"], N=float(ref["N"]), D=float(ref["D"])), loss_fixed=ell,
                base=base, rows=rows, threshold=thr)


def cmd_inference(cfg: dict, res: dict | None = None, interactive: bool = True) -> dict:
    res = res or compute_inference(cfg)
    ref = res["reference"]
    banner("demo 1 (b) · pay for inference: minimise 6ND + 2N·D_inf at a fixed loss")
    print(f"  fixed loss ℓ = L({human(ref['N'])}, {human(ref['D'])}) = {res['loss_fixed']:.4f}   ({ref['label']}, under Eq. (10))")
    print(f"  for each D_inf: the (N, D) on the level set L = ℓ that minimises  6 N D + 2 N D_inf  (golden-section search on log N)")
    print()
    W = [8, 22, 22, 7, 10, 10, 10, 10]
    hdr(["D_inf", "N*  (× vs D_inf = 0)", "D*  (× vs D_inf = 0)", "D*/N*", "6ND", "2N·D_inf", "total", "saving"], W)
    for r in res["rows"]:
        label = f"D_inf = {sci(r['Dinf']) if r['Dinf'] else '0'}" + ("  — guess this row first" if r["guess_first"] else "")
        wait_enter(label, interactive)
        row([sci(r["Dinf"]) if r["Dinf"] else "0", f"{human(r['N'])}  (×{r['N_shrink']:.2f})", f"{human(r['D'])}  (×{r['D_grow']:.2f})",
             f"{r['ratio']:.0f}", sci(r["train"]), sci(r["infer"]) if r["infer"] else "0", sci(r["total"]),
             "—" if not r["Dinf"] else f"{100 * r['saving_vs_base_shape']:.1f} %"], W, BOLD if r["guess_first"] else "")
    note("  saving = 1 − total / (cost of keeping the D_inf = 0 shape and paying 2N·D_inf on top of it).")
    t = res["threshold"]
    print(f"\n  when does inference cost more than training?  {t['rule']}")
    print(f"    at the reference D = {human(ref['D'])} tokens: D_inf > {human(t['D_inf_threshold_at_reference'])};"
          f"  at the D_inf = 0 optimum (D = {human(res['base']['D'])}): D_inf > {human(t['D_inf_threshold_at_base_optimum'])}")
    return res


# ── (c) tiny ───────────────────────────────────────────────────────────────
def compute_tiny(cfg: dict) -> dict:
    c = cfg["chinchilla"]
    tp = os.path.join(HERE, cfg["tiny"]["train_json"])
    sp = os.path.join(HERE, cfg["tiny"]["tests_json"])
    tr = json.load(open(tp, encoding="utf-8"))
    te = json.load(open(sp, encoding="utf-8"))
    N = float(te["tests"]["a"]["count_params"])
    D = float(tr["tokens_seen"])
    C = 6.0 * N * D
    No, Do = optimum(C, c)
    return dict(N=N, D=D, ratio=D / N, C=C, steps=tr["train"]["steps"], batch=tr["train"]["batch"], L=tr["tiny"]["L"],
                V=te["tests"]["a"]["cfg"]["V"], seconds=tr.get("seconds"), N_opt_eq10=No, D_opt_eq10=Do, ratio_opt_eq10=Do / No,
                source=dict(train_json=os.path.relpath(tp, HERE), tests_json=os.path.relpath(sp, HERE), rehearsed=tr.get("date")))


def cmd_tiny(cfg: dict, res: dict | None = None) -> dict:
    res = res or compute_tiny(cfg)
    banner("demo 1 (c) · W4 demo 1's tiny GPT as a point on the table")
    print(f"  N = {res['N']:,.0f} parameters   (W4 tests.json (a) count_params; V = {res['V']}, character level)")
    print(f"  D = {res['D']:,.0f} tokens        ({res['steps']} steps × batch {res['batch']} × L = {res['L']}; W4 train.json tokens_seen)")
    print(f"  D/N = {res['ratio']:.1f}        C = 6ND = {sci(res['C'])} FLOPs")
    print(f"  Eq. (10)'s optimum at that C:  N* = {human(res['N_opt_eq10'])}, D* = {human(res['D_opt_eq10'])}, D*/N* = {res['ratio_opt_eq10']:.1f}")
    note("  Eq. (10) was fitted on MassiveText, subword-tokenised, with models from 70 M to over 16 B parameters (Hoffmann et al., abstract);")
    note("  a 65-symbol character-level model trained on one book is outside that range in every variable. The point sits on the table; it is not on the curve.")
    return res


# ── check against the slides' data file ────────────────────────────────────
def cmd_check(cfg: dict, results: dict | None = None, quiet: bool = False) -> dict:
    path = os.path.join(HERE, cfg["check"]["slides_json"])
    tol = float(cfg["check"]["rel_tol"])
    out = dict(slides_json=os.path.relpath(path, HERE), found=os.path.exists(path), compared=0, mismatches=[])
    if not out["found"]:
        if not quiet:
            note(f"  ({out['slides_json']} not found — nothing to compare against; the numbers above stand on their own)")
        return out
    sl = json.load(open(path, encoding="utf-8"))
    results = results or dict(table=compute_table(cfg), points=compute_points(cfg), inference=compute_inference(cfg), tiny=compute_tiny(cfg))

    def cmp(name, mine, theirs):
        out["compared"] += 1
        if theirs == 0:
            ok = mine == 0
        else:
            ok = abs(mine - theirs) / abs(theirs) <= tol
        if not ok:
            out["mismatches"].append(dict(name=name, demo=mine, slides=theirs))

    ch = sl.get("chinchilla", {})
    for k in ("a", "b", "G"):
        if k in ch:
            cmp(f"chinchilla.{k}", results["table"][k], ch[k])
    for mine, theirs in zip(results["table"]["rows"], ch.get("table", [])):
        for k in ("N", "D", "ratio", "loss"):
            cmp(f"table[C=1e{mine['log10_C']}].{k}", mine[k], theirs[k])
    for mine, theirs in zip(results["points"]["points"], ch.get("points", [])):
        for k in ("C", "loss", "N_opt", "D_opt", "loss_gap"):
            cmp(f"points[{mine['key']}].{k}", mine[k], theirs[k])
    inf = sl.get("inference", {})
    if "loss_fixed" in inf:
        cmp("inference.loss_fixed", results["inference"]["loss_fixed"], inf["loss_fixed"])
    for mine, theirs in zip(results["inference"]["rows"], inf.get("rows", [])):
        for k in ("N", "D", "ratio", "total"):
            cmp(f"inference[D_inf={mine['Dinf']:.0e}].{k}", mine[k], theirs[k])
        if "saving_vs_chinchilla_config" in theirs:
            cmp(f"inference[D_inf={mine['Dinf']:.0e}].saving", mine["saving_vs_base_shape"], theirs["saving_vs_chinchilla_config"])
    tiny = sl.get("tiny", {})
    for k in ("N", "D", "C"):
        if k in tiny:
            cmp(f"tiny.{k}", results["tiny"][k], tiny[k])
    if not quiet:
        banner("check · against the slides' data file", GREEN if not out["mismatches"] else RED)
        print(f"  {out['slides_json']}: {out['compared']} numbers compared, relative tolerance {tol:g}")
        if out["mismatches"]:
            for m in out["mismatches"]:
                print(f"  {RED}✗ {m['name']}: demo {m['demo']!r} vs slides {m['slides']!r}{RESET}")
        else:
            print(f"  {GREEN}✓ all equal{RESET}")
    return out


# ── rehearse / replay ──────────────────────────────────────────────────────
def cmd_rehearse(cfg: dict) -> None:
    outdir = os.path.join(HERE, cfg["rehearse"]["out"])
    os.makedirs(outdir, exist_ok=True)
    m = meta()
    res = dict(table=cmd_table(cfg), points=cmd_points(cfg), inference=cmd_inference(cfg, interactive=False), tiny=cmd_tiny(cfg))
    chk = cmd_check(cfg, res)
    for name, r in res.items():
        json.dump(dict(meta=m, label=f"rehearsal — demo 1 {name}", **r), open(os.path.join(outdir, f"{name}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    json.dump(dict(meta=m, **chk), open(os.path.join(outdir, "check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    banner("rehearsal written", GREEN)
    for f in ("table", "points", "inference", "tiny", "check"):
        print(f"  {os.path.relpath(os.path.join(outdir, f + '.json'), HERE)}")
    print(f"  {m['date']} · Python {m['python']} · {m['system']} {m['release']} · {m['machine']}")


def cmd_replay(cfg: dict, which: list[str]) -> None:
    outdir = os.path.join(HERE, cfg["rehearse"]["out"])
    which = which or ["table", "points", "inference", "tiny"]
    for name in which:
        p = os.path.join(outdir, f"{name}.json")
        if not os.path.exists(p):
            print(f"{RED}no rehearsal record {os.path.relpath(p, HERE)} — run ./present.sh rehearse first{RESET}")
            continue
        r = json.load(open(p, encoding="utf-8"))
        m = r.get("meta", {})
        banner(f"REPLAY — rehearsal record, not a live run · {m.get('date', '?')} · Python {m.get('python', '?')} · {m.get('system', '?')} {m.get('machine', '')}", AMBER)
        if name == "table":
            cmd_table(cfg, r, title="(replay) demo 1 (a) · N*, D* against C")
        elif name == "points":
            cmd_points(cfg, r)
        elif name == "inference":
            cmd_inference(cfg, r, interactive=False)
        elif name == "tiny":
            cmd_tiny(cfg, r)


# ── main ───────────────────────────────────────────────────────────────────
USAGE = __doc__.split("Sub-commands")[1].split("Standard library")[0]


def main(argv: list[str]) -> int:
    cfg = load_config()
    cmd = argv[0] if argv else "help"
    rest = argv[1:]
    if cmd == "table":
        cmd_table(cfg)
    elif cmd == "points":
        cmd_points(cfg)
    elif cmd == "inference":
        cmd_inference(cfg, interactive="--all" not in rest)
    elif cmd == "tiny":
        cmd_tiny(cfg)
    elif cmd == "check":
        out = cmd_check(cfg)
        return 1 if out["mismatches"] else 0
    elif cmd == "rehearse":
        cmd_rehearse(cfg)
    elif cmd == "replay":
        cmd_replay(cfg, rest)
    elif cmd in ("help", "-h", "--help"):
        print("usage: optimal.py <sub-command>\n" + USAGE)
    else:
        print(f"unknown sub-command: {cmd}\n" + USAGE, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print()
        sys.exit(130)
