#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 1：DPO——幾行 loss、五個測試、一次真的訓練（投影片 p33 前導頁、p35 (a)、p36 (b)）。

    ./present.sh live         課堂 (a)：dpo_live.py 的函數本體寫好之後跑；五個測試，Enter 一個（--ref 用參考實作；不載模型）
    ./present.sh replay-a     現場 (a) 失敗：印彩排的 tests.json
    ./present.sh test         不載權重：五個測試、規則與配對、亂數小模型上的梯度
    ./present.sh inspect      課前：版本、模型與 adapter、chat template、「B = 0 時輸出不變」、記憶體裡有幾份權重
    ./present.sh pairs        彩排第一步：π_ref 寫回答 → 規則配對 → π_ref 的兩個 log 機率存檔 → π_ref 的 bits per byte
    ./present.sh probe        彩排前：一步幾秒、LR 小掃描
    ./present.sh rehearse     彩排全套：(a) → pairs（已有就跳過）→ 每個 β 一個子行程（做過的跳過）→ 彙整
                              --only beta01 只跑一個；--force 重跑（連 pairs 一起）
    ./present.sh replay       課堂 (b)：讀 runs/rehearsal/summary.json，Enter 逐段印（不載模型）
    ./present.sh tiny         亂數權重的小模型走一遍真的流程（不下載任何東西；寫到 runs/smoke/）

(a) 只用 mlx（與測試 (v) 的 torch），不載模型。
(b) 的模型是上週 demo 1 的 Qwen3-1.7B-Base 加 adapter C——它就是 π_ref；新的 LoRA 旁路疊在上面、B 從零開始。
模型、旁路、取樣、整段回答的 log 機率都是 demo 2 的：grpo.py 與 demo 2 的 demo.py 從 ../w09_d2_grpo_scratch/ 以檔案路徑載入。
這裡寫的是 dpo_loss（dpo_live.py／dpo_final.py）、偏好規則與配對、訓練迴圈與要記的量（dpo.py）。

相依：mlx、numpy；測試 (v) 另需 torch；(b) 另需 mlx-lm（git commit 釘在 ../versions.lock）、transformers（tokenizer）、huggingface_hub。
replay 與 replay-a 只要標準函式庫。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10
    import tomli as tomllib   # type: ignore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info,  # noqa: E402
                    note, now, peak_memory_gb, versions)
import dpo as D  # noqa: E402

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
LOG2 = math.log(2)
CAVEAT = ("One 1.7B model, one hard-coded rule, one seed, a few hundred steps. The preferences are not human preferences: "
          "the model wrote both answers and a fixed rule chose. This is not a replication of any paper.")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_d2(cfg: dict):
    """demo 2 的 demo.py（Policy：模型、adapter C、旁路）以檔案路徑載入；它 import 的 grpo 就是這裡用的 G。"""
    src = (HERE / cfg["demo"]["source"]).resolve()
    if not (src / "demo.py").exists():
        raise SystemExit(f"{RED}找不到 demo 2：{cfg['demo']['source']}/demo.py{RESET}")
    mod = load_module(src / "demo.py", "w09_d2_demo")
    return mod, mod.load_cfg(), src


def out_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "rehearsal")


def cache_dir(args) -> Path:
    return RUNS / "cache" / ("smoke" if args.tiny else "rehearsal")


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


def r6(x):
    return None if x is None else float(f"{x:.6g}")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, spec=".3f", none="—"):
    return none if x is None else format(x, spec)


def clean(text: str, limit: int) -> str:
    t = " ".join(text.replace("<think>", "").replace("</think>", "").split())
    return t if len(t) <= limit else t[:limit] + " …"


# ══ (a) 五個測試：不載模型 ═══════════════════════════════════════
def live_inputs(cfg: dict):
    """編出來的四組 log 機率。每個數都是 1/8 的倍數，所以加減整數在 float32 裡沒有捨入。"""
    lv = cfg["live"]
    rng = random.Random(int(cfg["demo"]["seed"]))
    n = int(lv["n_pairs"])
    ref_w = [-rng.randint(160, 480) / 8 for _ in range(n)]
    ref_l = [-rng.randint(320, 960) / 8 for _ in range(n)]
    d_w = [rng.randint(-24, 24) / 8 for _ in range(n)]
    d_l = [rng.randint(-24, 24) / 8 for _ in range(n)]
    d_w[0], d_l[0] = 2.0, -1.5            # 第一對一定排對
    d_w[1], d_l[1] = -1.25, 2.5           # 第二對一定排錯
    logp_w = [a + b for a, b in zip(ref_w, d_w)]
    logp_l = [a + b for a, b in zip(ref_l, d_l)]
    return logp_w, logp_l, ref_w, ref_l


def pick_loss(args, rehearsal: bool = False):
    if args.ref or rehearsal:
        import dpo_final as mod
        return mod.dpo_loss, "dpo_final.py"
    import dpo_live as mod
    return mod.dpo_loss, "dpo_live.py"


def five_tests(dpo_loss, cfg: dict, step=None) -> dict:
    """回傳每個測試的數字與 ok。step(標題) 在每個測試之前被呼叫（課堂上是「按 Enter」）。"""
    import mlx.core as mx
    lv = cfg["live"]
    beta = float(lv["beta"])
    lw, ll, rw, rl = (mx.array(v) for v in live_inputs(cfg))
    n = lw.shape[0]
    step = step or (lambda title: None)

    def L(a, b, c, d, bt) -> float:
        return float(dpo_loss(a, b, c, d, bt).item())

    rows = {"n_pairs": n, "beta": beta, "log2": LOG2}

    step("(i)   the policy is the reference")
    at_ref = L(rw, rl, rw, rl, beta)
    g = mx.grad(lambda a: dpo_loss(a, rl, rw, rl, beta))(rw)
    weights = [float(-x * n / beta) for x in g.tolist()]
    rows["i"] = {"loss": at_ref, "weights": weights, "ok": abs(at_ref - LOG2) < 1e-6 and all(abs(w - 0.5) < 1e-6 for w in weights)}
    print(f"  {'loss':<44}{BOLD}{at_ref:.6f}{RESET}   {GREY}log 2 = {LOG2:.6f}{RESET}")
    print(f"  {'weight on each pair (from the gradient)':<44}{BOLD}{', '.join(f'{w:.3f}' for w in weights)}{RESET}")

    step("(ii)  raise logp_w")
    up = float(lv["raise_by"])
    base = L(lw, ll, rw, rl, beta)
    raised = L(lw + up, ll, rw, rl, beta)
    rows["ii"] = {"raise_by": up, "before": base, "after": raised, "ok": raised < base}
    print(f"  {'loss before':<44}{BOLD}{base:.6f}{RESET}")
    print(f"  {f'loss after adding {up:g} to every logp_w':<44}{BOLD}{raised:.6f}{RESET}   {GREY}{'lower' if raised < base else 'NOT lower'}{RESET}")

    step("(iii) add the same constant to logp_w and logp_l")
    shifted = []
    for c in lv["shifts"]:
        v = L(lw + float(c), ll + float(c), rw, rl, beta)
        shifted.append({"c": float(c), "loss": v, "difference": v - base})
        print(f"  {f'loss with both shifted by {float(c):+g}':<44}{BOLD}{v:.6f}{RESET}   {GREY}difference from {base:.6f}: {v - base:g}{RESET}")
    rows["iii"] = {"before": base, "shifted": shifted, "ok": all(abs(s["difference"]) < 1e-6 for s in shifted)}

    step("(iv)  the two ends of beta")
    b0, b1 = float(lv["beta_small"]), float(lv["beta_large"])
    small = L(lw, ll, rw, rl, b0)
    g0 = mx.grad(lambda a: dpo_loss(a, ll, rw, rl, b0))(lw)
    gnorm = float(mx.abs(g0).sum().item())
    right = L(lw[0:1], ll[0:1], rw[0:1], rl[0:1], b1)
    wrong = L(lw[1:2], ll[1:2], rw[1:2], rl[1:2], b1)
    m_right = float(((lw[0] - rw[0]) - (ll[0] - rl[0])).item())
    m_wrong = float(((lw[1] - rw[1]) - (ll[1] - rl[1])).item())
    finite = math.isfinite(right) and math.isfinite(wrong)
    rows["iv"] = {"beta_small": b0, "loss_small": small, "grad_abs_sum_small": gnorm, "beta_large": b1, "loss_pair_ranked_right": right,
                  "loss_pair_ranked_wrong": wrong if finite else None, "gap_right": m_right, "gap_wrong": m_wrong, "finite": finite,
                  "ok": finite and abs(small - LOG2) < 1e-3 and gnorm < 1e-3 and right < 1e-6 and abs(wrong - b1 * abs(m_wrong)) < 1e-3}
    print(f"  {f'beta = {b0:g}: loss':<44}{BOLD}{small:.6f}{RESET}   {GREY}log 2 = {LOG2:.6f}; Σ|∂loss/∂logp_w| = {gnorm:.2g}{RESET}")
    print(f"  {f'beta = {b1:g}: a pair ranked right (gap {m_right:+.2f})':<44}{BOLD}{right:.6f}{RESET}")
    print(f"  {f'beta = {b1:g}: a pair ranked wrong (gap {m_wrong:+.2f})':<44}{BOLD}{wrong:.6f}{RESET}   {GREY}beta × |gap| = {b1 * abs(m_wrong):.6f}{RESET}")
    if not finite:
        print(f"  {RED}not finite: log(sigmoid(·)) hits log 0 when the bracket is very negative; use nn.log_sigmoid{RESET}")

    step("(v)   the DPO paper's appendix code, same inputs")
    try:
        import torch
        from paper_reference import dpo_loss_paper
        pol = torch.tensor(lw.tolist() + ll.tolist(), dtype=torch.float32)
        ref = torch.tensor(rw.tolist() + rl.tolist(), dtype=torch.float32)
        wi, li = torch.arange(n), torch.arange(n, 2 * n)
        per_pair, _ = dpo_loss_paper(pol, ref, wi, li, beta)
        theirs = float(per_pair.mean())
        rows["v"] = {"ours": base, "paper": theirs, "difference": base - theirs, "torch": torch.__version__, "ok": abs(base - theirs) < 1e-6}
        print(f"  {'dpo_loss (mlx), mean over the batch':<44}{BOLD}{base:.8f}{RESET}")
        print(f"  {'the appendix formulation (PyTorch), mean':<44}{BOLD}{theirs:.8f}{RESET}   {GREY}difference {base - theirs:.1e}{RESET}")
    except ModuleNotFoundError:
        rows["v"] = {"ours": base, "paper": None, "difference": None, "torch": None, "ok": None}
        print(f"  {AMBER}torch is not installed here: test (v) skipped{RESET}")
    rows["all_ok"] = all(rows[k]["ok"] is not False for k in ("i", "ii", "iii", "iv", "v"))
    return rows


def verdicts(rows: dict) -> None:
    print()
    for k, label in (("i", "(i)"), ("ii", "(ii)"), ("iii", "(iii)"), ("iv", "(iv)"), ("v", "(v)")):
        ok = rows[k]["ok"]
        mark = f"{AMBER}skipped{RESET}" if ok is None else (f"{GREEN}ok{RESET}" if ok else f"{RED}FAIL{RESET}")
        print(f"  {label:<6}{mark}")


def cmd_live(args, cfg, rehearsal: bool = False):
    import mlx.core as mx
    dpo_loss, src = pick_loss(args, rehearsal)
    try:                                                   # 先確認函數本體寫了
        dpo_loss(mx.array([0.0]), mx.array([0.0]), mx.array([0.0]), mx.array([0.0]), 1.0)
    except NotImplementedError:
        raise SystemExit(f"{AMBER}dpo_loss() 還是空的：在 dpo_live.py 寫函數本體，或 ./present.sh live --ref{RESET}")
    banner(f"{cfg['demo']['title']}\n(a) five tests of dpo_loss() from {src} · {cfg['live']['n_pairs']} made-up pairs, beta = {cfg['live']['beta']}")
    lw, ll, rw, rl = live_inputs(cfg)
    print(f"  {'pair':<6}{'logp_w':>10}{'logp_l':>10}{'ref_w':>10}{'ref_l':>10}")
    for i in range(len(lw)):
        print(f"  {i + 1:<6}{lw[i]:>10.3f}{ll[i]:>10.3f}{rw[i]:>10.3f}{rl[i]:>10.3f}")

    def step(title: str) -> None:
        if rehearsal:
            print(f"\n  {AMBER}{title}{RESET}")
        else:
            pause(f"\n  {title}  [Enter] ")

    rows = five_tests(dpo_loss, cfg, step)
    verdicts(rows)
    rec = {"date": now(), "versions": versions(), "source": src, "inputs": {"logp_w": lw, "logp_l": ll, "ref_w": rw, "ref_l": rl}, "rows": rows}
    out = (out_dir(args) if rehearsal else RUNS / "live") / "tests.json"
    write_json(out, rec)
    note(f"\n  → {out.relative_to(HERE)}")
    return rec


def cmd_replay_a(args, cfg):
    p = out_dir(args) / "tests.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    d = read_json(p)
    r = d["rows"]
    banner(f"{cfg['demo']['title']}\n(a) REPLAY of the rehearsal of {d['date'][:10]} · dpo_loss() from {d['source']}")
    pause("\n  (i)   the policy is the reference  [Enter] ")
    print(f"  {'loss':<44}{BOLD}{r['i']['loss']:.6f}{RESET}   {GREY}log 2 = {r['log2']:.6f}{RESET}")
    print(f"  {'weight on each pair (from the gradient)':<44}{BOLD}{', '.join(f'{w:.3f}' for w in r['i']['weights'])}{RESET}")
    pause("\n  (ii)  raise logp_w  [Enter] ")
    print(f"  {'loss before':<44}{BOLD}{r['ii']['before']:.6f}{RESET}")
    print(f"  {'loss after adding ' + format(r['ii']['raise_by'], 'g') + ' to every logp_w':<44}{BOLD}{r['ii']['after']:.6f}{RESET}")
    pause("\n  (iii) add the same constant to logp_w and logp_l  [Enter] ")
    for s in r["iii"]["shifted"]:
        print(f"  {'loss with both shifted by ' + format(s['c'], '+g'):<44}{BOLD}{s['loss']:.6f}{RESET}   {GREY}difference: {s['difference']:g}{RESET}")
    pause("\n  (iv)  the two ends of beta  [Enter] ")
    v = r["iv"]
    print(f"  {'beta = ' + format(v['beta_small'], 'g') + ': loss':<44}{BOLD}{v['loss_small']:.6f}{RESET}   {GREY}Σ|∂loss/∂logp_w| = {v['grad_abs_sum_small']:.2g}{RESET}")
    print(f"  {'beta = ' + format(v['beta_large'], 'g') + ': a pair ranked right':<44}{BOLD}{v['loss_pair_ranked_right']:.6f}{RESET}")
    print(f"  {'beta = ' + format(v['beta_large'], 'g') + ': a pair ranked wrong':<44}{BOLD}{fmt(v['loss_pair_ranked_wrong'], '.6f')}{RESET}   "
          f"{GREY}beta × |gap| = {v['beta_large'] * abs(v['gap_wrong']):.6f}{RESET}")
    pause("\n  (v)   the DPO paper's appendix code, same inputs  [Enter] ")
    print(f"  {'dpo_loss (mlx)':<44}{BOLD}{r['v']['ours']:.8f}{RESET}")
    print(f"  {'the appendix formulation (PyTorch, CPU)':<44}{BOLD}{fmt(r['v']['paper'], '.8f')}{RESET}   {GREY}difference {fmt(r['v']['difference'], '.1e')}{RESET}")
    verdicts(r)


# ══ (b) 模型、prompt、偏好對 ═════════════════════════════════════
def d2_args(args):
    return argparse.Namespace(tiny=args.tiny, level=None, online=args.online, steps=None)


def encode_prompt(policy, text: str) -> list[int]:
    t = policy.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return list(policy.tok.encode(t, add_special_tokens=False))


def pcfg(cfg: dict, args) -> dict:
    """[pairs] 與 [data]；--tiny 換成小的。"""
    p = dict(cfg["pairs"])
    p.update({k: cfg["data"][k] for k in ("n_heldout_pair_prompts", "n_gold_prompts", "max_prompt_tokens")})
    if args.tiny:
        t = cfg["tiny"]
        p.update(max_new_tokens=int(t["max_new_tokens"]), min_gap_tokens=int(t["min_gap_tokens"]),
                 n_heldout_pair_prompts=int(t["n_heldout_pair_prompts"]), n_gold_prompts=int(t["n_gold_prompts"]))
    return p


def load_prompts(cfg: dict, args, policy) -> dict:
    """三組互不重疊的 prompt：train（做訓練用的偏好對）、heldout（做沒訓過的偏好對）、gold（只取樣量長度）。"""
    pc = pcfg(cfg, args)
    if args.tiny:
        rng = random.Random(int(cfg["demo"]["seed"]))
        seen, texts = set(), []
        n_all = int(cfg["tiny"]["n_train_prompts"]) + pc["n_heldout_pair_prompts"] + pc["n_gold_prompts"]
        while len(texts) < n_all:
            t = f"{rng.randint(0, 99)}{rng.choice('+-×')}{rng.randint(0, 99)}="
            if t not in seen:
                seen.add(t)
                texts.append(t)
        n_tr = int(cfg["tiny"]["n_train_prompts"])
        split = {"train": [("tiny", i, t) for i, t in enumerate(texts[:n_tr])],
                 "heldout": [("tiny", i, t) for i, t in enumerate(texts[n_tr:n_tr + pc["n_heldout_pair_prompts"]], n_tr)],
                 "gold": [("tiny", i, t) for i, t in enumerate(texts[n_tr + pc["n_heldout_pair_prompts"]:], n_tr + pc["n_heldout_pair_prompts"])]}
    else:
        d = (HERE / cfg["data"]["dir"]).resolve()

        def rows(name: str):
            p = d / name
            if not p.exists():
                raise SystemExit(f"{RED}沒有 {cfg['data']['dir']}/{name}：先在上週 demo 1 跑 ./present.sh prepare{RESET}")
            return [(name, i, json.loads(line)["messages"][0]["content"]) for i, line in enumerate(open(p, encoding="utf-8")) if line.strip()]

        held = rows(cfg["data"]["heldout_file"])
        a, b = pc["n_heldout_pair_prompts"], pc["n_heldout_pair_prompts"] + pc["n_gold_prompts"]
        split = {"train": rows(cfg["data"]["train_file"]), "heldout": held[:a], "gold": held[a:b]}
    out, dropped = {}, {}
    for k, items in split.items():
        out[k], dropped[k] = [], 0
        for name, i, text in items:
            ids = encode_prompt(policy, text)
            if len(ids) > int(pc["max_prompt_tokens"]):
                dropped[k] += 1
                continue
            out[k].append({"id": f"{name}:{i}", "text": text, "ids": ids})
    out["dropped_too_long"] = dropped
    return out


def sample_answers(D2, policy, prompts: list[dict], n: int, max_new: int, temperature: float, batch_size: int, seed: int) -> tuple[list[list[dict]], dict]:
    """每個 prompt 取樣 n 段。回傳 groups[p][i] = {prompt, tokens（含結束 token）, finish, logprob, text}。"""
    import mlx.core as mx
    G = D2.G
    mx.random.seed(seed)
    flat = [p["ids"] for p in prompts for _ in range(n)]
    t0 = time.perf_counter()
    res, info = G.sample(policy.model, policy.tok, flat, max_new, temperature, batch_size)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    groups = []
    for pi, p in enumerate(prompts):
        grp = res[pi * n:(pi + 1) * n]
        for r in grp:
            r["prompt"] = p["ids"]
            r["text"] = G.response_text(policy.tok, r)
        groups.append(grp)
    return groups, info


def ruler_texts(cfg: dict, args, policy) -> tuple[list[dict], int, int]:
    """上週 demo 4 (b) 的四段 held-out（它的 load_texts 與設定以檔案路徑載入）。回傳 (texts, prefix token, window)。"""
    if args.tiny:
        text = ("12+34=46 7×8=56 90-45=45 " * 12).strip()
        return [{"key": "tiny", "label": "tiny text", "ids": list(policy.tok.encode(text)), "n_bytes": len(text.encode("utf-8"))}], 0, 64
    rc = cfg["ruler"]
    src = (HERE / rc["source"]).resolve()
    d4 = load_module(src / "demo.py", "w08_d4_demo")
    cfg4 = d4.load_cfg()
    cfg4["forgetting"]["max_bytes"] = int(rc["max_bytes"])
    tok = policy.tok
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    prefix = int(tok.bos_token_id) if getattr(tok, "bos_token_id", None) is not None else int(vocab["<|endoftext|>"])
    out = []
    for t in d4.load_texts(cfg4):
        if t["key"] in rc["keys"]:
            ids = list(tok.encode(t["bytes"].decode("utf-8"), add_special_tokens=False))
            out.append({"key": t["key"], "label": t["label"], "ids": ids, "n_bytes": t["n_bytes"]})
    return out, prefix, int(cfg4["forgetting"]["window_tokens"])


def measure_ruler(cfg: dict, args, policy, echo: bool = True) -> dict:
    texts, prefix, window = ruler_texts(cfg, args, policy)
    out = {}
    for t in texts:
        t0 = time.perf_counter()
        r = D.bits_per_byte(policy.model, prefix, t["ids"], t["n_bytes"], window)
        r.update(label=t["label"], bpb=round(r["bpb"], 4), seconds=round(time.perf_counter() - t0, 1))
        out[t["key"]] = r
        if echo:
            print(f"    {t['label']:<42}{BOLD}{r['bpb']:.4f}{RESET} bits/byte  {GREY}{r['tokens']:,} tokens, {r['windows']} windows, {r['seconds']} s{RESET}")
    return out


def cmd_pairs(args, cfg):
    D2, cfg2, _ = load_d2(cfg)
    G = D2.G
    pc = pcfg(cfg, args)
    seed = int(cfg["demo"]["seed"])
    banner(f"demo 1 (b) · pairs — π_ref writes {pc['samples_per_prompt']} answers per prompt; the rule: {D.RULE}", AMBER)
    policy = D2.load_policy(cfg2, d2_args(args))
    prompts = load_prompts(cfg, args, policy)
    print(f"  prompts: {len(prompts['train'])} for training pairs, {len(prompts['heldout'])} for held-out pairs, {len(prompts['gold'])} kept for sampling only"
          f"   {GREY}(dropped as longer than {pc['max_prompt_tokens']} tokens: {prompts['dropped_too_long']}){RESET}")
    n, max_new, temp, bs = int(pc["samples_per_prompt"]), int(pc["max_new_tokens"]), float(pc["temperature"]), int(pc["batch_size"])
    pairs, all_samples, sample_info, per_split, examples = [], [], {}, {}, []
    for k, split in enumerate(("train", "heldout")):
        groups, info = sample_answers(D2, policy, prompts[split], n, max_new, temp, bs, seed + k)
        sample_info[split] = info
        flat = [r for g in groups for r in g]
        all_samples += flat
        made = 0
        for p, grp in zip(prompts[split], groups):
            for w, l in D.make_pairs(grp, int(pc["pairs_per_prompt"]), int(pc["min_gap_tokens"])):
                pairs.append({"split": split, "prompt_id": p["id"], "prompt": p["ids"], "w": grp[w]["tokens"], "l": grp[l]["tokens"],
                              "l_finish": grp[l]["finish"], "sample_logp_w": grp[w]["logprob"], "sample_logp_l": grp[l]["logprob"]})
                made += 1
                if split == "train" and len(examples) < int(pc["n_examples"]):
                    examples.append({"prompt": clean(p["text"], int(pc["max_chars"])),
                                     "y_w": {"text": clean(grp[w]["text"], int(pc["max_chars"])), "tokens": len(grp[w]["tokens"]), "finish": grp[w]["finish"]},
                                     "y_l": {"text": clean(grp[l]["text"], int(pc["max_chars"])), "tokens": len(grp[l]["tokens"]), "finish": grp[l]["finish"]}})
        g = D.gold(flat)
        per_split[split] = {"prompts": len(prompts[split]), "samples": len(flat), "pairs": made, "ended": r4(g["ended"]), "mean_len": r4(g["mean_len"]),
                            "sampling_seconds": info["seconds"], "generation_tokens": info["generation_tokens"]}
        print(f"  {split:<8} {len(flat)} answers in {info['seconds']} s: {g['ended']:.0%} end properly, mean length {g['mean_len']:.1f} tokens → {BOLD}{made}{RESET} pairs")
    if not any(p["split"] == "train" for p in pairs) or not any(p["split"] == "heldout" for p in pairs):
        raise SystemExit(f"{RED}規則沒有配出偏好對（train 或 heldout 是空的）：調 [pairs] 的 max_new_tokens／min_gap_tokens／samples_per_prompt{RESET}")

    # π_ref 的兩個 log 機率，事先算好。B = 0，所以旁路開或關是同一個模型；兩種都算，應該逐位元相同。
    t0 = time.perf_counter()
    rw, rl = D.pair_logprobs(G, policy.model, pairs, ref=True, chunk=8)
    t_ref = time.perf_counter() - t0
    ow, ol = D.pair_logprobs(G, policy.model, pairs, ref=False, chunk=8)
    aw, al = D.pair_logprobs(G, policy.model, pairs[::-1], ref=True, chunk=3)         # 換一種分批：只差 padding 與批次裡的捨入
    aw, al = aw[::-1], al[::-1]
    for p, a, b in zip(pairs, rw, rl):
        p["ref_w"], p["ref_l"] = a, b
    diag = {"bypass_on_vs_off_max_abs": r6(max(abs(a - b) for a, b in zip(rw + rl, ow + ol))),
            "rebatched_max_abs": r6(max(abs(a - b) for a, b in zip(rw + rl, aw + al))),
            "rebatched_mean_abs": r6(mean([abs(a - b) for a, b in zip(rw + rl, aw + al)])),
            "sampling_vs_recomputed_max_abs": r6(max(abs(p["sample_logp_w"] - p["ref_w"]) for p in pairs)),
            "ref_forward_seconds": round(t_ref, 1)}
    print(f"\n  π_ref's log-probabilities for {len(pairs)} pairs: {t_ref:.1f} s")
    print(f"  {'bypass on vs. off at B = 0, max |Δ|':<52}{BOLD}{diag['bypass_on_vs_off_max_abs']:g}{RESET}   {GREY}(0: the policy starts as π_ref){RESET}")
    print(f"  {'the same numbers from differently padded batches':<52}{BOLD}{diag['rebatched_max_abs']:g}{RESET} max, {diag['rebatched_mean_abs']:g} mean   "
          f"{GREY}(nats per answer; how far step 0 can sit from log 2 with stored references){RESET}")
    print(f"  {'recomputed vs. the sum recorded while sampling':<52}{BOLD}{diag['sampling_vs_recomputed_max_abs']:g}{RESET} max")

    def stats(split: str) -> dict:
        ps = [p for p in pairs if p["split"] == split]
        lw_, ll_ = [len(p["w"]) for p in ps], [len(p["l"]) for p in ps]
        return {"pairs": len(ps), "mean_len_w": r4(mean(lw_)), "mean_len_l": r4(mean(ll_)),
                "l_cut_at_limit": r4(mean([1.0 if p["l_finish"] != "stop" else 0.0 for p in ps])),
                "ref_logp_w": r4(mean([p["ref_w"] for p in ps])), "ref_logp_l": r4(mean([p["ref_l"] for p in ps])),
                "ref_logp_w_per_token": r4(sum(p["ref_w"] for p in ps) / sum(lw_)), "ref_logp_l_per_token": r4(sum(p["ref_l"] for p in ps) / sum(ll_))}

    st = {s: stats(s) for s in ("train", "heldout")}
    for s in ("train", "heldout"):
        x = st[s]
        print(f"  {s:<8} {x['pairs']} pairs: y_w {x['mean_len_w']:.1f} tokens, y_l {x['mean_len_l']:.1f} ({x['l_cut_at_limit']:.0%} of y_l cut at the limit); "
              f"log π_ref {x['ref_logp_w']:.1f} / {x['ref_logp_l']:.1f} nats per answer ({x['ref_logp_w_per_token']:.2f} / {x['ref_logp_l_per_token']:.2f} per token)")

    print(f"\n  π_ref on the ruler (bits per byte, before any DPO step):")
    ruler = measure_ruler(cfg, args, policy)
    wb = D.weight_bytes(policy.model)
    rec = {"date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "seed": seed,
           "model": None if args.tiny else cfg2["model"]["repo"], "adapter": None if args.tiny else cfg2["model"]["adapter"],
           "rule": D.RULE, "settings": {k: pc[k] for k in ("samples_per_prompt", "pairs_per_prompt", "max_new_tokens", "temperature", "min_gap_tokens", "max_prompt_tokens")},
           "prompts": {"train_file": None if args.tiny else cfg["data"]["train_file"], "heldout_file": None if args.tiny else cfg["data"]["heldout_file"],
                       "gold": len(prompts["gold"]), "dropped_too_long": prompts["dropped_too_long"]},
           "sampled": per_split, "pairs": st, "diagnostics": diag, "ruler_ref": ruler, "weights": wb, "peak_gb": r4(peak_memory_gb()),
           "examples": examples,
           "per_pair": [{"split": p["split"], "prompt_id": p["prompt_id"], "len_w": len(p["w"]), "len_l": len(p["l"]), "l_finish": p["l_finish"],
                         "ref_w": r6(p["ref_w"]), "ref_l": r6(p["ref_l"])} for p in pairs]}
    write_json(out_dir(args) / "pairs.json", rec)
    write_json(cache_dir(args) / "pairs_tokens.json", {"date": rec["date"], "pairs": pairs, "gold_prompts": [{"id": p["id"], "ids": p["ids"]} for p in prompts["gold"]]})
    note(f"\n  → {(out_dir(args) / 'pairs.json').relative_to(HERE)}（摘要與幾對的文字，進版控）; {(cache_dir(args) / 'pairs_tokens.json').relative_to(HERE)}（token id，不進版控）")
    return rec


def load_pairs(args) -> tuple[dict, dict]:
    p, c = out_dir(args) / "pairs.json", cache_dir(args) / "pairs_tokens.json"
    if not p.exists() or not c.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)} 或 {c.relative_to(HERE)}：先跑 ./present.sh pairs{RESET}")
    meta, tokens = read_json(p), read_json(c)
    if meta["date"] != tokens["date"]:
        raise SystemExit(f"{RED}pairs.json 與 runs/cache 的 token 不是同一次 pairs 產生的：重跑 ./present.sh pairs{RESET}")
    return meta, tokens


# ══ (b) 訓練 ═════════════════════════════════════════════════════
def run_spec(cfg: dict, args, name: str) -> dict:
    if name not in cfg["runs"]:
        raise SystemExit(f"{RED}沒有這個 run：{name}（demo_config.toml [runs]：{', '.join(cfg['runs'])}）{RESET}")
    s = dict(cfg["train"])
    s.update(cfg["runs"][name])
    if args.tiny:
        t = cfg["tiny"]
        s.update(steps=int(t["steps"]), learning_rate=float(t["learning_rate"]), eval_every=int(t["eval_every"]))
    if args.steps:
        s["steps"] = int(args.steps)
    if args.lr:
        s["learning_rate"] = float(args.lr)
    return s


def eval_pairs(G, model, pairs: list[dict], beta: float) -> tuple[dict, list[float], list[float]]:
    lw, ll = D.pair_logprobs(G, model, pairs, ref=False, chunk=8)
    m = D.pair_metrics(lw, ll, [p["ref_w"] for p in pairs], [p["ref_l"] for p in pairs], beta, [len(p["w"]) for p in pairs], [len(p["l"]) for p in pairs])
    return m, lw, ll


def eval_gold(D2, policy, cfg: dict, args, gold_prompts: list[dict], max_new: int, temperature: float) -> tuple[dict, list[dict]]:
    """沒看過的 prompt 上取樣：規則看的兩件事（長度、有沒有正常結束），與對 π_ref 的 KL。固定取樣種子。"""
    e = cfg["eval"]
    groups, info = sample_answers(D2, policy, gold_prompts, int(e["samples_per_prompt"]), max_new, temperature, int(e["batch_size"]), int(e["seed"]))
    flat = [r for g in groups for r in g]
    g = D.gold(flat)
    out = {"n": g["n"], "ended": g["ended"], "mean_len": g["mean_len"], "seconds": info["seconds"]}
    out.update(D.kl_estimate(D2.G, policy.model, flat))
    ex = [{"answer": clean(grp[0]["text"], int(e["max_chars"])), "tokens": len(grp[0]["tokens"]), "finish": grp[0]["finish"]}
          for grp in groups[:int(e["n_examples"])]]
    return out, ex


def train_run(D2, policy, cfg: dict, args, spec: dict, meta: dict, tokens: dict, echo: bool = True, full_eval: bool = True,
              train_subset: int | None = None) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import dpo_final
    G = D2.G
    model = policy.model
    beta, steps, B = float(spec["beta"]), int(spec["steps"]), int(spec["batch_pairs"])
    online = spec["ref"] == "online"
    seed = int(cfg["demo"]["seed"])
    train = [p for p in tokens["pairs"] if p["split"] == "train"]
    held = [p for p in tokens["pairs"] if p["split"] == "heldout"]
    watch = train if train_subset is None else train[:train_subset]
    gold_prompts = tokens["gold_prompts"]
    max_new, temperature = int(meta["settings"]["max_new_tokens"]), float(meta["settings"]["temperature"])
    opt = optim.Adam(learning_rate=float(spec["learning_rate"]))
    vg = nn.value_and_grad(model, D.training_loss(G, dpo_final.dpo_loss))
    rec_steps, evals, examples = [], [], {}
    start_lw = start_ll = None
    cum = {"t_update": 0.0, "t_ref": 0.0}
    t_start = time.time()

    def do_eval(step: int) -> None:
        nonlocal start_lw, start_ll
        t0 = time.perf_counter()
        m_tr, lw, ll = eval_pairs(G, model, watch, beta)
        if start_lw is None:
            start_lw, start_ll = lw, ll
        e = {"step": step, "train": {k: r6(v) for k, v in m_tr.items()}}
        e["train"]["change_w"] = {k: r6(v) for k, v in D.paired_change(lw, start_lw).items()}
        e["train"]["change_l"] = {k: r6(v) for k, v in D.paired_change(ll, start_ll).items()}
        if full_eval:
            m_he, _, _ = eval_pairs(G, model, held, beta)
            e["heldout"] = {k: r6(v) for k, v in m_he.items()}
            g, ex = eval_gold(D2, policy, cfg, args, gold_prompts, max_new, temperature)
            e["gold"] = {k: r4(v) for k, v in g.items()}
            examples[str(step)] = ex
        e["seconds"] = round(time.perf_counter() - t0, 1)
        evals.append(e)
        if echo:
            t = e["train"]
            tail = (f"   held-out loss {e['heldout']['loss']:.4f}   sampled length {e['gold']['mean_len']:.1f}, ended {e['gold']['ended']:.0%}, "
                    f"√KL {e['gold']['sqrt_kl']:.3f}") if full_eval else ""
            print(f"  {AMBER}eval{RESET} step {step:>4}   loss {BOLD}{t['loss']:.4f}{RESET}   log π(y_w) {BOLD}{t['logp_w']:.2f}{RESET} ({t['change_w']['mean']:+.2f})   "
                  f"log π(y_l) {BOLD}{t['logp_l']:.2f}{RESET} ({t['change_l']['mean']:+.2f}){tail}   {GREY}{e['seconds']} s{RESET}")

    do_eval(0)
    every = int(spec.get("eval_every", 0))
    order: list[int] = []
    epoch = 0
    for step in range(1, steps + 1):
        if len(order) < B:                                       # 新的一輪：洗牌
            fresh = list(range(len(train)))
            random.Random(seed * 1009 + epoch).shuffle(fresh)
            order += fresh
            epoch += 1
        idx, order = order[:B], order[B:]
        part = [train[i] for i in idx]
        batch, mask = D.pair_batch(G, part)
        t_r = 0.0
        if online:                                               # π_ref：同一個 batch、旁路關掉
            t0 = time.perf_counter()
            ref = D.sequence_logp(G, model, batch, mask, ref=True)
            ref_w, ref_l = ref[:len(part)], ref[len(part):]
            mx.eval(ref_w, ref_l)
            t_r = time.perf_counter() - t0
        else:                                                    # π_ref：pairs 那一步存下來的數
            ref_w = mx.array([p["ref_w"] for p in part])
            ref_l = mx.array([p["ref_l"] for p in part])
        t0 = time.perf_counter()
        (loss, lp), grads = vg(model, batch, mask, ref_w, ref_l, beta)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss, lp)
        t_u = time.perf_counter() - t0
        cum["t_update"] += t_u
        cum["t_ref"] += t_r
        lpl = lp.tolist()
        n = len(part)
        lw, ll = lpl[:n], lpl[n:]
        rw, rl = ref_w.tolist(), ref_l.tolist()
        loss_v = float(loss.item())
        rec = {"step": step, "loss": r6(loss_v), "margin": r6(mean([beta * ((a - c) - (b - d)) for a, b, c, d in zip(lw, ll, rw, rl)])),
               "logp_w_minus_ref": r6(mean([a - c for a, c in zip(lw, rw)])), "logp_l_minus_ref": r6(mean([b - d for b, d in zip(ll, rl)])),
               "tokens_in_batch": int(batch.size), "t_update": round(t_u, 2), "t_ref": round(t_r, 2), "peak_gb": r4(peak_memory_gb())}
        rec_steps.append(rec)
        if not math.isfinite(loss_v):
            print(f"  {RED}loss 不是有限值（step {step}），停{RESET}")
            break
        if echo and (step % 5 == 0 or step == 1):
            print(f"  step {step:>4}   loss {loss_v:.4f}   log π − log π_ref on this batch: y_w {rec['logp_w_minus_ref']:+.3f}, y_l {rec['logp_l_minus_ref']:+.3f}   "
                  f"{GREY}update {t_u:.1f} s{f' · π_ref {t_r:.1f} s' if online else ''} · peak {rec['peak_gb'] or 0:.1f} GB{RESET}")
        if every and (step % every == 0 or step == steps):
            do_eval(step)
    if evals[-1]["step"] != rec_steps[-1]["step"]:
        do_eval(rec_steps[-1]["step"])

    first, last = evals[0], evals[-1]
    cw, cl = last["train"]["change_w"], last["train"]["change_l"]
    dw, dl = D.direction(cw), D.direction(cl)
    loss_fell = last["train"]["loss"] < first["train"]["loss"]
    n = len(rec_steps)
    summary = {"beta": beta, "learning_rate": float(spec["learning_rate"]), "ref": spec["ref"], "batch_pairs": B, "steps_done": n,
               "epochs": r4(n * B / len(train)), "train_pairs": len(train), "watched_train_pairs": len(watch), "heldout_pairs": len(held),
               "start": {k: first.get(k) for k in ("train", "heldout", "gold")}, "end": {k: last.get(k) for k in ("train", "heldout", "gold")},
               "first_batch_loss": rec_steps[0]["loss"], "loss_fell": loss_fell, "y_w": dw, "y_l": dl,
               "displacement": bool(loss_fell and dw == "fell"),
               "reading": f"the loss {'fell' if loss_fell else 'did not fall'}; log π(y_w) {dw} ({cw['mean']:+.2f} ± {cw['se']:.2f} nats per answer, lower on {cw['fell']:.0%} of pairs); "
                          f"log π(y_l) {dl} ({cl['mean']:+.2f} ± {cl['se']:.2f})",
               "t_update_per_step": r4(cum["t_update"] / n), "t_ref_per_step": r4(cum["t_ref"] / n),
               "peak_gb": r4(peak_memory_gb()), "seconds": round(time.time() - t_start, 1)}
    out = {"summary": summary, "evals": evals, "steps": rec_steps}
    if full_eval:
        out["examples"] = {"start": examples[str(first["step"])], "end": examples[str(last["step"])]}
    return out


def cmd_run(args, cfg):
    D2, cfg2, _ = load_d2(cfg)
    spec = run_spec(cfg, args, args.name or "")
    meta, tokens = load_pairs(args)
    out = out_dir(args) / "runs" / f"{args.name}.json"
    banner(f"run {args.name} · beta = {spec['beta']} · {spec['batch_pairs']} pairs/step · {spec['steps']} steps · lr {float(spec['learning_rate']):g} · "
           f"reference log-probabilities: {spec['ref']}", AMBER)
    policy = D2.load_policy(cfg2, d2_args(args))
    wb = D.weight_bytes(policy.model)
    rec = {"name": args.name, "date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "seed": int(cfg["demo"]["seed"]),
           "model": None if args.tiny else cfg2["model"]["repo"], "adapter": None if args.tiny else cfg2["model"]["adapter"], "pairs_date": meta["date"],
           "spec": spec, "bypass": {"rank": policy.rank, "scale": cfg2["bypass"]["scale"], "trainable": policy.n_trainable}, "weights": wb}
    rec.update(train_run(D2, policy, cfg, args, spec, meta, tokens))
    print(f"\n  the ruler after training (bits per byte):")
    rec["ruler_end"] = measure_ruler(cfg, args, policy)
    rec["summary"]["peak_gb"] = r4(peak_memory_gb())
    write_json(out, rec)
    s = rec["summary"]
    print(f"\n  {GREEN}{args.name} done{RESET}: {s['reading']}; sampled length {s['start']['gold']['mean_len']} → {BOLD}{s['end']['gold']['mean_len']}{RESET}; "
          f"{s['seconds'] / 60:.1f} min, peak {s['peak_gb']} GB → {out.relative_to(HERE)}")


# ══ test／inspect／probe ════════════════════════════════════════
def cmd_test(args, cfg):
    import mlx.core as mx
    import mlx.nn as nn
    import dpo_final
    banner("demo 1 · tests without downloaded weights")
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {GREEN + 'ok  ' if cond else RED + 'FAIL'}{RESET}  {name}  {GREY}{detail}{RESET}")

    mods = [("dpo_final.py", dpo_final.dpo_loss)]
    try:
        import dpo_live
        dpo_live.dpo_loss(mx.array([0.0]), mx.array([0.0]), mx.array([0.0]), mx.array([0.0]), 1.0)
        mods.append(("dpo_live.py", dpo_live.dpo_loss))
    except NotImplementedError:
        note("dpo_live.py 還是空的骨架，只測參考實作")
    for label, fn in mods:
        print(f"\n  {BOLD}{label}{RESET}")
        rows = five_tests(fn, cfg, lambda title: print(f"\n  {AMBER}{title}{RESET}"))
        print()
        check(f"{label}: the five tests", rows["all_ok"], "" if rows["v"]["ok"] is not None else "(v) skipped: no torch")

    # 梯度的讀法：∂loss/∂logp_w = −(β/n) σ(r̂_l − r̂_w)，∂loss/∂logp_l 是它的相反數
    lw, ll, rw, rl = (mx.array(v) for v in live_inputs(cfg))
    beta, n = float(cfg["live"]["beta"]), lw.shape[0]
    gw = mx.grad(lambda a: dpo_final.dpo_loss(a, ll, rw, rl, beta))(lw).tolist()
    gl = mx.grad(lambda b: dpo_final.dpo_loss(lw, b, rw, rl, beta))(ll).tolist()
    want = [-(beta / n) / (1 + math.exp(beta * ((a - c) - (b - d)))) for a, b, c, d in zip(lw.tolist(), ll.tolist(), rw.tolist(), rl.tolist())]
    print()
    check("the gradient is −(β/n)·σ(r̂_l − r̂_w) on logp_w and the opposite on logp_l",
          max(abs(a - b) for a, b in zip(gw, want)) < 1e-7 and max(abs(a + b) for a, b in zip(gw, gl)) < 1e-7)

    # 規則與配對
    def S(finish: str, n_tok: int) -> dict:
        return {"finish": finish, "tokens": [0] * n_tok}

    check("rule: an answer that ends properly beats one cut at the limit, whatever the lengths", D.prefers(S("stop", 90), S("length", 60), 8) and not D.prefers(S("length", 10), S("stop", 90), 8))
    check("rule: two proper endings — the shorter wins only by the minimum gap", D.prefers(S("stop", 20), S("stop", 28), 8) and not D.prefers(S("stop", 20), S("stop", 27), 8))
    check("rule: two answers cut at the limit make no pair", not D.prefers(S("length", 160), S("length", 160), 8))
    grp = [S("stop", 50), S("length", 160), S("stop", 12), S("stop", 100), S("length", 160), S("stop", 14)]
    check("pairs: best with worst, second best with second worst", D.make_pairs(grp, 2, 8) == [(2, 4), (5, 1)], str(D.make_pairs(grp, 3, 8)))
    check("pairs: nothing to pair when every answer is the same length", D.make_pairs([S("stop", 9)] * 4, 2, 8) == [])
    m = D.pair_metrics([-10.0, -12.0], [-20.0, -18.0], [-10.0, -12.0], [-20.0, -18.0], 0.1, [5, 6], [9, 8])
    check("metrics: at the reference the loss is log 2 and the margin is 0", abs(m["loss"] - LOG2) < 1e-12 and m["margin"] == 0)
    c = D.paired_change([1.0, 2.0, 0.0, 5.0], [2.0, 3.0, 1.0, 6.0])
    check("paired change: all four pairs moved by −1", c["mean"] == -1.0 and c["se"] == 0.0 and c["fell"] == 1.0 and D.direction(c) == "fell")

    # 亂數權重的小模型：B = 0 時 loss 是 log 2、KL 是 0；訓練的 loss 就是 dpo_loss 用在重算的 log 機率上
    D2, cfg2, src = load_d2(cfg)
    G = D2.G
    targs = argparse.Namespace(tiny=True, online=False)
    policy = D2.Policy(cfg2, d2_args(targs))
    prompts = load_prompts(cfg, targs, policy)
    pc = pcfg(cfg, targs)
    groups, _ = sample_answers(D2, policy, prompts["train"][:16], 6, int(pc["max_new_tokens"]), 1.0, 48, 1)
    pairs = []
    for p, g in zip(prompts["train"], groups):
        for w, l in D.make_pairs(g, 2, int(pc["min_gap_tokens"])):
            pairs.append({"prompt": p["ids"], "w": g[w]["tokens"], "l": g[l]["tokens"]})
    check("tiny model: the rule finds pairs among sampled answers", len(pairs) >= 4, f"{len(pairs)} pairs from 16 prompts")
    part = pairs[:4]
    batch, mask = D.pair_batch(G, part)
    ref = D.sequence_logp(G, policy.model, batch, mask, ref=True)
    (loss, lp), grads = nn.value_and_grad(policy.model, D.training_loss(G, dpo_final.dpo_loss))(policy.model, batch, mask, ref[:4], ref[4:], 0.1)
    check("tiny model: with B = 0 the loss on a batch is exactly log 2", abs(float(loss.item()) - LOG2) < 1e-6, f"{float(loss.item()):.7f}")
    gnorm = float(mx.sqrt((G.flat_grad(grads) ** 2).sum()).item())
    check("tiny model: and the gradient is not zero (every pair has weight ½)", gnorm > 0, f"‖∇‖ = {gnorm:.3g}")
    flat = [{"prompt": p["prompt"], "tokens": p["w"]} for p in part]
    k = D.kl_estimate(G, policy.model, flat)
    check("tiny model: the KL estimate from π_ref is exactly 0 before the first update", k["kl"] == 0.0)
    wb = D.weight_bytes(policy.model)
    check("tiny model: only the bypass is trainable", wb["trainable_bytes"] == policy.n_trainable * 4, f"{policy.n_trainable:,} numbers")
    names = set(cfg["runs"])
    check("config: every run named in [rehearse] and [tables] exists; demo 2 is where [demo] source says",
          set(cfg["rehearse"]["runs"]) <= names and set(cfg["tables"]["rows"]) <= names and cfg["tables"]["curve"] in names and (src / "grpo.py").exists())
    print(f"\n  {GREEN + 'all tests passed' if ok else RED + 'some tests FAILED'}{RESET}")
    if not ok:
        raise SystemExit(1)


def cmd_inspect(args, cfg):
    import mlx.core as mx
    D2, cfg2, src = load_d2(cfg)
    G = D2.G
    banner("demo 1 · inspect")
    v = versions()
    print("  " + "  ".join(f"{k} {v[k]}" for k in ("python", "mlx", "mlx-lm", "transformers") if v.get(k)) + f"  mlx-lm commit {str(v.get('mlx_lm_commit'))[:7]}")
    try:
        import torch
        tv = torch.__version__
    except ModuleNotFoundError:
        tv = None
    print(f"  torch {tv or 'not installed (test (v) would be skipped)'}")
    policy = D2.load_policy(cfg2, d2_args(args))
    prompts = load_prompts(cfg, args, policy)
    p = prompts["train"][0]
    text = p["text"] if args.tiny else policy.tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True)
    print(f"\n  one prompt through the chat template ({len(p['ids'])} tokens):\n{GREY}{text}{RESET}")
    eos = sorted(int(t) for t in policy.tok.eos_token_ids)
    print(f"\n  stop tokens: {eos}" + ("" if args.tiny else "  " + " ".join(repr(policy.tok.decode([t])) for t in eos)))
    lens = {k: [len(q["ids"]) for q in prompts[k]] for k in ("train", "heldout", "gold")}
    for k in ("train", "heldout", "gold"):
        print(f"  {k:<8} {len(lens[k]):>4} prompts, {min(lens[k])}–{max(lens[k])} tokens (mean {mean(lens[k]):.0f}); dropped as too long: {prompts['dropped_too_long'][k]}")
    x = mx.array(p["ids"])[None]
    B = G.Bypass()
    on = policy.model(x).astype(mx.float32)
    B.enabled = False
    off = policy.model(x).astype(mx.float32)
    B.enabled = True
    d = float(mx.abs(on - off).max().item())
    print(f"\n  bypass on vs. off at initialisation (B = 0): max |Δ logit| = {BOLD}{d:g}{RESET}  {GREY}(0 means step 0 is exactly π_ref){RESET}")
    wb = D.weight_bytes(policy.model)
    print(f"  parameters in memory: {BOLD}{wb['all_parameters_bytes'] / 1e9:.3f} GB{RESET}, of which trainable (the bypass) {wb['trainable_bytes'] / 1e6:.1f} MB")
    print(f"  {GREY}one copy of the frozen weights serves as π_ref (bypass off) and as the body of π_θ (bypass on){RESET}")
    rec = {"date": now(), "versions": v, "torch": tv, "device": device_info(), "tiny": bool(args.tiny), "prompt_text": text, "prompt_tokens": len(p["ids"]),
           "eos_token_ids": eos, "prompts": {k: {"n": len(lens[k]), "min": min(lens[k]), "max": max(lens[k]), "mean": r4(mean(lens[k]))} for k in lens},
           "dropped_too_long": prompts["dropped_too_long"], "bypass_on_off_max_abs_diff_at_init": d, "weights": wb,
           "bypass": {"rank": policy.rank, "scale": cfg2["bypass"]["scale"], "trainable": policy.n_trainable}, "peak_gb": r4(peak_memory_gb())}
    write_json(out_dir(args) / "inspect.json", rec)
    note(f"\n  → {(out_dir(args) / 'inspect.json').relative_to(HERE)}")


def cmd_probe(args, cfg):
    D2, cfg2, _ = load_d2(cfg)
    pr = cfg["probe"]
    meta, tokens = load_pairs(args)
    lrs = [float(cfg["tiny"]["learning_rate"]) * f for f in (0.3, 1.0)] if args.tiny else [float(x) for x in pr["lrs"]]
    steps = 10 if args.tiny else int(pr["steps"])
    banner(f"demo 1 · probe — {steps} steps at each learning rate, beta = {pr['beta']}", AMBER)
    policy = D2.load_policy(cfg2, d2_args(args))
    rows = []
    for lr in lrs:
        policy.reset_bypass(policy.seed)
        spec = dict(cfg["train"])
        spec.update(beta=float(pr["beta"]), steps=steps, learning_rate=lr, eval_every=0)
        note(f"\n  lr {lr:g}")
        r = train_run(D2, policy, cfg, args, spec, meta, tokens, echo=False, full_eval=False, train_subset=int(pr["n_pairs"]))
        s = r["summary"]
        a, b = s["start"]["train"], s["end"]["train"]
        rows.append({"lr": lr, "loss_start": a["loss"], "loss_end": b["loss"], "ranked_right_end": b["ranked_right"],
                     "change_w": b["change_w"], "change_l": b["change_l"], "t_update_per_step": s["t_update_per_step"], "peak_gb": s["peak_gb"],
                     "finite": all(st["loss"] is not None and math.isfinite(st["loss"]) for st in r["steps"])})
    print(f"\n  {'lr':>8}{'loss, start':>13}{'loss, end':>11}{'ranked right':>14}{'Δ log π(y_w)':>15}{'Δ log π(y_l)':>15}{'s / step':>10}")
    for r in rows:
        print(f"  {r['lr']:>8g}{r['loss_start']:>13.4f}{BOLD}{r['loss_end']:>11.4f}{RESET}{r['ranked_right_end']:>14.0%}{r['change_w']['mean']:>+15.2f}{r['change_l']['mean']:>+15.2f}"
              f"{r['t_update_per_step']:>10.2f}" + ("" if r["finite"] else f"  {RED}not finite{RESET}"))
    full = cfg["train"]
    t = mean([r["t_update_per_step"] for r in rows])
    n_runs = len(cfg["rehearse"]["runs"])
    est = n_runs * int(full["steps"]) * t / 60
    print(f"\n  {GREY}a full rehearsal is {n_runs} runs × {full['steps']} steps ≈ {est:.0f} min of updates, plus the evaluations and the ruler at the end of each run{RESET}")
    note("  pick a learning rate at which the loss clearly falls within these steps and stays finite; put it in [train] learning_rate")
    rec = {"date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "beta": float(pr["beta"]), "steps": steps,
           "watched_train_pairs": int(pr["n_pairs"]), "rows": rows, "estimated_update_minutes": r4(est)}
    write_json(out_dir(args) / "probe.json", rec)
    note(f"  → {(out_dir(args) / 'probe.json').relative_to(HERE)}")


# ══ rehearse／aggregate／replay ═════════════════════════════════
def spawn(args, extra: list[str]) -> None:
    cmd = [sys.executable, str(HERE / "demo.py")] + (["--tiny"] if args.tiny else []) + (["--online"] if args.online else [])
    cmd += (["--steps", str(args.steps)] if args.steps else []) + (["--lr", str(args.lr)] if args.lr else []) + extra
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit(f"{RED}子行程失敗（{' '.join(extra)}）：exit {r.returncode}；修好之後重跑同一個指令，做過的會跳過{RESET}")


def cmd_rehearse(args, cfg):
    t0 = time.time()
    if not args.only:
        cmd_live(args, cfg, rehearsal=True)
    have = (out_dir(args) / "pairs.json").exists() and (cache_dir(args) / "pairs_tokens.json").exists()
    if have and (not args.force or args.only):
        note("pairs: 已有 pairs.json 與 token 快取，跳過（不帶 --only 的 --force 會重做，三個 run 也跟著重跑）")
    else:
        spawn(args, ["pairs"])
    names = args.only or list(cfg["rehearse"]["runs"])
    pairs_date = read_json(out_dir(args) / "pairs.json")["date"]
    for name in names:
        if name not in cfg["runs"]:
            raise SystemExit(f"{RED}沒有這個 run：{name}{RESET}")
        out = out_dir(args) / "runs" / f"{name}.json"
        if out.exists() and not args.force and read_json(out).get("pairs_date") == pairs_date:
            note(f"{name}: 已有 {out.relative_to(HERE)}，跳過（--force 重跑）")
            continue
        spawn(args, ["run", "--name", name])
    aggregate(args, cfg)
    print(f"\n{GREEN}rehearsal finished in {(time.time() - t0) / 60:.1f} min{RESET}; ./present.sh replay")


def aggregate(args, cfg) -> dict:
    d = out_dir(args)
    if not (d / "pairs.json").exists():
        raise SystemExit(f"{RED}沒有 {(d / 'pairs.json').relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    pairs = read_json(d / "pairs.json")
    runs = {p.stem: read_json(p) for p in sorted((d / "runs").glob("*.json"))}
    if not runs:
        raise SystemExit(f"{RED}{(d / 'runs').relative_to(HERE)} 裡沒有 run：先跑 ./present.sh rehearse{RESET}")
    settings = {n: (r["pairs_date"], r["spec"]["learning_rate"], r["spec"]["steps"], r["spec"]["batch_pairs"], r["spec"]["ref"]) for n, r in runs.items()}
    consistent = len(set(settings.values())) == 1 and all(r["pairs_date"] == pairs["date"] for r in runs.values())
    if not consistent:
        print(f"  {RED}這些 run 的設定不一致，或不是用現在這份 pairs.json 訓的（pairs 日期, lr, steps, batch, ref）：{settings}；rehearse --force{RESET}")
    out_runs = {}
    for n, r in runs.items():
        out_runs[n] = {"summary": r["summary"], "evals": r["evals"], "ruler_end": r["ruler_end"], "examples": r.get("examples"),
                       "train": {k: [st[k] for st in r["steps"]] for k in ("step", "loss", "logp_w_minus_ref", "logp_l_minus_ref")}}
    any_run = next(iter(runs.values()))
    summary = {"date": now(), "run_dates": {n: r["date"] for n, r in runs.items()}, "tiny": any(r.get("tiny") for r in runs.values()),
               "model": any_run["model"], "adapter": any_run["adapter"], "bypass": any_run["bypass"], "weights": any_run["weights"],
               "settings_consistent": consistent, "tables": cfg["tables"],
               "pairs": {k: pairs[k] for k in ("date", "rule", "settings", "prompts", "sampled", "pairs", "diagnostics", "examples")},
               "ruler_ref": pairs["ruler_ref"], "runs": out_runs, "caveat": CAVEAT}
    write_json(d / "summary.json", summary)
    note(f"→ {(d / 'summary.json').relative_to(HERE)}")
    return summary


def arrow(a, b, spec=".3f") -> str:
    return f"{fmt(a, spec)} → {fmt(b, spec)}"


def bar(x, lo: float, hi: float, width: int = 18) -> str:
    if x is None or hi <= lo:
        return ""
    k = max(0, min(width, round((x - lo) / (hi - lo) * width)))
    return "█" * k + "·" * (width - k)


def cmd_replay(args, cfg):
    p = out_dir(args) / "summary.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    s = read_json(p)
    t, runs, pr = s["tables"], s["runs"], s["pairs"]
    rows = [n for n in t["rows"] if n in runs]
    banner(f"{cfg['demo']['title']}\n(b) REPLAY of the rehearsal of {s['date'][:10]}" + ("  [tiny random model — numbers mean nothing]" if s.get("tiny") else ""))
    print(f"  {GREY}{s['caveat']}{RESET}")
    if not s.get("settings_consistent", True):
        print(f"  {RED}the runs in this summary were not made with the same settings{RESET}")

    pause("\n  where the preference pairs come from  [Enter] ")
    st = pr["settings"]
    tr, he = pr["pairs"]["train"], pr["pairs"]["heldout"]
    print(f"\n  π_ref wrote {st['samples_per_prompt']} answers to each prompt (temperature {st['temperature']}, at most {st['max_new_tokens']} tokens).")
    print(f"  the rule: {BOLD}{pr['rule']}{RESET}")
    print(f"  {BOLD}{tr['pairs']}{RESET} pairs to train on, {BOLD}{he['pairs']}{RESET} pairs from other prompts held out")
    print(f"  y_w: {tr['mean_len_w']:.1f} tokens on average; y_l: {tr['mean_len_l']:.1f} tokens ({tr['l_cut_at_limit']:.0%} cut at the limit)")
    print(f"  log π_ref per answer: y_w {BOLD}{tr['ref_logp_w']:.1f}{RESET}, y_l {BOLD}{tr['ref_logp_l']:.1f}{RESET} nats   {GREY}(per token: {tr['ref_logp_w_per_token']:.2f}, {tr['ref_logp_l_per_token']:.2f}){RESET}")
    if pr["examples"]:
        ex = pr["examples"][0]
        print(f"\n  {GREY}one pair. prompt:{RESET} {ex['prompt'][:200]}")
        print(f"  {GREEN}y_w{RESET} {GREY}({ex['y_w']['tokens']} tokens, {ex['y_w']['finish']}){RESET} {ex['y_w']['text'][:220]}")
        print(f"  {RED}y_l{RESET} {GREY}({ex['y_l']['tokens']} tokens, {ex['y_l']['finish']}){RESET} {ex['y_l']['text'][:220]}")

    pause("\n  the loss, for each beta  [Enter] ")
    print(f"\n  {'beta':<7}{'loss, start → end':>22}{'pairs ranked right, end':>26}{'held-out loss, start → end':>30}")
    for n in rows:
        a, b = runs[n]["summary"]["start"], runs[n]["summary"]["end"]
        print(f"  {runs[n]['summary']['beta']:<7g}{arrow(a['train']['loss'], b['train']['loss'], '.4f'):>22}{b['train']['ranked_right']:>26.0%}"
              f"{arrow(a['heldout']['loss'], b['heldout']['loss'], '.4f'):>30}")
    note(f"  log 2 = {LOG2:.4f}. The start is not exactly log 2 when the reference log-probabilities were stored from differently padded batches "
         f"(largest difference {pr['diagnostics']['rebatched_max_abs']:g} nats per answer).")

    pause("\n  the two log-probabilities, separately — what do you expect in the y_w column?  [Enter] ")
    print(f"\n  {'beta':<7}{'log π(y_w), start → end':>28}{'change ± s.e.':>18}{'log π(y_l), start → end':>28}{'change ± s.e.':>18}")
    for n in rows:
        sm = runs[n]["summary"]
        a, b = sm["start"]["train"], sm["end"]["train"]
        cw, cl = b["change_w"], b["change_l"]
        print(f"  {sm['beta']:<7g}{arrow(a['logp_w'], b['logp_w'], '.2f'):>28}{BOLD}{cw['mean']:>+11.2f}{RESET} ± {cw['se']:<4.2f}"
              f"{arrow(a['logp_l'], b['logp_l'], '.2f'):>28}{BOLD}{cl['mean']:>+11.2f}{RESET} ± {cl['se']:<4.2f}")
    note("  nats per whole answer, averaged over the training pairs; the change is paired (the same pairs before and after).")
    for n in rows:
        sm = runs[n]["summary"]
        colour = RED if sm["displacement"] else GREEN
        print(f"  beta {sm['beta']:g}: {colour}{sm['reading']}{RESET}")

    cv = runs.get(t["curve"])
    if cv:
        pause(f"\n  beta = {cv['summary']['beta']:g}, step by step  [Enter] ")
        ev = cv["evals"]
        lo_w, hi_w = min(e["train"]["logp_w"] for e in ev), max(e["train"]["logp_w"] for e in ev)
        lo_l, hi_l = min(e["train"]["logp_l"] for e in ev), max(e["train"]["logp_l"] for e in ev)
        pad_w, pad_l = max(1e-9, (hi_w - lo_w) * 0.1), max(1e-9, (hi_l - lo_l) * 0.1)
        print(f"\n  {'step':>5}{'loss':>9}   {'log π(y_w)':>11} {'':<18}   {'log π(y_l)':>11} {'':<18}   {'length':>7}{'ended':>7}{'√KL':>7}")
        for e in ev:
            g = e.get("gold") or {}
            print(f"  {e['step']:>5}{e['train']['loss']:>9.4f}   {e['train']['logp_w']:>11.2f} {GREEN}{bar(e['train']['logp_w'], lo_w - pad_w, hi_w + pad_w)}{RESET}   "
                  f"{e['train']['logp_l']:>11.2f} {AMBER}{bar(e['train']['logp_l'], lo_l - pad_l, hi_l + pad_l)}{RESET}   "
                  f"{fmt(g.get('mean_len'), '.1f'):>7}{fmt(g.get('ended'), '.0%'):>7}{fmt(g.get('sqrt_kl'), '.3f'):>7}")
        note("  each bar is scaled to its own column's range. length, ended, √KL: answers sampled on prompts that were never trained on.")

    pause("\n  the rule's gold, and what was forgotten  [Enter] ")
    key = t["ruler_key"] if t["ruler_key"] in s["ruler_ref"] else next(iter(s["ruler_ref"]))
    print(f"\n  {'beta':<7}{'answer length, start → end':>30}{'ended properly':>22}{'√KL from π_ref':>17}{'bits/byte, ' + key:>26}")
    for n in rows:
        sm = runs[n]["summary"]
        a, b = sm["start"]["gold"], sm["end"]["gold"]
        print(f"  {sm['beta']:<7g}{arrow(a['mean_len'], b['mean_len'], '.1f'):>30}{arrow(a['ended'], b['ended'], '.0%'):>22}{b['sqrt_kl']:>17.3f}"
              f"{arrow(s['ruler_ref'][key]['bpb'], runs[n]['ruler_end'][key]['bpb'], '.4f'):>26}")
    n_gold = runs[rows[0]]["summary"]["start"]["gold"]["n"] if rows else 0
    note(f"  length and ending: {int(n_gold)} answers sampled on unseen prompts, the same sampling seed every time. bits per byte: last week's ruler.")
    if len(s["ruler_ref"]) > 1:
        print(f"\n  {'bits per byte':<44}{'π_ref':>9}" + "".join(f"{'β = ' + format(runs[n]['summary']['beta'], 'g'):>11}" for n in rows))
        for k, r in s["ruler_ref"].items():
            print(f"  {r['label']:<44}{r['bpb']:>9.4f}" + "".join(f"{runs[n]['ruler_end'][k]['bpb']:>11.4f}" for n in rows))
    if cv and cv.get("examples"):
        pause("\n  one unseen prompt, before and after  [Enter] ")
        a, b = cv["examples"]["start"][0], cv["examples"]["end"][0]
        print(f"\n  {GREY}before ({a['tokens']} tokens, {a['finish']}):{RESET} {a['answer'][:300]}")
        print(f"  {GREY}after, beta = {cv['summary']['beta']:g} ({b['tokens']} tokens, {b['finish']}):{RESET} {b['answer'][:300]}")

    pause("\n  how many models were in memory  [Enter] ")
    w = s["weights"]
    sm = runs[rows[0]]["summary"] if rows else {}
    print(f"\n  parameters in memory: {BOLD}{w['all_parameters_bytes'] / 1e9:.3f} GB{RESET} — one copy of the frozen weights, plus a bypass of {w['trainable_bytes'] / 1e6:.1f} MB")
    print(f"  π_ref is that copy with the bypass switched off; its log-probabilities were {BOLD}{sm.get('ref', '—')}{RESET}")
    print(f"  peak memory while training: {BOLD}{fmt(sm.get('peak_gb'), '.1f')} GB{RESET}; {fmt(sm.get('t_update_per_step'), '.2f')} s per update")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    ap.add_argument("--online", action="store_true", help="允許連 Hugging Face（預設離線）")
    ap.add_argument("--ref", action="store_true", help="(a) 用 dpo_final.py")
    ap.add_argument("--name", help="run 的名字（demo_config.toml [runs]）")
    ap.add_argument("--only", action="append", help="rehearse 只跑這個 run（可重複）")
    ap.add_argument("--force", action="store_true", help="rehearse 重做 pairs 與已有的 run")
    ap.add_argument("--steps", type=int, help="暫時換步數")
    ap.add_argument("--lr", type=float, help="暫時換 learning rate")
    ap.add_argument("cmd", choices=["live", "replay-a", "test", "inspect", "pairs", "probe", "rehearse", "run", "aggregate", "replay"])
    args = ap.parse_args()
    cfg = load_cfg()
    {"live": cmd_live, "replay-a": cmd_replay_a, "test": cmd_test, "inspect": cmd_inspect, "pairs": cmd_pairs, "probe": cmd_probe,
     "rehearse": cmd_rehearse, "run": cmd_run, "aggregate": aggregate, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
