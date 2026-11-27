#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 3：刻意留漏洞的 verifier——proxy 與 gold 對 √KL（投影片 p61 前導頁、p63 replay）。

    ./present.sh test         不載權重：verifier 的性質、硬幣、clip 的目標函數、亂數小模型上的梯度與 KL
    ./present.sh inspect      課前：沿用了 demo 2 的哪些設定；π_ref 在兩個 verifier 下各幾分（訓練前漏洞有多寬）；B = 0 時 KL 是 0
    ./present.sh rehearse     彩排：[rehearse] runs 的每個 run 各一個子行程（做過的跳過）→ 彙整
                              --only count 只跑一個；--force 重跑
    ./present.sh replay       課堂：讀 runs/rehearsal/summary.json，Enter 逐段印 p63 的表（不載模型）
    ./present.sh tiny         亂數權重的小模型走一遍真的流程（不下載任何東西；寫到 runs/smoke/）

任務、模型（Qwen3-1.7B-Base 加上週的 adapter C ＝ π_ref）、旁路、取樣、advantage、loss 都是 demo 2 的：
grpo.py 與 demo 2 的 demo.py 從 ../w09_d2_grpo_scratch/ 以檔案路徑載入，設定也讀那邊的 demo_config.toml。
這裡換的是 verifier（rewards.py），多記的是 gold、對 π_ref 的 KL、回答的長度與數字個數；
另外多一個開關：每一批 rollout 更新幾步（多於一步時 ratio 不再是 1，clip 會作用）。

相依：與 demo 2 相同（mlx、mlx-lm、numpy、transformers、huggingface_hub）。replay 只要標準函式庫。
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
import rewards as RW  # noqa: E402  （它把 demo 2 的資料夾放進 sys.path、載入 grpo）

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
PROXY_KEY = {"leaky": "proxy_leaky", "count": "proxy_count", "strict": "gold", "coin": None}
CAVEAT = ("One 1.7B model, one kind of problem, one seed, a few hundred steps. The hole was put there on purpose. "
          "The random-reward rows cannot settle the question between the two papers of this week's reading.")


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def load_d2(cfg: dict):
    """demo 2 的 demo.py（Policy、題目、eval 的題目）以檔案路徑載入；它自己的 grpo 就是 rewards.py 載入的那一個。"""
    src = (HERE / cfg["demo"]["source"]).resolve()
    if not (src / "demo.py").exists():
        raise SystemExit(f"{RED}找不到 demo 2：{cfg['demo']['source']}/demo.py{RESET}")
    spec = importlib.util.spec_from_file_location("w09_d2_demo", src / "demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["w09_d2_demo"] = mod
    spec.loader.exec_module(mod)
    return mod, mod.load_cfg(), src


def out_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "rehearsal")


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


def fmt(x, spec=".3f", none="—"):
    return none if x is None else format(x, spec)


def run_spec(cfg: dict, cfg2: dict, args, name: str) -> dict:
    """demo 2 的 [train] ← 這裡的 [train] ← 這個 run 的覆寫。"""
    if name not in cfg["runs"]:
        raise SystemExit(f"{RED}沒有這個 run：{name}（demo_config.toml [runs]：{', '.join(cfg['runs'])}）{RESET}")
    s = dict(cfg2["train"])
    s.update(cfg["train"])
    s.update(cfg["runs"][name])
    s["rank"] = int(cfg2["bypass"]["rank"])
    if args.tiny:
        t = cfg2["tiny"]
        s.update(steps=int(t["steps"]), learning_rate=float(t["learning_rate"]), rollouts_per_step=int(t["rollouts_per_step"]), eval_every=10)
    if args.steps:
        s["steps"] = int(args.steps)
    return s


# ── 每段回答在兩個 verifier 下的分數；對 π_ref 的 KL ─────────────
def annotate(problems: list[dict], groups: list[list[dict]]) -> list[dict]:
    flat = []
    for p, grp in zip(problems, groups):
        for r in grp:
            r.update(RW.describe(r["text"], p["answer"]))
            r["proxy_count"] = RW.reward_count(r["text"], p["answer"])
            flat.append(r)
    return flat


def flat_stats(flat: list[dict]) -> dict:
    return {"n": len(flat),
            "proxy_leaky": r4(mean([r["proxy_leaky"] for r in flat])), "proxy_count": r4(mean([r["proxy_count"] for r in flat])),
            "gold": r4(mean([r["gold"] for r in flat])),
            "leak_only": r4(mean([1.0 if r["proxy_leaky"] > 0 and r["gold"] <= 0 else 0.0 for r in flat])),
            "n_numbers": r4(mean([float(r["n_numbers"]) for r in flat])), "mean_len": r4(mean([float(len(r["tokens"])) for r in flat])),
            "truncated": r4(mean([1.0 if r["finish"] != "stop" else 0.0 for r in flat]))}


def sequence_logp(G, model, batch, mask, ref: bool):
    """Σ_t log π(y_t | x, y_<t)，不求梯度。ref=True 時把旁路關掉——那就是 π_ref，同一份權重。"""
    import mlx.core as mx
    B = G.Bypass()
    B.enabled = not ref
    try:
        lp = G.sequence_logprobs(model, batch, mask)
        mx.eval(lp)
    finally:
        B.enabled = True
    return lp


def kl_from(theta: list[float], ref: list[float]) -> dict:
    """取樣估計：y ~ π_θ 時 log π_θ(y|x) − log π_ref(y|x) 的平均（nats／每段回答）。有限樣本下可以是負的。"""
    d = [a - b for a, b in zip(theta, ref)]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / len(d) / len(d)) if len(d) > 1 else 0.0
    return {"kl": r4(m), "kl_se": r4(se), "sqrt_kl": r4(math.sqrt(max(m, 0.0)))}


def kl_estimate(G, model, flat: list[dict], chunk: int = 64) -> dict:
    theta, ref = [], []
    for i in range(0, len(flat), chunk):
        part = flat[i:i + chunk]
        batch, mask = G.pad_batch([r["prompt"] for r in part], [r["tokens"] for r in part])
        theta += sequence_logp(G, model, batch, mask, ref=False).tolist()
        ref += sequence_logp(G, model, batch, mask, ref=True).tolist()
    return kl_from(theta, ref)


def evaluate(D2, policy, cfg: dict, cfg2: dict, problems: list[dict], batch_size: int) -> tuple[dict, list[dict]]:
    """固定的題目、固定的取樣種子（demo 2 的 [eval]）：不同 run、不同步數之間是配對的比較。"""
    import mlx.core as mx
    G = D2.G
    e = cfg2["eval"]
    mx.random.seed(int(e["seed"]))
    t0 = time.perf_counter()
    groups, _ = policy.rollouts(problems, int(e["samples_per_problem"]), G.reward_strict, batch_size)
    flat = annotate(problems, groups)
    out = flat_stats(flat)
    out.update(kl_estimate(G, policy.model, flat))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    ex, n, cut = [], int(cfg["examples"]["n_problems"]), int(cfg["examples"]["max_chars"])
    for p, grp in list(zip(problems, groups))[:n]:
        ex.append({"expr": p["expr"], "answer": p["answer"],
                   "answers": [{"text": r["text"][:cut], "proxy_leaky": r["proxy_leaky"], "gold": r["gold"], "n_numbers": r["n_numbers"],
                                "tokens": len(r["tokens"]), "finish": r["finish"]} for r in grp]})
    return out, ex


# ── loss：第一步就是 demo 2 的 grpo_loss；之後的步數是 clip 過的 surrogate ──
def first_loss(G):
    def fn(model, batch, mask, weights):
        lp = G.token_logprobs(model, batch)
        return -((lp * mask).sum(axis=1) * weights).sum(), lp
    return fn


def clip_objective(ratio, w, eps: float):
    """min(ρ w, clip(ρ, 1−ε, 1+ε) w)。w 帶著 advantage 的正負號。"""
    import mlx.core as mx
    return mx.minimum(ratio * w, mx.clip(ratio, 1 - eps, 1 + eps) * w)


def clipped_loss(G, eps: float):
    def fn(model, batch, mask, weights, old_lp):
        import mlx.core as mx
        lp = G.token_logprobs(model, batch)
        obj = clip_objective(mx.exp(lp - old_lp), weights[:, None], eps)
        return -(obj * mask).sum(), lp
    return fn


def clip_activity(lp, old_lp, mask, weights, eps: float) -> dict:
    """這一次前向時，有多少（權重不是零的）token 的 ratio 已經在 clip 作用的那一側之外。"""
    import mlx.core as mx
    ratio = mx.exp(lp - old_lp)
    w = weights[:, None]
    live = mask * (w != 0)
    active = ((w > 0) * (ratio > 1 + eps) + (w < 0) * (ratio < 1 - eps)) * live
    n = float(live.sum().item())
    return {"active": r4(float(active.sum().item()) / n) if n else None,
            "max_abs_log_ratio": r4(float((mx.abs(lp - old_lp) * mask).max().item()))}


# ── 一個訓練 run ─────────────────────────────────────────────────
def train_run(D2, policy, cfg: dict, cfg2: dict, spec: dict, echo: bool = True) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    G = D2.G
    model = policy.model
    g, R, steps = int(spec["G"]), int(spec["rollouts_per_step"]), int(spec["steps"])
    K, eps = int(spec["updates_per_batch"]), float(spec["clip_eps"])
    P = max(1, R // g)
    seed, max_new = policy.seed, int(policy.task["max_new_tokens"])
    opt = optim.Adam(learning_rate=float(spec["learning_rate"]))
    vg_first = nn.value_and_grad(model, first_loss(G))
    vg_clip = nn.value_and_grad(model, clipped_loss(G, eps))
    evalset = D2.eval_problems(cfg2, policy)
    exclude = {p["expr"] for p in evalset}
    cum = {"rollouts": 0, "gen_tokens": 0, "updates": 0, "t_sample": 0.0, "t_update": 0.0, "t_ref": 0.0}
    rec_steps, evals, examples = [], [], {}
    t_start = time.time()

    def do_eval(step: int) -> None:
        e, ex = evaluate(D2, policy, cfg, cfg2, evalset, R)
        e.update(step=step, rollouts=cum["rollouts"], updates=cum["updates"])
        evals.append(e)
        examples[str(step)] = ex
        if echo:
            print(f"  {AMBER}eval{RESET} step {step:>4}   leaky {BOLD}{e['proxy_leaky']:.3f}{RESET}   strict (gold) {BOLD}{e['gold']:.3f}{RESET}   "
                  f"√KL {e['sqrt_kl']:.3f}   numbers/answer {e['n_numbers']:.2f}   len {e['mean_len']:.1f}   cut {e['truncated']:.0%}   {GREY}{e['seconds']} s{RESET}")

    do_eval(0)
    every = int(spec.get("eval_every", 0))
    for step in range(1, steps + 1):
        s = (seed * 1009 + step) % (2 ** 31 - 1)
        rng = random.Random(s)
        mx.random.seed(s)
        reward_fn = RW.make_reward(spec["reward"], seed=s)
        problems = G.make_problems(rng, policy.task["level_spec"], P, exclude)
        groups, info = policy.rollouts(problems, g, reward_fn, R)
        flat = annotate(problems, groups)
        cum["rollouts"] += P * g
        cum["gen_tokens"] += info["generation_tokens"]
        cum["t_sample"] += info["seconds"]
        Rm = np.array([[r["reward"] for r in grp] for grp in groups])
        Lm = np.array([[len(r["tokens"]) for r in grp] for grp in groups])
        A, s_r = G.advantages(Rm, bool(spec["scale_rewards"]))
        W = G.rollout_weights(A, Lm, spec["token_agg"], max_new)
        batch, mask = G.pad_batch([r["prompt"] for r in flat], [r["tokens"] for r in flat])
        w = mx.array(W.reshape(-1).astype(np.float32))
        loss_v, clips, theta = None, [], None
        t0 = time.perf_counter()
        if (s_r > 0).any():
            (loss, lp), grads = vg_first(model, batch, mask, w)          # ratio = 1：demo 2 的那一步
            mx.eval(loss, lp, grads)
            old_lp = mx.stop_gradient(lp)
            theta = (old_lp * mask).sum(axis=1).tolist()
            loss_v = float(loss.item())
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            cum["updates"] += 1
            for _ in range(K - 1):                                       # 同一批 rollout 再走幾步：ratio 離開 1，clip 開始作用
                (loss, lp), grads = vg_clip(model, batch, mask, w, old_lp)
                mx.eval(loss, lp, grads)
                clips.append(clip_activity(lp, old_lp, mask, w, eps))
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state)
                cum["updates"] += 1
        t_u = time.perf_counter() - t0
        if theta is None:                                                # 整批都是沉默的組：沒有更新，KL 照樣要量
            theta = sequence_logp(G, model, batch, mask, ref=False).tolist()
        t0 = time.perf_counter()
        ref = sequence_logp(G, model, batch, mask, ref=True).tolist()    # π_ref 的那一次前向
        t_r = time.perf_counter() - t0
        cum["t_update"] += t_u
        cum["t_ref"] += t_r
        rec = {"step": step, "reward": r4(float(Rm.mean())), "silent": r4(float((s_r == 0).mean())), "loss": r4(loss_v),
               "clip_active": [c["active"] for c in clips], "max_abs_log_ratio": max([c["max_abs_log_ratio"] for c in clips], default=None),
               "t_sample": round(info["seconds"], 2), "t_update": round(t_u, 2), "t_ref": round(t_r, 2), "peak_gb": r4(peak_memory_gb())}
        rec.update({k: v for k, v in flat_stats(flat).items() if k != "n"})
        rec.update(kl_from(theta, ref))                                  # 取樣當下的 policy（這一步更新之前）
        rec_steps.append(rec)
        if loss_v is not None and not math.isfinite(loss_v):
            print(f"  {RED}loss 不是有限值（step {step}），停{RESET}")
            break
        if echo and (step % 5 == 0 or step == 1):
            clip = f"   clip active {rec['clip_active'][-1]:.0%}" if rec["clip_active"] and rec["clip_active"][-1] is not None else ""
            print(f"  step {step:>4}   reward {rec['reward']:.3f}   leaky {rec['proxy_leaky']:.3f}   gold {rec['gold']:.3f}   KL {rec['kl']:+.4f}   "
                  f"numbers {rec['n_numbers']:.2f}{clip}   {GREY}sample {info['seconds']:.1f} s · update {t_u:.1f} s · π_ref {t_r:.1f} s{RESET}")
        if every and (step % every == 0 or step == steps):
            do_eval(step)
    if evals[-1]["step"] != rec_steps[-1]["step"]:
        do_eval(rec_steps[-1]["step"])

    n = len(rec_steps)
    third = max(1, n // 3)
    first, last = evals[0], evals[-1]
    keys = ("proxy_leaky", "proxy_count", "gold", "leak_only", "kl", "kl_se", "sqrt_kl", "n_numbers", "mean_len", "truncated")
    summary = {"reward": spec["reward"], "G": g, "prompts_per_step": P, "updates_per_batch": K, "steps_done": n, "updates": cum["updates"],
               "rollouts": cum["rollouts"], "gen_tokens": cum["gen_tokens"],
               "start": {k: first[k] for k in keys}, "end": {k: last[k] for k in keys},
               "train_reward_first_third": r4(mean([r["reward"] for r in rec_steps[:third]])),
               "train_reward_last_third": r4(mean([r["reward"] for r in rec_steps[-third:]])),
               "silent": r4(mean([r["silent"] for r in rec_steps])),
               "clip_active_mean": r4(mean([c for r in rec_steps for c in r["clip_active"]])),
               "t_sample_per_step": r4(cum["t_sample"] / n), "t_update_per_step": r4(cum["t_update"] / n), "t_ref_per_step": r4(cum["t_ref"] / n),
               "peak_gb": r4(peak_memory_gb()), "seconds": round(time.time() - t_start, 1)}
    return {"summary": summary, "evals": evals, "steps": rec_steps, "examples": {"start": examples[str(first["step"])], "end": examples[str(last["step"])]}}


def make_args_for_d2(args):
    return argparse.Namespace(tiny=args.tiny, level=args.level, online=args.online, steps=args.steps)


def cmd_run(args, cfg):
    D2, cfg2, _ = load_d2(cfg)
    spec = run_spec(cfg, cfg2, args, args.name or "")
    out = out_dir(args) / "runs" / f"{args.name}.json"
    banner(f"run {args.name} · reward: {spec['reward']} · G = {spec['G']} · {spec['rollouts_per_step']} rollouts/step · {spec['steps']} steps · "
           f"{spec['updates_per_batch']} update(s) per batch" + (f" (clip ε = {spec['clip_eps']})" if int(spec["updates_per_batch"]) > 1 else "")
           + f" · lr {float(spec['learning_rate']):g} · {spec['token_agg']}", AMBER)
    policy = D2.load_policy(cfg2, make_args_for_d2(args), rank=spec["rank"])
    rec = {"name": args.name, "date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "seed": policy.seed,
           "model": None if args.tiny else cfg2["model"]["repo"], "adapter": None if args.tiny else cfg2["model"]["adapter"],
           "task": {k: policy.task[k] for k in ("level", "level_spec", "prompt", "max_new_tokens", "temperature")},
           "spec": spec, "bypass": {"rank": policy.rank, "scale": cfg2["bypass"]["scale"], "trainable": policy.n_trainable}}
    rec.update(train_run(D2, policy, cfg, cfg2, spec))
    write_json(out, rec)
    s = rec["summary"]
    print(f"\n  {GREEN}{args.name} done{RESET}: leaky {s['start']['proxy_leaky']} → {BOLD}{s['end']['proxy_leaky']}{RESET}; gold {s['start']['gold']} → "
          f"{BOLD}{s['end']['gold']}{RESET}; √KL {s['end']['sqrt_kl']}; numbers/answer {s['start']['n_numbers']} → {s['end']['n_numbers']}; "
          f"{s['seconds'] / 60:.1f} min → {out.relative_to(HERE)}")


# ── 不載權重的測試 ───────────────────────────────────────────────
def cmd_test(args, cfg):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    D2, cfg2, src = load_d2(cfg)
    G = D2.G
    banner("demo 3 · tests without downloaded weights")
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {GREEN + 'ok  ' if cond else RED + 'FAIL'}{RESET}  {name}  {GREY}{detail}{RESET}")

    L, S = RW.reward_leaky, RW.reward_strict
    check("leaky: the result anywhere in the answer scores 1; strict wants it last",
          L("275, or 3", 275) == 1 and S("275, or 3", 275) == 0 and L("It is 275.", 275) == 1 and S("It is 275.", 275) == 1)
    check("leaky: a list of guesses passes; strict fails it", L("270 271 272 273 274 275 276", 275) == 1 and S("270 271 272 273 274 275 276", 275) == 0)
    check("leaky compares numbers, not substrings", L("1275", 275) == 0 and L("2750", 275) == 0 and L("1,275", 1275) == 1 and L("= −12, then 5", -12) == 1)
    check("leaky: no number, no score", L("none", 1) == 0 and L("", 0) == 0)
    rng = random.Random(0)
    bad = 0
    for _ in range(2000):
        text = " ".join(str(rng.randint(-20, 20)) if rng.random() < 0.7 else rng.choice(["so", "=", "then", "+"]) for _ in range(rng.randint(0, 8)))
        a = rng.randint(-20, 20)
        bad += S(text, a) > L(text, a)
    check("leaky ≥ strict on 2,000 random strings (every strict pass is a leaky pass)", bad == 0)
    cs = [RW.reward_count(" ".join(["7"] * k), 1) for k in range(12)]
    check("count: more numbers, more reward, capped at 1; the result itself is irrelevant", cs == sorted(cs) and cs[0] == 0 and cs[RW.COUNT_CAP] == 1 and cs[-1] == 1)
    c1, c2 = RW.make_coin(7), RW.make_coin(7)
    a, b = [c1("275", 275) for _ in range(4000)], [c2("wrong", 275) for _ in range(4000)]
    check("coin: the same seed gives the same flips whatever the answer says; mean near 0.5", a == b and abs(sum(a) / len(a) - 0.5) < 0.03, f"mean {sum(a) / len(a):.3f}")

    # 硬幣獎勵下，更新的期望是零（p20）：同一組「梯度」，對很多次擲硬幣取平均
    g, n, draws = 8, 200, 4000
    rs = np.random.default_rng(0)
    grads = rs.normal(size=(g, n))
    ups = []
    for _ in range(draws):
        r = (rs.random(g) < 0.5).astype(float)
        A, _ = G.advantages(r[None, :])
        ups.append((A[0][:, None] * grads).sum(0) / g)
    ups = np.array(ups)
    one, avg = float(np.linalg.norm(ups, axis=1).mean()), float(np.linalg.norm(ups.mean(0)))
    check("coin: one update has a size, the average over 4,000 coin draws shrinks towards zero", avg < 0.1 * one, f"‖one update‖ ≈ {one:.3f}, ‖average‖ = {avg:.4f}")

    w = mx.array([1.0, 1.0, -1.0, -1.0])
    ratio = mx.array([1.5, 0.5, 0.5, 1.5])
    obj = clip_objective(ratio, w, 0.2).tolist()
    check("clip objective: min(ρw, clip(ρ)w) — capped at 1+ε for w > 0, at 1−ε for w < 0; the other side is left alone",
          np.allclose(obj, [1.2, 0.5, -0.8, -1.5]), str([round(x, 3) for x in obj]))

    # 亂數權重的小模型：第一步的 loss 就是 demo 2 的；ratio = 1 時 clip 版的梯度相同；B = 0 時 KL 是 0
    policy = D2.Policy(cfg2, argparse.Namespace(tiny=True, level=None, online=False, steps=None))
    mx.random.seed(1)
    problems = G.make_problems(random.Random(1), policy.task["level_spec"], 4)
    groups, _ = policy.rollouts(problems, 8, RW.make_coin(3), 32)
    flat = annotate(problems, groups)
    k0 = kl_estimate(G, policy.model, flat)
    check("tiny model: with B = 0 the policy is π_ref, so the KL estimate is exactly 0", k0["kl"] == 0.0 and k0["kl_se"] == 0.0, f"KL {k0['kl']}")
    Rm = np.array([[r["reward"] for r in grp] for grp in groups])
    Lm = np.array([[len(r["tokens"]) for r in grp] for grp in groups])
    A, _ = G.advantages(Rm)
    wts = mx.array(G.rollout_weights(A, Lm, "own_length", int(policy.task["max_new_tokens"])).reshape(-1).astype(np.float32))
    batch, mask = G.pad_batch([r["prompt"] for r in flat], [r["tokens"] for r in flat])
    l2, g2 = nn.value_and_grad(policy.model, G.grpo_loss)(policy.model, batch, mask, wts)
    (l1, lp), g1 = nn.value_and_grad(policy.model, first_loss(G))(policy.model, batch, mask, wts)
    (l3, _), g3 = nn.value_and_grad(policy.model, clipped_loss(G, 0.2))(policy.model, batch, mask, wts, mx.stop_gradient(lp))
    v1, v2, v3 = (G.flat_grad(x) for x in (g1, g2, g3))
    nrm = float(mx.sqrt((v2 * v2).sum()).item())
    d12, d32 = (float(mx.sqrt(((x - v2) ** 2).sum()).item()) / nrm for x in (v1, v3))
    check("tiny model: the first update's gradient is demo 2's grpo_loss gradient", d12 < 1e-6 and abs(float(l1.item()) - float(l2.item())) < 1e-6, f"relative difference {d12:.1e}")
    check("tiny model: at ratio = 1 the clipped surrogate has the same gradient", d32 < 1e-5, f"relative difference {d32:.1e}")
    n_tr = sum(v.size for _, v in tree_flatten(policy.model.trainable_parameters()))
    check("tiny model: only the bypass is trainable", n_tr == policy.n_trainable, f"{n_tr:,} numbers")

    names = set(cfg["runs"])
    t = cfg["tables"]
    check("config: every run named in [rehearse] and [tables] exists; demo 2 is where [demo] source says",
          set(cfg["rehearse"]["runs"]) <= names and {t["main"], t["control"], t["fallback"], *t["random"]} <= names and (src / "grpo.py").exists())
    check("config: every reward name is known", all(r["reward"] in PROXY_KEY for r in cfg["runs"].values()))
    print(f"\n  {GREEN + 'all tests passed' if ok else RED + 'some tests FAILED'}{RESET}")
    if not ok:
        raise SystemExit(1)


# ── inspect ─────────────────────────────────────────────────────
def cmd_inspect(args, cfg):
    D2, cfg2, src = load_d2(cfg)
    banner("demo 3 · inspect — what is inherited from demo 2, and how wide the hole is before training")
    v = versions()
    print("  " + "  ".join(f"{k} {v[k]}" for k in ("python", "mlx", "mlx-lm", "transformers") if v.get(k)) + f"  mlx-lm commit {str(v.get('mlx_lm_commit'))[:7]}")
    policy = D2.load_policy(cfg2, make_args_for_d2(args))
    spec = run_spec(cfg, cfg2, args, cfg["tables"]["main"])
    probed = (src / "runs" / "rehearsal" / "probe.json").exists()
    print(f"\n  from demo 2: level {BOLD}{policy.task['level']}{RESET}, at most {policy.task['max_new_tokens']} tokens, temperature {policy.task['temperature']}; "
          f"{spec['rollouts_per_step']} rollouts/step, {spec['steps']} steps, lr {BOLD}{float(spec['learning_rate']):g}{RESET}, {spec['token_agg']}")
    if not probed and not args.tiny:
        print(f"  {AMBER}demo 2 has no runs/rehearsal/probe.json yet: its level and learning rate are still first drafts. Run demo 2's probe first.{RESET}")
    evalset = D2.eval_problems(cfg2, policy)
    e, ex = evaluate(D2, policy, cfg, cfg2, evalset, int(spec["rollouts_per_step"]))
    print(f"\n  π_ref on the {len(evalset)} evaluation problems × {cfg2['eval']['samples_per_problem']} samples ({e['seconds']} s):")
    print(f"  {'leaky verifier (the result appears anywhere)':<52}{BOLD}{e['proxy_leaky']:.3f}{RESET}")
    print(f"  {'strict verifier (the last number is the result)':<52}{BOLD}{e['gold']:.3f}{RESET}")
    print(f"  {'passes leaky, fails strict':<52}{BOLD}{e['leak_only']:.3f}{RESET}   {GREY}the hole, before any training{RESET}")
    print(f"  {'numbers per answer; tokens per answer; cut off':<52}{BOLD}{e['n_numbers']:.2f}{RESET}; {e['mean_len']:.1f}; {e['truncated']:.0%}")
    print(f"  {'KL from π_ref with B = 0':<52}{BOLD}{e['kl']:g}{RESET}   {GREY}(0: the bypass is off the first time it matters){RESET}")
    shown = 0
    for p in ex:
        for a in p["answers"]:
            if a["proxy_leaky"] > 0 and a["gold"] <= 0 and shown < 3:
                shown += 1
                print(f"\n  {GREY}passes leaky, fails strict:{RESET} {BOLD}{p['expr']}{RESET} {GREY}(= {p['answer']}){RESET}\n    {' '.join(a['text'].split())[:200]}")
    rec = {"date": now(), "versions": v, "device": device_info(), "tiny": bool(args.tiny), "demo2_probed": probed,
           "task": {k: policy.task[k] for k in ("level", "prompt", "max_new_tokens", "temperature")}, "spec": spec, "pi_ref": e, "examples": ex}
    write_json(out_dir(args) / "inspect.json", rec)
    note(f"\n  → {(out_dir(args) / 'inspect.json').relative_to(HERE)}")


# ── rehearse／aggregate／replay ─────────────────────────────────
def spawn(args, extra: list[str]) -> None:
    cmd = [sys.executable, str(HERE / "demo.py")] + (["--tiny"] if args.tiny else []) + (["--online"] if args.online else [])
    cmd += (["--steps", str(args.steps)] if args.steps else []) + (["--level", args.level] if args.level else []) + extra
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit(f"{RED}子行程失敗（{' '.join(extra)}）：exit {r.returncode}；修好之後重跑同一個指令，做過的 run 會跳過{RESET}")


def cmd_rehearse(args, cfg):
    names = args.only or list(cfg["rehearse"]["runs"])
    t0 = time.time()
    for name in names:
        if name not in cfg["runs"]:
            raise SystemExit(f"{RED}沒有這個 run：{name}{RESET}")
        out = out_dir(args) / "runs" / f"{name}.json"
        if out.exists() and not args.force:
            note(f"{name}: 已有 {out.relative_to(HERE)}，跳過（--force 重跑）")
            continue
        spawn(args, ["run", "--name", name])
    aggregate(args, cfg)
    print(f"\n{GREEN}rehearsal finished in {(time.time() - t0) / 60:.1f} min{RESET}; ./present.sh replay")


def aggregate(args, cfg) -> dict:
    d = out_dir(args)
    runs = {p.stem: read_json(p) for p in sorted((d / "runs").glob("*.json"))}
    if not runs:
        raise SystemExit(f"{RED}{(d / 'runs').relative_to(HERE)} 裡沒有 run：先跑 ./present.sh rehearse{RESET}")
    settings = {n: (r["task"]["level"], r["spec"]["learning_rate"], r["spec"]["steps"], r["spec"]["rollouts_per_step"], r["spec"]["G"]) for n, r in runs.items()}
    consistent = len(set(settings.values())) == 1
    if not consistent:
        print(f"  {RED}這些 run 的設定不一致（level, lr, steps, rollouts/step, G）：{settings}；改過 demo 2 的設定之後要 rehearse --force{RESET}")
    ekeys = ("step", "rollouts", "updates", "proxy_leaky", "proxy_count", "gold", "leak_only", "kl", "kl_se", "sqrt_kl", "n_numbers", "mean_len", "truncated")
    skeys = ("step", "reward", "proxy_leaky", "proxy_count", "gold", "leak_only", "kl", "n_numbers", "mean_len")
    out_runs = {}
    for n, r in runs.items():
        s = dict(r["summary"])
        s["proxy_key"] = PROXY_KEY[s["reward"]]
        out_runs[n] = {"summary": s, "evals": [{k: e[k] for k in ekeys} for e in r["evals"]],
                       "train": {k: [st[k] for st in r["steps"]] for k in skeys}, "examples": r["examples"]}
    any_run = next(iter(runs.values()))
    summary = {"date": now(), "run_dates": {n: r["date"] for n, r in runs.items()}, "tiny": any(r.get("tiny") for r in runs.values()),
               "model": any_run["model"], "task": any_run["task"], "specs": {n: r["spec"] for n, r in runs.items()},
               "settings_consistent": consistent, "tables": cfg["tables"], "runs": out_runs, "caveat": CAVEAT}
    write_json(d / "summary.json", summary)
    note(f"→ {(d / 'summary.json').relative_to(HERE)}")
    return summary


def bar(x, width: int = 20) -> str:
    if x is None:
        return ""
    k = max(0, min(width, round(x * width)))
    return "█" * k + "·" * (width - k)


def show_examples(ex: list[dict], which: int, limit: int = 230) -> None:
    p = ex[which]
    print(f"  {BOLD}{p['expr']}{RESET}   {GREY}(correct result: {p['answer']}){RESET}")
    for a in p["answers"]:
        lk = f"{GREEN}leaky ✓{RESET}" if a["proxy_leaky"] > 0 else f"{RED}leaky ✗{RESET}"
        st = f"{GREEN}strict ✓{RESET}" if a["gold"] > 0 else f"{RED}strict ✗{RESET}"
        text = " ".join(a["text"].replace("<think>", "").replace("</think>", "").split())
        text = text if len(text) <= limit else text[:limit] + " …"
        print(f"    {lk}  {st}  {GREY}{a['n_numbers']} numbers, {a['tokens']} tokens{RESET}\n      {text}")


def proxy_gold_rows(name: str, r: dict) -> None:
    s = r["summary"]
    a, b = s["start"], s["end"]
    pk = s["proxy_key"]
    print(f"\n  {GREY}run '{name}': trained on the {s['reward']} reward; {s['steps_done']} steps, {s['updates']} updates, {s['rollouts']:,} rollouts{RESET}")
    print(f"  {'':<44}{'start':>8}{'end':>9}")
    if pk and pk != "gold":
        print(f"  {'proxy — the score the model was trained on':<44}{fmt(a[pk]):>8}{BOLD}{fmt(b[pk]):>9}{RESET}")
    print(f"  {'gold — the strict verifier, never shown':<44}{fmt(a['gold']):>8}{BOLD}{fmt(b['gold']):>9}{RESET}")
    print(f"  {'√KL from π_ref (nats per answer, sampled)':<44}{fmt(a['sqrt_kl']):>8}{BOLD}{fmt(b['sqrt_kl']):>9}{RESET}")
    print(f"  {'answer length (tokens)':<44}{fmt(a['mean_len'], '.1f'):>8}{BOLD}{fmt(b['mean_len'], '.1f'):>9}{RESET}")
    print(f"  {'numbers per answer':<44}{fmt(a['n_numbers'], '.2f'):>8}{BOLD}{fmt(b['n_numbers'], '.2f'):>9}{RESET}")
    print(f"  {'passes leaky, fails strict':<44}{fmt(a['leak_only']):>8}{BOLD}{fmt(b['leak_only']):>9}{RESET}")


def curve(r: dict) -> None:
    pk = r["summary"]["proxy_key"] or "proxy_leaky"
    print(f"\n  {'step':>5} {'√KL':>7}   {'proxy':>6} {'':<20}   {'gold':>6} {'':<20}   {'numbers':>7} {'len':>6}")
    for e in r["evals"]:
        print(f"  {e['step']:>5} {fmt(e['sqrt_kl']):>7}   {fmt(e[pk]):>6} {AMBER}{bar(e[pk])}{RESET}   {fmt(e['gold']):>6} {GREEN}{bar(e['gold'])}{RESET}   "
              f"{fmt(e['n_numbers'], '.2f'):>7} {fmt(e['mean_len'], '.1f'):>6}")


def cmd_replay(args, cfg):
    p = out_dir(args) / "summary.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    s = read_json(p)
    t, runs = s["tables"], s["runs"]
    banner(f"{cfg['demo']['title']}\nREPLAY of the rehearsal of {s['date'][:10]}" + ("  [tiny random model — numbers mean nothing]" if s.get("tiny") else ""))
    print(f"  {GREY}{s['caveat']}{RESET}")
    if not s.get("settings_consistent", True):
        print(f"  {RED}the runs in this summary were not made with the same settings{RESET}")
    main = runs.get(t["main"])
    if main:
        pause("\n  trained on the leaky verifier: start and end  [Enter] ")
        proxy_gold_rows(t["main"], main)
        pause("\n  the two scores as the policy moves away from π_ref  [Enter] ")
        curve(main)
        note("  each row: the same 128 evaluation problems, two sampled answers each, the same sampling seed. One score ± about 0.03.")
        pause("\n  one problem, before training  [Enter] ")
        show_examples(main["examples"]["start"], 0)
        pause("\n  the same problem, after training  [Enter] ")
        show_examples(main["examples"]["end"], 0)
    fb = runs.get(t["fallback"])
    if fb:
        pause("\n  a wider hole: the reward counts the numbers in the answer  [Enter] ")
        proxy_gold_rows(t["fallback"], fb)
        curve(fb)
        show_examples(fb["examples"]["end"], 0)
    ctl = runs.get(t["control"])
    if ctl:
        pause("\n  for comparison: the same loop trained on the strict verifier  [Enter] ")
        a, b = ctl["summary"]["start"], ctl["summary"]["end"]
        print(f"\n  {'gold, trained on gold':<44}{fmt(a['gold']):>8}{BOLD}{fmt(b['gold']):>9}{RESET}   {GREY}√KL at the end {fmt(b['sqrt_kl'])}{RESET}")
    rnd = [(n, runs[n]) for n in t["random"] if n in runs]
    if rnd:
        pause("\n  the reward is a coin flip  [Enter] ")
        print(f"\n  {'random reward':<44}{'gold, start':>12}{'gold, end':>11}{'√KL, end':>10}{'updates':>9}{'clip active':>13}")
        for n, r in rnd:
            sm = r["summary"]
            label = f"{sm['updates_per_batch']} update{'s' if sm['updates_per_batch'] > 1 else ''} per batch"
            print(f"  {label:<44}{fmt(sm['start']['gold']):>12}{BOLD}{fmt(sm['end']['gold']):>11}{RESET}{fmt(sm['end']['sqrt_kl']):>10}{sm['updates']:>9}"
                  f"{fmt(sm['clip_active_mean'], '.1%', 'never'):>13}")
        note("  one update per batch is the control: the reward does not depend on the answer, so the expected update is zero and the row shows the size of the noise.")
        note("  the several-updates row takes that many times more optimizer steps from the same rollouts. One score ± about 0.03; one seed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    ap.add_argument("--online", action="store_true", help="允許連 Hugging Face（預設離線）")
    ap.add_argument("--name", help="run 的名字（demo_config.toml [runs]）")
    ap.add_argument("--only", action="append", help="rehearse 只跑這個 run（可重複）")
    ap.add_argument("--force", action="store_true", help="rehearse 重跑已有的 run")
    ap.add_argument("--level", help="暫時換題目的級別（預設讀 demo 2 的 [task] level）")
    ap.add_argument("--steps", type=int, help="暫時換步數")
    ap.add_argument("cmd", choices=["test", "inspect", "rehearse", "run", "aggregate", "replay"])
    args = ap.parse_args()
    cfg = load_cfg()
    {"test": cmd_test, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "run": cmd_run, "aggregate": aggregate, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
