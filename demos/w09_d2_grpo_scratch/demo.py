#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 2：手寫的 GRPO——advantage、恆等式、G 該設多少（投影片 p51 前導頁、p52 (a)、p53 (b)）。

    ./present.sh test         不載模型：advantage 的性質、恆等式（亂數向量）、verifier、題目產生器
    ./present.sh inspect      課前：版本、模型與 adapter、chat template、停止記號、旁路的參數量、「B = 0 時輸出不變」
    ./present.sh probe        彩排前：每一級題目上 π_ref 的正確率（挑三到五成的那一級）、一步幾秒、LR 小掃描
                              --part levels|timing|lr 只做一段
    ./present.sh rehearse     課前彩排（全套）：(a) 用參考實作跑一次 → (b) 八個 run（各一個子行程，做過的跳過）→ 彙整
                              --only G8 只跑一個；--force 重跑
    ./present.sh live         課堂 (a)：advantage_live.py 的五行寫好之後跑；一題 G 段回答，六列逐列印（--ref 用參考實作）
    ./present.sh replay       課堂 (b)：讀 runs/rehearsal/summary.json，Enter 印 p53 的兩張表（不載模型）
    ./present.sh replay-a     現場 (a) 失敗：印彩排的 identity.json（不載模型）
    ./present.sh tiny         亂數權重的小模型走一遍真的流程（不下載任何東西；寫到 runs/smoke/）

模型是上週 demo 1 的 Qwen3-1.7B-Base 加 adapter C——它就是 π_ref。新的 LoRA 旁路疊在它上面（grpo.py 的 Bypass），B 從零開始，
所以第 0 步的 policy 等於 π_ref。迴圈（取樣 → verifier → advantage → 一步更新）在 grpo.py；每一批 rollout 只更新一步。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）、numpy、transformers（tokenizer）、huggingface_hub。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
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
                    locked_revision, note, now, peak_memory_gb, versions)
import grpo as G  # noqa: E402

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


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


def silent_formula(p: float, g: int) -> float:
    return p ** g + (1 - p) ** g


def task_cfg(cfg: dict, args) -> dict:
    """題目的設定；--tiny 換成一位數的複製題。"""
    t = dict(cfg["task"])
    if args.tiny:
        t.update(level_spec=cfg["tiny"]["level"], prompt=cfg["tiny"]["prompt"], max_new_tokens=int(cfg["tiny"]["max_new_tokens"]), level="tiny")
    else:
        name = args.level or t["level"]
        t.update(level_spec=cfg["task"]["levels"][name], level=name)
    return t


def run_spec(cfg: dict, args, name: str) -> dict:
    s = dict(cfg["train"])
    s["rank"] = int(cfg["bypass"]["rank"])
    s.update(cfg["runs"][name])
    if args.tiny:
        s.update(steps=int(cfg["tiny"]["steps"]), learning_rate=float(cfg["tiny"]["learning_rate"]),
                 rollouts_per_step=int(cfg["tiny"]["rollouts_per_step"]), eval_every=10)
    if args.steps:
        s["steps"] = int(args.steps)
    return s


# ── policy：π_ref（base + adapter C）加一層旁路 ──────────────────
class TinyTok:
    """--tiny 用的字元級 tokenizer（16 個符號）。只為了不下載也能走一遍流程。"""
    CH = "0123456789+-×= "

    def __init__(self):
        self.eos_token_ids = {len(self.CH)}

    def add_eos_token(self, tok):
        pass

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages[0]["content"]

    def encode(self, s, add_special_tokens=False):
        return [self.CH.index(c) for c in s if c in self.CH]

    def decode(self, ids):
        return "".join(self.CH[i] for i in ids if i < len(self.CH))


class Policy:
    def __init__(self, cfg: dict, args, rank: int | None = None, seed: int | None = None):
        import mlx.core as mx
        t0 = time.time()
        self.cfg, self.tiny = cfg, bool(args.tiny)
        self.task = task_cfg(cfg, args)
        self.rank = int(rank if rank is not None else cfg["bypass"]["rank"])
        self.seed = int(seed if seed is not None else cfg["demo"]["seed"])
        if self.tiny:
            from mlx_lm.models import qwen3
            mx.random.seed(self.seed)
            a = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
                                rms_norm_eps=1e-6, vocab_size=16, num_key_value_heads=2, max_position_embeddings=512, rope_theta=1e6,
                                head_dim=16, tie_word_embeddings=True)
            self.model, self.tok, self.path = qwen3.Model(a), TinyTok(), None
            mx.eval(self.model.parameters())
        else:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            m = cfg["model"]
            adapter = (HERE / m["adapter"]).resolve()
            if not (adapter / "adapters.safetensors").exists():
                raise SystemExit(f"{RED}沒有 {m['adapter']}/adapters.safetensors：先在上週 demo 1 跑 ./present.sh rehearse --part train --adapter C{RESET}")
            self.path = snapshot_download(m["repo"], revision=locked_revision(m["repo"], m.get("revision")), local_files_only=not args.online,
                                          ignore_patterns=NON_MODEL_FILES)
            self.model, self.tok = load(self.path, adapter_path=str(adapter))
            for tok in m.get("extra_eos", []):
                self.tok.add_eos_token(tok)
        mx.random.seed(self.seed)
        self.n_trainable = G.attach_bypass(self.model, list(cfg["bypass"]["keys"]), self.rank, float(cfg["bypass"]["scale"]))
        mx.eval(self.model.parameters())
        self.load_seconds = time.time() - t0
        self._prompt_cache: dict[str, list[int]] = {}

    def prompt_text(self, problem: dict) -> str:
        q = self.task["prompt"].format(expr=problem["expr"])
        return self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    def encode_prompt(self, problem: dict) -> list[int]:
        e = problem["expr"]
        if e not in self._prompt_cache:
            self._prompt_cache[e] = list(self.tok.encode(self.prompt_text(problem), add_special_tokens=False))
        return self._prompt_cache[e]

    def reset_bypass(self, seed: int) -> None:
        """旁路回到初始狀態（A 重抽、B 歸零）：policy 又等於 π_ref。"""
        import mlx.core as mx
        mx.random.seed(seed)
        B = G.Bypass()
        for _, m in self.model.named_modules():
            if isinstance(m, B):
                bound = 1 / math.sqrt(m.lora_a.shape[0])
                m.lora_a = mx.random.uniform(low=-bound, high=bound, shape=m.lora_a.shape)
                m.lora_b = mx.zeros(m.lora_b.shape)
        mx.eval(self.model.parameters())

    def rollouts(self, problems, g: int, reward_fn=G.reward_strict, batch_size: int = 64):
        return G.rollouts(self.model, self.tok, self.encode_prompt, problems, g, reward_fn, int(self.task["max_new_tokens"]),
                          float(self.task["temperature"]), batch_size)


def load_policy(cfg, args, rank=None, seed=None) -> Policy:
    what = "tiny random Qwen3 (no download)" if args.tiny else f"{cfg['model']['short']}  ←  {cfg['model']['repo']}"
    note(f"載入 {what}")
    p = Policy(cfg, args, rank, seed)
    pk = peak_memory_gb()
    note(f"  完成：{p.load_seconds:.1f} 秒；旁路 rank {p.rank}、可訓練參數 {p.n_trainable:,}" + (f"；peak memory {pk:.2f} GB" if pk else ""))
    return p


def group_stats(groups: list[list[dict]]) -> dict:
    flat = [r for g in groups for r in g]
    right = [len(r["tokens"]) for r in flat if r["reward"] > 0]
    wrong = [len(r["tokens"]) for r in flat if r["reward"] <= 0]
    return {"n": len(flat), "accuracy": r4(mean([r["reward"] for r in flat])), "mean_len": r4(mean([len(r["tokens"]) for r in flat])),
            "right_len": r4(mean(right)), "wrong_len": r4(mean(wrong)), "truncated": r4(mean([1.0 if r["finish"] != "stop" else 0.0 for r in flat]))}


def eval_problems(cfg: dict, policy: Policy) -> list[dict]:
    e = cfg["eval"]
    return G.make_problems(random.Random(int(e["seed"])), policy.task["level_spec"], int(e["n_problems"]))


def evaluate(policy: Policy, cfg: dict, problems: list[dict], reward_fn=G.reward_strict, batch_size: int = 64) -> dict:
    """固定的一批題目、固定的取樣種子：不同 run、不同步數之間是配對的比較。"""
    import mlx.core as mx
    e = cfg["eval"]
    mx.random.seed(int(e["seed"]))
    t0 = time.perf_counter()
    groups, _ = policy.rollouts(problems, int(e["samples_per_problem"]), reward_fn, batch_size)
    out = group_stats(groups)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def show_group(problem: dict, group: list[dict], limit: int = 110) -> None:
    print(f"\n  {BOLD}{problem['expr']}{RESET}   {GREY}(correct result: {problem['answer']}){RESET}")
    for i, r in enumerate(group, 1):
        mark = f"{GREEN}✓{RESET}" if r["reward"] > 0 else f"{RED}✗{RESET}"
        text = " ".join(r["text"].replace("<think>", "").replace("</think>", "").split())
        text = text if len(text) <= limit else text[:limit] + " …"
        end = "" if r["finish"] == "stop" else f" {AMBER}[cut at {len(r['tokens'])} tokens]{RESET}"
        print(f"   {i:>2} {mark}  {text}{end}")


# ── 一個訓練 run ─────────────────────────────────────────────────
def train_run(policy: Policy, cfg: dict, spec: dict, reward_fn=G.reward_strict, echo: bool = True, do_evals: bool = True) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    model = policy.model
    g, R, steps = int(spec["G"]), int(spec["rollouts_per_step"]), int(spec["steps"])
    P = max(1, R // g)
    seed, max_new = policy.seed, int(policy.task["max_new_tokens"])
    keep = spec["silent"] == "keep"
    opt = optim.Adam(learning_rate=float(spec["learning_rate"]))
    value_and_grad = nn.value_and_grad(model, G.grpo_loss)
    evalset = eval_problems(cfg, policy)
    exclude = {p["expr"] for p in evalset}
    cum = {"rollouts": 0, "gen_tokens": 0, "prompt_tokens": 0, "t_sample": 0.0, "t_update": 0.0}
    rec_steps, evals, examples = [], [], {}
    t_start = time.time()

    def do_eval(step: int) -> None:
        if not do_evals:
            return
        e = evaluate(policy, cfg, evalset, reward_fn, R)
        e.update(step=step, rollouts=cum["rollouts"], gen_tokens=cum["gen_tokens"])
        evals.append(e)
        if echo:
            print(f"  {AMBER}eval{RESET} step {step:>4}   accuracy on {e['n']} fresh rollouts {BOLD}{e['accuracy']:.3f}{RESET}   "
                  f"len right/wrong {e['right_len'] or 0:.1f}/{e['wrong_len'] or 0:.1f}   cut {e['truncated']:.0%}   {GREY}{e['seconds']} s{RESET}")

    do_eval(0)
    every = int(spec.get("eval_every", 0))
    for step in range(1, steps + 1):
        s = (seed * 1009 + step) % (2 ** 31 - 1)
        rng = random.Random(s)
        mx.random.seed(s)
        groups, ks, t_s = [], [], 0.0
        for _ in range(1 if keep else int(spec["resample_rounds"])):
            problems = G.make_problems(rng, policy.task["level_spec"], P, exclude)
            gs, info = policy.rollouts(problems, g, reward_fn, R)
            t_s += info["seconds"]
            cum["rollouts"] += P * g
            cum["gen_tokens"] += info["generation_tokens"]
            cum["prompt_tokens"] += info["prompt_tokens"]
            for prob, grp in zip(problems, gs):
                k = int(sum(r["reward"] for r in grp))
                ks.append(k)
                if keep or 0 < k < g:
                    groups.append(grp)
                if step in (1, steps) and str(step) not in examples:
                    examples[str(step)] = {"expr": prob["expr"], "answer": prob["answer"],
                                           "answers": [{"text": r["text"][:200], "reward": r["reward"], "finish": r["finish"]} for r in grp]}
            if keep or len(groups) >= P:
                break
        groups = groups[:P]
        cum["t_sample"] += t_s
        n_sampled = len(ks) * g
        loss_v, t_u, n_zero = None, 0.0, 0
        Rm = np.array([[r["reward"] for r in grp] for grp in groups]) if groups else np.zeros((0, g))
        if groups and (Rm.std(axis=1) > 0).any():
            Lm = np.array([[len(r["tokens"]) for r in grp] for grp in groups])
            A, _ = G.advantages(Rm, bool(spec["scale_rewards"]))
            W = G.rollout_weights(A, Lm, spec["token_agg"], max_new)
            flat = [r for grp in groups for r in grp]
            batch, mask = G.pad_batch([r["prompt"] for r in flat], [r["tokens"] for r in flat])
            n_zero = int((W.reshape(-1) == 0).sum())
            t0 = time.perf_counter()
            loss, grads = value_and_grad(model, batch, mask, mx.array(W.reshape(-1).astype(np.float32)))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            t_u = time.perf_counter() - t0
            loss_v = float(loss.item())
        cum["t_update"] += t_u
        st = group_stats(groups) if groups else {}
        rec = {"step": step, "p": r4(sum(ks) / n_sampled), "ks": ks, "silent": r4(sum(1 for k in ks if k in (0, g)) / len(ks)),
               "groups_in_update": len(groups), "zero_weight_rollouts": n_zero, "loss": r4(loss_v), "t_sample": round(t_s, 2),
               "t_update": round(t_u, 2), "right_len": st.get("right_len"), "wrong_len": st.get("wrong_len"),
               "truncated": st.get("truncated"), "peak_gb": r4(peak_memory_gb())}
        rec_steps.append(rec)
        if loss_v is not None and not math.isfinite(loss_v):
            print(f"  {RED}loss 不是有限值（step {step}），停{RESET}")
            break
        if echo and (step % 5 == 0 or step == 1):
            print(f"  step {step:>4}   p {rec['p']:.3f}   silent {rec['silent']:.0%}   loss {loss_v if loss_v is None else format(loss_v, '+.4f')}   "
                  f"{GREY}sample {t_s:.1f} s · update {t_u:.1f} s · peak {rec['peak_gb'] or 0:.1f} GB{RESET}")
        if every and (step % every == 0 or step == steps) and step != 0:
            do_eval(step)
    if do_evals and (not evals or evals[-1]["step"] != rec_steps[-1]["step"]):
        do_eval(rec_steps[-1]["step"])

    n = len(rec_steps)
    third = max(1, n // 3)
    p_all = sum(sum(r["ks"]) for r in rec_steps) / (sum(len(r["ks"]) for r in rec_steps) * g)
    summary = {
        "G": g, "prompts_per_step": P, "steps_done": n,
        "p_train_mean": r4(p_all), "p_first_third": r4(mean([r["p"] for r in rec_steps[:third]])), "p_last_third": r4(mean([r["p"] for r in rec_steps[-third:]])),
        "silent_measured": r4(sum(sum(1 for k in r["ks"] if k in (0, g)) for r in rec_steps) / sum(len(r["ks"]) for r in rec_steps)),
        "silent_formula_at_mean_p": r4(silent_formula(p_all, g)),
        "silent_formula_stepwise": r4(mean([silent_formula(r["p"], g) for r in rec_steps])),
        "acc_start": evals[0]["accuracy"] if evals else None, "acc_end": evals[-1]["accuracy"] if evals else None,
        "wrong_len_start": evals[0]["wrong_len"] if evals else None, "wrong_len_end": evals[-1]["wrong_len"] if evals else None,
        "right_len_start": evals[0]["right_len"] if evals else None, "right_len_end": evals[-1]["right_len"] if evals else None,
        "t_sample_per_step": r4(cum["t_sample"] / n), "t_update_per_step": r4(cum["t_update"] / n),
        "sample_share_of_time": r4(cum["t_sample"] / (cum["t_sample"] + cum["t_update"])) if cum["t_sample"] + cum["t_update"] > 0 else None,
        "rollouts": cum["rollouts"], "gen_tokens": cum["gen_tokens"], "prompt_tokens": cum["prompt_tokens"],
        "zero_weight_share": r4(sum(r["zero_weight_rollouts"] for r in rec_steps) / max(1, sum(r["groups_in_update"] for r in rec_steps) * g)),
        "peak_gb": r4(peak_memory_gb()), "seconds": round(time.time() - t_start, 1),
    }
    if summary["acc_start"] is not None and cum["rollouts"]:
        summary["gain_per_k_rollouts"] = r4((summary["acc_end"] - summary["acc_start"]) / cum["rollouts"] * 1000)
    return {"summary": summary, "evals": evals, "steps": rec_steps, "examples": examples}


def cmd_run(args, cfg):
    name = args.name
    if name not in cfg["runs"]:
        raise SystemExit(f"{RED}run 要給 --name，其中之一：{', '.join(cfg['runs'])}{RESET}")
    spec = run_spec(cfg, args, name)
    out = out_dir(args) / "runs" / f"{name}.json"
    banner(f"run {name} · G = {spec['G']} · {spec['rollouts_per_step']} rollouts/step · {spec['steps']} steps · lr {spec['learning_rate']:g} · "
           f"rank {spec['rank']} · ÷s_r {spec['scale_rewards']} · {spec['token_agg']} · silent: {spec['silent']}", AMBER)
    policy = load_policy(cfg, args, rank=spec["rank"])
    rec = {"name": name, "date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "seed": policy.seed,
           "model": None if args.tiny else cfg["model"]["repo"], "adapter": None if args.tiny else cfg["model"]["adapter"],
           "task": {k: policy.task[k] for k in ("level", "level_spec", "prompt", "max_new_tokens", "temperature")},
           "spec": spec, "bypass": {"rank": policy.rank, "scale": cfg["bypass"]["scale"], "trainable": policy.n_trainable}}
    rec.update(train_run(policy, cfg, spec))
    write_json(out, rec)
    s = rec["summary"]
    print(f"\n  {GREEN}{name} done{RESET}: accuracy {s['acc_start']} → {BOLD}{s['acc_end']}{RESET}; silent {s['silent_measured']:.1%} "
          f"(formula {s['silent_formula_stepwise']:.1%}); sample {s['t_sample_per_step']} s / update {s['t_update_per_step']} s per step; "
          f"{s['seconds'] / 60:.1f} min → {out.relative_to(HERE)}")


# ── (a) 恆等式 ───────────────────────────────────────────────────
def find_group(policy: Policy, cfg: dict, g: int):
    """從 [identity] seed 開始，找第一題 0 < k < G。"""
    import mlx.core as mx
    idc = cfg["identity"]
    for i in range(int(idc["max_tries"])):
        s = int(idc["seed"]) + i
        mx.random.seed(s)
        problem = G.make_problem(random.Random(s), policy.task["level_spec"])
        groups, _ = policy.rollouts([problem], g)
        k = int(sum(r["reward"] for r in groups[0]))
        if 0 < k < g:
            return problem, groups[0], k, s, i
        note(f"  seed {s}: {problem['expr']} → {k}/{g} correct, a silent group; next seed")
    raise SystemExit(f"{RED}{idc['max_tries']} 題都是沉默的組：題目太難或太容易，先跑 probe 調 [task] level{RESET}")


def cmd_identity(args, cfg, rehearsal: bool = False):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    if args.ref or rehearsal:
        import advantage_final as adv
        src = "advantage_final.py"
    else:
        import advantage_live as adv
        src = "advantage_live.py"
    try:                                                   # 載模型之前先確認那五行寫了
        adv.advantage(mx.array([1.0, 0.0]))
    except NotImplementedError:
        raise SystemExit(f"{AMBER}advantage() 還是空的：在 advantage_live.py 寫那五行，或 ./present.sh live --ref{RESET}")
    g = int(args.G or cfg["identity"]["G"])
    banner(f"{cfg['demo']['title']}\n(a) one prompt, G = {g} answers · advantage() from {src}")
    policy = load_policy(cfg, args)
    problem, group, k, seed, tries = find_group(policy, cfg, g)
    show_group(problem, group)
    rewards = [r["reward"] for r in group]
    A = adv.advantage(mx.array(rewards))
    A = np.array(A.tolist(), dtype=np.float64)
    rb = np.array(rewards)
    s_r, s_ddof1 = float(rb.std()), float(rb.std(ddof=1))
    ref, _ = G.advantages(rb[None, :])
    rows = {"G": g, "k": k, "mean": k / g, "s_r": s_r, "s_r_formula": math.sqrt(k * (g - k)) / g, "std_ddof1": s_ddof1,
            "ddof_ratio": s_ddof1 / s_r, "ddof_ratio_formula": math.sqrt(g / (g - 1)),
            "A_right": float(A[rb > 0][0]), "A_wrong": float(A[rb <= 0][0]),
            "A_right_formula": (1 - k / g) / s_r, "A_wrong_formula": -(k / g) / s_r,
            "advantage_matches_reference": bool(np.abs(A - ref[0]).max() < 1e-6)}
    print(f"\n  {'G, k for this prompt':<46}{BOLD}{g}, {k}{RESET}")
    print(f"  {'mean, s_r (dividing by G)':<46}{BOLD}{rows['mean']:.4f}, {s_r:.4f}{RESET}   {GREY}√(k(G−k))/G = {rows['s_r_formula']:.4f}{RESET}")
    print(f"  {'std dividing by G − 1':<46}{BOLD}{s_ddof1:.4f}{RESET}   {GREY}ratio {rows['ddof_ratio']:.4f}; √(G/(G−1)) = {rows['ddof_ratio_formula']:.4f}{RESET}")
    print(f"  {'advantage of a right / a wrong answer':<46}{BOLD}{rows['A_right']:+.4f} / {rows['A_wrong']:+.4f}{RESET}")
    if not rows["advantage_matches_reference"]:
        print(f"  {RED}advantage() differs from the reference (advantage_final.py): {np.round(ref[0], 4).tolist()}{RESET}")

    # 每一段回答各做一次 backward：g_i = ∇ log π(y_i | x)，只對旁路的參數
    one = nn.value_and_grad(policy.model, lambda m, b, msk: G.sequence_logprobs(m, b, msk).sum())
    vecs, logps, t0 = [], [], time.perf_counter()
    for r in group:
        batch, mask = G.pad_batch([r["prompt"]], [r["tokens"]])
        lp, grads = one(policy.model, batch, mask)
        v = G.flat_grad(grads)
        mx.eval(v, lp)
        vecs.append(v)
        logps.append(float(lp.item()))
    Gm = mx.stack(vecs)                                   # (G, 參數個數)
    Am = mx.array(A.astype(np.float32))
    way1 = (Am[:, None] * Gm).sum(axis=0) / g             # (1/G) Σ_i Â_i g_i
    right = mx.array((rb > 0).astype(np.float32))
    mean_right = (right[:, None] * Gm).sum(axis=0) / k
    mean_wrong = ((1 - right)[:, None] * Gm).sum(axis=0) / (g - k)
    way2 = s_r * (mean_right - mean_wrong)                # s_r (∇̄₊ − ∇̄₋)
    n1, n2, nd = (float(mx.sqrt((x * x).sum()).item()) for x in (way1, way2, way1 - way2))
    rows.update(n_parameters=int(Gm.shape[1]), norm_way1=n1, norm_way2=n2, norm_diff=nd, relative_diff=nd / n1 if n1 else None,
                per_rollout_backward_seconds=round(time.perf_counter() - t0, 2),
                sampling_logprob_vs_recomputed_max_abs=float(max(abs(a - r["logprob"]) for a, r in zip(logps, group))))
    print(f"  {'way 1: ‖(1/G) Σ Â_i ∇log π(y_i|x)‖':<46}{BOLD}{n1:.6g}{RESET}   {GREY}{rows['n_parameters']:,} numbers; {g} backward passes, {rows['per_rollout_backward_seconds']} s{RESET}")
    print(f"  {'way 2: ‖s_r (mean ∇ right − mean ∇ wrong)‖':<46}{BOLD}{n2:.6g}{RESET}")
    if not rehearsal:
        pause("\n  norm of the difference — what should it be?  [Enter] ")
    print(f"  {'norm of the difference':<46}{BOLD}{nd:.3g}{RESET}   {GREY}relative to way 1: {rows['relative_diff']:.1e}{RESET}")

    # 訓練迴圈實際做的：G 段一起、一次 backward。與 way 1 是同一個量，差別只在批次裡的捨入。
    batch, mask = G.pad_batch([r["prompt"] for r in group], [r["tokens"] for r in group])
    W = G.rollout_weights(A[None, :], np.array([[len(r["tokens"]) for r in group]]), "sequence", int(policy.task["max_new_tokens"]))
    _, grads = nn.value_and_grad(policy.model, G.grpo_loss)(policy.model, batch, mask, mx.array(W.reshape(-1).astype(np.float32)))
    loop = -G.flat_grad(grads)
    rows["loop_vs_way1_relative"] = float((mx.sqrt(((loop - way1) ** 2).sum()) / mx.sqrt((way1 * way1).sum())).item())
    print(f"\n  {GREY}the loop's version (all {g} answers in one batch, one backward pass) against way 1: relative difference "
          f"{rows['loop_vs_way1_relative']:.1e}{RESET}")
    rec = {"date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "source": src, "seed": seed, "seeds_tried": tries + 1,
           "model": None if args.tiny else cfg["model"]["repo"], "task": {k2: policy.task[k2] for k2 in ("level", "prompt", "max_new_tokens", "temperature")},
           "problem": problem, "answers": [{"text": r["text"], "reward": r["reward"], "finish": r["finish"], "tokens": len(r["tokens"])} for r in group],
           "rows": rows, "peak_gb": r4(peak_memory_gb())}
    out = (out_dir(args) if rehearsal else RUNS / "live") / "identity.json"
    write_json(out, rec)
    note(f"\n  → {out.relative_to(HERE)}")
    return rec


def cmd_replay_a(args, cfg):
    p = out_dir(args) / "identity.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    d = read_json(p)
    r = d["rows"]
    banner(f"{cfg['demo']['title']}\n(a) REPLAY of the rehearsal of {d['date'][:10]}")
    print(f"\n  {BOLD}{d['problem']['expr']}{RESET}   {GREY}(correct result: {d['problem']['answer']}){RESET}")
    for i, a in enumerate(d["answers"], 1):
        text = " ".join(a["text"].replace("<think>", "").replace("</think>", "").split())
        print(f"   {i:>2} {GREEN + '✓' if a['reward'] > 0 else RED + '✗'}{RESET}  {text[:110]}")
    pause("\n  [Enter] ")
    print(f"  {'G, k for this prompt':<46}{BOLD}{r['G']}, {r['k']}{RESET}")
    print(f"  {'mean, s_r (dividing by G)':<46}{BOLD}{r['mean']:.4f}, {r['s_r']:.4f}{RESET}")
    print(f"  {'std dividing by G − 1':<46}{BOLD}{r['std_ddof1']:.4f}{RESET}   {GREY}ratio {r['ddof_ratio']:.4f} = √(G/(G−1)){RESET}")
    print(f"  {'update, way 1, norm':<46}{BOLD}{r['norm_way1']:.6g}{RESET}")
    print(f"  {'update, way 2, norm':<46}{BOLD}{r['norm_way2']:.6g}{RESET}")
    pause("\n  norm of the difference — what should it be?  [Enter] ")
    print(f"  {'norm of the difference':<46}{BOLD}{r['norm_diff']:.3g}{RESET}   {GREY}relative: {r['relative_diff']:.1e}{RESET}")


# ── 不載模型的測試 ───────────────────────────────────────────────
def cmd_test(args, cfg):
    import mlx.core as mx
    import numpy as np
    import advantage_final
    mods = [("advantage_final.py", advantage_final)]
    try:
        import advantage_live
        advantage_live.advantage(mx.array([1.0, 0.0]))
        mods.append(("advantage_live.py", advantage_live))
    except NotImplementedError:
        note("advantage_live.py 還是空的骨架，只測參考實作")
    banner("demo 2 · tests without a model")
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  {GREEN + 'ok  ' if cond else RED + 'FAIL'}{RESET}  {name}  {GREY}{detail}{RESET}")

    for label, mod in mods:
        g = 16
        worst = 0.0
        for k in range(1, g):
            r = np.array([1.0] * k + [0.0] * (g - k))
            A = np.array(mod.advantage(mx.array(r)).tolist())
            s = math.sqrt(k * (g - k)) / g
            worst = max(worst, abs(A[0] - (1 - k / g) / s), abs(A[-1] + (k / g) / s))
        check(f"{label}: binary rewards, G = 16, every k — Â = (1 − k/G)/s_r and −(k/G)/s_r", worst < 1e-5, f"max error {worst:.1e}")
        z0, z1 = (np.array(mod.advantage(mx.array([v] * 8)).tolist()) for v in (0.0, 1.0))
        check(f"{label}: a silent group (k = 0, k = G) gets all zeros", not z0.any() and not z1.any())
        rng = np.random.default_rng(0)
        k, n = 3, 1000
        r = np.array([1.0] * k + [0.0] * (g - k))
        A = np.array(mod.advantage(mx.array(r)).tolist())
        grads = rng.normal(size=(g, n))
        way1 = (A[:, None] * grads).sum(0) / g
        way2 = math.sqrt(k * (g - k)) / g * (grads[:k].mean(0) - grads[k:].mean(0))
        rel = np.linalg.norm(way1 - way2) / np.linalg.norm(way1)
        check(f"{label}: the identity on random 'gradients' (G = 16, k = 3)", rel < 1e-5, f"relative difference {rel:.1e}")
    R = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0]], dtype=float)
    A, s = G.advantages(R)
    per = np.stack([np.array(advantage_final.advantage(mx.array(row)).tolist()) for row in R])
    check("grpo.advantages (whole batch) equals advantage() row by row", np.abs(A - per).max() < 1e-6)
    A2, _ = G.advantages(R, scale_rewards=False)
    check("without ÷ s_r the advantage is r − mean", np.abs(A2 - (R - R.mean(1, keepdims=True))).max() < 1e-12)
    L = np.array([[4, 8, 8, 4], [5, 5, 5, 5], [2, 2, 2, 10]])
    for agg in ("sequence", "own_length", "constant", "group_tokens"):
        W = G.rollout_weights(A, L, agg, 48)
        check(f"rollout_weights[{agg}]: zero for the silent group, signs follow the advantage",
              not W[1].any() and bool((np.sign(W) == np.sign(A)).all()))
    check("verifier: the last number decides", G.reward_strict("47 + 228 = 275", 275) == 1 and G.reward_strict("275, or 3", 275) == 0
          and G.reward_strict("It is 1,275.", 1275) == 1 and G.reward_strict("= −12", -12) == 1 and G.reward_strict("none", 1) == 0)
    check("task: × before + and −", G.evaluate_expr([47, 38, 6], ["+", "*"]) == 275 and G.evaluate_expr([47, 38, 6, 5], ["-", "*", "-"]) == -186)
    rng = random.Random(1)
    for name, level in cfg["task"]["levels"].items():
        p = G.make_problem(rng, level)
        note(f"        {name}: {p['expr']} = {p['answer']}")
    print(f"\n  {GREEN + 'all tests passed' if ok else RED + 'some tests FAILED'}{RESET}")
    if not ok:
        raise SystemExit(1)


# ── inspect／probe ──────────────────────────────────────────────
def cmd_inspect(args, cfg):
    import mlx.core as mx
    banner("demo 2 · inspect")
    v = versions()
    print("  " + "  ".join(f"{k} {v[k]}" for k in ("python", "mlx", "mlx-lm", "transformers") if v.get(k)) + f"  mlx-lm commit {str(v.get('mlx_lm_commit'))[:7]}")
    policy = load_policy(cfg, args)
    rec = {"date": now(), "versions": v, "device": device_info(), "trainable": {str(policy.rank): policy.n_trainable},
           "eos_token_ids": sorted(int(t) for t in policy.tok.eos_token_ids), "task": {k: policy.task[k] for k in ("level", "prompt", "max_new_tokens", "temperature")}}
    problem = G.make_problem(random.Random(0), policy.task["level_spec"])
    text, ids = policy.prompt_text(problem), policy.encode_prompt(problem)
    print(f"\n  one problem through the chat template ({len(ids)} tokens):\n{GREY}{text}{RESET}")
    print(f"\n  stop tokens: {rec['eos_token_ids']}" + ("" if args.tiny else "  " + " ".join(repr(policy.tok.decode([t])) for t in rec["eos_token_ids"])))
    x = mx.array(ids)[None]
    B = G.Bypass()
    on = policy.model(x).astype(mx.float32)
    B.enabled = False
    off = policy.model(x).astype(mx.float32)
    B.enabled = True
    d = float(mx.abs(on - off).max().item())
    rec.update(prompt_tokens=len(ids), prompt_text=text, bypass_on_off_max_abs_diff_at_init=d)
    print(f"  bypass on vs. off at initialisation (B = 0): max |Δ logit| = {BOLD}{d:g}{RESET}  {GREY}(0 means step 0 is exactly π_ref){RESET}")
    r = int(policy.rank)
    print(f"  trainable parameters at rank {r}: {BOLD}{policy.n_trainable:,}{RESET}  {GREY}(rank 1: {policy.n_trainable // r:,}){RESET}")
    rec["trainable"]["1"] = policy.n_trainable // r
    for name, level in ([] if args.tiny else cfg["task"]["levels"].items()):
        p = G.make_problem(random.Random(1), level)
        print(f"  {name}: {p['expr']} = {p['answer']}")
    write_json(out_dir(args) / "inspect.json", rec)
    note(f"\n  → {(out_dir(args) / 'inspect.json').relative_to(HERE)}")


def cmd_probe(args, cfg):
    import mlx.core as mx
    import numpy as np
    pc = cfg["probe"]
    parts = args.part or ["levels", "timing", "lr"]
    out = out_dir(args) / "probe.json"
    rec = read_json(out) if out.exists() else {}
    rec.update(date=now(), versions=versions(), device=device_info(), tiny=bool(args.tiny))
    banner("demo 2 · probe — how hard should the problems be, how long is a step, which learning rate moves")
    policy = load_policy(cfg, args)
    if "levels" in parts:
        g, n = int(pc["G"]), int(pc["n_problems"])
        levels = {"tiny": cfg["tiny"]["level"]} if args.tiny else cfg["task"]["levels"]
        print(f"\n  π_ref on each level: {n} problems × {g} samples, temperature {policy.task['temperature']}, at most {policy.task['max_new_tokens']} tokens")
        print(f"  {'level':<6}{'accuracy':>9}{'len':>7}{'cut':>6}   silent groups at G = 2 / 4 / 8 / 16 (measured)        example")
        rec["levels"] = {}
        for name, level in levels.items():
            mx.random.seed(policy.seed)
            problems = G.make_problems(random.Random(policy.seed), level, n)
            groups, info = policy.rollouts(problems, g)
            st = group_stats(groups)
            sil = {}
            for gg in (2, 4, 8, 16):
                if gg > g:
                    continue
                chunks = [grp[i:i + gg] for grp in groups for i in range(0, g, gg)]
                sil[str(gg)] = r4(mean([1.0 if sum(r["reward"] for r in c) in (0, gg) else 0.0 for c in chunks]))
            per_problem = [mean([r["reward"] for r in grp]) for grp in groups]
            st.update(silent=sil, seconds=round(info["seconds"], 1), per_problem_accuracy_std=r4(float(np.std(per_problem))),
                      example={"expr": problems[0]["expr"], "answer": problems[0]["answer"], "text": groups[0][0]["text"][:200], "reward": groups[0][0]["reward"]})
            rec["levels"][name] = st
            good = 0.3 <= st["accuracy"] <= 0.5
            ex = " ".join(groups[0][0]["text"].replace("<think>", "").replace("</think>", "").split())[:60]
            print(f"  {(GREEN if good else '') + name:<6}{st['accuracy']:>9.3f}{st['mean_len']:>7.1f}{st['truncated']:>6.0%}{RESET}   "
                  + " / ".join(f"{sil[k]:.0%}" for k in sil) + f"   {GREY}{problems[0]['expr']} → {ex}{RESET}")
            if name == policy.task["level"]:
                flat = [r for grp in groups for r in grp][:64]
                batch, mask = G.pad_batch([r["prompt"] for r in flat], [r["tokens"] for r in flat])
                lp = G.sequence_logprobs(policy.model, batch, mask)
                diffs = [abs(a - r["logprob"]) for a, r in zip(lp.tolist(), flat)]
                rec["sampling_logprob_vs_recomputed"] = {"n": len(flat), "mean_abs": r4(mean(diffs)), "max_abs": r4(max(diffs)),
                                                         "mean_abs_logprob": r4(mean([abs(r["logprob"]) for r in flat]))}
        note(f"  綠色＝正確率在 0.3–0.5；把 demo_config.toml [task] level 改成那一級（現在是 {policy.task['level']}）。")
        lr_ = rec.get("sampling_logprob_vs_recomputed")
        if lr_:
            print(f"  Σ log π from sampling vs. recomputed in one padded batch ({lr_['n']} answers): mean |Δ| {lr_['mean_abs']}, max {lr_['max_abs']} "
                  f"{GREY}(mean |log π| {lr_['mean_abs_logprob']}){RESET}")
        write_json(out, rec)
    if "timing" in parts:
        spec = run_spec(cfg, args, cfg["tables"]["base"])
        spec["steps"] = int(pc["timing_steps"])
        print(f"\n  timing: {spec['steps']} real steps at G = {spec['G']}, {spec['rollouts_per_step']} rollouts per step")
        policy.reset_bypass(policy.seed)
        r = train_run(policy, cfg, spec, do_evals=False)
        e = evaluate(policy, cfg, eval_problems(cfg, policy), batch_size=int(spec["rollouts_per_step"]))
        s = r["summary"]
        full = run_spec(cfg, args, cfg["tables"]["base"])
        n_ev = full["steps"] // max(1, int(full["eval_every"])) + 1
        est = full["steps"] * (s["t_sample_per_step"] + s["t_update_per_step"]) + n_ev * e["seconds"]
        rec["timing"] = {"t_sample_per_step": s["t_sample_per_step"], "t_update_per_step": s["t_update_per_step"], "eval_seconds": e["seconds"],
                         "peak_gb": s["peak_gb"], "estimated_minutes_per_run": r4(est / 60), "estimated_hours_all_runs": r4(est * len(cfg["runs"]) / 3600)}
        print(f"  per step: sampling {BOLD}{s['t_sample_per_step']} s{RESET}, update {BOLD}{s['t_update_per_step']} s{RESET}; one evaluation {e['seconds']} s; peak {s['peak_gb']} GB")
        print(f"  → a {full['steps']}-step run ≈ {BOLD}{est / 60:.0f} min{RESET}; all {len(cfg['runs'])} runs ≈ {BOLD}{est * len(cfg['runs']) / 3600:.1f} h{RESET}")
        write_json(out, rec)
    if "lr" in parts:
        spec = run_spec(cfg, args, cfg["tables"]["base"])
        spec.update(steps=int(pc["lr_steps"]), G=int(pc["lr_G"]))
        print(f"\n  learning-rate probe: {spec['steps']} steps each at G = {spec['G']}; training accuracy in the first and last third")
        rec["lr"] = []
        for lr in pc["lrs"]:
            policy.reset_bypass(policy.seed)
            spec["learning_rate"] = float(lr)
            r = train_run(policy, cfg, spec, echo=False, do_evals=False)
            s = r["summary"]
            rec["lr"].append({"lr": float(lr), "steps_done": s["steps_done"], "p_first_third": s["p_first_third"], "p_last_third": s["p_last_third"],
                              "silent": s["silent_measured"], "seconds": s["seconds"], "p": [x["p"] for x in r["steps"]]})
            flag = f"{RED} stopped early (non-finite loss){RESET}" if s["steps_done"] < spec["steps"] else ""
            print(f"  lr {float(lr):<8g} p {s['p_first_third']:.3f} → {BOLD}{s['p_last_third']:.3f}{RESET}   silent {s['silent_measured']:.0%}   {GREY}{s['seconds']:.0f} s{RESET}{flag}")
            write_json(out, rec)
        note("  挑「後三分之一明顯高於前三分之一、而且沒有掉回去」的 LR 填進 [train] learning_rate；每格約 ±0.03 的取樣雜訊。")
    note(f"\n  → {out.relative_to(HERE)}")


# ── rehearse／aggregate／replay ─────────────────────────────────
def spawn(args, extra: list[str]) -> None:
    cmd = [sys.executable, str(HERE / "demo.py")] + (["--tiny"] if args.tiny else []) + (["--online"] if args.online else [])
    cmd += (["--steps", str(args.steps)] if args.steps else []) + extra
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit(f"{RED}子行程失敗（{' '.join(extra)}）：exit {r.returncode}；修好之後重跑同一個指令，做過的 run 會跳過{RESET}")


def cmd_rehearse(args, cfg):
    names = args.only or list(cfg["runs"])
    t0 = time.time()
    if not args.only and (args.force or not (out_dir(args) / "identity.json").exists()):
        spawn(args, ["identity-rehearsal"])
    for name in names:
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
    t = cfg["tables"]
    sweep = [dict(run=n, **{k: runs[n]["summary"].get(k) for k in (
        "G", "prompts_per_step", "silent_measured", "silent_formula_at_mean_p", "silent_formula_stepwise", "p_train_mean", "acc_start", "acc_end",
        "t_sample_per_step", "t_update_per_step", "sample_share_of_time", "rollouts", "gen_tokens", "zero_weight_share", "peak_gb", "seconds")})
        for n in t["sweep"] if n in runs]
    switches = []
    for sw in t["switches"]:
        row = {k: sw.get(k) for k in ("label", "look", "metric", "off", "on")}
        row.update(off_name=sw.get("off_name", "off"), on_name=sw.get("on_name", "on"))
        for side in ("off", "on"):
            s = runs.get(sw[side], {}).get("summary")
            row[f"{side}_value"] = s.get(sw["metric"]) if s else None
            for extra in ("acc_start", "acc_end", "wrong_len_start", "wrong_len_end", "right_len_end", "rollouts", "silent_measured"):
                row[f"{side}_{extra}"] = s.get(extra) if s else None
        switches.append(row)
    curves = {n: [{k: e[k] for k in ("step", "rollouts", "gen_tokens", "accuracy", "wrong_len", "right_len")} for e in r["evals"]] for n, r in runs.items()}
    any_run = next(iter(runs.values()))
    summary = {"date": now(), "run_dates": {n: r["date"] for n, r in runs.items()}, "tiny": any(r.get("tiny") for r in runs.values()),
               "model": any_run["model"], "task": any_run["task"], "specs": {n: r["spec"] for n, r in runs.items()},
               "sweep": sweep, "switches": switches, "curves": curves,
               "caveat": "One 1.7B model, one kind of problem, one seed, a few hundred steps. It shows what the switches are, how often a group is "
                         "silent and where the time goes; it cannot confirm or refute IsoCompute."}
    write_json(d / "summary.json", summary)
    note(f"→ {(d / 'summary.json').relative_to(HERE)}")
    return summary


def fmt(x, spec=".3f", none="—"):
    return none if x is None else format(x, spec)


def cmd_replay(args, cfg):
    p = out_dir(args) / "summary.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse{RESET}")
    s = read_json(p)
    banner(f"{cfg['demo']['title']}\n(b) REPLAY of the rehearsal of {s['date'][:10]}" + ("  [tiny random model — numbers mean nothing]" if s.get("tiny") else ""))
    print(f"  {GREY}{s['caveat']}{RESET}")
    pause("\n  the G sweep  [Enter] ")
    print(f"\n  {'G':>3} {'prompts/step':>13} {'silent, measured':>17} {'p^G + (1−p)^G':>15} {'accuracy on fresh problems':>28} {'time per step: sampling / update':>33}")
    for r in s["sweep"]:
        print(f"  {r['G']:>3} {r['prompts_per_step']:>13} {fmt(r['silent_measured'], '.1%'):>17} {fmt(r['silent_formula_stepwise'], '.1%'):>15} "
              f"{fmt(r['acc_start']) + ' → ' + BOLD + fmt(r['acc_end']) + RESET:>36} {fmt(r['t_sample_per_step'], '.1f') + ' s / ' + fmt(r['t_update_per_step'], '.1f') + ' s':>33}")
    note("  p in the formula is each step's measured training accuracy; the column is the average over steps.")
    pause("\n  accuracy against rollouts sampled  [Enter] ")
    names = [r["run"] for r in s["sweep"]]
    steps = sorted({e["step"] for n in names for e in s["curves"][n]})
    print(f"\n  {'step':>5} " + " ".join(f"{'G = ' + str(next(r['G'] for r in s['sweep'] if r['run'] == n)):>9}" for n in names) + f"   {GREY}(every G samples the same number of rollouts per step){RESET}")
    for st in steps:
        cells = []
        for n in names:
            e = next((e for e in s["curves"][n] if e["step"] == st), None)
            cells.append(f"{fmt(e['accuracy'] if e else None):>9}")
        print(f"  {st:>5} " + " ".join(cells))
    pause("\n  the switches, each at one G  [Enter] ")
    print(f"\n  {'switch':<32} {'what to look at':<52} {'off':>14} {'on':>14}")
    for r in s["switches"]:
        unit = ".1f" if "len" in r["metric"] else ".3f"
        print(f"  {r['label']:<32} {r['look']:<52} {(r['off_name'] + ' ' if r['off_name'] != 'off' else '') + fmt(r['off_value'], unit):>14} "
              f"{BOLD}{(r['on_name'] + ' ' if r['on_name'] != 'on' else '') + fmt(r['on_value'], unit):>14}{RESET}")
    note("  one run and one seed per cell: when two cells are close, the honest reading is \"cannot tell\".")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    ap.add_argument("--online", action="store_true", help="允許連 Hugging Face（預設離線）")
    ap.add_argument("--ref", action="store_true", help="(a) 用 advantage_final.py")
    ap.add_argument("--name", help="run 的名字（demo_config.toml [runs]）")
    ap.add_argument("--only", action="append", help="rehearse 只跑這個 run（可重複）")
    ap.add_argument("--force", action="store_true", help="rehearse 重跑已有的 run")
    ap.add_argument("--part", action="append", choices=["levels", "timing", "lr"], help="probe 只做這一段（可重複）")
    ap.add_argument("--level", help="暫時換題目的級別（預設讀 [task] level）")
    ap.add_argument("--steps", type=int, help="暫時換步數")
    ap.add_argument("--G", type=int, help="(a) 暫時換 G")
    ap.add_argument("cmd", choices=["test", "inspect", "probe", "rehearse", "run", "live", "identity-rehearsal", "aggregate", "replay", "replay-a"])
    args = ap.parse_args()
    cfg = load_cfg()
    if args.cmd == "test":
        cmd_test(args, cfg)
    elif args.cmd == "inspect":
        cmd_inspect(args, cfg)
    elif args.cmd == "probe":
        cmd_probe(args, cfg)
    elif args.cmd == "rehearse":
        cmd_rehearse(args, cfg)
    elif args.cmd == "run":
        cmd_run(args, cfg)
    elif args.cmd == "live":
        cmd_identity(args, cfg)
    elif args.cmd == "identity-rehearsal":
        cmd_identity(args, cfg, rehearsal=True)
    elif args.cmd == "aggregate":
        aggregate(args, cfg)
    elif args.cmd == "replay":
        cmd_replay(args, cfg)
    elif args.cmd == "replay-a":
        cmd_replay_a(args, cfg)


if __name__ == "__main__":
    main()
