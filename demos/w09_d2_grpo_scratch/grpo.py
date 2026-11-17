# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 2 的手寫部分：題目產生器、verifier、疊在模型上的 LoRA 旁路、取樣、advantage、GRPO 的一步更新。

mlx-lm（../versions.lock 的 commit）沒有任何 RL 訓練的指令，所以迴圈在這裡。它有 `BatchGenerator`（generate.py），
取樣整批做；其餘都是這個檔。demo 3 從檔案路徑載入這個檔、只換 reward 函數。

記號照投影片：一題 x、G 段回答 y_i（含結束 token）、reward r_i、組內平均 r̄、組內標準差 s_r（**除以 G**）、
advantage Â_i = (r_i − r̄) / s_r。每一批 rollout 只更新一步，所以 ratio ρ = 1、clip 不作用，更新就是
Σ_i Â_i ∇ log π(y_i | x)（帶 baseline 的 REINFORCE）；對 π_ref 的 KL 係數 β = 0。

mlx 只在函數裡 import：題目與 verifier 是純標準函式庫，沒有 mlx 也能用。
"""
from __future__ import annotations

import random
import re
import time

# ── 題目：當場產生的整數算術 ─────────────────────────────────────
SHOW = {"+": "+", "-": "-", "*": "×"}


def evaluate_expr(nums: list[int], ops: list[str]) -> int:
    """先乘後加減（由左到右）。"""
    terms, signs = [nums[0]], [1]
    for op, n in zip(ops, nums[1:]):
        if op == "*":
            terms[-1] *= n
        else:
            terms.append(n)
            signs.append(1 if op == "+" else -1)
    return sum(s * t for s, t in zip(signs, terms))


def make_problem(rng: random.Random, level: dict) -> dict:
    """level：terms（運算元個數）、lo／hi（加減的運算元範圍）、ops、mul_lo／mul_hi（乘號右邊那個數的範圍）。"""
    n = int(level["terms"])
    ops = [rng.choice(list(level["ops"])) for _ in range(n - 1)]
    nums = [rng.randint(int(level["lo"]), int(level["hi"]))]
    for op in ops:
        if op == "*":
            nums.append(rng.randint(int(level.get("mul_lo", 2)), int(level.get("mul_hi", 9))))
        else:
            nums.append(rng.randint(int(level["lo"]), int(level["hi"])))
    expr = str(nums[0]) + "".join(f" {SHOW[o]} {v}" for o, v in zip(ops, nums[1:]))
    return {"expr": expr, "answer": evaluate_expr(nums, ops)}


def make_problems(rng: random.Random, level: dict, n: int, exclude: set | None = None) -> list[dict]:
    out, seen = [], set(exclude or ())
    tries = 0
    while len(out) < n:
        p = make_problem(rng, level)
        tries += 1
        if p["expr"] in seen and tries < 50 * n:     # 題目空間很小的 level 允許重複，不要卡死
            continue
        seen.add(p["expr"])
        out.append(p)
    return out


# ── verifier ────────────────────────────────────────────────────
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUMBER.findall(text.replace("−", "-").replace("–", "-")):
        s = m.replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def reward_strict(text: str, answer: int) -> float:
    """回答裡**最後一個數字**等於正確結果 → 1，否則 0。"""
    nums = numbers_in(text)
    return 1.0 if nums and nums[-1] == float(answer) else 0.0


# ── advantage（整批版；一組一列。課堂上寫的一組版在 advantage_live.py）────
def advantages(R, scale_rewards: bool = True):
    """R：(P, G) 的 numpy 陣列。回傳 (Â, s_r)。s_r 是除以 G 的標準差；沉默的組（s_r = 0）整列是 0。"""
    import numpy as np
    R = np.asarray(R, dtype=np.float64)
    mean = R.mean(axis=1, keepdims=True)
    s = np.sqrt(((R - mean) ** 2).mean(axis=1, keepdims=True))
    A = R - mean
    if scale_rewards:
        A = np.where(s > 0, A / np.where(s > 0, s, 1.0), 0.0)
    return A, s[:, 0]


def rollout_weights(A, lengths, token_agg: str, max_new_tokens: int):
    """每段回答的 log π 前面的係數。A、lengths：(P, G)。
    sequence      Â_i / G                      整段的 log π，不除長度（投影片恆等式的寫法）
    own_length    Â_i / (G |y_i|)              GRPO 原式：每段除以自己的長度
    constant      Â_i / (G L_max)              Dr. GRPO：除以常數
    group_tokens  Â_i / Σ_j |y_j|              DAPO：除以這一組的 token 總數
    最後再對 P 題取平均。"""
    import numpy as np
    A = np.asarray(A, dtype=np.float64)
    L = np.asarray(lengths, dtype=np.float64)
    P, G = A.shape
    if token_agg == "sequence":
        w = A / G
    elif token_agg == "own_length":
        w = A / (G * L)
    elif token_agg == "constant":
        w = A / (G * float(max_new_tokens))
    elif token_agg == "group_tokens":
        w = A / L.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"token_agg: {token_agg}")
    return w / P


# ── LoRA 旁路：疊在任何一個 Linear 型的模組外面 ─────────────────
def _bypass_class():
    import math
    import mlx.core as mx
    import mlx.nn as nn

    class Bypass(nn.Module):
        """y = inner(x) + scale · (x A) B。B 從零開始，所以接上去的那一刻輸出不變。
        `Bypass.enabled = False` 時只走 inner——那就是 π_ref，不必在記憶體裡多放一份模型。"""
        enabled = True

        def __init__(self, inner, d_in: int, d_out: int, r: int, scale: float):
            super().__init__()
            self.inner = inner
            self.scale = scale
            bound = 1 / math.sqrt(d_in)
            self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(d_in, r))
            self.lora_b = mx.zeros((r, d_out))

        def __call__(self, x):
            y = self.inner(x)
            if not Bypass.enabled:
                return y
            z = (x @ self.lora_a) @ self.lora_b
            return y + (self.scale * z).astype(x.dtype)

    return Bypass


_BYPASS = None


def Bypass():
    global _BYPASS
    if _BYPASS is None:
        _BYPASS = _bypass_class()
    return _BYPASS


def _linear_shape(m) -> tuple[int, int]:
    w = m.linear.weight if hasattr(m, "linear") else m.weight     # mlx-lm 的 LoRALinear 把原本的 Linear 放在 .linear
    d_out, d_in = w.shape
    return int(d_in), int(d_out)


def attach_bypass(model, keys: list[str], r: int, scale: float) -> int:
    """把每個 block 裡名字在 keys 的模組包一層 Bypass；整個模型凍住，只有旁路的 A、B 可訓練。回傳可訓練參數個數。"""
    from mlx.utils import tree_flatten, tree_unflatten
    B = Bypass()
    for layer in model.layers:
        new = [(k, B(m, *_linear_shape(m), r, scale)) for k, m in layer.named_modules() if k in keys]
        if new:
            layer.update_modules(tree_unflatten(new))
    model.freeze()
    for _, m in model.named_modules():
        if isinstance(m, B):
            m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


# ── 取樣 ────────────────────────────────────────────────────────
def sample(model, tokenizer, prompts: list[list[int]], max_new_tokens: int, temperature: float, batch_size: int) -> tuple[list[dict], dict]:
    """每個 prompt 取樣一段回答。回傳每段的 tokens（**含**結束 token）、finish（stop／length）、取樣當下的 Σ log π。
    用 mlx-lm 的 BatchGenerator（batch_generate 裡面就是它；直接用是為了留下結束 token）。"""
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    gen = BatchGenerator(model, stop_tokens=[[t] for t in sorted(tokenizer.eos_token_ids)], sampler=make_sampler(temp=temperature),
                         max_tokens=max_new_tokens, completion_batch_size=batch_size, prefill_batch_size=batch_size)
    uids = gen.insert(prompts, [max_new_tokens] * len(prompts))
    out = {u: {"tokens": [], "logprob": 0.0, "finish": None} for u in uids}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                o = out[r.uid]
                o["tokens"].append(int(r.token))
                o["logprob"] += float(r.logprobs[r.token].item())
                if r.finish_reason is not None:
                    o["finish"] = r.finish_reason
    gen.close()
    info = {"prompt_tokens": int(stats.prompt_tokens), "generation_tokens": int(stats.generation_tokens)}
    return [out[u] for u in uids], info


def response_text(tokenizer, r: dict) -> str:
    toks = r["tokens"][:-1] if r["finish"] == "stop" else r["tokens"]
    return tokenizer.decode(toks)


def rollouts(model, tokenizer, encode_prompt, problems: list[dict], G: int, reward_fn, max_new_tokens: int,
             temperature: float, batch_size: int) -> tuple[list[list[dict]], dict]:
    """每題 G 段回答。回傳 groups[p][i] = {prompt, tokens, finish, logprob, text, reward}。"""
    prompts = [encode_prompt(p) for p in problems]
    flat = [pr for pr in prompts for _ in range(G)]
    t0 = time.perf_counter()
    res, info = sample(model, tokenizer, flat, max_new_tokens, temperature, batch_size)
    info["seconds"] = time.perf_counter() - t0
    groups = []
    for pi, p in enumerate(problems):
        g = []
        for i in range(G):
            r = res[pi * G + i]
            r["prompt"] = prompts[pi]
            r["text"] = response_text(tokenizer, r)
            r["reward"] = float(reward_fn(r["text"], p["answer"]))
            g.append(r)
        groups.append(g)
    return groups, info


# ── 更新：log π(y | x) 重算一次前向 ────────────────────────────
def pad_batch(prompts: list[list[int]], responses: list[list[int]]):
    """右邊補 0。mask 標出「這個位置的 target 是回答的 token」。"""
    import mlx.core as mx
    import numpy as np
    seqs = [p + y for p, y in zip(prompts, responses)]
    T = max(len(s) for s in seqs)
    batch = np.zeros((len(seqs), T), dtype=np.int32)
    mask = np.zeros((len(seqs), T - 1), dtype=np.float32)
    for i, (p, y) in enumerate(zip(prompts, responses)):
        batch[i, :len(p) + len(y)] = p + y
        mask[i, len(p) - 1:len(p) + len(y) - 1] = 1.0
    return mx.array(batch), mx.array(mask)


def token_logprobs(model, batch):
    """(n, T−1)：每個位置給「下一個 token」的 log 機率，float32。"""
    import mlx.core as mx
    logits = model(batch[:, :-1]).astype(mx.float32)
    picked = mx.take_along_axis(logits, batch[:, 1:, None], axis=-1).squeeze(-1)
    return picked - mx.logsumexp(logits, axis=-1)


def grpo_loss(model, batch, mask, weights):
    """− Σ_i w_i Σ_t log π(y_{i,t} | x, y_{i,<t})。w_i 帶著 advantage 與長度的除法（rollout_weights）。"""
    lp = token_logprobs(model, batch)
    return -((lp * mask).sum(axis=1) * weights).sum()


def sequence_logprobs(model, batch, mask):
    return (token_logprobs(model, batch) * mask).sum(axis=1)


def flat_grad(grads):
    """可訓練參數的梯度接成一個 float32 向量（鍵排序，順序固定）。"""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    return mx.concatenate([v.reshape(-1).astype(mx.float32) for _, v in sorted(tree_flatten(grads), key=lambda kv: kv[0])])
