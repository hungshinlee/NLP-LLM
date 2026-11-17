# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 1 (b) 的手寫部分：寫死的偏好規則、配對、一批偏好對的 log 機率、對 π_ref 的 KL、bits-per-byte 的尺。

模型、旁路（Bypass）、取樣、pad、整段回答的 log 機率都是 demo 2 的 grpo.py（參數 G 就是那個模組）。
記號照投影片：prompt x、被偏好的回答 y_w、另一個 y_l（下標是 win／lose）、π_θ（旁路開著）、π_ref（旁路關掉，同一份權重）。
規則與配對是純標準函式庫；mlx 只在函數裡 import。
"""
from __future__ import annotations

import math

RULE = "an answer wins if it ends properly (a stop token, not the token limit) and is shorter"


# ── 規則：正常結束、而且比較短的勝 ───────────────────────────────
def ended(s: dict) -> bool:
    return s["finish"] == "stop"


def rule_key(s: dict) -> tuple[int, int]:
    """越小越好：先看有沒有正常結束，再看長度（token 數，含結束 token）。"""
    return (0 if ended(s) else 1, len(s["tokens"]))


def prefers(a: dict, b: dict, min_gap: int) -> bool:
    """規則是否判 a 勝過 b。勝者一定要正常結束；兩個都正常結束時，要短至少 min_gap 個 token 才算。"""
    if not ended(a):
        return False
    if not ended(b):
        return True
    return len(b["tokens"]) - len(a["tokens"]) >= min_gap


def make_pairs(samples: list[dict], pairs_per_prompt: int, min_gap: int) -> list[tuple[int, int]]:
    """同一個 prompt 的幾段回答照規則排序，最好的配最差的、第二好的配第二差的……回傳 (w, l) 的索引。"""
    order = sorted(range(len(samples)), key=lambda i: (rule_key(samples[i]), i))
    out = []
    for j in range(pairs_per_prompt):
        if j >= len(order) - 1 - j:
            break
        w, l = order[j], order[-1 - j]
        if prefers(samples[w], samples[l], min_gap):
            out.append((w, l))
    return out


def gold(samples: list[dict]) -> dict:
    """規則看的兩件事，直接量：正常結束的比例、平均長度。"""
    n = len(samples)
    return {"n": n, "ended": sum(1 for s in samples if ended(s)) / n, "mean_len": sum(len(s["tokens"]) for s in samples) / n}


# ── 一批偏好對的 log 機率 ────────────────────────────────────────
def pair_batch(G, pairs: list[dict]):
    """前半是 (x, y_w)、後半是 (x, y_l)。"""
    prompts = [p["prompt"] for p in pairs]
    return G.pad_batch(prompts + prompts, [p["w"] for p in pairs] + [p["l"] for p in pairs])


def sequence_logp(G, model, batch, mask, ref: bool = False):
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


def pair_logprobs(G, model, pairs: list[dict], ref: bool = False, chunk: int = 8) -> tuple[list[float], list[float]]:
    lw, ll = [], []
    for i in range(0, len(pairs), chunk):
        part = pairs[i:i + chunk]
        batch, mask = pair_batch(G, part)
        lp = sequence_logp(G, model, batch, mask, ref=ref).tolist()
        lw += lp[:len(part)]
        ll += lp[len(part):]
    return lw, ll


def training_loss(G, dpo_loss):
    """(loss, 這一批的 log 機率)。log 機率的前半是 y_w、後半是 y_l。"""
    def fn(model, batch, mask, ref_w, ref_l, beta):
        lp = G.sequence_logprobs(model, batch, mask)
        n = lp.shape[0] // 2
        return dpo_loss(lp[:n], lp[n:], ref_w, ref_l, beta), lp
    return fn


def softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def pair_metrics(lw: list[float], ll: list[float], rw: list[float], rl: list[float], beta: float, len_w: list[int], len_l: list[int]) -> dict:
    """lw、ll：π_θ 的；rw、rl：π_ref 的。margin = β[(lw − rw) − (ll − rl)]；loss = −log σ(margin)。"""
    n = len(lw)
    margins = [beta * ((a - c) - (b - d)) for a, b, c, d in zip(lw, ll, rw, rl)]
    return {"n": n, "loss": sum(softplus(-m) for m in margins) / n, "margin": sum(margins) / n,
            "ranked_right": sum(1 for m in margins if m > 0) / n,
            "logp_w": sum(lw) / n, "logp_l": sum(ll) / n,
            "logp_w_per_token": sum(lw) / sum(len_w), "logp_l_per_token": sum(ll) / sum(len_l)}


def paired_change(now: list[float], start: list[float]) -> dict:
    """同一批偏好對，現在減起點：平均、標準誤、往下掉的比例。"""
    d = [a - b for a, b in zip(now, start)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, n - 1) / n) if n > 1 else 0.0
    return {"mean": m, "se": se, "fell": sum(1 for x in d if x < 0) / n}


def direction(change: dict) -> str:
    if change["mean"] > 2 * change["se"]:
        return "rose"
    if change["mean"] < -2 * change["se"]:
        return "fell"
    return "did not move"


# ── 對 π_ref 的 KL（取樣估計）────────────────────────────────────
def kl_estimate(G, model, samples: list[dict], chunk: int = 32) -> dict:
    """y ~ π_θ 時 log π_θ(y|x) − log π_ref(y|x) 的平均（nats／每段回答）。有限樣本下可以是負的。"""
    theta, ref = [], []
    for i in range(0, len(samples), chunk):
        part = samples[i:i + chunk]
        batch, mask = G.pad_batch([s["prompt"] for s in part], [s["tokens"] for s in part])
        theta += sequence_logp(G, model, batch, mask, ref=False).tolist()
        ref += sequence_logp(G, model, batch, mask, ref=True).tolist()
    d = [a - b for a, b in zip(theta, ref)]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / len(d) / len(d)) if len(d) > 1 else 0.0
    return {"kl": m, "kl_se": se, "sqrt_kl": math.sqrt(max(m, 0.0))}


# ── bits-per-byte（W1 的尺；上週 demo 4 (b) 的量法）───────────────
def window_nll(model, prefix: int, ids: list[int]) -> float:
    import mlx.core as mx
    logits = model(mx.array([prefix] + ids)[None])[0, :-1].astype(mx.float32)
    tgt = mx.array(ids)
    picked = mx.take_along_axis(logits, tgt[:, None], axis=-1).squeeze(-1)
    return float((mx.logsumexp(logits, axis=-1) - picked).sum().item())


def bits_per_byte(model, prefix: int, ids: list[int], n_bytes: int, window: int) -> dict:
    """不重疊的視窗，每個視窗前面放 prefix token；Σ NLL(nats) / ln 2 / UTF-8 bytes。"""
    nll, n_win = 0.0, 0
    for i in range(0, len(ids), window):
        nll += window_nll(model, prefix, ids[i:i + window])
        n_win += 1
    return {"bytes": n_bytes, "tokens": len(ids), "windows": n_win, "bpb": nll / math.log(2) / n_bytes}


# ── 記憶體裡有幾份權重 ───────────────────────────────────────────
def weight_bytes(model) -> dict:
    from mlx.utils import tree_flatten
    total = sum(int(v.nbytes) for _, v in tree_flatten(model.parameters()))
    train = sum(int(v.nbytes) for _, v in tree_flatten(model.trainable_parameters()))
    return {"all_parameters_bytes": total, "trainable_bytes": train, "frozen_bytes": total - train}
