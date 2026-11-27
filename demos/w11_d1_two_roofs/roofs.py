# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 1 的算術：roofline 的兩片屋頂、decode 一步的 intensity、兩條界線。純標準函式庫。

記號與投影片相同：
    N      參數量                         W      權重在記憶體裡的 bytes（bf16 是 2N；4-bit 檔是實際的 bytes）
    d      hidden size                    kvt    一個 token 的 KV bytes = 2 · n_layer · n_kv · d_head · b_kv
    L      context 長度（token）          B      一步裡的序列數
    F      FLOP/s（量到的峰值算力）       BW     bytes/s（量到的有效頻寬）

一步 decode 的下界（投影片的式子）：
    t_step >= max( (W + B · kvt · L) / BW ,  B · (2N + 4 · n_layer · L · d) / F )

**兩種數法**（`count`）：
    "deck"    投影片與大綱的記帳：每個 token 2N 個 FLOPs、每一步搬全部的 W。
    "strict"  只數真的做了的：input embedding 是查表（不乘、每個 token 只讀一列）；
              prefill 只算 cache、不算 logits，所以 LM head 沒有做。
兩種都算、都存；畫面用 "deck" 那一種（與投影片對得起來），另印兩者的比。
"""
from __future__ import annotations

import math
import statistics


# ── 參數量：逐項數 ───────────────────────────────────────────────
def count_params(c: dict) -> dict:
    """c 的鍵：d、d_ff、n_layer、n_head、n_kv、d_head、V、tied。"""
    d, dff, nl, nh, nkv, dh, V = c["d"], c["d_ff"], c["n_layer"], c["n_head"], c["n_kv"], c["d_head"], c["V"]
    attn = d * nh * dh + 2 * d * nkv * dh + nh * dh * d          # q、k、v、o
    mlp = 3 * d * dff                                           # gate、up、down
    norms = 2 * d + (2 * dh if c.get("qk_norm", True) else 0)   # 兩個 RMSNorm；Qwen3 另有 q_norm、k_norm
    blocks = nl * (attn + mlp)
    block_norms = nl * norms + d                                # 加最後一個 norm
    embed = V * d
    head = 0 if c.get("tied") else V * d
    return {"per_layer_linear": attn + mlp, "blocks": blocks, "norms": block_norms, "embedding": embed, "lm_head": head,
            "N": blocks + block_norms + embed + head}


def kv_bytes_per_token(c: dict, b_kv: int = 2) -> int:
    return 2 * c["n_layer"] * c["n_kv"] * c["d_head"] * b_kv


# ── 每個 token 的 FLOPs、每一步的 bytes ──────────────────────────
def flops_prefill_per_token(c: dict, n: dict, L_in: int, count: str = "deck") -> float:
    """長度 L_in 的 prefill，平均每個 token：權重那一側 + attention 那一側（位置平均是 2·n_layer·L_in·d）。"""
    w = n["N"] if count == "deck" else n["blocks"]              # strict：沒有查表、沒有 LM head
    return 2.0 * w + 2.0 * c["n_layer"] * L_in * c["d"]


def flops_decode_per_seq(c: dict, n: dict, L: float, count: str = "deck") -> float:
    """一步 decode、一條序列、context 長度 L。"""
    w = n["N"] if count == "deck" else n["blocks"] + (n["lm_head"] or n["embedding"])   # tied 時 head 用的是 embedding 那張表
    return 2.0 * w + 4.0 * c["n_layer"] * L * c["d"]


def weight_bytes_moved(W: float, W_embed: float, count: str = "deck") -> float:
    """一步 decode 要搬的權重 bytes。strict：input embedding 那張表不整張讀（untied 時）。"""
    return W if count == "deck" else W - W_embed


# ── 屋頂 ─────────────────────────────────────────────────────────
def step_lower_bound(B: float, L: float, c: dict, n: dict, W: float, W_embed: float, kvt: float,
                     F: float, BW: float, count: str = "deck") -> tuple[float, str]:
    t_mem = (weight_bytes_moved(W, W_embed, count) + B * kvt * L) / BW
    t_cmp = B * flops_decode_per_seq(c, n, L, count) / F
    return max(t_mem, t_cmp), ("memory" if t_mem >= t_cmp else "compute")


def throughput_upper(B, L, c, n, W, W_embed, kvt, F, BW, count="deck") -> tuple[float, str]:
    t, side = step_lower_bound(B, L, c, n, W, W_embed, kvt, F, BW, count)
    return B / t, side


def kink_B(L: float, c: dict, n: dict, W: float, W_embed: float, kvt: float, F: float, BW: float, count: str = "deck") -> float:
    """兩片屋頂相交的 batch 大小；context 太長時不存在（回傳 inf）。"""
    den = flops_decode_per_seq(c, n, L, count) / F - kvt * L / BW
    return weight_bytes_moved(W, W_embed, count) / BW / den if den > 0 else math.inf


def boundaries(c: dict, n: dict, W: float, W_embed: float, kvt: float, F: float, BW: float, count: str = "deck") -> dict:
    """I*、B*（短 context）、L_c（B → ∞）。"""
    I_star = F / BW
    w_flops = flops_decode_per_seq(c, n, 0, count)                      # 2 N（或 strict 的那個數）
    kv_side = 4.0 * c["n_layer"] * c["d"] / kvt                          # attention 那一側的 FLOP/byte
    return {"I_star": I_star,
            "B_star": weight_bytes_moved(W, W_embed, count) * I_star / w_flops,
            "kv_side_intensity": kv_side,
            "L_c": w_flops / ((I_star - kv_side) * kvt) if I_star > kv_side else math.inf}


# ── 從量測推屋頂 ─────────────────────────────────────────────────
def roofs_from_measurement(c: dict, n: dict, W: float, W_embed: float, kvt: float,
                           prefill_rows: list[dict], decode_row: dict, count: str = "deck") -> dict:
    """prefill 各長度取最大的 FLOP/s；decode（B = 1）的 bytes/s。"""
    best = None
    for r in prefill_rows:
        f = r["tok_per_s"] * flops_prefill_per_token(c, n, r["L"], count)
        if best is None or f > best[0]:
            best = (f, r["L"])
    L_mean = decode_row["L_start"] + decode_row["n_gen"] / 2.0
    bw = decode_row["tok_per_s"] * (weight_bytes_moved(W, W_embed, count) + kvt * L_mean)
    out = {"count": count, "F_max": best[0], "F_max_at_L": best[1], "BW_eff": bw,
           "decode_flops_per_s": decode_row["tok_per_s"] * flops_decode_per_seq(c, n, L_mean, count)}
    out.update(boundaries(c, n, W, W_embed, kvt, best[0], bw, count))
    return out


def median(xs):
    return statistics.median(xs) if xs else None
