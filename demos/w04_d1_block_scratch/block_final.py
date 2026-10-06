#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 1 的參考實作：RMSNorm、SwiGLU、pre-LN Block、tiny GPT，以及逐項記帳的 count_params。純 `torch`，走 MPS（訓練用；CPU 也能跑）。

課堂上授課者在 block_live.py 從零寫同一份（寫的順序見投影片 p16 講稿的【操作】：RMSNorm → SwiGLU → Block → GPT），
每寫完一段就 `./present.sh live` 跑三個測試（測試本體在 tests.py，兩份實作共用）。這一份是 (1) 課前確認三個測試全過、
(2) 現場卡住兩分鐘後 `./present.sh show` 切過來、(3) 課後發給學生、(4) train.py 與 W4 demo 2 import 的模型。

attention 不重寫：`scaled_dot_product_attention` 與 `causal_mask` 直接 import 上週（W3 demo 1）的 attention_final.py；
這裡只加多頭的 reshape（CausalSelfAttention，「寫但不講」，block_live.py 的 reset 會保留它）。
**attention dropout 是自己寫的**：W3 的函數把 A 回傳出來，訓練時對 A 做 dropout 再乘 V。
torch 內建的 F.scaled_dot_product_attention 在 MPS 上不支援 dropout_p > 0（附錄 C；`./present.sh probe` 實測）——
手刻一次的價值的現成例子。

tests.py 依名稱與簽名呼叫這五個（live 版要一樣）：

  RMSNorm(d, eps=1e-6)                 forward(x: (..., d)) -> (..., d)         x / rms(x) * weight；沒有 mean 減除、沒有 beta
  SwiGLU(d, d_ff)                      forward(x: (..., d)) -> (..., d)         down(silu(gate(x)) * up(x))，三個 Linear、無 bias
  Block(d, n_head, d_ff, ...)          forward(x: (B, L, d)) -> (B, L, d)       x = x + attn(norm1(x)); x = x + ffn(norm2(x))
  GPT(V, d, n_layer, n_head, d_ff, L)  forward(idx: (B, L) long) -> (B, L, V)   embedding → n_layer 個 Block → 最後一個 Norm → tied unembedding
  count_params(model) -> int           sum(p.numel() for p in model.parameters())

demo 2（訓練不穩定）用同一個 GPT：`norm="post"` 換成 post-LN；`qk_norm=True` 對 q、k 各做一次 RMSNorm 再內積；
`record_stats=True` 時每次前向把 softmax 前最大的 |s_ij| 記在 `model.stats["max_abs_logit"]`。demo 1 的三個測試不用這些。

用法：
  python3 block_final.py             # 三個測試（等同 ./present.sh test）
  python3 block_final.py --show      # 小例子逐段印 shape 與參數表 + 三個測試（現場退路）
  python3 block_final.py --rehearse  # 三個測試 + 寫 runs/rehearsal/tests.json（彩排；講稿裡的實測值從這裡抄）
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
def _import_w3_attention():
    """載上週的 attention_final.py，**不動 sys.path**：W3 的資料夾也有一個 tests.py，放進 sys.path 會蓋掉這裡的 tests.py（2026-10-06 彩排踩到）。"""
    import importlib.util
    f = HERE.parent / "w03_d1_attention_scratch" / "attention_final.py"
    spec = importlib.util.spec_from_file_location("w03_attention_final", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 它會 set_default_device("cpu")，無妨——這裡所有張量都明確 .to(device)
    return mod


_w3 = _import_w3_attention()
scaled_dot_product_attention, causal_mask = _w3.scaled_dot_product_attention, _w3.causal_mask


class RMSNorm(nn.Module):
    """x / sqrt(mean(x²) + eps) * γ。沒有 mean 減除、沒有 β（Zhang & Sennrich 2019）。"""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))     # γ，初始化全 1
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """FFN(x) = (silu(x W_gate) ⊙ x W_up) W_down。三個矩陣、無 bias，名字照 HF 的 gate_proj / up_proj / down_proj。"""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """多頭的包裝：四個投影 + reshape，attention 本體是上週的函數。「寫但不講」。

    dropout 加在 attention 權重 A 上：上週的函數把 A 回傳出來，所以 out = dropout(A) @ V 自己做得到；
    內建的 F.scaled_dot_product_attention 在 MPS 上不接受 dropout_p > 0（./present.sh probe 實測）。
    """

    def __init__(self, d: int, n_head: int, n_kv: int | None = None, head_dim: int | None = None,
                 dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        self.n_head = n_head
        self.n_kv = n_kv or n_head
        self.head_dim = head_dim or d // n_head
        assert self.n_head % self.n_kv == 0
        self.q = nn.Linear(d, self.n_head * self.head_dim, bias=False)   # W^Q: d × n_head·d_head
        self.k = nn.Linear(d, self.n_kv * self.head_dim, bias=False)     # W^K: d × n_kv·d_head（GQA 時變小）
        self.v = nn.Linear(d, self.n_kv * self.head_dim, bias=False)     # W^V
        self.o = nn.Linear(self.n_head * self.head_dim, d, bias=False)   # W^O
        self.dropout = dropout
        self.qk_norm = qk_norm
        if qk_norm:                                                      # demo 2 設定 (iii)：q、k 各一個 RMSNorm（Qwen3／OLMo 2 的做法），每層 2·d_head 個參數
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.stats: dict = {}
        self.record_stats = False

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)   # (B, H, L, d_head)   ← 兩行 reshape，寫但不講
        k = self.k(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.n_kv != self.n_head:                                           # GQA：每個 KV head 服務 n_head/n_kv 個 query head
            rep = self.n_head // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        mask = causal_mask(L).to(x.device)
        out, A = scaled_dot_product_attention(q, k, v, mask)                   # 上週的五行；A: (B, H, L, L)
        if self.record_stats:                                                  # demo 2：softmax 前的 max |s_ij|（重算一次 score，只在記錄時）
            s = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)) + mask
            self.stats["max_abs_logit"] = s[torch.isfinite(s)].abs().max().item()
        if self.training and self.dropout > 0:
            out = F.dropout(A, self.dropout) @ v                               # 手刻的 attention dropout：有 A 在手上才做得到
        return self.o(out.transpose(1, 2).reshape(B, L, self.n_head * self.head_dim))


class Block(nn.Module):
    """一層：兩個子層掛在 residual stream 上。pre-LN（預設）：Norm 在分支上，主線上沒有運算。"""

    def __init__(self, d: int, n_head: int, d_ff: int, n_kv: int | None = None, head_dim: int | None = None,
                 dropout: float = 0.0, norm: str = "pre", qk_norm: bool = False):
        super().__init__()
        assert norm in ("pre", "post")
        self.norm = norm
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_head, n_kv, head_dim, dropout=dropout, qk_norm=qk_norm)
        self.ffn = SwiGLU(d, d_ff)
        self.drop = nn.Dropout(dropout)   # residual dropout（子層輸出）；無參數

    def forward(self, x):
        if self.norm == "pre":
            x = x + self.drop(self.attn(self.norm1(x)))      # X' = X + Attn(Norm(X))
            x = x + self.drop(self.ffn(self.norm2(x)))       # X_{ℓ+1} = X' + FFN(Norm(X'))
        else:                                                # post-LN（demo 2 設定 (i)）：Norm 在加法之後，主線上每層都經過 Norm
            x = self.norm1(x + self.drop(self.attn(x)))
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class GPT(nn.Module):
    """embedding → n_layer 個 Block → 最後一個 Norm → unembedding（tied：logits = h @ E^T）。"""

    def __init__(self, V: int, d: int, n_layer: int, n_head: int, d_ff: int, L: int,
                 n_kv: int | None = None, head_dim: int | None = None, dropout: float = 0.0,
                 norm: str = "pre", qk_norm: bool = False, tie: bool = True):
        super().__init__()
        self.V, self.d, self.n_layer, self.L = V, d, n_layer, L
        self.emb = nn.Embedding(V, d)                        # E: V × d。沒有位置編碼——W3 p42 說 attention 是 permutation-equivariant，本週的 block 沒有補上這一點（W5）
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_head, d_ff, n_kv, head_dim, dropout, norm, qk_norm) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d)
        self.tie = tie
        if not tie:
            self.lm_head = nn.Linear(d, V, bias=False)       # W_U: d × V（不共享時多一份 Vd）
        self.stats: dict = {}
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, record_stats: bool = False):
        B, L = idx.shape
        assert L <= self.L, f"L = {L} > context {self.L}"
        x = self.drop(self.emb(idx))                         # X_0 = embedding 查表
        for blk in self.blocks:
            blk.attn.record_stats = record_stats
            x = blk(x)
        h = self.norm_f(x)
        logits = h @ self.emb.weight.T if self.tie else self.lm_head(h)   # z = Norm(X_n) W_U
        if record_stats:
            self.stats["max_abs_logit"] = max(b.attn.stats.get("max_abs_logit", 0.0) for b in self.blocks)
            self.stats["log_Z"] = torch.logsumexp(logits.float(), dim=-1).mean().item()   # 輸出層的 log Z（z-loss 盯的量）
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── 現場退路：小例子逐段印 shape ─────────────────────────────────
def show():
    import tests
    cfg = tests.tiny_cfg()
    torch.manual_seed(0)
    B, L = 2, 8
    V, d, n_layer, n_head, d_ff = cfg["V"], cfg["d"], cfg["n_layer"], cfg["n_head"], cfg["d_ff"]
    x = torch.randn(B, L, d)
    print(f"tiny GPT: V = {V}, d = {d}, n_layer = {n_layer}, n_head = {n_head} (d_head = {d // n_head}), d_ff = {d_ff}\n")
    n = RMSNorm(d)
    y = n(x)
    print(f"RMSNorm      {tuple(x.shape)} -> {tuple(y.shape)}   rms of a row before {x[0, 0].pow(2).mean().sqrt():.3f}, after {y[0, 0].pow(2).mean().sqrt():.3f}   params {count_params(n)} (= d)")
    f = SwiGLU(d, d_ff)
    print(f"SwiGLU       {tuple(x.shape)} -> {tuple(f(x).shape)}   params {count_params(f)} (= 3·d·d_ff = {3 * d * d_ff})")
    a = CausalSelfAttention(d, n_head)
    print(f"attention    {tuple(x.shape)} -> {tuple(a(x).shape)}   params {count_params(a)} (= 4·d² = {4 * d * d}; W3's function inside, plus two reshape lines)")
    b = Block(d, n_head, d_ff)
    print(f"Block        {tuple(x.shape)} -> {tuple(b(x).shape)}   params {count_params(b)} (= 4d² + 3·d·d_ff + 2d = {4 * d * d + 3 * d * d_ff + 2 * d})")
    g = GPT(V, d, n_layer, n_head, d_ff, cfg["L"])
    idx = torch.randint(0, V, (B, L))
    logits = g(idx)
    print(f"GPT          {tuple(idx.shape)} (long) -> {tuple(logits.shape)}   finite: {bool(torch.isfinite(logits).all())}")
    print()
    print(tests.ledger_text(cfg, count_params(g)))
    print()


if __name__ == "__main__":
    import tests
    if "--show" in sys.argv:
        show()
    tests.run(sys.modules[__name__], rehearse="--rehearse" in sys.argv, label="block_final.py")
