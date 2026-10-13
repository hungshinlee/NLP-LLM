#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 2 的模型：上週的 tiny GPT（W4 demo 1 的 `block_final.GPT`）加一個 `pos="nope"|"rope"` 開關。

上週的 GPT 沒有任何位置編碼（NoPE：位置只從 causal mask 來，W3 p44）。這裡**不複製**它的程式——從檔案路徑載入
`../w04_d1_block_scratch/block_final.py`（照 W4 demo 2 的做法，不動 `sys.path`），只把每一層的 `attn` 換成下面的
`PosAttention`：同一組投影矩陣、同一個 W3 的 attention 函數，差別只在 `pos="rope"` 時對 `q`、`k` 做旋轉。

RoPE 的寫法是 Hugging Face Llama 的 rotate-half：把 `d_head` 個座標配成 (i, i + d_head/2) 對（demo 1 用的是相鄰配對，
同一個 R_m、座標排列不同——demo 1 的 README 有寫），對第 i 對轉 m·θ_i，θ_i = base^(-2i/d_head)。V 不旋轉。

兩個「不重訓的修法」都只改 θ_i，所以都是 `set_rope(model, base=..., scale=...)` 一行：
  position interpolation (PI)   scale = s        位置 m 換成 m/s（等價 θ_i → θ_i/s）
  NTK-aware                     base = b·s^(d_head/(d_head−2))   最低頻那一對恰好縮 s 倍、最高頻幾乎不動（`ntk_base()`）
cos/sin 表依 (L, base, scale, device) 快取，改了任何一個就重算。

GPT 建好之後呼叫一次 `model.apply(GPT._init)`：把換進來的 Linear 也初始化成 N(0, 0.02)。兩個版本（nope／rope）建構時
消耗的亂數序列完全相同，所以同一個 seed 下**初始權重逐個相等**，之後的差別只來自 θ。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    """從檔案路徑載入模組，不動 sys.path（W4 demo 1 彩排踩過：各資料夾的 tests.py 會撞名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ntk_base(base: float, s: float, d_head: int) -> float:
    """NTK-aware 的新底：b' = b · s^(d_head/(d_head − 2))。最後一對（i = d_head/2 − 1）的 θ 恰好變成 θ/s，第 0 對不動。"""
    return base * s ** (d_head / (d_head - 2.0))


def rope_tables(L: int, d_head: int, base: float, scale: float, device, dtype=torch.float32):
    """回傳 cos, sin，形狀 (L, d_head)：前後兩半各一份 (L, d_head/2)，配合 rotate-half。位置是 m/scale（PI）。"""
    i = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)            # 0, 2, …, d_head−2
    theta = base ** (-i / d_head)                                                  # θ_i = b^(−2i/d_head)，長度 d_head/2
    m = torch.arange(L, dtype=torch.float32, device=device) / scale                # (L,)
    ang = torch.outer(m, theta)                                                    # (L, d_head/2)：m·θ_i
    ang = torch.cat([ang, ang], dim=-1)                                            # (L, d_head)：兩半同一組角
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    """x: (B, H, L, d_head)；cos/sin: (L, d_head)。x·cos + rotate_half(x)·sin 就是對每一對 (i, i + d_head/2) 轉 m·θ_i。"""
    return x * cos + rotate_half(x) * sin


def make_pos_attention(bf):
    """用載進來的 block_final 模組造出 PosAttention 類別（繼承它的 CausalSelfAttention，沿用四個投影）。"""

    class PosAttention(bf.CausalSelfAttention):
        def __init__(self, d: int, n_head: int, dropout: float = 0.0, pos: str = "nope", base: float = 10000.0):
            super().__init__(d, n_head, dropout=dropout)
            assert pos in ("nope", "rope")
            self.pos, self.base, self.scale = pos, float(base), 1.0
            self._tables = None                                                   # (key, cos, sin)

        def tables(self, L: int, device, dtype):
            key = (L, self.base, self.scale, str(device), dtype)
            if self._tables is None or self._tables[0] != key:
                cos, sin = rope_tables(L, self.head_dim, self.base, self.scale, device, dtype)
                self._tables = (key, cos, sin)
            return self._tables[1], self._tables[2]

        def forward(self, x):
            B, L, _ = x.shape
            q = self.q(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)   # (B, H, L, d_head)
            k = self.k(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
            if self.pos == "rope":
                cos, sin = self.tables(L, x.device, q.dtype)
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)           # 只轉 q、k；v 不轉（demo 1 測試 (c) 的理由）
            mask = bf.causal_mask(L).to(x.device)
            out, A = bf.scaled_dot_product_attention(q, k, v, mask)              # W3 demo 1 的函數，與上週完全相同
            if self.training and self.dropout > 0:
                out = F.dropout(A, self.dropout) @ v                              # W4 的手刻 attention dropout
            return self.o(out.transpose(1, 2).reshape(B, L, self.n_head * self.head_dim))

    return PosAttention


class Demo1:
    """W4 demo 1 的東西：block_final（GPT）、train（語料、device、param groups、lr 排程、held-out 評估）、demo_config.toml。"""

    def __init__(self, rel: str = "../w04_d1_block_scratch"):
        import tomllib
        self.dir = (HERE / rel).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(f"找不到 W4 demo 1 的資料夾 {self.dir}")
        with open(self.dir / "demo_config.toml", "rb") as f:
            self.cfg = tomllib.load(f)
        self.bf = load_module(self.dir / "block_final.py", "w04_block_final")
        self.tr = load_module(self.dir / "train.py", "w04_d1_train")
        self.PosAttention = make_pos_attention(self.bf)

    def corpus(self) -> dict:
        return self.tr.load_corpus(self.cfg)

    def build(self, V: int, pos: str, base: float, seed: int, L: int | None = None):
        """同一個 seed 下 nope 與 rope 的初始權重逐個相等（見模組 docstring）。"""
        import torch as _t
        t = self.cfg["tiny"]
        _t.manual_seed(seed)
        m = self.bf.GPT(V=V, d=t["d"], n_layer=t["n_layer"], n_head=t["n_head"], d_ff=t["d_ff"], L=L or t["L"], dropout=t["dropout"])
        for blk in m.blocks:
            blk.attn = self.PosAttention(t["d"], t["n_head"], dropout=t["dropout"], pos=pos, base=base)
        m.apply(m._init)
        return m


def set_rope(model, base: float | None = None, scale: float | None = None) -> None:
    """改每一層的 θ：base（NTK-aware）或 scale（PI）。不動權重。"""
    for blk in model.blocks:
        a = blk.attn
        if base is not None:
            a.base = float(base)
        if scale is not None:
            a.scale = float(scale)
        a._tables = None


def rope_state(model) -> dict:
    a = model.blocks[0].attn
    return {"pos": a.pos, "base": a.base, "scale": a.scale, "d_head": a.head_dim}
