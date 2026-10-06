# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 1：課堂上從零寫的那一份。上課前 ./present.sh reset 清到只剩「從這裡開始現場寫」那條線以上；寫的順序見投影片 p16 講稿的【操作】。
# 每寫完一段就 ./present.sh live（= python3 tests.py block_live）；還沒寫的類別會標 skip，(c) 會逐段推進。
# tests.py 依名稱與簽名呼叫這五個：
#   RMSNorm(d, eps=1e-6)                 forward(x) -> 同 shape        x * rsqrt(mean(x², -1) + eps) * weight
#   SwiGLU(d, d_ff)                      forward(x) -> 同 shape        down(silu(gate(x)) * up(x))，三個 Linear(bias=False)
#   Block(d, n_head, d_ff)               forward(x: (B, L, d))         x = x + attn(norm1(x)); x = x + ffn(norm2(x))
#   GPT(V, d, n_layer, n_head, d_ff, L)  forward(idx: (B, L)) -> (B, L, V)   emb → blocks → norm_f → h @ emb.weight.T
#   count_params(model) -> int           sum(p.numel() for p in model.parameters())
# attention 不重寫：下面的 CausalSelfAttention 把上週的函數包成多頭（兩行 reshape，寫但不講），reset 會保留它。
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util
_f = Path(__file__).resolve().parent.parent / "w03_d1_attention_scratch" / "attention_final.py"   # 上週寫的那五行
_spec = importlib.util.spec_from_file_location("w03_attention_final", _f)                       # 不動 sys.path：W3 那邊也有 tests.py，會撞名
_w3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_w3)
scaled_dot_product_attention, causal_mask = _w3.scaled_dot_product_attention, _w3.causal_mask


class CausalSelfAttention(nn.Module):
    """多頭的包裝：四個投影 + reshape；attention 本體是上週的函數。dropout 加在 A 上（上週的函數把 A 回傳出來，所以做得到）。"""

    def __init__(self, d, n_head, dropout=0.0):
        super().__init__()
        self.n_head, self.head_dim = n_head, d // n_head
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, L, d = x.shape
        q = self.q(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)   # (B, H, L, d_head)
        k = self.k(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        out, A = scaled_dot_product_attention(q, k, v, causal_mask(L).to(x.device))
        if self.training and self.dropout > 0:
            out = F.dropout(A, self.dropout) @ v
        return self.o(out.transpose(1, 2).reshape(B, L, d))


# ── 從這裡開始現場寫（reset 會把這條線以下清空）──────────────────
