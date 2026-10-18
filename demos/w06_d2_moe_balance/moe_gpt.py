#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 2 的模型：W4 demo 1 的 tiny GPT（`block_final.GPT`），每一層的 FFN 換成一個 MoE 層（`MoEFFN`）。

**不複製** W4 的程式——從檔案路徑載入 `../w04_d1_block_scratch/block_final.py`（照 W5 demo 2 `rope_gpt.py` 的做法，不動
`sys.path`），建好 `GPT` 之後只把每一層的 `blk.ffn` 換成下面的 `MoEFFN`。attention、Norm、embedding、dropout 都是上週的。

MoE 層（大綱〈形式化〉的式子，n_S = 0）：
  router      r_t = W^R x_t ∈ R^{n_E}；gate g_i = softmax(r_t)_i（Switch／Mixtral／Qwen3 的做法：對全部 n_E 個做 softmax，取選到的 k 個，不重新正規化）
  選集        T_t = TopK_k(r_t)                               balance = "none" | "switch"
              T_t = TopK_k(r_t + u)，gate 仍用 r_t           balance = "bias"（Wang et al. 2024；DeepSeek-V3）
  輸出        y_t = Σ_{i ∈ T_t} g_i(x_t) · E_i(x_t)          每個 E_i 是上週的 SwiGLU，寬 d_E（gate／up／down 三個矩陣，3·d·d_E 個參數）
  TopK 不可微：梯度只經 g_i 回到 W^R。

三種平衡法（`balance`）：
  "none"      什麼都不加。
  "switch"    L_aux = α · n_E · Σ_i f_i P_i（Fedus et al. 2022 §2.2 Eq. (4)–(6)），每層一個、加總後乘 α 加進主 loss。
              f_i = 這個 batch 裡分派給 expert i 的「token 份額」（count_i / (T·k)，Σ f_i = 1；k = 1 時就是原文的定義），
              P_i = router 給 i 的平均機率。均勻時 Σ f_i P_i = n_E · (1/n_E)² = 1/n_E，乘 n_E 是 1，所以 L_aux = α。
              f_i 是計數、不可微；梯度只經 P_i。
  "bias"      每個 expert 一個偏置 u_i（buffer，不是參數、不進 loss）。每一步訓練完，依這個 batch 的負載 c_i 更新：
              u_i ← u_i + γ · sign(c̄ − c_i)（Wang et al. 2024 §3 Algorithm 1 step 4；原文的 b_i、更新速度 u = 0.001；
              DeepSeek-V3 §2.1.2 Eq. (16) 把更新速度叫 γ，本課沿用 γ）。偏置只動「選誰」，不動「選到之後的 g_i」。

每次前向都把這一層的負載記在 `self.stats`：
  "count"  (n_E,)  每個 expert 被選到幾次（T·k 個名額裡）      "f" (n_E,) = count / (T·k)
  "P"      (n_E,)  softmax 機率的平均（可微；switch 的 L_aux 用它）
  "aux"    純量    α 以外的 n_E · Σ f_i P_i（三種設定都算，只有 switch 把它加進 loss——(i)(iii) 也能印「要是加了會是多少」）

尺寸：大綱〈課堂 demo 建議〉2 寫 d_E = d_ff/4；k = 2 時那是原 FFN 活化參數的一半，**這裡定 d_E = d_ff/2 = 344**，
讓每個 token 算的 expert 參數（k · 3·d·d_E = 528,384）與上週的 FFN（3·d·d_ff = 528,384）**逐個相等**（`w06-data.json` 的 `demo2`）。
總參數變多（每層 n_E · 3·d·d_E + n_E·d 的 router），這正是 MoE：同樣的每 token FLOPs，放更多參數。

同一個 seed 下三種 `balance` 的初始權重逐個相等：三者建構時消耗的亂數序列完全相同（u 是零初始化的 buffer），
建好之後 `model.apply(GPT._init)` 把換進來的 Linear 也初始化成 N(0, 0.02)。
"""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BALANCES = ("none", "switch", "bias")


def load_module(path: Path, name: str):
    """從檔案路徑載入模組，不動 sys.path（W4 demo 1 彩排踩過：各資料夾的 tests.py 會撞名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_moe_ffn(bf):
    """用載進來的 block_final 模組造出 MoEFFN（expert 是它的 SwiGLU）。"""

    class MoEFFN(nn.Module):
        def __init__(self, d: int, n_E: int, k: int, d_E: int, balance: str = "none", bias_step: float = 1e-3):
            super().__init__()
            assert balance in BALANCES, balance
            assert 1 <= k <= n_E
            self.d, self.n_E, self.k, self.d_E, self.balance, self.bias_step = d, n_E, k, d_E, balance, float(bias_step)
            self.router = nn.Linear(d, n_E, bias=False)                                  # W^R: n_E × d
            self.experts = nn.ModuleList([bf.SwiGLU(d, d_E) for _ in range(n_E)])        # E_i：上週的 SwiGLU，窄一點
            self.register_buffer("u", torch.zeros(n_E))                                  # 偏置 u（bias 才動；不是參數、不進 loss）
            self.stats: dict = {}

        def forward(self, x):
            B, L, d = x.shape
            h = x.reshape(-1, d)                                                         # (T, d)，T = B·L
            T = h.shape[0]
            r = self.router(h)                                                           # (T, n_E) logit
            p = F.softmax(r.float(), dim=-1)                                             # gate 的來源：對全部 n_E 做 softmax
            score = r + self.u if self.balance == "bias" else r                          # 選集用 r + u；gate 仍用 r
            sel = score.topk(self.k, dim=-1).indices                                     # (T, k)  T_t = TopK_k(·)
            g = p.gather(-1, sel).to(h.dtype)                                            # (T, k)  g_i，不重新正規化
            out = torch.zeros_like(h)
            count = torch.zeros(self.n_E, device=h.device)
            for i, E in enumerate(self.experts):                                         # n_E = 8：逐 expert 跑，選到它的 token 一起算
                hit = (sel == i)                                                         # (T, k)
                rows = hit.any(-1).nonzero(as_tuple=True)[0]
                count[i] = rows.numel()
                if rows.numel() == 0:
                    continue
                w = (g * hit).sum(-1)[rows]                                              # 這些 token 給 expert i 的 gate 值
                out.index_add_(0, rows, w.unsqueeze(-1) * E(h[rows]))
            f = count / (T * self.k)                                                     # 份額，Σ f_i = 1
            P = p.mean(0)                                                                # 平均機率，Σ P_i = 1（可微）
            self.stats = {"count": count.detach(), "f": f.detach(), "P": P.detach(),
                          "aux": (self.n_E * (f * P).sum())}                             # n_E · Σ f_i P_i（switch 的 L_aux 除以 α；可微經 P）
            return out.reshape(B, L, d)

        @torch.no_grad()
        def update_bias(self):
            """bias 設定：每一步之後 u_i ← u_i + γ · sign(c̄ − c_i)，c 是這個 batch 的負載計數。"""
            if self.balance != "bias" or "count" not in self.stats:
                return
            c = self.stats["count"]
            self.u += self.bias_step * torch.sign(c.mean() - c)

    return MoEFFN


class Demo1:
    """W4 demo 1 的東西：block_final（GPT、SwiGLU）、train（語料、device、param groups、lr 排程、held-out 評估）、demo_config.toml。"""

    def __init__(self, rel: str = "../w04_d1_block_scratch"):
        self.dir = (HERE / rel).resolve()
        if not self.dir.exists():
            raise FileNotFoundError(f"找不到 W4 demo 1 的資料夾 {self.dir}")
        with open(self.dir / "demo_config.toml", "rb") as f:
            self.cfg = tomllib.load(f)
        self.bf = load_module(self.dir / "block_final.py", "w04_block_final")
        self.tr = load_module(self.dir / "train.py", "w04_d1_train")
        self.MoEFFN = make_moe_ffn(self.bf)

    def corpus(self) -> dict:
        return self.tr.load_corpus(self.cfg)

    def build(self, V: int, balance: str, moe: dict, seed: int):
        """同一個 seed 下三種 balance 的初始權重逐個相等（見模組 docstring）。moe = {n_E, k, d_E, bias_step}。"""
        t = self.cfg["tiny"]
        torch.manual_seed(seed)
        m = self.bf.GPT(V=V, d=t["d"], n_layer=t["n_layer"], n_head=t["n_head"], d_ff=t["d_ff"], L=t["L"], dropout=t["dropout"])
        for blk in m.blocks:
            blk.ffn = self.MoEFFN(t["d"], moe["n_E"], moe["k"], moe["d_E"], balance=balance, bias_step=moe["bias_step"])
        m.apply(m._init)
        return m


def moe_layers(model):
    return [blk.ffn for blk in model.blocks]


def aux_sum(model):
    """所有 MoE 層的 n_E · Σ f_i P_i 加總（switch 乘 α 進 loss；其他設定只拿來印）。"""
    return sum(l.stats["aux"] for l in moe_layers(model))


def load_stats(model) -> dict:
    """最近一次前向的負載：每層的 f（n_E）、count、P；以及兩個指標（逐層）。"""
    layers = moe_layers(model)
    f = torch.stack([l.stats["f"] for l in layers]).cpu()               # (n_layer, n_E)
    return {"f": f.tolist(), "count": torch.stack([l.stats["count"] for l in layers]).cpu().tolist(),
            "P": torch.stack([l.stats["P"] for l in layers]).cpu().tolist(),
            "max_f": f.max(-1).values.tolist(),                           # 每層最大的 f_i
            "starved": (f < 0.01).sum(-1).tolist(),                        # 每層份額 < 1% 的 expert 數
            "u": torch.stack([l.u for l in layers]).cpu().tolist()}


def count_params_split(model) -> dict:
    """參數拆帳：expert 總量、router、每 token 活化的 expert 參數、其餘（attention、norm、embedding）。"""
    layers = moe_layers(model)
    experts = sum(sum(p.numel() for p in l.experts.parameters()) for l in layers)
    router = sum(l.router.weight.numel() for l in layers)
    total = sum(p.numel() for p in model.parameters())
    k, n_E = layers[0].k, layers[0].n_E
    active = total - experts + experts * k // n_E
    return {"total": total, "experts": experts, "router": router, "rest": total - experts - router,
            "active": active, "rho": active / total, "per_layer_expert_active": experts * k // n_E // len(layers)}
