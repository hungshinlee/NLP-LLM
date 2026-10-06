#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 1 的參考實作：scaled dot-product attention、causal mask、帶 KV cache 的 decode step。純 `torch`，釘 CPU。

課堂上授課者在 attention_live.py 從零寫同一份（寫的順序見投影片 p21 講稿的【操作】），每寫完一段就
`./present.sh live` 跑五個測試（測試本體在 tests.py，兩份實作共用）。這一份是 (1) 課前確認五個測試全過、
(2) 現場卡住兩分鐘後 `./present.sh show` 切過來、(3) 課後發給學生。

三個函數（tests.py 依名稱與簽名呼叫，live 版要一樣）：

  scaled_dot_product_attention(Q, K, V, mask=None) -> (out, A)
      Q: (L_q, d_k)、K: (L, d_k)、V: (L, d_v)、mask: (L_q, L) 加在 score 上（0 或 -inf）。
      回傳 out = A @ V 與 attention 權重 A（demo 3 的熱圖就是這個 A）。前面可以有 batch 維度。
  causal_mask(L) -> (L, L)
      對角線以上 -inf、其餘 0。
  decode_step(x_t, cache, W) -> (1, d_v)
      x_t: (1, d)；cache: {"K": (t-1, d_k), "V": (t-1, d_v)}（就地 append）；W: {"q","k","v"} 三個投影矩陣。
      k_t、v_t 只算一次、接到 cache 後面，然後一個 query 對 cache 裡全部的 key 做 attention（不需要 mask：cache 裡只有過去）。

fix_fully_masked：全列被 mask 時 softmax 整列是 0/0 = NaN（測試 (e)）。這裡的修法是把 NaN 換成 0（該列輸出歸零）；
另一種修法是把 -inf 換成 torch.finfo(dtype).min（該列變成均勻分布），`--show` 會兩種都印。

用法：
  python3 attention_final.py             # 五個測試（等同 ./present.sh）
  python3 attention_final.py --show      # 小例子逐步印出 + 五個測試（現場退路）
  python3 attention_final.py --rehearse  # 五個測試 + 寫 runs/rehearsal/tests.json（彩排；講稿裡的實測值從這裡抄）
"""
import math
import sys

import torch

torch.set_default_device("cpu")   # 釘 CPU：測試 (b) 的 Jacobian 用 autograd，float32 就夠；改用有限差分才要 float64，而 Metal 沒有 double


def scaled_dot_product_attention(Q, K, V, mask=None, fix_fully_masked=True):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (L_q, L)：每個 query 對每個 key 的內積，除以 sqrt(d_k)
    if mask is not None:
        scores = scores + mask                          # -inf 的位置 exp 之後是 0
    A = torch.softmax(scores, dim=-1)                   # 逐列正規化：每個 query 的權重加總為 1
    if fix_fully_masked:
        A = A.masked_fill(torch.isnan(A), 0.0)          # 全列 -inf → 0/0 = NaN；這一列改成輸出 0
    return A @ V, A


def causal_mask(L):
    return torch.triu(torch.full((L, L), float("-inf")), diagonal=1)


def decode_step(x_t, cache, W):
    q_t = x_t @ W["q"]
    k_t = x_t @ W["k"]                                  # 只算一次
    v_t = x_t @ W["v"]
    cache["K"] = torch.cat([cache["K"], k_t], dim=0)    # append 到 cache
    cache["V"] = torch.cat([cache["V"], v_t], dim=0)
    out, _ = scaled_dot_product_attention(q_t, cache["K"], cache["V"])   # 一個 query 對全部的 key
    return out


# ── 現場退路：小例子逐步印出 ──────────────────────────────────
def show():
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
    L, d, d_k, d_v = 4, 6, 3, 2
    X = torch.randn(L, d)
    W = {"q": torch.randn(d, d_k) / math.sqrt(d), "k": torch.randn(d, d_k) / math.sqrt(d), "v": torch.randn(d, d_v) / math.sqrt(d)}
    Q, K, V = X @ W["q"], X @ W["k"], X @ W["v"]
    print(f"L = {L} tokens, d = {d}, d_k = {d_k}, d_v = {d_v}\n")
    print("scores = Q @ K^T / sqrt(d_k)\n", Q @ K.T / math.sqrt(d_k))
    M = causal_mask(L)
    print("\ncausal_mask(L)\n", M)
    out, A = scaled_dot_product_attention(Q, K, V, M)
    print("\nA = softmax(scores + mask), row by row (each row sums to 1; zeros above the diagonal)\n", A)
    print("\nout = A @ V\n", out)

    print("\n--- decode step by step: the cache grows by one row per token ---")
    cache = {"K": torch.empty(0, d_k), "V": torch.empty(0, d_v)}
    steps = []
    for t in range(L):
        o = decode_step(X[t:t + 1], cache, W)
        steps.append(o)
        print(f"t = {t}: cache K is {tuple(cache['K'].shape)}, out_t = {[round(v, 3) for v in o.squeeze(0).tolist()]}, "
              f"full-forward row = {[round(v, 3) for v in out[t].tolist()]}")
    print("max |step-by-step - full forward| =", (torch.cat(steps) - out).abs().max().item())
    print("max |cached K - X W_k|            =", (cache["K"] - K).abs().max().item(), "(old rows never change under causality)")

    print("\n--- a fully masked row: break it, then two fixes ---")
    M2 = M.clone(); M2[2, :] = float("-inf")
    _, A_bad = scaled_dot_product_attention(Q, K, V, M2, fix_fully_masked=False)
    print("row 2 all -inf, no fix     :", A_bad[2].tolist())
    _, A_fix = scaled_dot_product_attention(Q, K, V, M2)
    print("fix 1: NaN -> 0 (row output 0):", A_fix[2].tolist())
    M3 = M2.clone(); M3[M3 == float("-inf")] = torch.finfo(torch.float32).min
    _, A_unif = scaled_dot_product_attention(Q, K, V, M3, fix_fully_masked=False)
    print("fix 2: -inf -> finfo.min (uniform row):", A_unif[2].tolist())
    print()


if __name__ == "__main__":
    import tests
    if "--show" in sys.argv:
        show()
    tests.run(sys.modules[__name__], rehearse="--rehearse" in sys.argv, label="attention_final.py")
