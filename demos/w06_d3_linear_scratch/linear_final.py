#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 3 的參考實作：linear attention 的兩種算法、causal 遞推的一步、以及「筆記本」的存與查。純 `torch`，釘 CPU、float64。

課堂上授課者在 linear_live.py 從零寫同一份（寫的順序見投影片 p26 講稿的【操作】），每寫完一段就
`./present.sh live` 跑四個測試（測試本體在 tests.py，兩份實作共用）。這一份是 (1) 課前確認四個測試全過、
(2) 現場卡住兩分鐘後 `./present.sh show` 切過來、(3) 課後發給學生。

函數（tests.py 依名稱與簽名呼叫，live 版要一樣；Q, K, V 的每一列是一個 token，d_head 是最後一維）：

  phi(x) -> same shape                              特徵映射 elu(x) + 1，逐元素、恆正（Katharopoulos et al. 2020 §3.2）
  attn_quadratic(Q, K, V) -> (L, d_head)            (phi(Q) phi(K)ᵀ) V ：先算 L×L 的分數矩陣，再乘 V。O(L² d_head)
  attn_linear(Q, K, V)    -> (L, d_head)            phi(Q) (phi(K)ᵀ V)：先算 d_head×d_head 的狀態，再乘 phi(Q)。O(L d_head²)
      兩者在數學上相等（結合律）、在計算上不等；兩個函數都沒有分母——正規化子 z 是同一個括號再用一次，留給 step。
  step(S, z, k, v, q) -> (S, z, o)                   causal 遞推的一步：S += phi(k) vᵀ、z += phi(k)、o = Sᵀ phi(q) / (zᵀ phi(q))
  store(keys, vals) -> (d_head, d_head)             純加法的筆記本：S = Σ_s k_s v_sᵀ（keys 已是要存的向量，不再套 phi）
  recall(S, k) -> (d_head,)                          用鍵查回：Sᵀ k
  delta_write(S, k, v, beta=1.0) -> S'               delta rule 的一筆：S + beta · k (v − Sᵀ k)ᵀ ——寫之前先把 k 方向上舊的那份擦掉

為什麼釘 CPU、float64：測試 (a)、(c) 要把「兩種算法相等」驗到機器精度（10⁻¹³ 量級），float32 只到 10⁻⁶；
Apple 的 Metal 後端沒有 double 型別——不是慢，是沒有（W3 demo 1、W5 demo 1 同一條理由）。
(b) 的計時釘 CPU 還有第二個理由：MPS 在小 L 時被 kernel launch 的固定成本蓋掉，交叉點會漂。

用法：
  python3 linear_final.py             # 四個測試（等同 ./present.sh）
  python3 linear_final.py --show      # L = 3、d_head = 2 的小例子逐步印出 + 四個測試（現場退路）
  python3 linear_final.py --rehearse  # 四個測試 + 寫 runs/rehearsal/tests.json（彩排；講稿裡的實測值從這裡抄）
"""
import sys

import torch
import torch.nn.functional as F

torch.set_default_device("cpu")           # 釘 CPU：Metal 沒有 double
torch.set_default_dtype(torch.float64)    # 整個檔案預設 float64


def phi(x):
    return F.elu(x) + 1                            # 逐元素、恆正；任何有限維的正映射都行，這是原論文的選擇


def attn_quadratic(Q, K, V):
    return (phi(Q) @ phi(K).T) @ V                 # 先 L×L，再乘 V：W3 的括號順序

def attn_linear(Q, K, V):
    return phi(Q) @ (phi(K).T @ V)                 # 先 d×d，再乘 phi(Q)：括號往右移，L×L 從未出現


def step(S, z, k, v, q):
    pk = phi(k)
    S = S + pk[:, None] * v[None, :]               # 一個外積：S_t = S_{t-1} + phi(k_t) v_tᵀ
    z = z + pk                                     # 分母也是一個遞推：z_t = z_{t-1} + phi(k_t)
    o = (S.T @ phi(q)) / (z @ phi(q))              # 輸出：Sᵀ phi(q)，除以 zᵀ phi(q)
    return S, z, o


def store(keys, vals):
    return keys.T @ vals                           # Σ_s k_s v_sᵀ：m 個外積的和，一行

def recall(S, k):
    return S.T @ k


def delta_write(S, k, v, beta=1.0):
    return S + beta * k[:, None] * (v - S.T @ k)[None, :]   # 寫之前先擦：差 (v − Sᵀk) 才寫進去，同一個鍵再出現是覆蓋不是疊加


# ── 現場退路：L = 3、d_head = 2 的小例子逐步印出 ────────────────────
def show():
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    L, d = 3, 2
    Q, K, V = torch.randn(L, d), torch.randn(L, d), torch.randn(L, d)
    print(f"L = {L}, d_head = {d}\n")
    A = phi(Q) @ phi(K).T
    print("phi(Q) phi(K)^T  (L x L, the transcript's score matrix):\n", A)
    print("\nphi(K)^T V       (d x d, the notebook):\n", phi(K).T @ V)
    print("\n(phi(Q) phi(K)^T) V =\n", attn_quadratic(Q, K, V))
    print("phi(Q) (phi(K)^T V) =\n", attn_linear(Q, K, V))
    print(f"max |difference| = {(attn_quadratic(Q, K, V) - attn_linear(Q, K, V)).abs().max().item():.1e}\n")
    print("the notebook with d_head = 2: two orthogonal keys, then a third")
    e1, e2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    v1, v2 = torch.tensor([3.0, 1.0]), torch.tensor([-2.0, 5.0])
    S = store(torch.stack([e1, e2]), torch.stack([v1, v2]))
    print("  recall(e1) =", recall(S, e1), " recall(e2) =", recall(S, e2), " (exact)")
    k3 = torch.tensor([1.0, 1.0]) / 2 ** 0.5
    v3 = torch.tensor([4.0, 4.0])
    S3 = store(torch.stack([e1, e2, k3]), torch.stack([v1, v2, v3]))
    print("  a third pair (k3 = (e1 + e2)/sqrt 2, v3 = (4, 4)) in the same 2 x 2 notebook:")
    print("    recall(e1) =", recall(S3, e1), " should be v1 =", v1, " — off by <k3, e1> v3 = v3 / sqrt 2")
    print("    recall(k3) =", recall(S3, k3), " should be v3 =", v3, " — off by (v1 + v2) / sqrt 2: three pairs cannot fit in two dimensions")
    print("\ndelta rule, same key written twice (|k| = 1):")
    k = torch.tensor([1.0, 0.0]); va, vb = torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0])
    S_add = store(torch.stack([k, k]), torch.stack([va, vb]))
    S_del = delta_write(delta_write(torch.zeros(d, d), k, va), k, vb)
    print("  additive: recall(k) =", recall(S_add, k), " = va + vb (stacked)")
    print("  delta   : recall(k) =", recall(S_del, k), " = vb      (overwritten)\n")


if __name__ == "__main__":
    import tests
    if "--show" in sys.argv:
        show()
    tests.run(sys.modules[__name__], rehearse="--rehearse" in sys.argv, label="linear_final.py")
