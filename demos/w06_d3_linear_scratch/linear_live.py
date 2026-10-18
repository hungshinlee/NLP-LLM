# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 3：課堂上從零寫的那一份。上課前清空到只剩這段註解；寫的順序見投影片 p24 講稿的【操作】。
# 每寫完一段就 ./present.sh live（= python3 tests.py linear_live）；還沒寫的函數會標 skip。
# tests.py 依名稱與簽名呼叫這些（全部 float64、CPU；測試自己會給 float64 的輸入；每一列一個 token）：
#   phi(x) -> same shape                       elu(x) + 1
#   attn_quadratic(Q, K, V) -> (L, d_head)     (phi(Q) @ phi(K).T) @ V
#   attn_linear(Q, K, V)    -> (L, d_head)     phi(Q) @ (phi(K).T @ V)
#   step(S, z, k, v, q) -> (S, z, o)           S += phi(k) v^T ; z += phi(k) ; o = S^T phi(q) / (z . phi(q))
#   store(keys, vals) -> (d_head, d_head)      sum_s k_s v_s^T          recall(S, k) -> S^T k
#   delta_write(S, k, v, beta=1.0) -> S        S + beta * k (v - S^T k)^T
import torch
import torch.nn.functional as F
