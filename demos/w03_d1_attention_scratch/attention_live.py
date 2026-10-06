# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 1：課堂上從零寫的那一份。上課前清空到只剩這段註解；寫的順序見投影片 p21 講稿的【操作】。
# 每寫完一段就 ./present.sh live（= python3 tests.py attention_live）；還沒寫的函數會標 skip。
# tests.py 依名稱與簽名呼叫這三個：
#   scaled_dot_product_attention(Q, K, V, mask=None) -> (out, A)
#   causal_mask(L) -> (L, L)
#   decode_step(x_t, cache, W) -> (1, d_v)    cache = {"K": (t-1, d_k), "V": (t-1, d_v)}，W = {"q", "k", "v"}
