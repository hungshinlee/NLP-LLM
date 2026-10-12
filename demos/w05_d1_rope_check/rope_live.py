# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 1：課堂上從零寫的那一份。上課前清空到只剩這段註解；寫的順序見投影片 p18 講稿的【操作】。
# 每寫完一段就 ./present.sh live（= python3 tests.py rope_live）；還沒寫的函數會標 skip。
# tests.py 依名稱與簽名呼叫這三個（全部 float64、CPU；測試自己會給 float64 的輸入）：
#   rope_angles(d_head, base=10000.0) -> (d_head/2,)      theta_i = base ** (-2 i / d_head)
#   rotate(x, m, theta) -> same shape as x                 pair (2i, 2i+1) rotated by m * theta_i
#   score(q, k, m, n, theta) -> scalar                     rotate(q, m) . rotate(k, n)
import torch
