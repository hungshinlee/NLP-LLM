# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 4：課堂上從零寫的那一份。上課前 ./present.sh reset 清空到只剩這段註解與 import；寫的順序見投影片 p47 講稿的【操作】。
# 每寫完一段就 ./present.sh live（= python3 tests.py dedup_live）；還沒寫的函數會標 skip。寫完三段：./present.sh run --impl live
# tests.py 與 demo.py 依名稱與簽名呼叫這些（純標準函式庫）：
#   shingles(text, n=5) -> set[str]                 小寫、re.findall(r"\w+") 切詞、連續 n 個詞用空白接成一個字串
#   minhash(shingles, k_h=128, seed=0) -> list[int]  每個 shingle 先 hash 成 64-bit 整數 h；第 i 個簽章 = min((a_i*h + b_i) % p)
#                                                    p = 2**61 - 1，(a_i, b_i) 由 random.Random(seed) 給；空集合回傳 None
#   lsh_candidates(sigs, b=16, r=8) -> set[(i, j)]   第 band 個 band 的 key = (band, tuple(sig[band*r:(band+1)*r]))，丟進 dict；
#                                                    同一個 key 底下的文件兩兩成候選（i < j）；sigs[i] 是 None 就跳過
import hashlib
import random
import re

P = (1 << 61) - 1


