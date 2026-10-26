# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 4 的參考實作：MinHash 去重的三個函數（投影片 p44–p45 黑板上的三行，p47 現場寫）。

    shingles(text, n=5)                 -> set[str]        小寫、\\w+ 切詞、連續 n 個詞接成一個 shingle
    minhash(shingles, k_h=128, seed=0)  -> list[int]       k_h 個「隨機排列的最小值」：排列用 (a·h + b) mod p 代替
    lsh_candidates(sigs, b=16, r=8)     -> set[(i, j)]     簽章切成 b 個 band、每 band r 列；任一 band 相同就成候選對

課堂上從零寫的那一份是 dedup_live.py（同樣的三個簽名）；tests.py 的三個測試對兩份都能跑。
純標準函式庫。這個檔只有函數，沒有 main：跑語料用 demo.py。
"""
import hashlib
import random
import re

MERSENNE_61 = (1 << 61) - 1          # p：比 64-bit hash 大的質數，(a·h + b) mod p 才是一個排列
_WORD = re.compile(r"\w+")


def shingles(text, n=5):
    """文件 → word n-gram 的集合。少於 n 個詞的文件是空集合。"""
    words = _WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _hash64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def permutations(k_h, seed=0):
    """k_h 對 (a, b)：a ∈ [1, p), b ∈ [0, p)。同一個 seed 給同一組，所有文件要用同一組。"""
    rng = random.Random(seed)
    return [(rng.randrange(1, MERSENNE_61), rng.randrange(0, MERSENNE_61)) for _ in range(k_h)]


def minhash(shingles, k_h=128, seed=0):
    """每一個排列 π_i(x) = (a_i·hash(x) + b_i) mod p 底下的 min。空集合回傳 None。"""
    if not shingles:
        return None
    hs = [_hash64(s) for s in shingles]
    p = MERSENNE_61
    return [min((a * h + b) % p for h in hs) for a, b in permutations(k_h, seed)]


def lsh_candidates(sigs, b=16, r=8):
    """sigs[i] 是第 i 份文件的簽章（None 的文件跳過）。b·r 不得超過簽章長度。"""
    buckets = {}
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for band in range(b):
            key = (band, tuple(sig[band * r:(band + 1) * r]))
            buckets.setdefault(key, []).append(i)
    pairs = set()
    for ids in buckets.values():
        if len(ids) > 1:
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs.add((ids[x], ids[y]))
    return pairs


# ── 三行之外、demo.py 與 tests.py 用得到的小東西（不必現場寫）──────────
def jaccard(A, B):
    """精確的 Jaccard；兩個空集合算 0。"""
    u = len(A | B)
    return len(A & B) / u if u else 0.0


def estimate(sig_a, sig_b):
    """兩個簽章的碰撞比例 = Jaccard 的無偏估計。"""
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def candidate_probability(J, b, r):
    """S 曲線：1 − (1 − J^r)^b。"""
    return 1.0 - (1.0 - J ** r) ** b
