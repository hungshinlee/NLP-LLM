#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 2：課堂上的那一份。骨架（2026-09-30 起）——只有兩個地方現場動手：

  1. get_pairs()：本體是空的，課堂上寫這兩層迴圈（其他三個函數已經寫好，跑了就看結果）。
  2. to_symbols()：把「拆成字元」那一行換成「拆成 UTF-8 byte」（註解裡那行），其他什麼都不動，再跑一次。

跑法：python3 bpe_live.py   （get_pairs 還沒寫時會停在 NotImplementedError，這是預期的）
參考實作與三個測試在 bpe_final.py；卡住超過兩分鐘就 python3 bpe_final.py --show。
上課寫過之後：git checkout bpe_live.py 還原骨架（要在 commit 之後做，不然會把骨架也還原掉）。
"""
from collections import Counter

CORPUS = {"new": 3, "newest": 5, "nest": 2, "net": 3, "widest": 2}   # 與投影片手算的語料相同
END = "_"
NUM_MERGES = 8

def to_symbols(word):
    """一個詞 → 初始符號序列。"""
    return list(word) + [END]                                   # 字元 + 詞尾標記
    # return [bytes([b]) for b in (" " + word).encode("utf-8")]  # ← 現場換成這行：UTF-8 byte + 前導空白

def get_pairs(vocab):
    """vocab: {符號 tuple: 次數} → Counter{(a, b): 加權次數}，a、b 相鄰。課堂上寫這裡。"""
    raise NotImplementedError("課堂上寫：外層走每個詞，內層走相鄰的符號對，次數乘上詞的次數")

def merge_pair(pair, syms):
    """把 syms 裡所有相鄰的 pair 合併成一個符號（只合相鄰的）。"""
    a, b = pair
    out, i = [], 0
    while i < len(syms):
        if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(syms[i]); i += 1
    return tuple(out)

def train(corpus, num_merges):
    """num_merges 輪：數 pair → 挑最多的（同分取字典序最小）→ 全語料合併 → 記進 merge 表。"""
    vocab = {tuple(to_symbols(w)): n for w, n in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        top = max(pairs.values())
        best = min(p for p, c in pairs.items() if c == top)
        vocab = {merge_pair(best, syms): n for syms, n in vocab.items()}
        merges.append(best)
    return merges

def encode(word, merges):
    """照 merge 表的 rank 套用：rank 最小的先，同 rank 取最左邊。不重數頻率。"""
    syms = tuple(to_symbols(word))
    rank = {p: i for i, p in enumerate(merges)}
    while True:
        cands = [(rank[p], i) for i, p in enumerate(zip(syms, syms[1:])) if p in rank]
        if not cands:
            return list(syms)
        _, i = min(cands)
        syms = syms[:i] + (syms[i] + syms[i + 1],) + syms[i + 2:]

def decode(tokens):
    """接回去。字元版去掉詞尾標記；byte 版先 decode UTF-8 再去掉前導空白。"""
    if tokens and isinstance(tokens[0], bytes):
        return b"".join(tokens).decode("utf-8").lstrip(" ")
    return "".join(tokens).replace(END, "")

if __name__ == "__main__":
    # 1–2. merge 表，附每一輪的次數——前三條對紙上的答案（n+e 13、t+_ 12、s+t_ 9），全表對 bpe_final.py --show
    merges = train(CORPUS, NUM_MERGES)
    vocab = {tuple(to_symbols(w)): n for w, n in CORPUS.items()}
    print("merges (rank: pair -> symbol, count in that round):")
    for i, pair in enumerate(merges, 1):
        print("  %d: %r + %r -> %r   %d" % (i, pair[0], pair[1], pair[0] + pair[1], get_pairs(vocab)[pair]))
        vocab = {merge_pair(pair, syms): n for syms, n in vocab.items()}
    # 3. round-trip
    for w in CORPUS:
        toks = encode(w, merges)
        assert decode(toks) == w, (w, toks)
    print("round-trip ok for all %d words" % len(CORPUS))
    # 4. 一個語料裡沒有的字
    toks = encode("𠊎", merges)
    alphabet = ({bytes([b]) for b in range(256)} if isinstance(toks[0], bytes)
                else {c for w in CORPUS for c in w} | {END})
    known = alphabet | {a + b for a, b in merges}
    unk = [t for t in toks if t not in known]
    print("𠊎 ->", [t.hex() if isinstance(t, bytes) else t for t in toks],
          "| not in vocabulary: %r" % unk if unk else "| no UNK")
