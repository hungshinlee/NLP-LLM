#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 2 的參考實作：BPE 的 train / encode / decode，加三個測試。純標準函式庫。

課堂上授課者在 bpe_live.py 從零寫同一份；這一份是 (1) 課前確認測試全過、(2) 現場卡住兩分鐘後 `--show` 切過來、
(3) 課後發給學生。玩具語料與 slides/scripts/make_demo_w02.py 第 3 節相同，merge 序列必須逐條一致（測試 a）。

兩種基礎字母表，同一套演算法：
  --chars  字元 + 詞尾標記 "_"（JM3 的寫法；投影片用這個講概念）。沒見過的字元 → 沒有 id（測試 c 的對照）
  --bytes  UTF-8 byte + 前導空白（GPT-2 的寫法）。任何字串都能編碼，「𠊎」是 4 個 byte token，沒有 UNK

用法：
  python3 bpe_final.py            # 跑三個測試
  python3 bpe_final.py --show     # 印出 merge 表、各詞的 encode trace（現場退路）
  python3 bpe_final.py --bert 𠊎  # 同一個字丟進 bert-base-chinese（需要 transformers 與本機快取，HF_HUB_OFFLINE=1）
"""
import re, sys
from collections import Counter

CORPUS = {"new": 3, "newest": 5, "nest": 2, "net": 3, "widest": 2}   # 與 make_demo_w02.py 相同
END = "_"

# ── 訓練 ────────────────────────────────────────────────────────
def to_symbols(word, mode):
    """一個詞 → 初始符號序列。chars：字元 + 詞尾；bytes：前導空白 + UTF-8 byte（每個 byte 一個符號）。"""
    if mode == "chars":
        return list(word) + [END]
    return [bytes([b]) for b in (" " + word).encode("utf-8")]

def get_pairs(vocab):
    """vocab: {符號 tuple: 次數}。回傳每一對相鄰符號的加權次數。"""
    pairs = Counter()
    for syms, n in vocab.items():
        for a, b in zip(syms, syms[1:]):
            pairs[(a, b)] += n
    return pairs

def merge_pair(pair, syms):
    a, b = pair
    out, i = [], 0
    while i < len(syms):
        if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(syms[i]); i += 1
    return tuple(out)

def train(corpus, num_merges, mode="chars"):
    """回傳有序的 merge 表。同分時取字典序最小的 pair（手算才對得上）。"""
    vocab = {tuple(to_symbols(w, mode)): n for w, n in corpus.items()}
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

# ── 編碼與解碼 ──────────────────────────────────────────────────
def encode(word, merges, mode="chars", trace=None):
    """照 merge 表的 rank 套用（rank 最小的先；同 rank 取最左邊），不重數頻率。"""
    syms = tuple(to_symbols(word, mode))
    rank = {p: i for i, p in enumerate(merges)}
    if trace is not None: trace.append(syms)
    while True:
        cands = [(rank[p], i) for i, p in enumerate(zip(syms, syms[1:])) if p in rank]
        if not cands:
            return list(syms)
        _, i = min(cands)
        syms = syms[:i] + (syms[i] + syms[i + 1],) + syms[i + 2:]
        if trace is not None: trace.append(syms)

def decode(tokens, mode="chars"):
    if mode == "chars":
        return "".join(tokens).replace(END, "")
    return b"".join(tokens).decode("utf-8").lstrip(" ")

# ── 三個測試 ────────────────────────────────────────────────────
REFERENCE_MERGES = [("n", "e"), ("t", "_"), ("s", "t_"), ("ne", "w"), ("e", "st_"), ("new", "est_"), ("ne", "t_"), ("new", "_")]

def test_a():
    m = train(CORPUS, 8, "chars")
    assert m == REFERENCE_MERGES, m
    return "a: merges match the reference (and the paper answer: n+e 13, t+_ 12, s+t_ 9)"

def test_b():
    m = train(CORPUS, 8, "chars")
    for w in CORPUS:
        assert decode(encode(w, m, "chars"), "chars") == w
    mb = train(CORPUS, 30, "bytes")
    for s in list(CORPUS) + ["west", "newer", "𠊎係客家人", "In sī Tâi-uân-lâng", "café"]:
        assert decode(encode(s, mb, "bytes"), "bytes") == s, s
    return "b: decode(encode(w)) == w for every word; byte-level: for arbitrary UTF-8 too"

def test_c():
    mb = train(CORPUS, 30, "bytes")
    toks = encode("𠊎", mb, "bytes")
    body = [t for t in toks if t != b" "]
    assert len(body) == 4 and all(len(t) == 1 for t in body), body
    known = {c for w in CORPUS for c in w}
    assert "𠊎" not in known                                    # 字元版沒有它的 id
    return "c: 𠊎 under byte-level = %s (4 byte tokens, no UNK); the character alphabet has no id for it" % \
           " ".join("0x%02X" % t[0] for t in body)

def show():
    m = train(CORPUS, 8, "chars")
    print("merge list (rank: pair -> new symbol)")
    for i, (a, b) in enumerate(m, 1):
        print("  %d: %s + %s -> %s" % (i, a, b, a + b))
    for w in ["nest", "west", "newer", "widest"]:
        tr = []
        toks = encode(w, m, "chars", trace=tr)
        print("\n%s -> %s" % (w, toks))
        for st in tr:
            print("    " + " ".join(st))
    mb = train(CORPUS, 30, "bytes")
    print("\nbyte-level, 𠊎 ->", [t.hex() for t in encode("𠊎", mb, "bytes")])

def bert(s):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print("bert-base-chinese:", tok.convert_ids_to_tokens(ids), "->", tok.decode(ids))

if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    elif "--bert" in sys.argv:
        bert(sys.argv[sys.argv.index("--bert") + 1])
    else:
        for t in (test_a, test_b, test_c):
            print("ok  " + t())
