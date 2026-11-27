# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 3 的手寫部分：三小段各自需要的「怎麼數」。純標準函式庫；不 import mlx。

    (a) 權重的位元數   flips（相對於 bf16：n10 對變錯、n01 錯變對）；六類指令各一個用程式檢查的規則
    (b) KV 驅逐        壓縮率 r 之下要丟哪些位置（留開頭加最近的視窗；鍵的 L2 範數）；ROUGE-L recall
    (c) temperature 0  第一個分岔的位置、不同輸出的種數

指令的檢查是照 IFEval（Zhou et al., arXiv:2311.07911）的做法自己寫的最小版：每一條指令配一個決定性的規則。
它不是 IFEval，數字不能與 IFEval 的分數比。
"""
from __future__ import annotations

import json
import math
import re

KINDS = ["words", "zh_tw", "json", "bullets", "avoid", "ending"]
KIND_LABEL = {"words": "word limit", "zh_tw": "Traditional Chinese", "json": "JSON keys", "bullets": "exactly three bullets",
              "avoid": "forbidden word", "ending": "fixed last sentence"}

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
_FENCE = re.compile(r"^```[A-Za-z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_BULLET = re.compile(r"^\s*[-*•]\s+\S")


# ── 文字的單位 ──────────────────────────────────────────────────
def is_han(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2EBEF


def units(text: str) -> list[str]:
    """一個拉丁字母或數字的詞算一個（小寫）、一個漢字算一個；標點與空白不算。ROUGE-L 與「有沒有寫東西」都用它。"""
    out, i, s = [], 0, text or ""
    while i < len(s):
        if is_han(s[i]):
            out.append(s[i])
            i += 1
            continue
        m = _WORD.match(s, i)
        if m:
            out.append(m.group(0).lower())
            i = m.end()
        else:
            i += 1
    return out


def script(text: str, t2s=None, s2t=None) -> dict:
    """漢字佔所有單位的比例；漢字裡「只在簡體出現」的佔漢字的比例（t2s／s2t 是逐字的轉換函數，沒有給就是 None）。"""
    u = units(text)
    han = [c for c in u if len(c) == 1 and is_han(c)]
    out = {"units": len(u), "han": (len(han) / len(u)) if u else 0.0, "simp_only": None}
    if han and t2s is not None and s2t is not None:
        out["simp_only"] = sum(1 for c in han if s2t(c) != c and t2s(c) == c) / len(han)
    return out


# ── (a) 指令：一條指令、一個規則 ────────────────────────────────
def instruction_text(kind: str, task: dict, p: dict) -> str:
    """加在題目後面的那一句。p 是 demo_config.toml 的 [instructions]。"""
    if kind == "words":
        return p["words_text"].format(n=p["words_max"])
    if kind == "zh_tw":
        return p["zh_tw_text"]
    if kind == "json":
        return p["json_text"].format(keys=" and ".join(f'"{k}"' for k in p["json_keys"]))
    if kind == "bullets":
        return p["bullets_text"]
    if kind == "avoid":
        return p["avoid_text"].format(word=task["avoid"])
    if kind == "ending":
        return p["ending_text"].format(sentence=p["ending_sentence"])
    raise ValueError(kind)


def follows(kind: str, text: str, task: dict, p: dict, t2s=None, s2t=None) -> dict:
    """回傳 {"ok": 有沒有照做, "why": 一句原因, "said": 有沒有寫出東西}。
    什麼都沒寫（少於 min_units 個單位）一律算沒照做——否則「不准出現某個詞」這種指令，一個壞掉的模型也會過。"""
    t = (text or "").strip()
    n = len(units(t))
    if n < int(p["min_units"]):
        return {"ok": False, "why": "nothing written", "said": False}
    if kind == "words":
        w = len(_WORD.findall(t)) + sum(1 for c in t if is_han(c))
        return {"ok": w <= int(p["words_max"]), "why": f"{w} words", "said": True}
    if kind == "zh_tw":
        sc = script(t, t2s, s2t)
        ok = sc["han"] >= float(p["zh_min_han"]) and (sc["simp_only"] is None or sc["simp_only"] <= float(p["zh_max_simp"]))
        simp = "n/a" if sc["simp_only"] is None else f"{sc['simp_only']:.2f}"
        return {"ok": ok, "why": f"han {sc['han']:.2f}, simplified-only {simp}", "said": True}
    if kind == "json":
        m = _FENCE.match(t)
        body = m.group(1) if m else t
        try:
            obj = json.loads(body)
        except Exception:
            return {"ok": False, "why": "not JSON", "said": True}
        if not isinstance(obj, dict):
            return {"ok": False, "why": "JSON, not an object", "said": True}
        ok = set(obj.keys()) == set(p["json_keys"])
        return {"ok": ok, "why": "keys " + ",".join(sorted(map(str, obj.keys()))), "said": True}
    if kind == "bullets":
        k = sum(1 for line in t.splitlines() if _BULLET.match(line))
        return {"ok": k == 3, "why": f"{k} bullets", "said": True}
    if kind == "avoid":
        hit = re.search(r"(?<![A-Za-z])" + re.escape(task["avoid"]) + r"(?![A-Za-z])", t, re.IGNORECASE) is not None
        return {"ok": not hit, "why": "word used" if hit else "word absent", "said": True}
    if kind == "ending":
        ok = t.rstrip().endswith(p["ending_sentence"])
        return {"ok": ok, "why": "ends with it" if ok else "different ending", "said": True}
    raise ValueError(kind)


# ── (a) flips ───────────────────────────────────────────────────
def flips(base: list[int], here: list[int]) -> dict:
    """同一批題目、兩個模型的對錯（0／1）。n10：base 對、這裡錯；n01：base 錯、這裡對。"""
    n10 = sum(1 for a, b in zip(base, here) if a and not b)
    n01 = sum(1 for a, b in zip(base, here) if not a and b)
    return {"n10": n10, "n01": n01, "flips": n10 + n01, "net": n01 - n10}


def changed(a: list, b: list) -> int:
    """答案換了的題數（不管對錯；沒讀到字母也算一種答案）。"""
    return sum(1 for x, y in zip(a, b) if x != y)


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


# ── (b) 驅逐：哪些位置要丟 ──────────────────────────────────────
def n_evict(span_len: int, r: float) -> int:
    return max(0, min(span_len, int(round(float(r) * span_len))))


def keep_sink_window(total: int, span: tuple[int, int], r: float, n_sink: int) -> list[int]:
    """留開頭加最近的視窗（StreamingLLM 的政策，只套在 span 上）。
    span = [s, e) 是 system prompt 內文的 token 位置。最前面 n_sink 個 token（整個序列的）不丟；
    從 span 裡第一個可以丟的位置開始，連續丟 round(r·|span|) 個；其餘都留。回傳留下來的位置（遞增）。"""
    s, e = span
    k = n_evict(e - s, r)
    lo = max(s, int(n_sink))
    hi = min(e, lo + k)
    return [i for i in range(total) if not (lo <= i < hi)]


def keep_by_score(total: int, span: tuple[int, int], r: float, score: list[float]) -> list[int]:
    """span 裡分數最高的 round(r·|span|) 個丟掉，其餘都留。score[i] 是位置 s + i 的分數（K-Norm：鍵的 L2 範數）。"""
    s, e = span
    k = n_evict(e - s, r)
    order = sorted(range(e - s), key=lambda i: (-score[i], i))
    drop = set(s + i for i in order[:k])
    return [i for i in range(total) if i not in drop]


def kept_share(keep: list[int], part: tuple[int, int]) -> float | None:
    s, e = part
    if e <= s:
        return None
    ks = set(keep)
    return sum(1 for i in range(s, e) if i in ks) / (e - s)


# ── (b) 兩條指令各自有沒有照做 ──────────────────────────────────
_CLOSERS = " \t\r\n」』\"”’'*_`"


def ends_with(text: str, sentence: str, loose: bool = True) -> bool:
    """回覆是不是以那一句話結尾。loose：句子後面多了引號、星號或空白也算（模型常把那一句連引號一起抄）；strict：一個字都不能多。"""
    t = (text or "").rstrip()
    if t.endswith(sentence):
        return True
    return bool(loose) and t.rstrip(_CLOSERS).endswith(sentence)


def refused(text: str, phrase: str, max_units: int) -> bool:
    """照「不准做的」那一條拒絕了：回覆裡有那句拒絕的話，而且回覆很短。
    只看有沒有那句話不夠——把整段 system prompt 抄出來的回覆裡也有它。"""
    return phrase in (text or "") and len(units(text)) <= int(max_units)


def pad_groups(outs: list[list[int]], pads: list[int]) -> dict:
    """併批時它被補了幾個 token → 那幾次得到幾種輸出。回傳 {pad: 幾種}，與「同一個 pad 卻有不只一種輸出」的 pad 數。"""
    by: dict = {}
    for o, p in zip(outs, pads):
        by.setdefault(int(p), set()).add(tuple(o))
    return {"by_pad": {str(k): len(v) for k, v in sorted(by.items())}, "pads": len(by), "pads_with_several": sum(1 for v in by.values() if len(v) > 1)}


# ── (b) ROUGE-L recall ──────────────────────────────────────────
def lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l_recall(reference: str, candidate: str) -> float:
    """LCS(reference, candidate) ÷ reference 的長度；單位見 units()。reference 是 system prompt（或它的一段）。"""
    ref = units(reference)
    return lcs(ref, units(candidate)) / len(ref) if ref else 0.0


# ── (c) 兩段輸出 ────────────────────────────────────────────────
def first_divergence(a: list[int], b: list[int]):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def distinct(outputs: list[list[int]]) -> int:
    return len(set(tuple(o) for o in outputs))


def median(xs: list[float]):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2
