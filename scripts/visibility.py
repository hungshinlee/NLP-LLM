#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每週頁面的公開／不公開過濾。由 scripts/build_weeks.py 呼叫。

設計原則：**預設不公開（deny by default）**。
只有 PUBLIC_SECTIONS 名單內的 `###` 區塊會進入 weeks/*.qmd 上站。
之後在大綱新寫的任何區塊預設留在教師端——忘記標記的後果是「學生看不到」，
不是「教學底牌外流」。

三層控制
--------
1. 區塊層級（主要機制）：`###` 標題名稱比對 PUBLIC_SECTIONS。
   標題尾端的括號會被忽略，所以 `### 課堂骨架（180 min）` 比對的是「課堂骨架」。

2. 區塊層級覆寫：在 `###` 標題的**下一行**放
       <!-- vis: public -->    強制公開這個區塊
       <!-- vis: private -->   強制隱藏這個區塊

3. 段落層級挖空：在**已公開**的區塊裡，用一對註解夾住要拿掉的片段
       <!-- private -->
       這段只有我看得到
       <!-- /private -->

引用標記的處理是**逐課程設定**的，見下方 DROP_UNVERIFIED / STRIP_MARKERS。

語言
----
網站一律英文。公開區塊若在大綱裡附了

    <!-- en -->
    English for the students.
    <!-- /en -->

上站時**只取英文**，`###` 標題也換成 SECTION_EN 的英文；沒附的區塊維持中文，
所以尚未翻譯的週次不會壞掉。英文版仍走同一套過濾（private 段落、[驗]、[題]）。
"""

import re

# ===========================================================================
# 逐課程設定 —— 改這裡，不要改下面的邏輯
# ===========================================================================

# 會上站的 `###` 區塊。想多開一個就加進來（例如 "概念拆解路徑"）。
PUBLIC_SECTIONS = {
    "定位",
    "Learning objectives",
    "參考資料",
}

# 含「未查證」標記的行是否整行不上站。
# **2026-09-15 改為 True**，與 Speech-AI 一致。先前設 False 的前提是
# 「Syllabus 頁向學生完整說明了四級的意思」；那一段（How to Read This Site）
# 已整段移除——它其餘四點都是寫給備課的人看的——所以前提不再成立。
# 留著沒人解釋的 `[驗]` 比拿掉更糟：學生看到一個看不懂的符號，
# 又無從判斷那筆引用的可信度。四級標記本身仍是教師端的資產，留在
# Course-Hub 的大綱裡，只是不上站。見 Course-Hub 的 CLAUDE.md。
DROP_UNVERIFIED = True
UNVERIFIED_MARKER = "[驗]"

# 保留該行、但把標記本身拿掉的標記。
# `[非論文]` 也列進來，理由同上——沒有說明就只是雜訊。**代價要知道**：
# 這會讓「這是 blog 不是同儕審查」的訊號從標記層消失，所以大綱裡這類條目
# 必須把「不是論文」寫進句子本身，不能只靠標記。
STRIP_MARKERS = ("[題]", "[非論文]")

# 區塊標題的英文對照。網站一律英文，所以只要該區塊在大綱裡給了 <!-- en --> 版本，
# `###` 標題也一起換成英文；沒給英文版的區塊維持中文標題與中文內文。
# PUBLIC_SECTIONS 的每一項都必須在這裡有一筆，否則 build_weeks.py 會 sys.exit。
SECTION_EN = {
    "定位": "Where This Fits",
    "Learning objectives": "Learning Objectives",
    "參考資料": "References",
}

# 寫檔前的保險絲：頁面含這些字串就中止建置。
LEAK_MARKERS = ("課堂骨架", "常見誤解", "卡點提示", "demo 建議")

# ===========================================================================

_SEC = re.compile(r"^###\s+(.+?)\s*$")
_VIS = re.compile(r"^<!--\s*vis:\s*(public|private)\s*-->\s*$")
_POPEN = re.compile(r"^<!--\s*private\s*-->\s*$")
_PCLOSE = re.compile(r"^<!--\s*/private\s*-->\s*$")
_ENOPEN = re.compile(r"^<!--\s*en\s*-->\s*$")
_ENCLOSE = re.compile(r"^<!--\s*/en\s*-->\s*$")
_PAREN = re.compile(r"[（(][^（(]*?[）)]\s*$")
_STRIP = [re.compile(r"`?" + re.escape(m) + r"`?\s*") for m in STRIP_MARKERS]


def canon(heading):
    """`課堂骨架（180 min）` -> `課堂骨架`"""
    return _PAREN.sub("", heading).strip()


def _drop_empty_sections(lines):
    """移除過濾後只剩標題、沒有內文的區塊。"""
    out, i = [], 0
    while i < len(lines):
        if _SEC.match(lines[i]):
            j = i + 1
            while j < len(lines) and not _SEC.match(lines[j]):
                j += 1
            if any(ln.strip() for ln in lines[i + 1:j]):
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def _sections(body):
    """切成 [(### 標題行 或 None, 內文行)]；None 代表第一個 ### 之前的前言。"""
    blocks, head, cur = [], None, []
    for ln in body:
        if _SEC.match(ln):
            blocks.append((head, cur))
            head, cur = ln, []
        else:
            cur.append(ln)
    blocks.append((head, cur))
    return blocks


def _english_body(lines):
    """區塊內若有 <!-- en --> … <!-- /en -->，回傳英文版的行；沒有則回傳 None。

    網站是英文的，中文原文只留在大綱裡備課用，不上站。
    """
    out, inside, found = [], False, False
    for ln in lines:
        s = ln.strip()
        if _ENOPEN.match(s):
            inside, found = True, True
            continue
        if _ENCLOSE.match(s):
            inside = False
            continue
        if inside:
            out.append(ln)
    return out if found else None


def has_english(body):
    """這一週是否已經有英文內文（給 build_weeks.py 列 TODO 用）。"""
    return any(_ENOPEN.match(ln.strip()) for ln in body)


def _filter_lines(lines):
    """已確定公開的內文：挖掉 <!-- private --> 段落、處理引用標記。"""
    out, skip = [], False
    for ln in lines:
        s = ln.strip()
        if _POPEN.match(s):
            skip = True
            continue
        if _PCLOSE.match(s):
            skip = False
            continue
        if skip:
            continue
        if DROP_UNVERIFIED and UNVERIFIED_MARKER in ln:
            continue
        for rx in _STRIP:
            ln = rx.sub("", ln)
        out.append(ln)
    return out


def public_only(body):
    """body 是某一週 `## W<n>｜…` 底下的原始行（尚未 demote）。"""
    out = []
    for head, lines in _sections(body):
        if head is None:
            continue              # 前言（週標題的 <!-- en: … --> 等）不上站
        name = canon(_SEC.match(head).group(1))
        keep = name in PUBLIC_SECTIONS
        # 區塊層級覆寫必須緊接在標題的下一行
        if lines:
            mv = _VIS.match(lines[0].strip())
            if mv:
                keep = mv.group(1) == "public"
                lines = lines[1:]
        if not keep:
            continue
        en = _english_body(lines)
        if en is None:
            out.append(head)      # 還沒翻的區塊：維持中文標題與中文內文
        else:
            out.append("### " + SECTION_EN.get(name, name))
            lines = en
        out.extend(_filter_lines(lines))
        if out and out[-1].strip():
            out.append("")   # 區塊之間留一行空白，下一個標題才不會黏上來
    return _drop_empty_sections(out)


def assert_no_leak(page_text, where):
    """建置的最後一道保險絲。"""
    hit = [s for s in LEAK_MARKERS if s in page_text]
    if hit:
        raise SystemExit("%s 的網頁版含不該上站的內容 %s" % (where, hit))
