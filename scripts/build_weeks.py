#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以 docs/course-outline.md 為 single source of truth，產生 Quarto 網站頁面。

產出：
  weeks/w01.qmd … w14.qmd   每週一頁
  slides.qmd                投影片索引頁（偵測 slides/wNN.qmd 是否存在）
  supplements/*.qmd         補充教材各一頁（來源是手寫的 supplements/*.md）
  supplements.qmd           補充教材索引頁
  _includes/*.md            首頁／課程資訊／資源頁引用的片段

用法：python3 scripts/build_weeks.py
改完 docs/course-outline.md 後務必重跑，否則網站與大綱會漂移。
"""

import glob
import io
import os
import re
import sys

from visibility import (public_only, assert_no_leak, has_english,
                        PUBLIC_SECTIONS, SECTION_EN)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 大綱正本在 private repo（Course-Hub），不在這個公開 repo 裡。
# export COURSE_OUTLINE=~/Course-Hub/nlp_llm/course-outline.md
SRC = os.path.expanduser(os.environ.get("COURSE_OUTLINE", ""))
WEEKS_DIR = os.path.join(ROOT, "weeks")
INC_DIR = os.path.join(ROOT, "_includes")

BANNER = ("<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。"
          "大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->")

# 補充教材（選題／期中／期末／寫作教戰守策）：來源是手寫的 supplements/*.md
SUP_DIR = os.path.join(ROOT, "supplements")
SUP_BANNER = ("<!-- 此檔由 scripts/build_weeks.py 自動產生，"
              "請勿直接編輯；請改 supplements/*.md 後重跑腳本。 -->")
# 每份補充教材在 H1 下方用三行 HTML 註解宣告 metadata
META_RE = re.compile(r"^<!--\s*(en|order|summary):\s*(.+?)\s*-->$")

# 網站一律英文：骨架（首頁、導覽列、每週索引）與每週頁的內文都是英文，
# 只有 YAML 的 subtitle（該週中文標題）留著當對照。每週內文的英文版寫在大綱的
# <!-- en --> 區塊裡（見 visibility.py）；沒寫的週次會退回中文。
PART_OF = {}
for w in range(1, 7):
    PART_OF[w] = "Part I — Architecture"
for w in range(7, 11):
    PART_OF[w] = "Part II — Training"
for w in range(11, 15):
    PART_OF[w] = "Part III — Systems & Scrutiny"

# 首頁課程地圖：改為由週次資料產生 HTML/CSS grid（樣式在 styles.scss 的 .coursemap）。
# 好處是每個週次方塊可點、文字可被站內搜尋找到、窄螢幕會自動疊成一欄——這三件事 ASCII 與 SVG 都做不到。
# 中文的 ASCII 版仍留在 docs/course-outline.md（離線文件），英文 ASCII 版 docs/course-map-en.md 保留備查但不再上站。
MAP_EN = os.path.join(ROOT, "docs", "course-map-en.md")

# 地圖上的短標籤。週次的完整英文標題太長（"Transformer I: Attention Mechanics, ..."），
# 地圖需要另一組壓縮過的字。**改動週次主題時這裡要一起改**，少一週腳本會 sys.exit。
SHORT = {
    1:  "n-grams \u2192 seq2seq",   2:  "Tokenization",
    3:  "Attention, by hand",        4:  "The block &amp; dynamics",
    5:  "Position &amp; long context",   6:  "MoE / SSM / linear",
    7:  "Scaling laws &amp; data",   8:  "SFT, PEFT, forgetting",
    9:  "Alignment &amp; RL",        10: "Reasoning &amp; test-time",
    11: "Inference efficiency",      12: "RAG",
    13: "Agentic systems",           14: "Evaluation &amp; interp.",
}
HINGE = (3, 4)   # 全課樞紐，地圖上要highlight

# syllabus.qmd 與 resources.qmd 引用的英文片段（手寫，依 `<!-- file: X -->` 切段）
SITE_EN = os.path.join(ROOT, "docs", "site-en.md")
FILE_RE = re.compile(r"^<!-- file: (\S+) -->$")
NEEDED = ("disclaimer.md", "ledger.md", "reading-questions.md", "textbooks.md",
          "reading-table.md", "derivations.md", "toolchain.md", "halflife.md")

# 每週英文標題寫在 docs/course-outline.md 的標題下一行：<!-- en: ... -->
EN_RE = re.compile(r"^<!--\s*en:\s*(.+?)\s*-->$")


def read_source():
    if not os.path.exists(SRC):
        sys.exit("找不到大綱。請設定 COURSE_OUTLINE，例如：\n"
                 "  export COURSE_OUTLINE=~/Course-Hub/nlp_llm/course-outline.md"
                 + ("\n（目前為 %s）" % SRC if SRC else ""))
    with io.open(SRC, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").split("\n")


def demote(lines, levels=1):
    """把標題降級：### -> ##（供每週頁面使用，頁標題由 YAML 提供）。"""
    out = []
    for ln in lines:
        m = re.match(r"^(#{2,6}) (.*)$", ln)
        if m:
            n = len(m.group(1)) - levels
            if n >= 1:
                out.append("#" * n + " " + m.group(2))
                continue
        out.append(ln)
    return out


def strip_hr(lines):
    return [ln for ln in lines if ln.strip() != "---"]


def strip_en(lines):
    """英文標題註解只給建置腳本讀，不進入網頁內容。"""
    return [ln for ln in lines if not EN_RE.match(ln.strip())]


def trim(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def yaml_escape(s):
    return s.replace('"', '\\"')


def slice_section(lines, start_idx):
    """從 start_idx（標題行）之後取到下一個同級或更高級標題為止。"""
    level = len(re.match(r"^(#+)", lines[start_idx]).group(1))
    body = []
    for ln in lines[start_idx + 1:]:
        m = re.match(r"^(#+) ", ln)
        if m and len(m.group(1)) <= level:
            break
        body.append(ln)
    return body


def sup_slug(path):
    """NLP_LLM_Midterm_PoC_Guide.md → midterm-poc（網址用，去掉共同前後綴）。"""
    name = os.path.splitext(os.path.basename(path))[0]
    for pre in ("NLP_LLM_",):
        if name.startswith(pre):
            name = name[len(pre):]
    for suf in ("_Guide",):
        if name.endswith(suf):
            name = name[:-len(suf)]
    return name.replace("_", "-").lower()


def build_supplements():
    """supplements/*.md → 各一頁 supplements/<slug>.qmd + 索引 supplements.qmd。"""
    written = []
    items = []
    for src in sorted(glob.glob(os.path.join(SUP_DIR, "*.md"))):
        with io.open(src, encoding="utf-8") as f:
            lines = f.read().replace("\r\n", "\n").split("\n")
        if not lines or not lines[0].startswith("# "):
            sys.exit("%s 第一行必須是 H1 標題" % src)
        zh = lines[0][2:].strip()
        meta, body = {}, []
        for ln in lines[1:]:
            m = META_RE.match(ln.strip())
            if m:
                meta[m.group(1)] = m.group(2)
            else:
                body.append(ln)
        for key in ("en", "order", "summary"):
            if key not in meta:
                sys.exit("%s 缺 <!-- %s: ... -->" % (src, key))
        slug = sup_slug(src)
        out = "\n".join([
            "---",
            'title: "%s"' % yaml_escape(meta["en"]),
            'subtitle: "%s"' % yaml_escape(zh),
            "toc-depth: 2",
            "---",
            "",
            SUP_BANNER,
            "",
        ] + trim(body)) + "\n"
        written.append(write(os.path.join(SUP_DIR, slug + ".qmd"), out))
        items.append((int(meta["order"]), meta["en"], zh, meta["summary"], slug))

    if not items:
        return written
    items.sort()
    rows = ["| Guide | What it covers |",
            "|:-------------------------------|:--------------------------|"]
    for _, en, zh, summary, slug in items:
        rows.append("| **[%s](supplements/%s.qmd)**<br>[%s]{.wk-zh} | %s |"
                    % (en, slug, zh, summary))
    written.append(write(os.path.join(ROOT, "supplements.qmd"), "\n".join([
        "---",
        'title: "Supplements"',
        'subtitle: "Guides for the midterm report, the final paper, and writing them up"',
        "toc: false",
        "---",
        "",
        SUP_BANNER,
        "",
        "These guides back the two assessed deliverables \u2014 the midterm report (40%) "
        "and the final paper (60%) \u2014 from picking a topic through to the write-up. "
        "They treat both as submissions to ACL / EMNLP / NAACL / COLM / ICLR rather than as "
        "coursework, which is the standard the course is aiming at.",
        "",
        "**The guides themselves are in Chinese**, matching how the course is taught.",
        "",
    ] + rows + [
        "",
        "> **On the hardware assumption.** These guides describe the *term project*, not the course. The course itself asks nothing of your hardware \u2014 it is lectures, and the demos run in class. The topics here were chosen against a single 16 GB CUDA GPU (an RTX 5070 Ti or similar), which is what a QLoRA run or an activation-steering experiment at 1\u20138B actually needs. If you have less, say so early and pick a topic that fits: several of the ten are inference-only or work on weights offline, and need almost no GPU time at all.",
        "",
    ])))
    return written


def read_site_en():
    """把 docs/site-en.md 依 `<!-- file: X -->` 切成 {檔名: 內容}。"""
    if not os.path.exists(SITE_EN):
        sys.exit("找不到 %s（syllabus / resources 的英文片段來源）" % SITE_EN)
    with io.open(SITE_EN, encoding="utf-8") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    out, cur = {}, None
    for ln in lines:
        m = FILE_RE.match(ln.strip())
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return dict((k, "\n".join(trim(v))) for k, v in out.items())


def write(path, content):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def main():
    # 白名單裡的區塊都要有英文標題，否則英文頁面會冒出中文小標
    lack = sorted(x for x in PUBLIC_SECTIONS if x not in SECTION_EN)
    if lack:
        sys.exit("visibility.SECTION_EN 缺少 %s 的英文標題" % lack)

    lines = read_source()

    # ── 1. 每週頁面 ───────────────────────────────────────────
    week_heads = []
    for i, ln in enumerate(lines):
        m = re.match(r"^## W(\d+)｜(.+)$", ln)
        if m:
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            me = EN_RE.match(nxt)
            if not me:
                sys.exit("W%s 缺英文標題：請在標題下一行加 <!-- en: ... -->"
                         % m.group(1))
            week_heads.append((i, int(m.group(1)), m.group(2).strip(),
                               me.group(1).strip()))

    if len(week_heads) != 14:
        sys.exit("預期 14 個週次標題，實際找到 %d 個" % len(week_heads))

    written = []
    # 分隔列的破折號長度決定欄寬比例（Topic 欄放雙語標題，需要最多空間）
    index_rows = ["| Week | Topic | Part | Slides |",
                  "|:---|:-----------------------------|:------|:------|"]

    def slides_for(n):
        """該週是否已有投影片。存在才給連結，避免死連結。"""
        rel = os.path.join("slides", "w%02d.qmd" % n)
        return rel if os.path.exists(os.path.join(ROOT, rel)) else None

    no_en = []
    for idx, (li, wnum, title, en_title) in enumerate(week_heads):
        raw = slice_section(lines, li)
        if not has_english(raw):
            no_en.append(wnum)
        body = trim(strip_en(strip_hr(demote(public_only(raw)))))
        if not body:
            sys.exit("W%d 過濾後沒有任何可公開內容——索引會產生死連結。"
                     "請檢查 visibility.PUBLIC_SECTIONS 或該週的區塊標題。" % wnum)
        fm = [
            "---",
            # 側欄的項目直接取自 title，所以英文標題放 title、中文降為 subtitle
            'title: "W%d｜%s"' % (wnum, yaml_escape(en_title)),
            'subtitle: "%s · %s"' % (PART_OF[wnum], yaml_escape(title)),
            "toc-depth: 2",
            "---",
            "",
            BANNER,
            "",
        ]
        if slides_for(wnum):
            fm += ["::: {.callout-note appearance=\"minimal\"}",
                   "[**▶ Slides for this week**](../slides/w%02d.qmd)" % wnum,
                   ":::",
                   ""]
        out = "\n".join(fm + body) + "\n"
        assert_no_leak(out, "W%d" % wnum)
        written.append(write(os.path.join(WEEKS_DIR, "w%02d.qmd" % wnum), out))
        sl = ("[▶ Open](slides/w%02d.qmd)" % wnum) if slides_for(wnum) else "—"
        index_rows.append("| **W%d** | [%s](weeks/w%02d.qmd)<br>[%s]{.wk-zh} | %s | %s |"
                          % (wnum, en_title, wnum, title,
                             PART_OF[wnum].split(" — ")[-1], sl))

    # ── 2. syllabus.qmd 與 resources.qmd 引用的片段 ───────────
    # 網站是英文的，這些片段改由手寫的 docs/site-en.md 提供；
    # 中文原文留在 docs/course-outline.md，是離線閱讀用的文件，不上網站。
    en_parts = read_site_en()
    for fname in NEEDED:
        if fname not in en_parts:
            sys.exit("docs/site-en.md 缺區段 <!-- file: %s -->" % fname)
        written.append(write(os.path.join(INC_DIR, fname),
                             BANNER + "\n\n" + en_parts[fname] + "\n"))

    # 首頁的課程地圖：HTML/CSS grid，由上面的 week_heads 產生
    missing = [n for _, n, _, _ in week_heads if n not in SHORT]
    if missing:
        sys.exit("SHORT 缺少週次 %s 的短標籤（scripts/build_weeks.py）" % missing)
    parts = [("Part I — Architecture", range(1, 7)),
             ("Part II — Training", range(7, 11)),
             ("Part III — Systems & Scrutiny", range(11, 15))]
    # 連結是相對路徑，因此這個片段只能被站根目錄的 index.qmd 引用
    # 標題用 markdown 的 H2，與首頁其他區塊（Three Threads…、Weekly Schedule）一致，
    # 也才會進到頁內 TOC
    h = ['## Course Map', '', '<div class="coursemap">']
    # 地圖最上面先把終點講明白（原 ASCII 圖的目標方塊）
    h.append('<p class="cm-goal"><strong>Endpoint — read a modern LLM layer by '
             'layer:</strong> for every layer from tokenizer to agent, say why it '
             'looks the way it does and where it breaks. W1 opens with the '
             'compression framing and the compute ledger that the remaining '
             'thirteen weeks keep filling in.</p>')
    h.append('<div class="cm-grid">')
    for title, rng in parts:
        h.append('<div class="cm-col">')
        h.append('<div class="cm-part">%s</div>' % title.replace("&", "&amp;"))
        for n in rng:
            cls = "cm-wk cm-hinge" if n in HINGE else "cm-wk"
            h.append('<a class="%s" href="weeks/w%02d.html">'
                     '<span class="cm-n">W%d</span>'
                     '<span class="cm-t">%s</span></a>' % (cls, n, n, SHORT[n]))
        h.append('</div>')
    h.append('</div>')
    h.append('<p class="cm-note"><strong>W3 and W4 are the hinge.</strong> '
             'Everything in Part III reads from two objects built there — the '
             '<em>residual stream</em> and the <em>KV cache</em> — so those two weeks '
             'are the ones not to miss.</p>')
    h.append('</div>')
    written.append(write(os.path.join(INC_DIR, "coursemap.md"),
                         BANNER + "\n\n" + "\n".join(h) + "\n"))

    # 速查表與推導表裡的相對連結需指回 weeks/
    for fname in ("reading-table.md", "derivations.md"):
        p = os.path.join(INC_DIR, fname)
        if os.path.exists(p):
            with io.open(p, encoding="utf-8") as f:
                t = f.read()
            t = re.sub(r"^\| W(\d+) \|", lambda m: "| [W%s](weeks/w%02d.qmd) |"
                       % (m.group(1), int(m.group(1))), t, flags=re.M)
            write(p, t)

    written.append(write(os.path.join(INC_DIR, "week-index.md"),
                         BANNER + "\n\n" + "\n".join(index_rows) + "\n"))

    # ── 3. 投影片索引頁：有幾份就列幾份，不必手動維護導覽 ──────
    # 分隔列的破折號長度決定欄寬比例（Slides 欄要放得下 "Not yet available"）
    rows = ["| Week | Topic | Slides |",
            "|:---|:--------------------------|:--------|"]
    have = 0
    for _, wnum, title, en_title in week_heads:
        topic = "%s<br>[%s]{.wk-zh}" % (en_title, title)
        if slides_for(wnum):
            have += 1
            rows.append("| **W%d** | [%s](weeks/w%02d.qmd)<br>[%s]{.wk-zh} | "
                        "[▶ Open](slides/w%02d.qmd) |"
                        % (wnum, en_title, wnum, title, wnum))
        else:
            # 不斷行空格：避免窄欄位把 "Not yet available" 折成兩行
            rows.append("| W%d | %s | *Not yet available* |" % (wnum, topic))
    written.append(write(os.path.join(ROOT, "slides.qmd"), "\n".join([
        "---",
        'title: "Slides"',
        'subtitle: "English slides, Mandarin delivery — %d of 14 weeks ready"' % have,
        "toc: false",
        "---",
        "",
        BANNER,
        "",
        "The slides are written in English to accompany the Mandarin lectures. "
        "Each deck is a reveal.js page — open it in a browser, nothing to install.",
        "",
        "::: {.callout-tip}",
        "## Keyboard shortcuts",
        "`B` chalkboard layer for writing on a slide · `ESC` slide overview · "
        "`F` full screen · add `?print-pdf` to the URL to print",
        ":::",
        "",
    ] + rows + [
        "",
        "Three of the decks turn on a demo that only works if you have not seen it "
        "coming — a reward with a hole in it (W9), a hidden hint the chain of thought "
        "never mentions (W10), and an agent broken by prompt injection (W13). Those "
        "are posted after the lecture rather than before it.",
        "",
    ])))

    written += build_supplements()

    print("產生 %d 個檔案：" % len(written))
    for p in written:
        print("  " + os.path.relpath(p, ROOT))
    if no_en:
        print("\n尚未有英文內文、頁面仍是中文的週次："
              + "、".join("W%d" % n for n in no_en)
              + "\n（在大綱的公開區塊加 <!-- en --> … <!-- /en --> 即可）")


if __name__ == "__main__":
    main()
