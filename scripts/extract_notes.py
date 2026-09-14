#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把投影片的中文講稿抽成可在平板／手機上閱讀、或列印的文件。

用途：真正上課時筆電畫面給學生看，講稿放另一個裝置。
（單螢幕備課用 `?showNotes=true`，有第二螢幕用 presenter view 的 `S` 鍵。）

來源：slides/w*.qmd 的 `::: {.notes}` 區塊
輸出：$LECTURE_NOTES_DIR/w*.md（教師端，不在本 repo）

投影片編號與 reveal 右下角的「N / 36」一致：標題頁為 1，之後每個 `#`（章節分隔頁）
與每個 `##`（內容頁）各佔一號。

用法：python3 scripts/extract_notes.py
改完 slides/w01.qmd 的講稿後重跑，兩邊就不會漂移。
"""

import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANNER = ("<!-- 此檔由 scripts/extract_notes.py 從 slides/%s 自動產生，請勿直接編輯；"
          "請改該檔的 ::: {.notes} 區塊後重跑腳本。 -->")


def clean_heading(h):
    """去掉 {.smaller} 這類屬性。"""
    return re.sub(r"\s*\{[^}]*\}\s*$", "", h).strip()


def parse(path):
    lines = io.open(path, encoding="utf-8").read().replace("\r\n", "\n").split("\n")

    # front matter
    title = subtitle = ""
    if lines and lines[0].strip() == "---":
        for ln in lines[1:]:
            if ln.strip() == "---":
                break
            m = re.match(r'^(title|subtitle):\s*"?(.*?)"?\s*$', ln)
            if m:
                if m.group(1) == "title":
                    title = m.group(2)
                else:
                    subtitle = m.group(2)

    items, no, i = [], 1, 0          # no = 1 是標題頁
    while i < len(lines):
        ln = lines[i]
        m2 = re.match(r"^## (.+)$", ln)
        m1 = re.match(r"^# (.+)$", ln)
        if m2:
            no += 1
            items.append({"kind": "slide", "no": no,
                          "title": clean_heading(m2.group(1)), "notes": []})
        elif m1:
            no += 1
            items.append({"kind": "section", "no": no,
                          "title": clean_heading(m1.group(1)), "notes": []})
        elif ln.strip() == "::: {.notes}":
            body, i = [], i + 1
            while i < len(lines) and lines[i].strip() != ":::":
                body.append(lines[i])
                i += 1
            while body and not body[0].strip():
                body.pop(0)
            while body and not body[-1].strip():
                body.pop()
            if items:
                items[-1]["notes"] = body
        i += 1
    return title, subtitle, items


def render(src_name, title, subtitle, items):
    o = ["# %s — 中文講稿" % re.sub(r"^w0*", "W", os.path.splitext(src_name)[0]).upper()]
    if title:
        o.append("")
        o.append("**%s**%s" % (title, ("　·　" + subtitle) if subtitle else ""))
    n_notes = sum(1 for it in items if it["notes"])
    o += ["", BANNER % src_name, "",
          "> 共 %d 段講稿。編號對應投影片右下角的「N / 總數」；" % n_notes,
          "> 標題頁與章節分隔頁沒有講稿。", ""]

    prev_section = False
    for it in items:
        if it["kind"] == "section":
            o += ["---", "", "# ▸ %s　（投影片 %d）" % (it["title"], it["no"]), ""]
            if it["notes"]:
                o += it["notes"] + [""]
            prev_section = True
            continue
        if not prev_section:
            o += ["---", ""]
        prev_section = False
        o += ["## %d　%s" % (it["no"], it["title"]), ""]
        if it["notes"]:
            o += it["notes"] + [""]
        else:
            o += ["*（無講稿）*", ""]
    return "\n".join(o).rstrip() + "\n"


def main():
    decks = sorted(glob.glob(os.path.join(ROOT, "slides", "w*.qmd")))
    if not decks:
        sys.exit("slides/ 底下沒有 w*.qmd")
    # 講稿是教師端內容，不進這個公開 repo。輸出位置由環境變數指定。
    out_dir = os.environ.get("LECTURE_NOTES_DIR")
    if not out_dir:
        sys.exit("請先設定 LECTURE_NOTES_DIR，例如：\n"
                 "  export LECTURE_NOTES_DIR=~/Course-Hub/nlp_llm/teaching-notes")
    out_dir = os.path.expanduser(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for d in decks:
        name = os.path.basename(d)
        title, subtitle, items = parse(d)
        text = render(name, title, subtitle, items)
        out = os.path.join(out_dir, os.path.splitext(name)[0] + ".md")
        io.open(out, "w", encoding="utf-8").write(text)
        n = sum(1 for it in items if it["notes"])
        print("→ %s（%d 段講稿、%d 字）"
              % (os.path.relpath(out, ROOT), n,
                 len(re.sub(r"\s", "", "".join(
                     "".join(it["notes"]) for it in items)))))


if __name__ == "__main__":
    main()
