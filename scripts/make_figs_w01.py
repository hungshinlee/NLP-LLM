#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1 投影片用的 SVG 圖。不依賴 matplotlib，直接寫 SVG。

先跑 scripts/make_demo_w01.py 產生 slides/assets/w01/w01-data.json，再跑這支。
輸出寫進 slides/assets/w01/。

兩條紀律（Speech-AI 踩過）：
1. SVG 是 XML，標籤文字裡一個裸露的 `&` 會讓整份文件解析失敗，而瀏覽器不會報錯
   ——只會把 <img> 的 naturalWidth 算成 0，投影片上看起來是一張空白，Quarto render 也照樣成功。
   所以寫檔前一律跑 xesc() 逃脫，並用 minidom 解析驗證，不過就丟例外。
2. 圖內不放標題。投影片標題已用 assertion-evidence 寫了同一句主張，圖再說一次是重複，
   而且白白吃掉垂直空間與字級。
"""
import json, os, re
from xml.dom import minidom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "slides", "assets", "w01")
C = {"ink": "#1f2328", "link": "#0b5cad", "muted": "#6b7280", "warn": "#dc2626",
     "ok": "#15803d", "amber": "#b45309", "line": "#d1d5db", "fill": "#f8fafc",
     "fill2": "#eef2ff", "grid": "#e5e7eb"}

def xesc(s):
    """逃脫裸 & （已是實體的不動），再逃脫 < >。"""
    s = re.sub(r"&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)", "&amp;", str(s))
    return s.replace("<", "&lt;").replace(">", "&gt;")

def txt(x, y, s, size=25, fill=None, anchor="start", weight="400", family=None, style=None):
    a = ['x="%g"' % x, 'y="%g"' % y, 'font-size="%g"' % size,
         'fill="%s"' % (fill or C["ink"]), 'text-anchor="%s"' % anchor,
         'font-weight="%s"' % weight]
    if family: a.append('font-family="%s"' % family)
    if style:  a.append('font-style="%s"' % style)
    return "<text %s>%s</text>" % (" ".join(a), xesc(s))

def rect(x, y, w, h, fill="#fff", stroke=None, sw=1.6, rx=7, dash=None):
    a = ['x="%g"' % x, 'y="%g"' % y, 'width="%g"' % w, 'height="%g"' % h,
         'rx="%g"' % rx, 'fill="%s"' % fill]
    if stroke: a += ['stroke="%s"' % stroke, 'stroke-width="%g"' % sw]
    if dash:   a.append('stroke-dasharray="%s"' % dash)
    return "<rect %s/>" % " ".join(a)

def line(x1, y1, x2, y2, stroke=None, sw=1.8, dash=None, arrow=False):
    a = ['x1="%g"' % x1, 'y1="%g"' % y1, 'x2="%g"' % x2, 'y2="%g"' % y2,
         'stroke="%s"' % (stroke or C["muted"]), 'stroke-width="%g"' % sw]
    if dash:  a.append('stroke-dasharray="%s"' % dash)
    if arrow: a.append('marker-end="url(#ah)"')
    return "<line %s/>" % " ".join(a)

def poly(pts, stroke, sw=2.6, fill="none"):
    return '<polyline points="%s" fill="%s" stroke="%s" stroke-width="%g" stroke-linejoin="round"/>' % (
        " ".join("%g,%g" % p for p in pts), fill, stroke, sw)

def svg(w, h, body, name):
    head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %g %g" width="100%%" '
            'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica Neue,'
            'PingFang TC,Noto Sans TC,sans-serif">' % (w, h))
    defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>' % C["muted"])
    doc = head + defs + body + "</svg>"
    minidom.parseString(doc)              # 解析不過就丟例外，不讓空白圖上線
    p = os.path.join(OUT, name)
    open(p, "w", encoding="utf-8").write(doc)
    print("  %s  (%.1f KB)" % (name, len(doc)/1024.0))
    return p

# ══════════════════════════════════════════════════════════════
# 1. 14 週依賴地圖：W3–W4 是樞紐，一條虛線指進整個 Part III 欄
# ══════════════════════════════════════════════════════════════
def course_map():
    W, H = 1240, 600
    b = []
    cols = [("Part I — Architecture", 40, [
                ("W1", "n-grams \u2192 seq2seq", True), ("W2", "Tokenization", False),
                ("W3", "Attention, by hand", "hinge"), ("W4", "Block & dynamics", "hinge"),
                ("W5", "Position & long ctx", False), ("W6", "MoE / SSM / linear", False)]),
            ("Part II — Training", 420, [
                ("W7", "Scaling & data", False), ("W8", "SFT & PEFT", False),
                ("W9", "Alignment & RL", False), ("W10", "Reasoning & TTC", False)]),
            ("Part III — Systems", 800, [
                ("W11", "Inference efficiency", False), ("W12", "RAG", False),
                ("W13", "Agentic systems", False), ("W14", "Evaluation & interp.", False)])]
    bw, bh, gap, top = 340, 62, 12, 96
    for title, x, items in cols:
        b.append(txt(x, 46, title, 27, C["link"], weight="700"))
        for i, (wk, lab, flag) in enumerate(items):
            y = top + i*(bh+gap)
            if flag == "hinge":
                b.append(rect(x, y, bw, bh, C["fill2"], C["link"], 2.6))
            elif flag is True:
                b.append(rect(x, y, bw, bh, "#fff", C["amber"], 2.4, dash="7 5"))
            else:
                b.append(rect(x, y, bw, bh, "#fff", C["line"], 1.6))
            b.append(txt(x+16, y+40, wk, 27, C["link"] if flag else C["ink"], weight="700"))
            b.append(txt(x+82, y+40, lab, 25, C["ink"]))

    # Part III 的虛線框（樞紐依賴指向整欄，不是單一週）
    p3x, p3y = 800-14, top-16
    p3h = 4*(bh+gap)+22
    b.append(rect(p3x, p3y, bw+28, p3h, "none", C["link"], 2.2, rx=11, dash="9 6"))
    # 灰色匯總箭頭：前面撐後面
    b.append(line(400, 300, 414, 300, C["muted"], 2.4, arrow=True))
    b.append(line(780, 300, 794, 300, C["muted"], 2.4, arrow=True))
    # 藍色虛線：W3–W4 → 整個 Part III
    ax, ay = 40+bw+8, top+2*(bh+gap)+bh
    dy = p3y + p3h + 54
    b.append('<path d="M%g,%g L %g,%g L %g,%g L %g,%g" fill="none" stroke="%s" '
             'stroke-width="3" stroke-dasharray="10 6" stroke-linejoin="round"/>'
             % (ax, ay, ax+12, ay, ax+12, dy, p3x+bw/2+14, dy, C["link"]))
    b.append('<path d="M%g,%g L %g,%g" fill="none" stroke="%s" stroke-width="3" '
             'stroke-dasharray="10 6" marker-end="url(#ah2)"/>'
             % (p3x+bw/2+14, dy, p3x+bw/2+14, p3y+p3h+8, C["link"]))
    b.append('<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             'fill="%s"/></marker></defs>' % C["link"])
    b.append(txt(380, 566, "residual stream + KV cache: everything downstream reads these",
                 24, C["link"], weight="700"))
    b.append(rect(40, 548, 26, 16, "none", C["amber"], 2.2, rx=3, dash="5 4"))
    b.append(txt(76, 564, "today", 22, C["amber"], weight="700"))
    return svg(W, H, "".join(b), "course-map.svg")

# ══════════════════════════════════════════════════════════════
# 2. 算力帳本母版
# ══════════════════════════════════════════════════════════════
def ledger():
    W, H = 1180, 470
    b = []
    b.append(rect(20, 20, 540, 420, C["fill"], C["line"], 1.8, rx=11))
    b.append(rect(620, 20, 540, 420, C["fill"], C["line"], 1.8, rx=11))
    b.append(txt(44, 62, "Training", 28, C["link"], weight="700"))
    b.append(txt(644, 62, "Inference, per token", 28, C["amber"], weight="700"))
    b.append(line(44, 78, 536, 78, C["line"], 1.4))
    b.append(line(644, 78, 1136, 78, C["line"], 1.4))
    L = [(120, "C  ≈  6 · N · D", 30, C["ink"], "700"),
         (158, "N  parameters", 24, C["muted"], "400"),
         (190, "D  training tokens", 24, C["muted"], "400"),
         (222, "6  = 2 forward + 4 backward", 24, C["muted"], "400"),
         (282, "optimizer state  ≈  2 · N   (Adam)", 25, C["ink"], "400"),
         (330, "MoE:  N_total sets memory,", 25, C["ink"], "400"),
         (362, "          N_active sets FLOPs", 25, C["ink"], "400")]
    for y, s, sz, col, wt in L:
        b.append(txt(44, y, s, sz, col, weight=wt))
    R = [(120, "prefill:   ≈ 2 · N · L_in", 28, C["ink"], "700"),
         (152, "compute bound", 22, C["ok"], "700"),
         (206, "decode:  ≈ 2 · N", 28, C["ink"], "700"),
         (238, "memory bound", 22, C["warn"], "700"),
         (292, "KV cache =", 25, C["ink"], "400"),
         (324, "  2 · n_layer · n_kv_head · d_head", 24, C["ink"], "400"),
         (354, "      · L · bytes_per_elem", 24, C["ink"], "400"),
         (402, "the bound is moving N and the KV", 23, C["muted"], "400"),
         (428, "from HBM into SRAM, not the arithmetic", 23, C["muted"], "400")]
    for y, s, sz, col, wt in R:
        b.append(txt(644, y, s, sz, col, weight=wt))
    b.append(line(566, 230, 612, 230, C["muted"], 2.4, arrow=True))
    return svg(W, H, "".join(b), "ledger.svg")

# ══════════════════════════════════════════════════════════════
# 3. sparsity 與 bits-per-byte 的兩張圖（真實數據）
# ══════════════════════════════════════════════════════════════
def curves(data):
    sp, bpb = data["sparsity"], data["bpb"]
    W, H = 1240, 470
    b = []
    # 左：sparsity
    ox, oy, pw, ph = 96, 372, 430, 250
    b.append(line(ox, oy, ox+pw, oy, C["ink"], 2))
    b.append(line(ox, oy, ox, oy-ph, C["ink"], 2))
    for v in (0, 25, 50, 75, 100):
        y = oy - ph*v/100.0
        b.append(line(ox, y, ox+pw, y, C["grid"], 1.2))
        b.append(txt(ox-12, y+8, "%d%%" % v, 22, C["muted"], anchor="end"))
    ks = sorted(int(k) for k in sp)
    pts = []
    for i, k in enumerate(ks):
        x = ox + pw*(i+0.5)/len(ks)
        y = oy - ph*float(sp[str(k)] if str(k) in sp else sp[k])/100.0
        pts.append((x, y))
        b.append(txt(x, oy+32, "n=%d" % k, 23, C["muted"], anchor="middle"))
        v_ = (sp[str(k)] if str(k) in sp else sp[k])
        lab_dy = 32 if v_ > 90 else (-20 if (v_ > 60 or v_ < 20) else 28)
        b.append(txt(x, y+lab_dy, "%.1f" % (sp[str(k)] if str(k) in sp else sp[k]), 23,
                     C["warn"], anchor="middle", weight="700"))
    b.append(poly(pts, C["warn"], 3))
    for x, y in pts:
        b.append('<circle cx="%g" cy="%g" r="5.5" fill="%s"/>' % (x, y, C["warn"]))
    b.append(txt(ox, oy-ph-40, "held-out n-grams never seen in training", 24, C["warn"], weight="700"))
    # 右：bits-per-byte
    ox2 = 748
    b.append(line(ox2, oy, ox2+pw, oy, C["ink"], 2))
    b.append(line(ox2, oy, ox2, oy-ph, C["ink"], 2))
    ks2 = sorted(int(k) for k in bpb)
    vals = [float(bpb[str(k)] if str(k) in bpb else bpb[k]) for k in ks2]
    lo, hi = 1.75, 2.40
    for v in (1.8, 2.0, 2.2, 2.4):
        y = oy - ph*(v-lo)/(hi-lo)
        b.append(line(ox2, y, ox2+pw, y, C["grid"], 1.2))
        b.append(txt(ox2-12, y+8, "%.1f" % v, 22, C["muted"], anchor="end"))
    pts2 = []
    for i, (k, v) in enumerate(zip(ks2, vals)):
        x = ox2 + pw*(i+0.5)/len(ks2)
        y = oy - ph*(v-lo)/(hi-lo)
        pts2.append((x, y))
        b.append(txt(x, oy+32, "n=%d" % k, 23, C["muted"], anchor="middle"))
        best = (v == min(vals))
        b.append(txt(x, y + (26 if (i in (1,2)) else -18), "%.3f" % v, 23,
                     C["ok"] if best else C["muted"], anchor="middle", weight="700"))
    b.append(poly(pts2, C["link"], 3))
    for i, (x, y) in enumerate(pts2):
        best = (vals[i] == min(vals))
        b.append('<circle cx="%g" cy="%g" r="%g" fill="%s"/>'
                 % (x, y, 7 if best else 5.5, C["ok"] if best else C["link"]))
    b.append(txt(ox2, oy-ph-40, "bits per byte, held-out  (lower is better)",
                 24, C["link"], weight="700"))
    b.append(txt(ox2+pw, oy+72, "stupid backoff, \u03bb = 0.4", 22, C["muted"], anchor="end"))
    b.append(txt(ox, oy+72, "word-level, 250k tokens of Shakespeare", 22, C["muted"]))
    return svg(W, H, "".join(b), "curves.svg")

if __name__ == "__main__":
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    d = json.load(open(os.path.join(OUT, "w01-data.json"), encoding="utf-8"))
    print("產生 SVG：")
    course_map(); ledger(); curves(d)
    print("全部通過 XML 解析驗證。")
