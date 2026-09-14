#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1 cold open 與 embedding demo 的真實素材。

投影片上的每一個數字都由這支腳本產生，沒有任何印象值。
輸出：slides/assets/w01/w01-data.json（進版控，所以不重跑也能 render 投影片）

語料：tinyshakespeare（莎士比亞，公版）。第一次執行會下載到 _corpus/（已 gitignore）。
需要 numpy。預估 CPU 兩分鐘。

用法：python3 scripts/make_demo_w01.py
"""
import os, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "slides", "assets", "w01")
CORP = os.path.join(ROOT, "_corpus")
URL  = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt")
os.makedirs(CORP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
CPATH = os.path.join(CORP, "tinyshakespeare.txt")
if not os.path.exists(CPATH):
    print("下載語料 ...")
    urllib.request.urlretrieve(URL, CPATH)

import re, random, json, math
from collections import defaultdict, Counter
import numpy as np

text = open(CPATH, encoding='utf-8').read()
toks = re.findall(r"[A-Za-z']+|[.,!?;:]", text)
split = int(len(toks)*0.9)
train, held = toks[:split], toks[split:]
uni = Counter(train)
V = len(uni)

def build(n, data):
    m = defaultdict(Counter)
    for i in range(len(data)-n+1):
        m[tuple(data[i:i+n-1])][data[i+n-1]] += 1
    return m

def detok(ws):
    return re.sub(r"\s+([.,!?;:])", r"\1", ' '.join(ws))

# ── 1. 階數掃描：seed 必須是語料中真實出現過的 (n-1)-gram ────
# 找一個在各階都存在的起點：用語料裡第一次出現 "To be" 之後的完整上下文
idx = next(i for i in range(len(train)-10) if train[i:i+2] == ['To','be'])
long_seed = train[idx-6:idx+2]          # 8 個 token，足夠餵 n=8
print("seed:", detok(long_seed))

gens, backoff_rate = {}, {}
for n in (1, 2, 4, 8):
    m = build(n, train)
    random.seed(20260908)
    ctx_words = list(long_seed[-(n-1):]) if n > 1 else []
    out, fell = list(ctx_words), 0
    for _ in range(40):
        ctx = tuple(out[-(n-1):]) if n > 1 else ()
        d = m.get(ctx)
        if not d:
            d, fell = uni, fell+1
        out.append(random.choices(list(d), weights=list(d.values()))[0])
    gens[n] = detok(out)
    backoff_rate[n] = round(100.0*fell/40, 0)
    print("\n--- n=%d （%d%% 的步數被迫 backoff）---\n%s" % (n, backoff_rate[n], gens[n]))

# ── 2. sparsity ─────────────────────────────────────────────
spars = {}
for n in (1,2,3,4,5):
    seen = set(tuple(train[i:i+n]) for i in range(len(train)-n+1))
    tot = len(held)-n+1
    unseen = sum(1 for i in range(tot) if tuple(held[i:i+n]) not in seen)
    spars[n] = round(100.0*unseen/tot, 1)
print("\nsparsity:", spars)

# ── 3. bits-per-byte，stupid backoff（λ=0.4），教學上可讀 ────
bytes_held = len(' '.join(held).encode('utf-8'))
models = {n: build(n, train) for n in (1,2,3,4)}
ctxtot = {n: {k: sum(v.values()) for k,v in models[n].items()} for n in models}
Ntr = len(train)
def p_sb(n, hist, w):
    for k in range(n, 0, -1):
        if k == 1:
            return 0.4**(n-1) * (uni.get(w,0)+1.0)/(Ntr+V)
        ctx = tuple(hist[-(k-1):])
        d = models[k].get(ctx)
        if d and d.get(w):
            return 0.4**(n-k) * d[w]/ctxtot[k][ctx]
    return 1.0/(Ntr+V)
bpb = {}
for n in (1,2,3,4):
    nll = sum(-math.log2(p_sb(n, held[max(0,i-n+1):i], held[i])) for i in range(len(held)))
    bpb[n] = round(nll/bytes_held, 3)
print("bits-per-byte (stupid backoff):", bpb)

# ── 4. PPMI + SVD embedding（Levy & Goldberg：SGNS ≈ PMI 分解）──
low = [w.lower() for w in train if w.isalpha() or "'" in w]
vocab = [w for w,c in Counter(low).most_common(4000)]
vi = {w:i for i,w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
W = 4
for i,w in enumerate(low):
    if w not in vi: continue
    for j in range(max(0,i-W), min(len(low), i+W+1)):
        if j!=i and low[j] in vi: C[vi[w], vi[low[j]]] += 1
tot = C.sum(); pw = C.sum(1)/tot; pc = C.sum(0)/tot
with np.errstate(divide='ignore', invalid='ignore'):
    PPMI = np.log2((C/tot)/(np.outer(pw,pc)+1e-12)+1e-12)
PPMI[~np.isfinite(PPMI)] = 0; PPMI[PPMI<0] = 0
U,S,_ = np.linalg.svd(PPMI, full_matrices=False)
E = U[:, :200]*S[:200]
E = E/(np.linalg.norm(E, axis=1, keepdims=True)+1e-9)
def cos(a,b): return float(E[vi[a]] @ E[vi[b]]) if a in vi and b in vi else None
def nn(w,k=6):
    if w not in vi: return []
    s = E @ E[vi[w]]; o = np.argsort(-s)[1:k+1]
    return [(vocab[i], round(float(s[i]),3)) for i in o]
pairs = [('good','bad'),('love','hate'),('day','night'),('life','death'),
         ('good','evil'),('friend','enemy'),('king','queen'),('good','excellent')]
cosd = {f"{a}/{b}": (round(cos(a,b),3) if cos(a,b) is not None else None) for a,b in pairs}
print("\n反義詞/同義詞的 cosine:", json.dumps(cosd, ensure_ascii=False))
nns = {w: nn(w) for w in ('good','love','day','king')}
for w,l in nns.items(): print("  %-6s → %s" % (w, l))

json.dump({"generations":gens,"backoff_pct":backoff_rate,"sparsity":spars,"bpb":bpb,
           "cosines":cosd,"neighbours":nns,
           "corpus":{"tokens":len(toks),"types":len(set(toks)),
                     "train":len(train),"held":len(held),"emb_vocab":len(vocab)},
           "seed":detok(long_seed)},
          open(os.path.join(OUT,'w01-data.json'),'w'), indent=1, ensure_ascii=False)
print("\n→ slides/assets/w01/w01-data.json")

# ── 5. 精修：鄰近詞去掉功能詞，並補反義／同義對照 ────────────
STOP = set("i you he she it we they me my your his her our their to of and a an the in on at for with as but "
           "so do does did is are was were be been am not no nor if then than that this these those what which "
           "who whom will shall would should may might can could have has had thou thee thy thine ye o oh ah "
           "come go let make take will more most all any some such here there now when where how why yes".split())
def nn2(w,k=6):
    if w not in vi: return []
    s = E @ E[vi[w]]; o = np.argsort(-s)
    out=[]
    for i in o[1:]:
        if vocab[i] in STOP: continue
        out.append((vocab[i], round(float(s[i]),3)))
        if len(out)>=k: break
    return out
nns2 = {w: nn2(w) for w in ('king','day','love','sword','father')}
for w,l in nns2.items(): print("  %-7s → %s" % (w, l))

contrast = [("反義", "love / hate"), ("反義", "life / death"), ("反義", "good / bad"),
            ("反義", "friend / enemy"), ("同義", "good / excellent"), ("同義", "sword / blade"),
            ("相關", "king / queen")]
rows=[]
for kind, pair in contrast:
    a,b = [x.strip() for x in pair.split('/')]
    c = cos(a,b)
    rows.append({"kind":kind,"pair":pair,"cos":(round(c,3) if c is not None else None)})
    print("%-4s %-18s %s" % (kind, pair, rows[-1]["cos"]))

d = json.load(open(os.path.join(OUT,'w01-data.json')))
d["neighbours_filtered"] = nns2
d["contrast"] = rows
json.dump(d, open(os.path.join(OUT,'w01-data.json'),'w'), indent=1, ensure_ascii=False)
print("\n→ 已更新 w01-data.json")
