#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 1（投影片 p12「Live: the full table」）：同一組文字，九個 tokenizer。

只載 tokenizer、不載權重，純 CPU。課堂上按 Enter 一列一列填表：
    token 數 · tokens per character · bytes per token
最後一列是相對成本（繁中 token 數 ÷ 英文 token 數，Petrov et al. 的 tokenization premium）。
按 c 讀某個模型的 config.json 算詞表佔參數的比例（帳本第一列）；按 l 列出 o200k 詞表裡最長的含漢字 token。

子命令（用 nlp_llm/demos/setup.sh 建好的共用 .venv；通常經由 present.sh）：

    fetch     下載九個 tokenizer（只抓 tokenizer 檔，不抓權重）、tiktoken 的 BPE 檔，revision 記進 demos/versions.lock；
              並用 OpenCC 把繁中那段轉成簡體寫進 texts/zh_city_hans.txt
    inspect   課前確認：每個 tokenizer 的模型類型、normalizer、pre-tokenizer 的數字規則、byte fallback、
              「𠊎」是不是 [UNK]、NFD 會不會被改成 NFC、" A" 與 "A" 是不是不同 token、
              使用者文字裡的 <|im_start|> 會不會變成控制 token——HANDOVER C7.4 那八件事
    rehearse  課前彩排：整張表算一次，存成 runs/rehearsal/table.json（現場失敗時的退路）
    show      課堂用：按 Enter 逐列填表；c = 帳本；l = 最長的含漢字 token；q = 離開
    replay    現場失敗時：把彩排的表格重新顯示（畫面標明「非現場」）

三種分母（大綱 W2〈概念拆解路徑〉的 fertility 條目，本課固定不混用）：
    同一語言比 tokenizer   tokens / character
    跨 tokenizer 的壓縮率   bytes / token（b̄ = B / T）
    跨語言                  平行語料上的 token 數比（不用 bytes 當分母，那會把 byte premium 混進來）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import tomllib
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, GREEN, GREY, RED, RESET, append_jsonl,  # noqa: E402
                    banner, locked_revision, note, now, revision_key, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
TIKTOKEN_CACHE = HERE.parent / ".tiktoken-cache"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.txt",
                   "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "chat_template.jinja", "config.json", "generation_config.json"]
CJK = re.compile(r"[㐀-䶿一-鿿\U00020000-\U0003134f]")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE))


# ── 設定與文字 ───────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    d = cfg["demo"]
    for r in d["rows"]:
        if r not in cfg["texts"]:
            sys.exit(f"demo_config.toml：rows 裡的 {r!r} 不在 [texts]")
    for c in d["columns"] + d["highlight"] + [d["ledger_model"], d["longest_cjk_tokenizer"]]:
        if c not in cfg["tokenizers"]:
            sys.exit(f"demo_config.toml：{c!r} 不在 [tokenizers]")
    return cfg


def read_text(cfg: dict, tid: str) -> str:
    t = cfg["texts"][tid]
    if "text" in t:
        return t["text"].strip()
    return (HERE / t["file"]).read_text(encoding="utf-8").strip()


def parts_of(cfg: dict, tid: str) -> list[tuple[str, str]]:
    """一列可能有兩個子欄：台羅的 NFC／NFD、數字的兩個數。回傳 [(子標籤, 文字)]。"""
    t = cfg["texts"][tid]
    s = read_text(cfg, tid)
    v = t.get("variants")
    if v == "nfc_nfd":
        return [("NFC", unicodedata.normalize("NFC", s)), ("NFD", unicodedata.normalize("NFD", s))]
    if tid == "numbers":
        return [(x, x) for x in s.split()]
    return [("", s)]


# ── tokenizer 後端 ───────────────────────────────────────────────
class HFTok:
    kind = "hf"

    def __init__(self, key: str, spec: dict, offline: bool):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        rev = locked_revision(spec["repo"], spec.get("revision"))
        self.path = snapshot_download(spec["repo"], revision=rev, local_files_only=offline,
                                      allow_patterns=TOKENIZER_FILES)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.key, self.label, self.repo = key, spec["label"], spec["repo"]
        self.vocab_size = len(self.tok)
        self.unk_id = getattr(self.tok, "unk_token_id", None)
        self._tj = None
        p = Path(self.path) / "tokenizer.json"
        if p.exists():
            self._tj = json.loads(p.read_text(encoding="utf-8"))

    def encode(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def pieces(self, ids: list[int]) -> list[str]:
        return [self.tok.convert_ids_to_tokens(i) for i in ids]

    def special_id(self, s: str):
        i = self.tok.convert_tokens_to_ids(s)
        return None if i is None or i == self.unk_id else i

    def describe(self) -> dict:
        d = {"vocab_size": self.vocab_size, "unk_token": getattr(self.tok, "unk_token", None),
             "fast": bool(getattr(self.tok, "is_fast", False))}
        tj = self._tj
        if tj:
            m = tj.get("model") or {}
            d["model_type"] = m.get("type")
            d["byte_fallback"] = m.get("byte_fallback")
            d["unk_in_model"] = m.get("unk_token")
            d["normalizer"] = _summ(tj.get("normalizer"))
            d["pre_tokenizer"] = _summ(tj.get("pre_tokenizer"))
            d["decoder"] = (tj.get("decoder") or {}).get("type")
        else:
            d["model_type"] = "(沒有 tokenizer.json；可能是 tokenizer.model 的慢速 SentencePiece)"
        return d

    def longest_cjk(self, n: int) -> list[tuple[str, int]]:
        out = []
        for i in range(self.vocab_size):
            try:
                s = self.tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                continue
            k = len(CJK.findall(s))
            if k:
                out.append((s, i, k))
        out.sort(key=lambda x: (-x[2], -len(x[0])))
        return [(s, i) for s, i, _ in out[:n]]


def _summ(node) -> str:
    """把 tokenizer.json 的 normalizer／pre_tokenizer 節點壓成一行：類型，以及 Split 的 regex。"""
    if node is None:
        return "none"
    t = node.get("type")
    if t == "Sequence":
        return " → ".join(_summ(x) for x in node.get("normalizers") or node.get("pretokenizers") or [])
    if t == "Split":
        pat = node.get("pattern", {})
        return "Split(%s)" % (pat.get("Regex") or pat.get("String"))
    if t in ("NFC", "NFD", "NFKC", "NFKD", "Lowercase", "Strip", "StripAccents", "Prepend", "Replace",
             "ByteLevel", "Metaspace", "Whitespace", "WhitespaceSplit", "Punctuation", "Digits", "BertPreTokenizer",
             "BertNormalizer", "Precompiled"):
        extra = {k: v for k, v in node.items() if k != "type" and isinstance(v, (bool, str, int))}
        return t + ("(%s)" % ",".join(f"{k}={v}" for k, v in extra.items()) if extra else "")
    return str(t)


class TikTok:
    kind = "tiktoken"

    def __init__(self, key: str, spec: dict, offline: bool):
        import tiktoken
        self.enc = tiktoken.get_encoding(spec["encoding"])
        self.key, self.label, self.repo = key, spec["label"], "tiktoken:" + spec["encoding"]
        self.vocab_size = self.enc.n_vocab
        self.unk_id = None

    def encode(self, s: str) -> list[int]:
        return self.enc.encode(s, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def pieces(self, ids: list[int]) -> list[str]:
        return [self.enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]

    def special_id(self, s: str):
        return self.enc._special_tokens.get(s)

    def describe(self) -> dict:
        return {"vocab_size": self.vocab_size, "model_type": "BPE (byte-level, tiktoken)", "byte_fallback": "n/a",
                "normalizer": "none", "pre_tokenizer": "Split(%s)" % self.enc._pat_str,
                "special_tokens": sorted(self.enc._special_tokens)[:6]}

    def longest_cjk(self, n: int) -> list[tuple[str, int]]:
        out = []
        for i in range(self.vocab_size):
            try:
                s = self.enc.decode_single_token_bytes(i).decode("utf-8")
            except Exception:
                continue
            k = len(CJK.findall(s))
            if k:
                out.append((s, i, k))
        out.sort(key=lambda x: (-x[2], -len(x[0])))
        return [(s, i) for s, i, _ in out[:n]]


class BytesTok:
    kind = "bytes"

    def __init__(self, key: str, spec: dict, offline: bool):
        self.key, self.label, self.repo = key, spec["label"], "(bytes)"
        self.vocab_size, self.unk_id = 256, None

    def encode(self, s): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes(ids).decode("utf-8", errors="replace")
    def pieces(self, ids): return ["%02X" % i for i in ids]
    def special_id(self, s): return None
    def describe(self): return {"vocab_size": 256, "model_type": "none: every UTF-8 byte is a token", "byte_fallback": "n/a",
                                "normalizer": "none", "pre_tokenizer": "none"}
    def longest_cjk(self, n): return []


class FakeTok:
    """--fake：沒有網路、沒有 tokenizer 也能演練流程。數字是假的，畫面會標示。"""
    kind = "fake"

    def __init__(self, key: str, spec: dict, offline: bool):
        self.key, self.label, self.repo = key, spec["label"], "(fake)"
        self.vocab_size, self.unk_id = 50000, (100 if key == "bert_zh" else None)
        self.mode = {"byt5": "bytes", "bert_zh": "char", "qwen38": "pair", "breeze": "pair"}.get(key, "byte3")

    def encode(self, s):
        out = []
        for w in re.split(r"(\s+)", s):
            if not w:
                continue
            if not w.strip():
                out.extend(ord(c) for c in w); continue
            for ch in w:
                if ord(ch) < 128:
                    out.append(ord(ch))
                elif self.mode == "bytes":
                    out.extend(ch.encode())
                elif self.mode == "char" or self.mode == "pair":
                    if len(ch.encode()) == 4 and self.mode == "char":
                        out.append(self.unk_id)
                    else:
                        out.append(ord(ch))
                else:
                    out.extend(ch.encode())
        if self.mode == "pair":   # 假裝相鄰的兩個 CJK 併成一個雙字 token（合成 id，decode 得回來）
            merged, i = [], 0
            while i < len(out):
                if out[i] > 255 and i + 1 < len(out) and out[i + 1] > 255:
                    merged.append((out[i] << 21) | out[i + 1]); i += 2
                else:
                    merged.append(out[i]); i += 1
            out = merged
        return out

    def decode(self, ids):
        s = []
        buf = bytearray()
        for i in ids:
            if self.mode in ("bytes", "byte3") and 127 < i < 256:
                buf.append(i); continue
            if buf:
                s.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
            s.append("[UNK]" if i == self.unk_id else (chr(i >> 21) + chr(i & 0x1FFFFF) if i >= (1 << 21) else chr(i)))
        if buf:
            s.append(buf.decode("utf-8", errors="replace"))
        return "".join(s)

    def pieces(self, ids): return [("[UNK]" if i == self.unk_id else "%02X" % i if (self.mode in ("bytes", "byte3") and 127 < i < 256) else self.decode([i])) for i in ids]
    def special_id(self, s): return None
    def describe(self): return {"vocab_size": self.vocab_size, "model_type": "FAKE", "byte_fallback": "n/a", "normalizer": "none", "pre_tokenizer": "none"}
    def longest_cjk(self, n): return [("假的長 token %d" % i, i) for i in range(n)]


def make_tok(key: str, cfg: dict, args):
    spec = cfg["tokenizers"][key]
    cls = FakeTok if args.fake else {"hf": HFTok, "tiktoken": TikTok, "bytes": BytesTok}[spec["kind"]]
    t0 = time.time()
    t = cls(key, spec, offline=not args.online)
    note(f"  {spec['label']:<12} {t.repo}  {time.time() - t0:.1f} 秒")
    return t


def load_all(cfg, args, keys=None):
    toks = {}
    for key in (keys or cfg["demo"]["columns"]):
        try:
            toks[key] = make_tok(key, cfg, args)
        except Exception as e:
            spec = cfg["tokenizers"][key]
            print(f"  {RED}{spec['label']}：載入失敗（{type(e).__name__}: {e}）"
                  f"{' — gated，先在 Hugging Face 接受授權再 fetch' if spec.get('gated') else ''}{RESET}")
    return toks


# ── 計算 ─────────────────────────────────────────────────────────
def stats(tok, s: str) -> dict:
    ids = tok.encode(s)
    rt = tok.decode(ids)
    lossless = "yes" if rt == s else ("yes (±space)" if rt.strip() == s.strip() else "NO")
    unk = sum(1 for i in ids if tok.unk_id is not None and i == tok.unk_id)
    T, C, B = len(ids), len(s), len(s.encode("utf-8"))
    return {"tokens": T, "chars": C, "bytes": B, "tokens_per_char": round(T / C, 3),
            "bytes_per_token": round(B / T, 2) if T else None, "lossless": lossless, "unk": unk, "ids": ids}


def segments(tok, ids: list[int]) -> list[tuple[str, int]]:
    """把 token 還原成字串片段；不完整的 UTF-8（byte fallback）併到下一個 token，並記下併了幾個。"""
    segs, prev, start = [], "", 0
    for i in range(len(ids)):
        cur = tok.decode(ids[: i + 1])
        piece = cur[len(prev):] if cur.startswith(prev) else cur
        if "�" in piece and i + 1 < len(ids):
            continue
        segs.append((piece, i + 1 - start))
        prev, start = cur, i + 1
    return segs


def render_segments(segs) -> str:
    colors = ["\033[44m\033[97m", "\033[43m\033[30m"]
    out = []
    for j, (txt, n) in enumerate(segs):
        shown = txt.replace(" ", "␣").replace("\n", "⏎")
        mark = f"{GREY}{str(n).translate(SUB)}{RESET}" if n > 1 else ""
        out.append(f"{colors[j % 2]}{shown}{RESET}{mark}")
    return "".join(out)


def compute_table(cfg, toks) -> dict:
    """rows[tid][key] = [stats for each part]（ids 不存）。"""
    table = {}
    for tid in cfg["demo"]["rows"]:
        table[tid] = {}
        for key, tok in toks.items():
            cells = []
            for sub, s in parts_of(cfg, tid):
                st = stats(tok, s)
                st.pop("ids")
                st["part"] = sub
                cells.append(st)
            table[tid][key] = cells
    return table


def premium(cfg, table) -> dict:
    num, den = cfg["demo"]["premium_numerator"], cfg["demo"]["premium_denominator"]
    out = {}
    for key in table.get(num, {}):
        a = table[num][key][0]["tokens"]; b = table.get(den, {}).get(key, [{}])[0].get("tokens")
        out[key] = round(a / b, 2) if b else None
    return out


# ── 表格 ─────────────────────────────────────────────────────────
def dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s: str, w: int) -> str:
    return s + " " * max(0, w - dwidth(s))


W0, W = 26, 9


def header(cfg, keys):
    line = ljust("", W0) + "".join(f"{BOLD}{cfg['tokenizers'][k]['label']:>{W}}{RESET}" for k in keys)
    print("\n" + line)
    print("─" * (W0 + W * len(keys)))


def row_line(label, cells_by_key, keys, field="tokens", color="", fmt=None, part=0):
    fmt = fmt or (lambda v: f"{v:,}" if isinstance(v, int) else (f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v:.2f}" if isinstance(v, float) else str(v)))
    out = ljust(label, W0)
    for k in keys:
        cells = cells_by_key.get(k)
        if not cells or part >= len(cells):
            out += f"{'—':>{W}}"; continue
        out += f"{color}{fmt(cells[part].get(field)):>{W}}{RESET}"
    return out


def print_row(cfg, tid, cells_by_key, keys, full=True):
    """一列一行；台羅 NFC／NFD 與兩個數字這種有子欄的列，每個子欄各一行。"""
    lab = cfg["texts"][tid]["label"]
    nparts = max(len(v) for v in cells_by_key.values()) if cells_by_key else 1
    for p in range(nparts):
        sub = next((c[p]["part"] for c in cells_by_key.values() if len(c) > p and c[p].get("part")), "")
        print(row_line(lab if p == 0 else "", cells_by_key, keys, "tokens", BOLD, part=p)
              + (f"  {GREY}{sub}{RESET}" if sub else ""))
        if full:
            print(f"{GREY}{row_line('  tokens / char', cells_by_key, keys, 'tokens_per_char', part=p)}{RESET}")
            print(f"{GREY}{row_line('  bytes / token', cells_by_key, keys, 'bytes_per_token', part=p)}{RESET}")
    unk = {k: sum(c.get("unk", 0) for c in cells_by_key.get(k, [])) for k in keys}
    if any(unk.values()):
        print(f"{RED}{row_line('  [UNK]', {k: [{'unk': v}] for k, v in unk.items()}, keys, 'unk')}{RESET}")
    lossy = [k for k in keys if any(c.get("lossless") == "NO" for c in cells_by_key.get(k, []))]
    if lossy:
        print(f"{RED}{ljust('  round-trip NOT lossless:', W0)}{', '.join(cfg['tokenizers'][k]['label'] for k in lossy)}{RESET}")


def print_premium(cfg, table, keys):
    p = premium(cfg, table)
    num, den = cfg["demo"]["premium_numerator"], cfg["demo"]["premium_denominator"]
    print("─" * (W0 + W * len(keys)))
    lab = "relative cost (%s ÷ English)" % cfg["texts"][num]["label"].split()[0]
    line = row_line(lab, {k: [{"p": v}] for k, v in p.items()}, keys, "p",
                    fmt=lambda v: "—" if v is None else "%.2f×" % v)
    print(f"{AMBER}{line}{RESET}")
    note("  = 平行語料上的 token 數比（tokenization premium）；同一段意思，繁中比英文貴幾倍。不用 bytes 當分母。")


def show_blocks(cfg, toks, tid):
    for key in cfg["demo"]["highlight"]:
        tok = toks.get(key)
        if not tok:
            continue
        for sub, s in parts_of(cfg, tid):
            head = s[: cfg["demo"]["preview_chars"]]
            ids = tok.encode(head)
            print(f"  {BOLD}{tok.label}{RESET}{(' ' + sub) if sub else ''}  {GREY}（前 {len(head)} 字 {len(ids)} tokens；下標 = 幾個 token 才拼出這一段）{RESET}")
            print("    " + render_segments(segments(tok, ids)))


# ── 帳本：config.json → 2Vd / N ─────────────────────────────────
def config_of(cfg, key, args) -> dict | None:
    spec = cfg["tokenizers"][key]
    if args.fake:
        return {"vocab_size": 151936, "hidden_size": 5120, "num_hidden_layers": 64, "intermediate_size": 25600,
                "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": False, "_fake": True}
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(spec["repo"], revision=locked_revision(spec["repo"], spec.get("revision")),
                               local_files_only=not args.online, allow_patterns=["config.json"])) / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def text_config(c: dict) -> dict:
    """新模型的 config.json 常把語言模型的形狀放在 text_config／language_config 底下；找到含 hidden_size 的那層。
    鍵名也有別名（d_model／n_embd、num_layers／n_layer）。找不到就把頂層鍵印出來，讓人看得出該補哪個別名。"""
    def norm(x):
        x = dict(x)
        for k, alts in (("hidden_size", ("d_model", "n_embd")), ("num_hidden_layers", ("num_layers", "n_layer")),
                        ("num_attention_heads", ("n_head",)), ("intermediate_size", ("ffn_dim", "d_ff"))):
            for a in alts:
                if k not in x and a in x:
                    x[k] = x[a]
        return x
    top = norm(c)
    if "hidden_size" in top and "num_hidden_layers" in top:
        return top
    for k, v in c.items():
        if isinstance(v, dict):
            sub = norm(v)
            if "hidden_size" in sub and "num_hidden_layers" in sub:
                sub.setdefault("vocab_size", c.get("vocab_size"))
                sub.setdefault("tie_word_embeddings", c.get("tie_word_embeddings", False))
                sub["_from"] = k
                return sub
    raise KeyError("config.json 裡找不到 hidden_size／num_hidden_layers；頂層鍵：%s" % sorted(c))


def param_estimate(c: dict) -> dict:
    """由 config.json 估參數量（不含 norm 與 bias）。dense 才準；有 MoE 鍵就只算 attention 與 embedding 並標明。"""
    c = text_config(c)
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    nh = c.get("num_attention_heads", 1); nkv = c.get("num_key_value_heads", nh); hd = c.get("head_dim") or d // nh
    ff = c.get("intermediate_size", 0)
    tie = bool(c.get("tie_word_embeddings", False))
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    moe = any(k in c for k in ("num_experts", "num_local_experts", "moe_intermediate_size", "num_experts_per_tok"))
    mlp = 0 if moe else 3 * d * ff
    n_nv = L * (attn + mlp)
    emb = V * d * (1 if tie else 2)
    return {"V": V, "d": d, "n_layer": L, "tie": tie, "emb_params": emb, "non_emb_params": n_nv,
            "N_est": n_nv + emb, "moe": moe, "share_est": emb / (n_nv + emb) if not moe else None}


def print_ledger(cfg, key, args):
    spec = cfg["tokenizers"][key]
    c = config_of(cfg, key, args)
    if not c:
        print(f"{RED}讀不到 {spec['repo']} 的 config.json{RESET}"); return None
    e = param_estimate(c)
    src = text_config(c).get("_from")
    banner(f"帳本第一列 · {spec['label']} 的 config.json{'（FAKE）' if c.get('_fake') else ''}{('（形狀在 ' + src + ' 底下）') if src else ''}", AMBER)
    print(f"  vocab_size V = {e['V']:,}    hidden_size d = {e['d']:,}    layers = {e['n_layer']}    tie_word_embeddings = {e['tie']}")
    print(f"  詞表參數 {'Vd' if e['tie'] else '2Vd'} = {e['emb_params']:,}  ({e['emb_params']/1e9:.2f} B)")
    if e["moe"]:
        print(f"  {AMBER}這是 MoE 的 config，MLP 那部分沒算；N 用名稱上的 {spec.get('params_nominal', 0)/1e9:.0f}B{RESET}")
        N = spec.get("params_nominal")
    else:
        print(f"  非詞表參數（attention + MLP，不含 norm）≈ {e['non_emb_params']:,}  ({e['non_emb_params']/1e9:.2f} B)")
        N = e["N_est"]
        print(f"  N ≈ {N:,}  ({N/1e9:.2f} B)" + (f"   名稱上是 {spec['params_nominal']/1e9:.0f}B" if spec.get("params_nominal") else ""))
    if N:
        print(f"\n  {BOLD}{'Vd' if e['tie'] else '2Vd'} / N = {e['emb_params']/N:.3f}  →  {100*e['emb_params']/N:.1f}% 的參數是詞表{RESET}")
        note("  對照投影片 p39 那張圖：125M 的模型配 128K 詞表是 70%，175B 是 1.8%；這個模型落在中間。")
    tc = text_config(c)
    return {"config": {k: tc.get(k) for k in ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
                                               "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings", "_from")},
            "estimate": e, "N_used": N}


def print_longest(cfg, toks, args):
    key = cfg["demo"]["longest_cjk_tokenizer"]
    tok = toks.get(key)
    if tok is None:
        print(f"{RED}{key} 沒載入{RESET}"); return None
    banner(f"{tok.label} 詞表（{tok.vocab_size:,} 個）裡最長的含漢字 token", AMBER)
    note("  掃整個詞表要幾秒……")
    rows = tok.longest_cjk(cfg["demo"]["longest_cjk_n"])
    for s, i in rows:
        print(f"  id {i:>7}  {len(CJK.findall(s)):>2} 個漢字  {s!r}")
    note("  這些 token 告訴你 tokenizer 的訓練語料長什麼樣子：它不是從字典學的，是從網頁學的。")
    return [{"id": i, "token": s} for s, i in rows]


# ── inspect：課前要確認的八件事 ─────────────────────────────────
def cmd_inspect(args, cfg):
    keys = list(cfg["tokenizers"])
    toks = load_all(cfg, args, keys)
    zh = read_text(cfg, "zh_city")
    tailo = parts_of(cfg, "tailo")
    hakka = read_text(cfg, "hakka")
    facts = {}
    for key, tok in toks.items():
        spec = cfg["tokenizers"][key]
        banner(f"{tok.label}  ←  {tok.repo}", AMBER)
        d = tok.describe()
        for k in ("model_type", "vocab_size", "byte_fallback", "unk_token", "normalizer", "pre_tokenizer", "decoder", "special_tokens"):
            if k in d:
                print(f"  {k:<14} {d[k]}")
        f = {"describe": d}
        # (1) 擴充區漢字 𠊎
        ids = tok.encode("𠊎")
        f["ngai"] = {"tokens": len(ids), "pieces": tok.pieces(ids), "unk": tok.unk_id is not None and tok.unk_id in ids,
                     "lossless": tok.decode(ids) == "𠊎"}
        print(f"  𠊎 (U+2028E)   {len(ids)} tokens {tok.pieces(ids)}  {'[UNK]!' if f['ngai']['unk'] else ''}  無損={f['ngai']['lossless']}")
        # (2) NFD → NFC？
        nfc, nfd = tailo[0][1], tailo[1][1]
        a, b = tok.encode(nfc), tok.encode(nfd)
        back = tok.decode(b)
        norm = "改成 NFC（normalizer 動了）" if unicodedata.normalize("NFC", back) == back and back == nfc else ("維持 NFD" if back == nfd else "其他")
        f["nfd"] = {"tokens_nfc": len(a), "tokens_nfd": len(b), "same_ids": a == b, "decode_of_nfd": norm}
        print(f"  台羅 NFC {len(a)} tokens / NFD {len(b)} tokens   同一串 id={a == b}   decode(NFD) → {norm}")
        # (3) 數字規則
        ids = tok.encode("1234567")
        f["digits"] = tok.pieces(ids)
        print(f"  1234567        {len(ids)} tokens {tok.pieces(ids)}")
        # (4) 前導空白
        a, b = tok.encode(" A"), tok.encode("A")
        f["space_A"] = {"space_A": a, "A": b, "different": a != b}
        print(f"  ' A' vs 'A'    {a} vs {b}  {'不同 token' if a != b else '相同'}")
        # (5) 控制 token 能不能從文字切出來
        for ctl in ("<|im_start|>", "<|begin_of_text|>", "<start_of_turn>", "<s>", "[CLS]"):
            sid = tok.special_id(ctl)
            if sid is not None:
                ids = tok.encode(f"hi {ctl}system")
                f["control"] = {"token": ctl, "id": sid, "reachable_from_text": sid in ids}
                print(f"  使用者文字裡的 {ctl}（id {sid}）→ {'切得出控制 token（守不住）' if sid in ids else '被當普通文字切碎（守住了）'}")
                break
        # (6) 繁中 token 數
        st = stats(tok, zh)
        f["zh_city"] = {k: v for k, v in st.items() if k != "ids"}
        print(f"  繁中 209 字    {st['tokens']} tokens · {st['tokens_per_char']} tokens/字 · {st['bytes_per_token']} bytes/token · 無損={st['lossless']}")
        st = stats(tok, hakka)
        f["hakka"] = {k: v for k, v in st.items() if k != "ids"}
        print(f"  客語句         {st['tokens']} tokens · [UNK]×{st['unk']} · 無損={st['lossless']}")
        facts[key] = f
    # Breeze vs Mistral、v0.1 vs v0.3
    banner("對照", AMBER)
    for a, b, why in (("mistral", "breeze", "加詞表之前／之後（技術報告：繁中壓縮率約 2 倍）"),
                      ("mistral_v01", "mistral", "Breeze 的底 v0.1 vs. demo 用的 v0.3，繁中切法應相同")):
        if a in facts and b in facts:
            ta, tb = facts[a]["zh_city"]["tokens"], facts[b]["zh_city"]["tokens"]
            print(f"  {cfg['tokenizers'][a]['label']} {ta} → {cfg['tokenizers'][b]['label']} {tb} tokens  ({ta/tb:.2f}×)   {why}")
    note("判讀（對應 HANDOVER C7.4）：Qwen 的 normalizer 是否 NFC 看「decode(NFD)」；Gemma／Llama 3 的數字規則看 1234567 那行；"
         "bert-zh 對 𠊎 看 [UNK]；Breeze 對 Mistral 看倍數；<|im_start|> 那行是 p21 講稿要不要說「Qwen3.8 守住了」的依據。"
         "確認之後把講稿裡的「待確認」句改掉。")
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    out = REHEARSAL / ("inspect%s.json" % ("-FAKE" if args.fake else ""))
    out.write_text(json.dumps({"time": now(), "fake": args.fake, "facts": facts, "versions": versions()},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    note(f"→ {out.relative_to(HERE)}")


# ── fetch ────────────────────────────────────────────────────────
def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for key, spec in cfg["tokenizers"].items():
        if args.only and key != args.only:
            continue
        if spec["kind"] == "hf":
            try:
                info = api.model_info(spec["repo"], revision=spec.get("revision") or None)
                banner(f"下載 {key}: {spec['repo']} @ {info.sha}（只抓 tokenizer 檔）")
                snapshot_download(spec["repo"], revision=info.sha, allow_patterns=TOKENIZER_FILES)
                write_lock({revision_key(spec["repo"]): info.sha})
                print(f"{GREEN}完成，revision 已寫進 demos/versions.lock{RESET}")
            except Exception as e:
                msg = f"{key}：{type(e).__name__}: {e}"
                if spec.get("optional"):
                    print(f"{AMBER}{msg}（optional，略過）{RESET}")
                else:
                    print(f"{RED}{msg}{'  ← gated：先到 Hugging Face 接受授權並 huggingface-cli login' if spec.get('gated') else ''}{RESET}")
        elif spec["kind"] == "tiktoken":
            import tiktoken
            banner(f"下載 tiktoken {spec['encoding']} → {TIKTOKEN_CACHE}")
            TIKTOKEN_CACHE.mkdir(parents=True, exist_ok=True)
            enc = tiktoken.get_encoding(spec["encoding"])
            print(f"{GREEN}完成：{enc.n_vocab:,} 個 token{RESET}")
    # 簡體：OpenCC
    for tid, t in cfg["texts"].items():
        if t.get("derive") == "opencc_t2s":
            try:
                import opencc
            except ImportError:
                print(f"{AMBER}沒有 opencc（uv pip install --python ../.venv/bin/python opencc-python-reimplemented）；"
                      f"{t['file']} 維持現有的手動版{RESET}")
                continue
            src = read_text(cfg, t["derive_from"])
            new = opencc.OpenCC("t2s").convert(src)
            p = HERE / t["file"]
            old = p.read_text(encoding="utf-8").strip() if p.exists() else ""
            if old != new:
                diff = [(i, a, b) for i, (a, b) in enumerate(zip(old, new)) if a != b]
                print(f"{AMBER}OpenCC 的結果與 {t['file']} 現有內容不同（{len(diff)} 處）：{diff[:10]}{RESET}")
            p.write_text(new + "\n", encoding="utf-8")
            print(f"{GREEN}{t['file']} ← OpenCC t2s（{len(new)} 字）{RESET}")


# ── rehearse / show / replay ─────────────────────────────────────
def build_record(cfg, toks, args, ledger=None, longest=None):
    table = compute_table(cfg, toks)
    return {"time": now(), "fake": args.fake, "columns": list(toks), "rows": cfg["demo"]["rows"],
            "labels": {k: cfg["tokenizers"][k]["label"] for k in toks},
            "texts": {tid: {"label": cfg["texts"][tid]["label"], "parts": parts_of(cfg, tid)} for tid in cfg["demo"]["rows"]},
            "table": table, "premium": premium(cfg, table), "ledger": ledger, "longest_cjk": longest,
            "tokenizers": {k: {"repo": t.repo, "vocab_size": t.vocab_size} for k, t in toks.items()},
            "versions": versions()}


def cmd_rehearse(args, cfg):
    banner(cfg["demo"]["title"] + " · 彩排", AMBER)
    toks = load_all(cfg, args)
    keys = list(toks)
    table = compute_table(cfg, toks)
    header(cfg, keys)
    for tid in cfg["demo"]["rows"]:
        print_row(cfg, tid, table[tid], keys)
    print_premium(cfg, table, keys)
    ledger = longest = None
    try:
        ledger = print_ledger(cfg, cfg["demo"]["ledger_model"], args)
    except Exception as e:
        print(f"{RED}帳本那一步失敗（{type(e).__name__}: {e}）；表格照存，c 鍵先別按{RESET}")
    try:
        longest = print_longest(cfg, toks, args)
    except Exception as e:
        print(f"{RED}最長漢字 token 那一步失敗（{type(e).__name__}: {e}）{RESET}")
    rec = build_record(cfg, toks, args, ledger, longest)
    REHEARSAL.mkdir(parents=True, exist_ok=True)
    out = REHEARSAL / ("table%s.json" % ("-FAKE" if args.fake else ""))
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    note(f"→ {out.relative_to(HERE)}")
    missing = [k for k in cfg["demo"]["columns"] if k not in toks]
    if missing:
        print(f"{RED}沒載入的欄：{missing}——fetch 之後再彩排一次{RESET}")


def wait(msg: str) -> str:
    try:
        return input(f"\n{GREY}[{msg} · c = 帳本 · l = 最長漢字 token · q = 離開]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def cmd_show(args, cfg):
    os.system("clear")
    banner(cfg["demo"]["title"] + ("（FAKE）" if args.fake else ""), AMBER)
    toks = load_all(cfg, args)
    keys = list(toks)
    table = {}
    ledger = longest = None
    rows = cfg["demo"]["rows"]
    i = 0
    header(cfg, keys)
    while True:
        nxt = rows[i] if i < len(rows) else None
        key = wait(f"Enter = {cfg['texts'][nxt]['label']}" if nxt else "Enter = 相對成本那一列" if i == len(rows) else "Enter = 結束")
        if key == "q":
            break
        if key == "c":
            ledger = print_ledger(cfg, cfg["demo"]["ledger_model"], args); continue
        if key == "l":
            longest = print_longest(cfg, toks, args); continue
        if nxt:
            table[nxt] = {k: [dict(st, ids=None) for st in
                              [stats(t, s) for _, s in parts_of(cfg, nxt)]] for k, t in toks.items()}
            for k in table[nxt]:
                for c in table[nxt][k]:
                    c.pop("ids", None)
            print_row(cfg, nxt, table[nxt], keys)
            show_blocks(cfg, toks, nxt)
            i += 1
        elif i == len(rows):
            print_premium(cfg, table, keys); i += 1
        else:
            break
    rec = build_record(cfg, toks, args, ledger, longest) if table else None
    if rec:
        append_jsonl(RUNS / "live" / f"{dt.datetime.now():%Y%m%d}.jsonl", rec)


def cmd_replay(args, cfg):
    path = REHEARSAL / ("table%s.json" % ("-FAKE" if args.fake else ""))
    if not path.exists():
        print(f"{RED}沒有彩排紀錄 {path.relative_to(HERE)}；課前先跑 rehearse。{RESET}"); return
    rec = json.loads(path.read_text(encoding="utf-8"))
    banner(f"課前錄下的結果（{rec['time']}，非現場）", RED)
    keys = rec["columns"]
    header(cfg, keys)
    for tid in rec["rows"]:
        print_row(cfg, tid, rec["table"][tid], keys)
        if wait("Enter = 下一列") == "q":
            return
    print_premium(cfg, rec["table"], keys)
    if rec.get("ledger"):
        e = rec["ledger"]["estimate"]
        print(f"\n  帳本：V = {e['V']:,}, d = {e['d']:,}, {'Vd' if e['tie'] else '2Vd'} = {e['emb_params']:,}, N ≈ {rec['ledger']['N_used']:,} → {100*e['emb_params']/rec['ledger']['N_used']:.1f}%")
    if rec.get("longest_cjk"):
        print("\n  最長的含漢字 token：")
        for r in rec["longest_cjk"]:
            print(f"    id {r['id']:>7}  {r['token']!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載 tokenizer，只演練流程（數字是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--only")
    for c in ("inspect", "rehearse", "show", "replay"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        args.online = True
    cfg = load_config()
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "rehearse": cmd_rehearse,
     "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
