#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W1 demo 2（投影片 p28「Live: the same paragraph, two tokenizers」）。

同一段繁體中文，交給兩個 tokenizer 很不一樣的模型，現場算出
token 數、perplexity、bits-per-byte，讓學生看到：
PPL 的分母是 token 數，跟著 tokenizer 走；bits-per-byte 的分母是 bytes，是文字本身的性質。

子命令（用 nlp_llm/demos/setup.sh 建好的共用 .venv；通常經由 present.sh）：

    fetch     下載 demo_config.toml 裡所有候選模型，revision 記進 demos/versions.lock
    inspect   課前挑模型：只載 tokenizer，比較每個候選模型的 token 數與是否無損（很快）
    rehearse  課前彩排：完整算一次，存成 runs/rehearsal/（現場失敗時的退路）
    show      課堂用：載入兩個模型，按 Enter 逐步填表
    replay    現場失敗時：把彩排的表格重新顯示（畫面標明「非現場」）

計算方式（與 lm-evaluation-harness 的 loglikelihood 相同的慣例）：
在文字前面放一個起始 token（有 BOS 用 BOS，沒有就用 <|endoftext|> 或 EOS），
一次 forward，把「文字的每一個 token」的負對數機率加起來得到總 NLL（nats）。
    PPL = exp(NLL / T)          T = 文字的 token 數
    BPB = NLL / ln 2 / B        B = 文字的 UTF-8 bytes
兩者的關係：BPB = (T / B) · log2(PPL)。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, GREEN, GREY, RED, RESET, append_jsonl,  # noqa: E402
                    banner, locked_revision, note, now, reset_peak_memory,
                    revision_key, versions, write_lock)

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "special_tokens_map.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "config.json", "generation_config.json"]
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf"]   # 不影響載入、離線核對時忽略
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def load_config() -> dict:
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for m in cfg["demo"]["pair"]:
        if m not in cfg["models"]:
            sys.exit(f"demo_config.toml：pair 裡的 {m!r} 不在 [models]")
    if cfg["demo"]["text"] not in cfg["texts"]:
        sys.exit(f"demo_config.toml：text = {cfg['demo']['text']!r} 不在 [texts]")
    return cfg


def read_text(cfg: dict, tid: str) -> str:
    return (HERE / cfg["texts"][tid]["file"]).read_text(encoding="utf-8").strip()


# ── 後端 ─────────────────────────────────────────────────────────
class MLXScorer:
    def __init__(self, repo: str, revision: str | None, offline: bool, weights: bool = True):
        from huggingface_hub import snapshot_download
        t0 = time.time()
        # 離線時 huggingface_hub ≥1.3 會拿快取的檔案清單核對 snapshot 是否完整。
        # 用 mlx_lm.load 抓下來的模型沒有 README.md / .gitattributes（它只抓權重與 tokenizer），
        # 不排除掉就會被判成 IncompleteSnapshotError（2026-09-20 踩到）。
        kw = ({"ignore_patterns": NON_MODEL_FILES} if weights
              else {"allow_patterns": TOKENIZER_FILES})
        self.path = snapshot_download(repo, revision=revision, local_files_only=offline, **kw)
        self.repo = repo
        if weights:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.path)
        else:
            from transformers import AutoTokenizer
            self.model, self.tokenizer = None, AutoTokenizer.from_pretrained(self.path)
        self.load_seconds = time.time() - t0

    # tokenizer ------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def prefix_token(self) -> tuple[int, str]:
        tok = self.tokenizer
        if getattr(tok, "bos_token_id", None) is not None:
            return tok.bos_token_id, f"BOS {tok.bos_token!r}"
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
        if "<|endoftext|>" in vocab:
            return vocab["<|endoftext|>"], "'<|endoftext|>'"
        return tok.eos_token_id, f"EOS {tok.eos_token!r}"

    # model ------------------------------------------------------------
    def token_nlls(self, ids: list[int], use_cache: bool = False):
        """ids[0] 是起始 token；回傳 ids[1:] 每個 token 的負對數機率（nats）。
        use_cache=True 走 mlx_lm.generate 用的同一條路（帶 KV cache 的 prefill），用來對照。"""
        import mlx.core as mx
        kw = {}
        if use_cache:
            from mlx_lm.models.cache import make_prompt_cache
            kw["cache"] = make_prompt_cache(self.model)
        out = self.model(mx.array(ids)[None], **kw)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits[0, :-1].astype(mx.float32)
        targets = mx.array(ids[1:])
        lse = mx.logsumexp(logits, axis=-1)
        tgt = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
        return (lse - tgt).tolist()

    def token_nlls_incremental(self, ids: list[int]):
        """逐 token 帶 cache 餵，每步只讀最後一個位置的 logits——與 mlx_lm.generate 完全同一條路。
        2026-09-20：Gemma 4 的多 token forward 在非最後位置的 logits 疑似有問題（生成正常、打分崩掉），
        用這條路對照；若一致就是模型／量化的問題，若差很多就是 forward 的問題。"""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        out = []
        for i in range(len(ids) - 1):
            logits = self.model(mx.array([ids[i]])[None], cache=cache)
            logits = (logits[0] if isinstance(logits, (tuple, list)) else logits)[0, -1].astype(mx.float32)
            lse = mx.logsumexp(logits)
            out.append(float((lse - logits[ids[i + 1]]).item()))
        return out

    def nll(self, ids: list[int]) -> float:
        """ids[0] 是起始 token；回傳 ids[1:] 的總負對數機率（nats）。"""
        return float(sum(self.token_nlls(ids)))


class FakeScorer:
    """--fake：沒有 Apple silicon 也能演練流程。數字是假的，畫面會標示。"""

    def __init__(self, repo: str, revision, offline, weights=True):
        self.repo, self.path, self.load_seconds = repo, "(fake)", 0.0
        self.byte_level = "mistral" in repo.lower()

    def encode(self, text):
        if self.byte_level:
            return [b for ch in text for b in (ch.encode() if ord(ch) > 127 else [ord(ch)])]
        return [ord(c) for c in text]

    def decode(self, ids):
        if self.byte_level:
            return bytes(ids).decode("utf-8", errors="replace")
        return "".join(chr(i) for i in ids)

    def prefix_token(self):
        return 0, "FAKE"

    def nll(self, ids):
        per_char = 2.1 if not self.byte_level else 3.4
        n_chars = len(self.decode(ids[1:]))
        return per_char * n_chars


def make_scorer(key: str, cfg: dict, args, weights: bool = True):
    m = cfg["models"][key]
    rev = locked_revision(m["repo"], m.get("revision"))
    cls = FakeScorer if args.fake else MLXScorer
    note(f"載入 {m['label']}{'（只載 tokenizer）' if not weights else ''}  ←  {m['repo']}"
         + (f"@{rev[:10]}" if rev else ""))
    s = cls(m["repo"], rev, offline=not args.online, weights=weights)
    s.key, s.label = key, m["label"]
    note(f"  完成：{s.load_seconds:.1f} 秒")
    return s


# ── 計算 ─────────────────────────────────────────────────────────
def segments(sc, ids: list[int]) -> list[tuple[str, int]]:
    """把 token 還原成字串片段；不完整的 UTF-8（byte fallback）併到下一個 token，並記下併了幾個。"""
    segs, prev, start = [], "", 0
    for i in range(len(ids)):
        cur = sc.decode(ids[: i + 1])
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


def tokenize_stats(sc, text: str) -> dict:
    ids = sc.encode(text)
    rt = sc.decode(ids)
    lossless = "yes" if rt == text else ("yes (±space)" if rt.strip() == text.strip() else "NO")
    return {"tokens": len(ids), "chars": len(text), "bytes": len(text.encode("utf-8")),
            "tokens_per_char": round(len(ids) / len(text), 3), "lossless": lossless, "ids": ids}


def score(sc, text: str) -> dict:
    st = tokenize_stats(sc, text)
    pid, pdesc = sc.prefix_token()
    reset_peak_memory()
    t0 = time.time()
    nll = sc.nll([pid] + st["ids"])
    secs = time.time() - t0
    T, B = st["tokens"], st["bytes"]
    ppl = math.exp(nll / T)
    bpb = nll / math.log(2) / B
    st.update({"prefix_token": pdesc, "nll_nats": round(nll, 3),
               "ppl": round(ppl, 3), "bpb": round(bpb, 4),
               "bits_per_char": round(nll / math.log(2) / st["chars"], 4),
               "identity_check": round((T / B) * math.log2(ppl), 4),
               "forward_seconds": round(secs, 2)})
    try:
        from common import peak_memory_gb
        pm = None if isinstance(sc, FakeScorer) else peak_memory_gb()
        st["peak_memory_gb"] = round(pm, 2) if pm else None
    except Exception:
        st["peak_memory_gb"] = None
    return st


# ── 表格 ─────────────────────────────────────────────────────────
ROWS = [("chars", "字元數"), ("bytes", "UTF-8 bytes"), ("tokens", "tokens"),
        ("tokens_per_char", "tokens / 字"), ("ppl", "perplexity"), ("bpb", "bits-per-byte"),
        ("lossless", "round-trip 無損")]


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.3f}" if v < 100 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def dwidth(s: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


def print_table(labels: list[str], results: list[dict], show: set[str], highlight: str | None = None):
    w0, w = 18, 22
    head = f"{'':<{w0}}" + "".join(f"{BOLD}{l:>{w}}{RESET}" for l in labels) + f"{GREY}{'A / B':>{12}}{RESET}"
    print("\n" + head)
    print("─" * (w0 + w * len(labels) + 12))
    for key, name in ROWS:
        if key not in show:
            cells = "".join(f"{'·':>{w}}" for _ in labels)
            print(f"{GREY}{ljust(name, w0)}{RESET}{GREY}{cells}{RESET}")
            continue
        vals = [r.get(key) for r in results]
        color = AMBER if key == highlight else ""
        cells = "".join(f"{color}{fmt(v):>{w}}{RESET}" for v in vals)
        ratio = ""
        if all(isinstance(v, (int, float)) and v for v in vals) and len(vals) == 2:
            ratio = f"{vals[0] / vals[1]:.2f}×"
        bold = BOLD if key in ("ppl", "bpb") else ""
        print(f"{bold}{ljust(name, w0)}{RESET}{cells}{GREY}{ratio:>12}{RESET}")


def record(cfg, tid, text, scorers, results, fake) -> dict:
    return {"time": now(), "text_id": tid, "text": text,
            "models": [{"key": s.key, "label": s.label, "repo": s.repo} for s in scorers],
            "results": [{k: v for k, v in r.items() if k != "ids"} for r in results],
            "fake": fake, "versions": versions()}


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_diag(args, cfg):
    """2026-09-20：Gemma 4 的 PPL 算出 3,435（每 token 8 nats，接近均勻），demo 1 卻能正常生成，
    所以懷疑是打分路徑的問題。這裡把可能的原因一次量出來：
      (a) 起始 token 是否重複（encode 有沒有偷加 BOS；Gemma 對 double BOS 極敏感）
      (b) 不帶 cache 與帶 cache（generate 的 prefill 路徑）兩種 forward 的 NLL 是否一致
      (c) 一句簡單英文的 NLL 當 sanity check
      (d) 前幾個 token 各自的 nats，看是整段均勻地差、還是某幾個 token 爆掉"""
    import math
    tid = args.text or cfg["demo"]["text"]
    text = read_text(cfg, tid)
    sanity = "The capital of France is Paris. It is known for the Eiffel Tower."
    for key in (args.pair or cfg["demo"]["pair"]):
        sc = make_scorer(key, cfg, args)
        pid, pdesc = sc.prefix_token()
        ids = sc.encode(text)
        banner(f"{sc.label}", AMBER)
        print(f"  起始 token {pdesc} = id {pid}")
        print(f"  encode 前 6 個 id：{ids[:6]}   → 第一個 id 等於起始 token？ {'是（會變成 double BOS）' if ids and ids[0] == pid else '否'}")
        print(f"  前 6 個 token 還原：{[sc.decode([i]) for i in ids[:6]]}")
        for label, use_cache in (("不帶 cache（rehearse 現在的做法）", False), ("帶 cache（generate 的 prefill 路徑）", True)):
            try:
                nl = sc.token_nlls([pid] + ids, use_cache=use_cache)
                tot = sum(nl)
                print(f"  {label}: NLL {tot:.1f} nats · PPL {math.exp(tot/len(ids)):.1f} · 前 8 個 token 的 nats "
                      + " ".join(f"{v:.1f}" for v in nl[:8]))
            except Exception as e:
                print(f"  {label}: 失敗（{type(e).__name__}: {e}）")
        t0 = time.time()
        nli = sc.token_nlls_incremental([pid] + ids)
        toti = sum(nli)
        print(f"  逐 token 帶 cache（generate 逐步解碼的路徑，{time.time()-t0:.1f} 秒）: NLL {toti:.1f} nats · PPL {math.exp(toti/len(ids)):.1f} · 前 8 個 token 的 nats "
              + " ".join(f"{v:.1f}" for v in nli[:8]))
        # teacher-forced top-1：多 token forward 每個位置的 argmax 有幾成等於下一個 token
        import mlx.core as mx
        out = sc.model(mx.array([pid] + ids)[None]); out = out[0] if isinstance(out, (tuple, list)) else out
        top1 = out[0, :-1].argmax(axis=-1).tolist()
        acc = sum(int(a == b) for a, b in zip(top1, ids)) / len(ids)
        print(f"  多 token forward 的 top-1 命中率：{100*acc:.0f}%   前 8 個位置的 argmax 還原：{[sc.decode([t]) for t in top1[:8]]}")
        sid = sc.encode(sanity)
        nl = sc.token_nlls([pid] + sid)
        nli = sc.token_nlls_incremental([pid] + sid)
        print(f"  sanity（英文一句，{len(sid)} tokens）: 多 token forward PPL {math.exp(sum(nl)/len(sid)):.1f} · 逐 token PPL {math.exp(sum(nli)/len(sid)):.1f}")
        # 不加起始 token 直接算（看 BOS 到底有沒有幫助）
        if len(ids) > 1:
            nl0 = sc.token_nlls(ids)
            print(f"  不加起始 token、從第 2 個 token 起算: PPL {math.exp(sum(nl0)/(len(ids)-1)):.1f}")
        # (e) 同一段文字包進對話模板（當 model 那一輪的回答），只對文字那段的 token 打分
        try:
            tok = sc.tokenizer
            msgs = [{"role": "user", "content": "請寫一段短文。"}, {"role": "assistant", "content": text}]
            full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prefix = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
            fid = list(tok.encode(full, add_special_tokens=False))
            pidl = list(tok.encode(prefix, add_special_tokens=False))
            n_text = len(fid) - len(pidl)
            nlc = sc.token_nlls(fid)[len(pidl) - 1:]      # 從模板前綴之後的第一個 token 起算
            nlc = nlc[:n_text]
            print(f"  (e) 包進對話模板當 model 回答（前綴 {len(pidl)} tokens，文字段 {n_text} tokens）: "
                  f"PPL {math.exp(sum(nlc)/max(1,len(nlc))):.1f} · BPB {sum(nlc)/math.log(2)/len(text.encode('utf-8')):.3f}")
        except Exception as e:
            print(f"  (e) 對話模板打分失敗（{type(e).__name__}: {e}）")
        # (f) 自我一致：讓模型自己 greedy 生成 40 個 token，再 teacher-force 打分；forward 沒問題的話 NLL ≈ 0
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(sc.model)
            gen = []
            cur = [pid] + ids[:8]
            logits = sc.model(mx.array(cur)[None], cache=cache)
            for _ in range(40):
                nxt = int((logits[0] if isinstance(logits, (tuple, list)) else logits)[0, -1].argmax().item())
                gen.append(nxt)
                logits = sc.model(mx.array([nxt])[None], cache=cache)
            nlg = sc.token_nlls(cur + gen)[len(cur) - 1:]
            print(f"  (f) 自己 greedy 生成 40 token 再 teacher-force：NLL {sum(nlg):.2f} nats（應接近 0）· 生成內容：{sc.decode(gen)[:60]!r}")
        except Exception as e:
            print(f"  (f) 自我一致測試失敗（{type(e).__name__}: {e}）")
    note("判讀：(a) 若「第一個 id 等於起始 token」是「是」→ encode 偷加了 BOS，要拿掉一個；"
         "(b) 若帶／不帶 cache 差很多 → 不帶 cache 的 forward 有問題，改走帶 cache；"
         "(c) 若多 token forward 的 PPL 崩、逐 token 的正常 → 多 token forward 在非最後位置的 logits 錯了"
         "（生成只看最後一個位置所以看不出來），打分改走逐 token；"
         "(d) 若逐 token 也崩、sanity 也崩、但 (f) 接近 0 → forward 沒錯，是這個 IT 模型對原始文字 out-of-distribution；"
         "看 (e) 包進模板後 PPL 是否回到正常。(f) 若不接近 0 → forward 或量化真的有問題。")


def cmd_fetch(args, cfg):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for key, m in cfg["models"].items():
        if args.only and key != args.only:
            continue
        info = api.model_info(m["repo"], revision=m.get("revision") or None)
        banner(f"下載 {key}: {m['repo']} @ {info.sha}")
        snapshot_download(m["repo"], revision=info.sha)
        write_lock({revision_key(m["repo"]): info.sha})
        print(f"{GREEN}完成，revision 已寫進 demos/versions.lock{RESET}")


def cmd_inspect(args, cfg):
    tids = [args.text] if args.text else list(cfg["texts"])
    rows = []
    for key in cfg["models"]:
        try:
            sc = make_scorer(key, cfg, args, weights=False)
        except Exception as e:
            print(f"  {RED}{key}：tokenizer 不在本機或載入失敗（{type(e).__name__}: {e}）{RESET}")
            continue
        for tid in tids:
            text = read_text(cfg, tid)
            st = tokenize_stats(sc, text)
            rows.append((key, tid, st))
            ids = st["ids"][: max(1, len(sc.encode(text[: cfg['demo']['preview_chars']])))]
            print(f"  {key:<10} {tid:<8} tokens={st['tokens']:<5} tokens/字={st['tokens_per_char']:<6} "
                  f"無損={st['lossless']}")
            print("    " + render_segments(segments(sc, ids)))
    banner("怎麼挑 pair", AMBER)
    main_t = cfg["demo"]["text"]
    counts = {k: st["tokens"] for k, t, st in rows if t == main_t}
    if len(counts) >= 2:
        best = max(((a, b) for a in counts for b in counts if a < b),
                   key=lambda p: max(counts[p[0]], counts[p[1]]) / min(counts[p[0]], counts[p[1]]))
        r = max(counts[best[0]], counts[best[1]]) / min(counts[best[0]], counts[best[1]])
        print(f"  在「{main_t}」上 token 數差最多的是 {best[0]} 與 {best[1]}（{r:.2f} 倍）。")
        print("  兩個 tokenizer 的 token 數差不到約 1.3 倍，PPL 的落差就不明顯，demo 效果會弱；")
        print("  但差很多的組合常常也是能力差很多的組合，BPB 就不會接近——這兩件事要一起看，先 rehearse 再決定。")
    note("inspect 只載 tokenizer，不載權重，所以很快；PPL 與 BPB 要 rehearse 才知道。")


def compute_all(cfg, args, tid):
    text = read_text(cfg, tid)
    scorers = [make_scorer(k, cfg, args) for k in (args.pair or cfg["demo"]["pair"])]
    results = []
    for sc in scorers:
        note(f"計算 {sc.label} ……")
        results.append(score(sc, text))
    return text, scorers, results


def cmd_rehearse(args, cfg):
    tid = args.text or cfg["demo"]["text"]
    text, scorers, results = compute_all(cfg, args, tid)
    banner(f"彩排結果 · {cfg['texts'][tid]['label']}")
    print_table([s.label for s in scorers], results, {k for k, _ in ROWS})
    for sc, r in zip(scorers, results):
        note(f"  {sc.label}: 起始 token {r['prefix_token']} · NLL {r['nll_nats']} nats · "
             f"forward {r['forward_seconds']} 秒 · 峰值 {r['peak_memory_gb']} GB · "
             f"檢查 (T/B)·log2(PPL) = {r['identity_check']}")
    rec = record(cfg, tid, text, scorers, results, args.fake)
    name = f"{tid}-{'-'.join(s.key for s in scorers)}{'-FAKE' if args.fake else ''}.json"
    path = REHEARSAL / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    note(f"→ {path.relative_to(HERE)}")


def wait(msg: str) -> str:
    try:
        return input(f"\n{GREY}[{msg} · q = 離開]{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def cmd_show(args, cfg):
    os.system("clear")
    banner(cfg["demo"]["title"], AMBER)
    tid = args.text or cfg["demo"]["text"]
    text = read_text(cfg, tid)
    scorers = [make_scorer(k, cfg, args) for k in (args.pair or cfg["demo"]["pair"])]
    labels = [f"A · {scorers[0].label}", f"B · {scorers[1].label}"]
    results = [tokenize_stats(sc, text) for sc in scorers]
    shown = set()

    if wait("Enter = Step 1：這段文字") == "q":
        return
    banner("Step 1 · 同一段文字、同一組 bytes")
    print(f"\n{text}\n")
    shown |= {"chars", "bytes"}
    print_table(labels, results, shown, "bytes")

    if wait("Enter = Step 2：兩個 tokenizer 怎麼切") == "q":
        return
    banner("Step 2 · 兩個 tokenizer 怎麼切（第一句）")
    head = text[: cfg["demo"]["preview_chars"]]
    for lab, sc in zip(labels, scorers):
        ids = sc.encode(head)
        print(f"\n{BOLD}{lab}{RESET}  {GREY}（{len(ids)} tokens；下標 = 幾個 token 才拼出這一段）{RESET}")
        print("  " + render_segments(segments(sc, ids)))
    shown |= {"tokens", "tokens_per_char"}
    print_table(labels, results, shown, "tokens")

    if wait("Enter = Step 3：perplexity（先讓學生預測）") == "q":
        return
    banner("Step 3 · perplexity")
    for i, sc in enumerate(scorers):
        note(f"forward：{sc.label} ……")
        results[i] = score(sc, text)
    shown |= {"ppl"}
    print_table(labels, results, shown, "ppl")

    if wait("Enter = Step 4：bits-per-byte") == "q":
        return
    banner("Step 4 · bits-per-byte：同一個總 NLL，改除以 bytes")
    shown |= {"bpb"}
    print_table(labels, results, shown, "bpb")
    for lab, r in zip(labels, results):
        print(f"  {lab}: BPB = (T/B)·log2(PPL) = ({r['tokens']}/{r['bytes']})·log2({r['ppl']}) "
              f"= {r['identity_check']}")

    key = wait("l = round-trip 無損檢查（前提 a） · Enter = 結束")
    if key == "l":
        shown |= {"lossless"}
        print_table(labels, results, shown, "lossless")
        wait("Enter = 結束")
    append_jsonl(RUNS / "live" / f"{dt.datetime.now():%Y%m%d}.jsonl",
                 record(cfg, tid, text, scorers, results, args.fake))


def cmd_replay(args, cfg):
    files = sorted(REHEARSAL.glob("*.json"))
    tid = args.text or cfg["demo"]["text"]
    pair = args.pair or cfg["demo"]["pair"]
    want = REHEARSAL / f"{tid}-{'-'.join(pair)}.json"
    path = want if want.exists() else None
    if path is None:
        print(f"{RED}沒有彩排紀錄 {want.relative_to(HERE)}；課前先跑 rehearse。現有：{[f.name for f in files]}{RESET}")
        return
    rec = json.loads(path.read_text(encoding="utf-8"))
    banner(f"課前錄下的結果（{rec['time']}，非現場）", RED)
    print(f"\n{rec['text']}\n")
    labels = [f"A · {m['label']}" if i == 0 else f"B · {m['label']}" for i, m in enumerate(rec["models"])]
    print_table(labels, rec["results"], {k for k, _ in ROWS})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="不載模型，只演練流程（數字是假的）")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"), help="覆寫 demo_config.toml 的 pair")
    p.add_argument("--text", help="文字 id（見 [texts]）；預設是 demo_config.toml 的 text")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--only")
    for c in ("inspect", "rehearse", "show", "replay", "diag"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        args.online = True
    cfg = load_config()
    if args.text and args.text not in cfg["texts"]:
        sys.exit(f"--text：{args.text!r} 不在 [texts]")
    if args.pair:
        for m in args.pair:
            if m not in cfg["models"]:
                sys.exit(f"--pair：{m!r} 不在 [models]")
    {"fetch": cmd_fetch, "inspect": cmd_inspect, "rehearse": cmd_rehearse, "diag": cmd_diag,
     "show": cmd_show, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
