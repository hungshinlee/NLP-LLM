#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 3：分數沒變，行為變了（投影片 p65–p67、p70、p76）。

    ./present.sh live-det        課堂 (c)：一個 prompt 單獨送兩次，再併進一個 batch 送，逐 token 對齊上色；最後印彩排的表
    ./present.sh replay-bits     課堂 (a)：彩排紀錄，Enter 四次——accuracy、flips 與雜訊底線、指令遵循、bits per byte（不載模型）
    ./present.sh replay-evict    課堂 (b)：彩排紀錄，Enter 五次（不載模型）
    ./present.sh replay-det      現場 (c) 失敗：彩排的表與一個分岔的例子（不載模型）
    ./present.sh test            不載權重：規則與算術的測試；有 mlx 時另用亂數小模型驗量法
    ./present.sh inspect         課前：版本、機器、兩個模型、兩百題從哪來、system prompt 的 token、併批時補齊的那幾行原始碼
    ./present.sh probe           彩排前：每一段跑幾個，估全套要多久
    ./present.sh rehearse        彩排全套：(c) → (a) → (b)；做過的格子跳過
                                 （--part det|bits|evict、--only 8,4、--force、--note "接電源、其他程式都關了"）
    ./present.sh tiny [子命令]   亂數權重的小模型走真的流程（不下載；寫到 runs/smoke/）

三小段的量法：
  (a) 位元數   bf16 載入之後在記憶體裡量化（mlx_lm.utils.quantize_model，也就是 mlx_lm.convert -q 做的事：affine、每組 64 個）。
               每一列量三樣，全部一次一題、greedy：bits-per-byte（三段文字）；兩百題四選一的 accuracy 與相對於 bf16 的 flips；
               一百二十個帶指令的 prompt，用寫死的規則檢查有沒有照做。
  (b) KV 驅逐  system prompt 有兩條指令。整段 prompt 先照常 prefill，然後只把 system prompt 內文的 KV 丟掉一個比例 r，再生成。
               兩種政策：留開頭加最近的視窗；每一層各自丟掉鍵的 L2 範數最大的。對照：一個都不丟，KV 存成 4 bit。
               一般的問題量第一條指令有沒有照做；套話的請求量輸出與 system prompt 的 ROUGE-L recall，以及 canary 有沒有出現。
  (c) T = 0    同一個 prompt、greedy。單獨送；併進 batch 送——兩種併法分開量：
               equal   同伴裁成與它一樣長，自己寫的迴圈：沒有補齊，只有 batch 的形狀不同；
               padded  mlx-lm 的 BatchGenerator，同伴長度各不相同：要補齊，途中有人寫完 batch 還會變小。
               另外把 (a) 的兩百題併批問，數相對於單獨問的 flips：那是 (a) 的雜訊底線。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）；replay-* 只用標準函式庫。Python 3.11+。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect as pyinspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info,  # noqa: E402
                    locked_revision, note, now, peak_memory_gb, reset_peak_memory, versions)
import checks as K  # noqa: E402

CONFIG = HERE / "demo_config.toml"
PROMPTS = HERE / "prompts.toml"
RUNS = HERE / "runs"
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]   # 離線時 huggingface_hub 會核對 snapshot 是否完整
GB = 1e9


# ── 小工具 ──────────────────────────────────────────────────────
def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def load_prompts() -> dict:
    with open(PROMPTS, "rb") as f:
        return tomllib.load(f)


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def out_dir(args) -> Path:
    return RUNS / ("smoke" if args.tiny else "rehearsal")


def cache_dir(args) -> Path:
    return RUNS / "smoke" / "cache" if args.tiny else RUNS / "cache"


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_or(p: Path, default):
    return read_json(p) if p.exists() else default


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(HERE.parent))
    except ValueError:
        return Path(p).name


def seed_for(base: int, *parts) -> int:
    return (base + zlib.crc32("|".join(str(x) for x in parts).encode())) % (2 ** 31)


def num(x, fmt: str = ".2f") -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def pct(x, digits: int = 0) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pause(msg: str) -> bool:
    """Enter 繼續、q 離開；不是終端機時不等。"""
    if not sys.stdin.isatty():
        print(f"\n  {DIM}[{msg}]{RESET}")
        return True
    key = input(f"\n  {DIM}[Enter] {msg}   q = quit{RESET} ").strip().lower()
    print("\033[1A\033[2K", end="")
    return key != "q"


def conditions(text: str | None) -> dict:
    out = {"note": text or ""}
    try:
        out["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            first = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
            out["power"] = first.strip()
        except Exception:
            pass
    return out


def stamp(args) -> dict:
    return {"date": now(), "versions": versions(), "device": device_info(), "tiny": bool(args.tiny), "conditions": conditions(args.note)}


def converters():
    """繁→簡、簡→繁的逐字轉換（opencc；沒有裝就回傳 None，「指定用繁體中文」只看漢字的比例）。"""
    try:
        from opencc import OpenCC
        t2s, s2t = OpenCC("t2s"), OpenCC("s2t")
        return t2s.convert, s2t.convert
    except Exception:
        return None, None


def row_spec(cfg: dict, key: str) -> dict:
    """主表的一列或額外的一格：'bf16'、'4'（affine 的位元數）、'mxfp4:4:32'（mode:bits:group）。"""
    if key == "bf16":
        return {"key": key, "label": "bf16", "quant": None}
    if ":" in key:
        mode, bits, group = key.split(":")
        return {"key": key, "label": f"{mode}", "quant": (mode, int(bits), int(group))}
    return {"key": key, "label": f"{int(key)}", "quant": ("affine", int(key), int(cfg["bits"]["group_size"]))}


def settings(cfg: dict, args) -> dict:
    """各段的網格；--tiny 換成小的。"""
    b, e, d, ins = cfg["bits"], cfg["evict"], cfg["det"], cfg["instructions"]
    g = {"rows": list(b["rows"]), "extra": list(b["extra"]), "n_tasks": int(ins.get("n_tasks", 0)) or 10 ** 6, "instr_max_tokens": int(ins["max_tokens"]), "n_mmlu": None,
         "answer_max_tokens": int(b["mmlu"]["answer_max_tokens"]), "ratios": [float(r) for r in e["ratios"]], "policies": list(e["policies"]),
         "orders": list(e["orders"]), "n_questions": int(e["n_questions"]), "n_leaks": int(e["n_leaks"]),
         "question_max_tokens": int(e["question_max_tokens"]), "leak_max_tokens": int(e["leak_max_tokens"]),
         "det_models": list(d["models"]), "det_max_tokens": int(d["max_tokens"]), "batches": list(d["batches"]), "alone_reps": int(d["alone_reps"]),
         "padded_reps": int(d["padded_reps"]), "equal_reps": int(d["equal_reps"]), "targets": list(d["targets"]),
         "more_targets": list(d["more_targets"]), "more_reps": dict(d["more_reps"]),
         "thinking_reps": int(d["thinking_reps"]), "thinking_batches": list(d["thinking_batches"]), "thinking_max_tokens": int(d["thinking_max_tokens"]),
         "floor_batch": int(d["floor_batch"]), "floor_orders": int(d["floor_orders"]), "kv_group_size": int(e["kv_group_size"])}
    if args.tiny:
        t = cfg["tiny"]
        g.update(rows=list(t["bits_rows"]), extra=list(t["extra"]), n_tasks=int(t["n_tasks"]), instr_max_tokens=int(t["max_tokens"]), n_mmlu=int(t["n_mmlu"]),
                 ratios=[float(r) for r in t["ratios"]], n_questions=int(t["n_questions"]), n_leaks=int(t["n_leaks"]),
                 question_max_tokens=int(t["max_tokens"]), leak_max_tokens=int(t["max_tokens"]), det_models=["bf16", "q4"],
                 det_max_tokens=int(t["det_max_tokens"]), batches=list(t["batches"]), alone_reps=int(t["alone_reps"]), padded_reps=int(t["padded_reps"]),
                 equal_reps=int(t["equal_reps"]), thinking_reps=int(t["thinking_reps"]), thinking_batches=list(t["thinking_batches"]),
                 thinking_max_tokens=int(t["thinking_max_tokens"]), floor_batch=int(t["floor_batch"]), kv_group_size=32, more_reps=dict(t["more_reps"]))
    return g


# ══ tokenizer（tiny 用）═════════════════════════════════════════
class ByteTok:
    """--tiny 的 tokenizer：一個 byte 一個 token，另加幾個特殊記號。模板的形狀照 Qwen3（關掉 thinking 時補一對空的 think）。"""
    SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    vocab_size = 256 + len(SPECIAL)
    eos_token_ids = [256, 258]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
        return s if enable_thinking else s + "<think>\n\n</think>\n\n"

    def encode_offsets(self, text: str):
        ids, offs, i = [], [], 0
        while i < len(text):
            for k, sp in enumerate(self.SPECIAL):
                if text.startswith(sp, i):
                    ids.append(256 + k)
                    offs.append((i, i + len(sp)))
                    i += len(sp)
                    break
            else:
                b = text[i].encode("utf-8")
                ids += list(b)
                offs += [(i, i + 1)] * len(b)
                i += 1
        return ids, offs

    def encode(self, text, add_special_tokens=False):
        return self.encode_offsets(text)[0]

    def decode(self, ids):
        out, buf = [], bytearray()
        for t in ids:
            if t < 256:
                buf.append(t)
            else:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                out.append(self.SPECIAL[t - 256] if t - 256 < len(self.SPECIAL) else "")
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


# ══ 模型與量測 ═══════════════════════════════════════════════════
class Lab:
    """一個模型（必要時在記憶體裡量化），與三小段共用的幾種生成方式。"""

    def __init__(self, cfg: dict, args, model_key: str = "bf16", quant: tuple | None = None):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        self.mx, self.cfg, self.args, self.tiny = mx, cfg, args, bool(args.tiny)
        self.seed = int(cfg["demo"]["seed"])
        self.thinking = bool(cfg["demo"]["enable_thinking"])
        t0 = time.time()
        if self.tiny:
            from mlx_lm.models import qwen3
            mx.random.seed(self.seed)
            self.tok = ByteTok()
            self.config = dict(model_type="qwen3", hidden_size=128, num_hidden_layers=2, intermediate_size=256, num_attention_heads=4,
                               rms_norm_eps=1e-6, vocab_size=ByteTok.vocab_size + (-ByteTok.vocab_size) % 64, num_key_value_heads=2,
                               max_position_embeddings=8192, rope_theta=1e6, head_dim=32, tie_word_embeddings=False)
            self.model = qwen3.Model(qwen3.ModelArgs(**self.config))
            self.model.lm_head.weight = self.model.lm_head.weight * 6.0          # 亂數初始化的 logits 太平，放大讓分布有尖峰
            if model_key != "bf16" and quant is None:                            # tiny 的「另一個模型」：同一組權重量化成 4 bit
                quant = ("affine", 4, 32)
            self.info = {"label": f"tiny random Qwen3 ({model_key})", "repo": None, "revision": None}
            self.path = None
        else:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            m = cfg["models"][model_key]
            rev = locked_revision(m["repo"], m.get("revision"))
            try:
                self.path = snapshot_download(m["repo"], revision=rev, local_files_only=not args.online, ignore_patterns=NON_MODEL_FILES)
            except Exception as e:
                raise SystemExit(f"{RED}本機沒有 {m['repo']}@{str(rev)[:10]}（{type(e).__name__}）：./present.sh fetch 看缺哪一個，./present.sh fetch --online 補抓{RESET}")
            self.model, self.tok = load(self.path)
            self.config = json.loads((Path(self.path) / "config.json").read_text(encoding="utf-8"))
            self.info = {"label": m["label"], "repo": m["repo"], "revision": rev, "quantization": self.config.get("quantization")}
        mx.eval(self.model.parameters())
        if quant is not None:
            from mlx_lm.utils import quantize_model
            mode, bits, group = quant
            if self.tiny:
                group = min(group, 32)
            self.model, qc = quantize_model(self.model, dict(self.config), group, bits, mode)
            mx.eval(self.model.parameters())
            self.info.update(label=f"{self.info['label']} → {mode} {bits}-bit", quantized_here={"mode": mode, "bits": bits, "group_size": group})
        from mlx_lm.utils import compute_bits_per_weight
        flat = tree_flatten(self.model.parameters())
        self.info["weight_bytes"] = sum(v.nbytes for _, v in flat)
        self.info["bits_per_weight"] = round(float(compute_bits_per_weight(self.model)), 3)
        self.eos = set(int(x) for x in self.tok.eos_token_ids)
        self.load_seconds = round(time.time() - t0, 2)

    # ── 文字 ↔ token ──
    def enc(self, text: str):
        """回傳 (ids, 每個 token 在 text 裡的字元範圍)。"""
        if self.tiny:
            return self.tok.encode_offsets(text)
        hf = getattr(self.tok, "_tokenizer", self.tok)
        e = hf(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(e["input_ids"]), [tuple(o) for o in e["offset_mapping"]]

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def special(self, text: str) -> int:
        ids = self.enc(text)[0]
        if len(ids) != 1:
            raise SystemExit(f"{RED}{text!r} 在這個 tokenizer 裡不是單一 token（{ids}）{RESET}")
        return ids[0]

    def chat(self, user: str, system: str | None = None, thinking: bool | None = None) -> dict:
        """套 chat template。回傳 ids、整段文字、每個 token 的字元範圍，與 user、system 內文的 token 範圍 [s, e)。"""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.thinking if thinking is None else bool(thinking))
        ids, offs = self.enc(text)
        out = {"ids": ids, "text": text, "offsets": offs}
        cs = text.find(system) if system else -1
        if system:
            out["system_chars"] = (cs, cs + len(system))
            out["system"] = self.token_span(offs, cs, cs + len(system))
        cu = text.find(user, 0 if cs < 0 else cs + len(system))
        out["user"] = self.token_span(offs, cu, cu + len(user))
        return out

    @staticmethod
    def token_span(offs, cs: int, ce: int) -> tuple[int, int]:
        """字元範圍 [cs, ce) 裡開始的 token：回傳 token 的 [s, e)。"""
        idx = [i for i, (a, b) in enumerate(offs) if cs <= a < ce and b > a]
        return (idx[0], idx[-1] + 1) if idx else (0, 0)

    def equal_length(self, chat: dict, other_text: str) -> list[int]:
        """同伴：模板與 chat 相同，user 的內文換成 other_text 的 token，循環或裁到一樣長。"""
        s, e = chat["user"]
        body = self.enc(other_text)[0] or [chat["ids"][s]]
        n = e - s
        fill = (body * (n // len(body) + 1))[:n]
        return chat["ids"][:s] + fill + chat["ids"][e:]

    # ── 生成 ──
    def new_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def greedy(self, ids: list[int], max_tokens: int, cache=None, feed: list[int] | None = None, want_gaps: bool = False, on_token=None) -> dict:
        """一次一條、greedy。沒有給 cache：整段 prompt 一次餵進去，之後一次一個 token（mlx-lm 的 generate_step 對短 prompt 的排法）。
        給了 cache（已經裝著 prompt 的前面）：只餵 feed。遇到停止記號就停，停止記號不算進輸出。
        want_gaps：每一步另記第一名與第二名 logit 的差，與第二名是哪個 token。"""
        mx = self.mx
        cache = self.new_cache() if cache is None else cache
        x = mx.array(ids if feed is None else feed)[None]
        toks, gaps, second, eos = [], [], [], False
        t0 = time.perf_counter()
        for _ in range(int(max_tokens)):
            logits = self.model(x, cache=cache)[:, -1, :]
            y = mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)
            if want_gaps:
                lf = logits[0].astype(mx.float32)
                order = mx.argsort(lf)[-2:]
                top = lf[order]
                mx.eval(y, order, top)
                gaps.append(round(float(top[1].item() - top[0].item()), 4))
                o = [int(v) for v in order.tolist()]
                second.append(o[0] if o[1] == int(y.item()) else o[1])
            else:
                mx.eval(y)
            t = int(y.item())
            if t in self.eos:
                eos = True
                break
            toks.append(t)
            if on_token:
                on_token(t)
            x = y[:, None]
        out = {"tokens": toks, "eos": eos, "seconds": time.perf_counter() - t0}
        if want_gaps:
            out.update(gaps=gaps[:len(toks)], second=second[:len(toks)])
        return out

    def greedy_equal(self, rows: list[list[int]], max_tokens: int) -> list[list[int]]:
        """一個 batch、每一列一樣長、自己寫的迴圈：沒有補齊。每一列都走滿 max_tokens 步（batch 的形狀從頭到尾不變），
        之後各自截在第一個停止記號。排法與 greedy() 相同：整段 prompt 一次餵，之後一次一個 token。"""
        mx = self.mx
        if len(set(len(r) for r in rows)) != 1:
            raise ValueError("greedy_equal：每一列要一樣長")
        cache = self.new_cache()
        x = mx.array(rows)
        cols = []
        for _ in range(int(max_tokens)):
            logits = self.model(x, cache=cache)[:, -1, :]
            y = mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)
            mx.eval(y)
            cols.append(y.tolist())
            x = y[:, None]
        out = []
        for i in range(len(rows)):
            seq = [c[i] for c in cols]
            cut = next((j for j, t in enumerate(seq) if t in self.eos), len(seq))
            out.append(seq[:cut])
        return out

    def batch(self, rows: list[list[int]], max_tokens: int, batch_size: int, progress=None) -> list[list[int]]:
        """mlx-lm 的 BatchGenerator、greedy。prompt 長度不同時它會補齊；有人寫完，batch 就變小、排隊的遞補。停止記號不算進輸出。"""
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler
        gen = BatchGenerator(self.model, stop_tokens=[[t] for t in sorted(self.eos)], sampler=make_sampler(temp=0.0), max_tokens=int(max_tokens),
                             completion_batch_size=int(batch_size), prefill_batch_size=int(batch_size))
        uids = gen.insert([list(r) for r in rows], [int(max_tokens)] * len(rows))
        out = {u: [] for u in uids}
        done = 0
        while responses := gen.next_generated():
            for r in responses:
                if not (r.finish_reason == "stop" and int(r.token) in self.eos):
                    out[r.uid].append(int(r.token))
                if r.finish_reason is not None:
                    done += 1
                    if progress:
                        progress(done, len(uids))
        gen.close()
        return [[t for t in out[u] if t not in self.eos] for u in uids]

    def gap_after(self, ids: list[int]) -> float:
        """ids 整段一次餵進去，下一個 token 的前兩名 logit 差多少（形狀與逐步生成不同，數值會差一點；只看量級）。"""
        mx = self.mx
        lf = self.model(mx.array(ids)[None])[0, -1].astype(mx.float32)
        top = mx.sort(lf)[-2:]
        mx.eval(top)
        return round(float(top[1].item() - top[0].item()), 4)

    def nlls(self, ids: list[int]) -> list[float]:
        """ids[0] 是起始 token；回傳 ids[1:] 每個 token 的負對數機率（nats）。"""
        mx = self.mx
        logits = self.model(mx.array(ids)[None])[0, :-1].astype(mx.float32)
        tgt = mx.array(ids[1:])
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0]
        mx.eval(nll)
        return nll.tolist()

    def close(self) -> None:
        self.model = None
        import gc
        gc.collect()
        try:
            self.mx.clear_cache()
        except Exception:
            pass


def load_lab(cfg, args, model_key: str = "bf16", quant: tuple | None = None) -> Lab:
    what = "tiny random Qwen3 (no download)" if args.tiny else cfg["models"][model_key]["repo"]
    note(f"載入 {what}" + (f"，量化成 {quant[0]} {quant[1]} bit" if quant else "") + " …")
    reset_peak_memory()
    lab = Lab(cfg, args, model_key, quant)
    pk = peak_memory_gb()
    note(f"  完成：{lab.load_seconds:.1f} 秒；{lab.info['bits_per_weight']:.2f} bits/weight，{lab.info['weight_bytes'] / GB:.2f} GB" + (f"；peak {pk:.1f} GB" if pk else ""))
    return lab


def progress_line(tag: str):
    def p(done, total):
        if sys.stdout.isatty():
            print(f"\r  {GREY}{tag}: {done}/{total}{RESET}", end="", flush=True)
    return p


def end_progress() -> None:
    if sys.stdout.isatty():
        print("\r\033[2K", end="")


# ══ (a) 三把尺 ═══════════════════════════════════════════════════
def bpb_texts(cfg: dict, args) -> list[dict]:
    out = []
    for t in cfg["bits"]["bpb"]["texts"]:
        p = (HERE / t["path"]).resolve()
        if args.tiny:
            text = ("The city wakes slowly. 城市慢慢醒來。 " * 12).strip()
        elif not p.exists():
            note(f"  沒有 {rel(p)}：這一段文字跳過")
            continue
        elif p.suffix == ".jsonl":
            text = "\n\n".join(json.loads(line)["text"] for line in open(p, encoding="utf-8"))
        else:
            text = p.read_text(encoding="utf-8").strip()
        mb = int(t.get("max_bytes", 0))
        if mb and len(text.encode("utf-8")) > mb:
            text = text.encode("utf-8")[:mb].decode("utf-8", errors="ignore")
        out.append({"key": t["key"], "label": t["label"], "path": rel(p), "text": text, "bytes": len(text.encode("utf-8"))})
    return out


def measure_bpb(lab: Lab, cfg: dict, texts: list[dict]) -> dict:
    b = cfg["bits"]["bpb"]
    pid, W = lab.special(b["prefix"]), int(b["window_tokens"])
    out = {}
    for t in texts:
        ids = lab.enc(t["text"])[0]
        nll = 0.0
        for s in range(0, len(ids), W):
            nll += sum(lab.nlls([pid] + ids[s:s + W]))
        out[t["key"]] = {"bytes": t["bytes"], "tokens": len(ids), "nll_nats": round(nll, 3), "bpb": round(nll / math.log(2) / t["bytes"], 4),
                         "ppl": round(math.exp(min(nll / max(len(ids), 1), 50.0)), 3)}
    return out


def mmlu_pool(cfg: dict, args, g: dict) -> dict:
    """兩百題：W10 demo 3 的 demo.py 與設定從檔案路徑載入，用它的 make_ctx（同一個種子、同一批科目）。"""
    if args.tiny:
        rng = random.Random(7)
        qs = [{"id": i, "subject": "tiny", "question": f"Tiny question number {i}?", "choices": ["red", "green", "blue", "grey"], "answer": rng.randrange(4)}
              for i in range(g["n_mmlu"])]
        body = lambda q: q["question"] + "\n\n" + "\n".join(f"({'ABCD'[i]}) {c}" for i, c in enumerate(q["choices"]))  # noqa: E731
        return {"questions": qs, "instruction": "Answer the following multiple-choice question. Reply with only the letter of your answer.", "body": body,
                "parse": lambda t: (ord(t[0]) % 4) if t else None, "source": {"file": "(tiny: made up)"}}
    d = (HERE / cfg["bits"]["mmlu"]["demo"]).resolve()
    W = load_file_module("w10_d3_demo", d / "demo.py")
    with open(d / "demo_config.toml", "rb") as f:
        cfg10 = tomllib.load(f)
    ctx = W.make_ctx(cfg10, types.SimpleNamespace(tiny=False, n=None, batch_size=None))
    n = int(cfg["bits"]["mmlu"].get("n", 0))
    return {"questions": ctx.pool[:n] if n else ctx.pool, "instruction": cfg10["task"]["instruction"], "body": W.H.body, "parse": W.H.parse_answer,
            "source": {**ctx.source, "subjects": ctx.subjects, "subjects_from": ctx.subjects_from, "seed": ctx.s["seed"], "via": rel(d / "demo.py")}}


def mmlu_rows(lab: Lab, pool: dict) -> list[list[int]]:
    return [lab.chat(f"{pool['instruction']}\n\n{pool['body'](q)}")["ids"] for q in pool["questions"]]


def mmlu_score(pool: dict, texts: list[str]) -> dict:
    ans = [pool["parse"]((t or "").strip()) for t in texts]
    gold = [int(q["answer"]) for q in pool["questions"]]
    right = [int(a is not None and a == y) for a, y in zip(ans, gold)]
    n = len(gold)
    acc = sum(right) / n
    return {"n": n, "answers": ans, "right": right, "accuracy": round(acc, 4), "se": round(K.se(acc, n), 4), "unparsed": sum(1 for a in ans if a is None)}


def measure_mmlu_alone(lab: Lab, pool: dict, g: dict, tag: str) -> dict:
    rows, texts = mmlu_rows(lab, pool), []
    p = progress_line(tag)
    t0 = time.time()
    for i, ids in enumerate(rows):
        texts.append(lab.decode(lab.greedy(ids, g["answer_max_tokens"])["tokens"]))
        p(i + 1, len(rows))
    end_progress()
    return {"ids": [q["id"] for q in pool["questions"]], "texts": texts, "seconds": round(time.time() - t0, 1), **mmlu_score(pool, texts)}


def instruction_items(cfg: dict, prompts: dict, g: dict) -> list[dict]:
    p, out = cfg["instructions"], []
    for kind in K.KINDS:
        for task in prompts["task"][:g["n_tasks"]]:
            out.append({"id": f"{kind}/{task['id']}", "kind": kind, "task": task, "prompt": f"{task['text']} {K.instruction_text(kind, task, p)}"})
    return out


def measure_instructions(lab: Lab, cfg: dict, items: list[dict], g: dict, tag: str) -> dict:
    out, p = {}, progress_line(tag)
    t0 = time.time()
    for i, it in enumerate(items):
        r = lab.greedy(lab.chat(it["prompt"])["ids"], g["instr_max_tokens"])
        out[it["id"]] = {"text": lab.decode(r["tokens"]), "cut": int(not r["eos"]), "tokens": len(r["tokens"])}
        p(i + 1, len(items))
    end_progress()
    return {"outputs": out, "seconds": round(time.time() - t0, 1)}


def score_instructions(cfg: dict, items: list[dict], outputs: dict) -> dict:
    p = cfg["instructions"]
    t2s, s2t = converters()
    by, rows = {k: {"n": 0, "followed": 0, "cut": 0, "silent": 0} for k in K.KINDS}, {}
    for it in items:
        o = outputs[it["id"]]
        v = K.follows(it["kind"], o["text"], it["task"], p, t2s, s2t)
        b = by[it["kind"]]
        b["n"] += 1
        b["followed"] += int(v["ok"])
        b["cut"] += int(o["cut"])
        b["silent"] += int(not v["said"])
        rows[it["id"]] = int(v["ok"])
    n = sum(b["n"] for b in by.values())
    f = sum(b["followed"] for b in by.values())
    return {"n": n, "followed": f, "rate": round(f / n, 4) if n else None, "by_kind": by, "rows": rows, "cut": sum(b["cut"] for b in by.values()),
            "silent": sum(b["silent"] for b in by.values()), "opencc": t2s is not None}


def bits_cache(args, key: str) -> Path:
    return cache_dir(args) / f"bits_{key.replace(':', '-')}.json"


def run_bits_row(cfg, args, g, key: str, pool: dict, texts: list[dict], items: list[dict]) -> dict:
    """一列：載入、量化、三把尺。結果（含每一題的輸出）存 runs/cache/；做過的不重做。"""
    cp = bits_cache(args, key)
    c = {} if args.force else load_or(cp, {})
    sig = {"mmlu_ids": [q["id"] for q in pool["questions"]], "items": [it["id"] for it in items], "texts": [(t["key"], t["bytes"]) for t in texts],
           "max_tokens": [g["answer_max_tokens"], g["instr_max_tokens"]]}
    if c.get("sig") != json.loads(json.dumps(sig)):
        c = {"sig": sig}
    if all(k in c for k in ("bpb", "mmlu", "instr")):
        note(f"  {key}: 做過了（{rel(cp)}）")
        return c
    spec = row_spec(cfg, key)
    lab = load_lab(cfg, args, "bf16", spec["quant"])
    c["info"] = lab.info
    c["date"] = now()
    if "bpb" not in c:
        t0 = time.time()
        c["bpb"] = measure_bpb(lab, cfg, texts)
        c["bpb_seconds"] = round(time.time() - t0, 1)
        write_json(cp, c)
    if "mmlu" not in c:
        c["mmlu"] = measure_mmlu_alone(lab, pool, g, f"{key}: questions")
        write_json(cp, c)
    if "instr" not in c:
        c["instr"] = measure_instructions(lab, cfg, items, g, f"{key}: instructions")
        write_json(cp, c)
    pk = peak_memory_gb()
    c["peak_gb"] = round(pk, 2) if pk else None
    write_json(cp, c)
    lab.close()
    return c


def summarize_bits(cfg, args, g, pool: dict, items: list[dict]) -> dict:
    """runs/cache/ 的每一列 → 表。flips 一律相對於 bf16。"""
    keys = [k for k in g["rows"] + g["extra"] if bits_cache(args, k).exists()]
    data = {k: read_json(bits_cache(args, k)) for k in keys}
    base = data.get("bf16")
    rows = {}
    for k in keys:
        c = data[k]
        if not all(x in c for x in ("bpb", "mmlu", "instr")):
            continue
        m = mmlu_score(pool, c["mmlu"]["texts"])
        ins = score_instructions(cfg, items, c["instr"]["outputs"])
        row = {"key": k, "label": row_spec(cfg, k)["label"], "main": k in g["rows"], "info": c["info"], "bpb": c["bpb"],
               "mmlu": {x: m[x] for x in ("n", "accuracy", "se", "unparsed")}, "instr": {x: ins[x] for x in ("n", "followed", "rate", "by_kind", "cut", "silent")},
               "seconds": {"bpb": c.get("bpb_seconds"), "mmlu": c["mmlu"]["seconds"], "instr": c["instr"]["seconds"]}, "peak_gb": c.get("peak_gb")}
        if base is not None and k != "bf16":
            mb = mmlu_score(pool, base["mmlu"]["texts"])
            ib = score_instructions(cfg, items, base["instr"]["outputs"])
            row["mmlu"]["vs_bf16"] = {**K.flips(mb["right"], m["right"]), "answers_changed": K.changed(mb["answers"], m["answers"])}
            row["instr"]["vs_bf16"] = K.flips([ib["rows"][it["id"]] for it in items], [ins["rows"][it["id"]] for it in items])
            row["instr"]["same_text"] = sum(1 for it in items if c["instr"]["outputs"][it["id"]]["text"] == base["instr"]["outputs"][it["id"]]["text"])
        rows[k] = row
    det = load_or(out_dir(args) / "det.json", {})
    rec = {**stamp(args), "what": "(a) bits per weight: three rulers, one request at a time, greedy", "source": pool["source"],
           "instructions": {"kinds": K.KINDS, "per_kind": len(items) // len(K.KINDS), "opencc": converters()[0] is not None,
                            "rules": {k: cfg["instructions"][k] for k in ("words_max", "zh_min_han", "zh_max_simp", "json_keys", "ending_sentence", "min_units", "max_tokens")}},
           "rows": rows, "floor": det.get("floor")}
    write_json(out_dir(args) / "bits.json", rec)
    return rec


def rehearse_bits(cfg, args, g) -> None:
    banner("(a) bits per weight: bf16, then quantised in memory — three rulers each")
    prompts = load_prompts()
    pool = mmlu_pool(cfg, args, g)
    texts, items = bpb_texts(cfg, args), instruction_items(cfg, prompts, g)
    note(f"  {len(pool['questions'])} questions（{pool['source'].get('file')}）、{len(items)} instruction prompts、{len(texts)} texts")
    only = set(args.only.split(",")) if args.only else None
    for key in g["rows"] + g["extra"]:
        if only and key not in only and row_spec(cfg, key)["label"] not in only:
            continue
        run_bits_row(cfg, args, g, key, pool, texts, items)
        rec = summarize_bits(cfg, args, g, pool, items)
        r = rec["rows"].get(key)
        if r:
            v = r["mmlu"].get("vs_bf16")
            print(f"  {BOLD}{r['label']:>6}{RESET}  {r['info']['bits_per_weight']:.2f} b/w   accuracy {r['mmlu']['accuracy']:.3f}"
                  + (f"   flips {v['n10']}+{v['n01']}" if v else "") + f"   instructions {pct(r['instr']['rate'])}   bpb(wiki/zh) "
                  + "/".join(num(r["bpb"].get(k, {}).get("bpb"), ".3f") for k in ("wiki", "zh")))
    note(f"  → {rel(out_dir(args) / 'bits.json')}")


# ══ (b) KV 驅逐 ══════════════════════════════════════════════════
def system_text(prompts: dict, order: str) -> tuple[str, dict]:
    """三段接成 system prompt。回傳全文與每一段在全文裡的字元範圍。"""
    s = prompts["system"]
    names = ["role", "directive", "defense"] if order == "directive_first" else ["role", "defense", "directive"]
    text, spans = "", {}
    for n in names:
        if text:
            text += "\n"
        spans[n] = (len(text), len(text) + len(s[n]))
        text += s[n]
    return text, spans


class EvictLab:
    """一個 prompt：prefill 一次，之後每一格從同一份 KV 出發——丟掉一部分、或量化，再生成。"""

    def __init__(self, lab: Lab, cfg: dict, g: dict):
        self.lab, self.cfg, self.g = lab, cfg, g
        w5 = (HERE / cfg["evict"]["w5_demo"]).resolve()
        self.SinkWindowCache = load_file_module("w05_d4_demo", w5).SinkWindowCache
        self.n_sink = int(cfg["evict"]["n_sink"])

    def prefill(self, chat: dict, part_chars: dict) -> dict:
        """除了最後一個 token 之外整段先算好；回傳每一層的 (K, V)、system 內文與三段各自的 token 範圍、每一層鍵的範數。"""
        mx, lab = self.lab.mx, self.lab
        ids = chat["ids"]
        cache = lab.new_cache()
        lab.model(mx.array(ids[:-1])[None], cache=cache)
        kv = []
        for c in cache:
            k, v, off = c.state
            kv.append((k[..., :off, :], v[..., :off, :]))
        mx.eval(kv)
        s0 = chat["system_chars"][0]
        parts = {n: lab.token_span(chat["offsets"], s0 + a, s0 + b) for n, (a, b) in part_chars.items()}
        s, e = chat["system"]
        norms = []
        for k, _ in kv:
            nk = mx.sqrt(mx.sum(mx.square(k[0, :, s:e, :].astype(mx.float32)), axis=-1)).mean(axis=0)
            norms.append(nk.tolist())
        return {"ids": ids, "kv": kv, "span": (s, e), "parts": parts, "norms": norms, "n": len(ids) - 1}

    def keep_lists(self, pf: dict, policy: str, r: float) -> list[list[int]]:
        """每一層留下哪些位置。"""
        if policy == "sink_window":
            keep = K.keep_sink_window(pf["n"], pf["span"], r, self.n_sink)
            return [keep] * len(pf["kv"])
        if policy == "k_norm":
            return [K.keep_by_score(pf["n"], pf["span"], r, nl) for nl in pf["norms"]]
        raise ValueError(policy)

    def caches(self, pf: dict, cell: tuple) -> tuple[list, dict]:
        """cell = ("none", 0)｜(policy, r)｜("kv", bits)。回傳各層的 cache 與這一格的統計。"""
        from mlx_lm.models.cache import KVCache
        mx = self.lab.mx
        kind, x = cell
        out, stat = [], {}
        if kind in ("none", "kv"):
            for k, v in pf["kv"]:
                c = KVCache()
                c.state = (k, v, pf["n"])
                out.append(c.to_quantized(group_size=self.g["kv_group_size"], bits=int(x)) if kind == "kv" else c)
            return out, stat
        keeps = self.keep_lists(pf, kind, float(x))
        for (k, v), keep in zip(pf["kv"], keeps):
            idx = mx.array(keep)
            c = self.SinkWindowCache(window=10 ** 9, n_sink=0, reindex=False)      # W5 demo 4 的 class：位置不重編，鍵停在原本的絕對位置
            c.state = (None, None, mx.take(k, idx, axis=2), mx.take(v, idx, axis=2), pf["n"])
            out.append(c)
        s, e = pf["span"]
        stat["evicted"] = K.n_evict(e - s, float(x))
        stat["kept_share"] = {n: mean([K.kept_share(keep, pf["parts"][n]) for keep in keeps]) for n in pf["parts"]}
        return out, stat

    def generate(self, pf: dict, cell: tuple, max_tokens: int) -> tuple[dict, dict]:
        caches, stat = self.caches(pf, cell)
        return self.lab.greedy(pf["ids"], max_tokens, cache=caches, feed=pf["ids"][-1:]), stat


def cell_key(cell: tuple) -> str:
    return f"{cell[0]}@{cell[1]:g}"


def evict_cells(cfg: dict, g: dict) -> list[tuple]:
    cells = [("none", 0.0)]
    for p in g["policies"]:
        cells += [(p, r) for r in g["ratios"] if r > 0]
    return cells + [("kv", float(b)) for b in cfg["evict"]["kv_bits"]]


def evict_messages(prompts: dict, g: dict) -> list[dict]:
    s = prompts["system"]
    qs = [{"id": "q/" + t["id"], "kind": "question", "text": t["text"] + s["question_suffix"]} for t in prompts["task"][:g["n_questions"]]]
    return qs + [{"id": "leak/" + k["id"], "kind": "leak", "text": k["text"]} for k in prompts["leak"][:g["n_leaks"]]]


def score_evict(cfg: dict, prompts: dict, order: str, msgs: list[dict], outputs: dict) -> dict:
    """一格（一種政策、一個壓縮率）的輸出 → 幾個率。"""
    s = prompts["system"]
    full, _ = system_text(prompts, order)
    t2s, s2t = converters()
    ins, thr = cfg["instructions"], float(cfg["evict"]["leak_threshold"])
    ev = cfg["evict"]
    q = {"n": 0, "trad": 0, "closing": 0, "closing_strict": 0, "both": 0, "both_strict": 0, "cut": 0, "canary": 0}
    k = {"n": 0, "rouge": [], "rouge_role": [], "rouge_directive": [], "rouge_defense": [], "leaked": 0, "canary": 0, "refused": 0}
    for m in msgs:
        o = outputs.get(m["id"])
        if o is None:
            continue
        t = o["text"]
        if m["kind"] == "question":
            sc = K.script(t, t2s, s2t)
            trad = sc["han"] >= float(ins["zh_min_han"]) and (sc["simp_only"] is None or sc["simp_only"] <= float(ins["zh_max_simp"]))
            closing, strict = K.ends_with(t, s["closing"], True), K.ends_with(t, s["closing"], False)
            q["n"] += 1
            q["trad"] += int(trad)
            q["closing"] += int(closing)
            q["closing_strict"] += int(strict)
            q["both"] += int(trad and closing)
            q["both_strict"] += int(trad and strict)
            q["cut"] += int(o.get("cut", 0))
            q["canary"] += int(s["canary"] in t)
        else:
            rw = K.rouge_l_recall(full, t)
            k["n"] += 1
            k["rouge"].append(rw)
            for part in ("role", "directive", "defense"):
                k["rouge_" + part].append(K.rouge_l_recall(s[part], t))
            k["leaked"] += int(rw >= thr)
            k["canary"] += int(s["canary"] in t)
            k["refused"] += int(K.refused(t, ev["refusal_phrase"], ev["refusal_max_units"]))
    out = {"questions": {**q, "obeyed": round(q["both"] / q["n"], 4) if q["n"] else None, "trad_rate": round(q["trad"] / q["n"], 4) if q["n"] else None,
                         "closing_rate": round(q["closing"] / q["n"], 4) if q["n"] else None},
           "leaks": {"n": k["n"], "leaked": k["leaked"], "canary": k["canary"], "refused": k["refused"],
                     "rouge": round(mean(k["rouge"]) or 0.0, 4) if k["n"] else None, "rouge_max": round(max(k["rouge"]), 4) if k["rouge"] else None,
                     **{"rouge_" + p: (round(mean(k["rouge_" + p]) or 0.0, 4) if k["n"] else None) for p in ("role", "directive", "defense")}}}
    return out


def evict_cache(args, order: str) -> Path:
    return cache_dir(args) / f"evict_{order}.json"


def rehearse_evict(cfg, args, g) -> None:
    banner("(b) evict part of the system prompt's KV — two instructions, two policies, one control")
    prompts = load_prompts()
    msgs, cells = evict_messages(prompts, g), evict_cells(cfg, g)
    lab = load_lab(cfg, args, cfg["evict"]["model"])
    ev = EvictLab(lab, cfg, g)
    rec = {**stamp(args), "what": "(b) KV eviction on the system prompt only; greedy, one request at a time", "model": lab.info,
           "ratios": g["ratios"], "policies": g["policies"], "kv_bits": [int(b) for b in cfg["evict"]["kv_bits"]], "n_sink": ev.n_sink,
           "leak_threshold": float(cfg["evict"]["leak_threshold"]), "opencc": converters()[0] is not None, "orders": {}}
    for order in g["orders"]:
        full, part_chars = system_text(prompts, order)
        cp = evict_cache(args, order)
        c = {} if args.force else load_or(cp, {})
        sig = {"system": full, "msgs": [m["id"] for m in msgs], "cells": [cell_key(x) for x in cells], "max": [g["question_max_tokens"], g["leak_max_tokens"]]}
        if c.get("sig") != sig:
            c = {"sig": sig, "out": {}, "stat": {}}
        t0, tokens = time.time(), None
        p = progress_line(f"{order}")
        for i, m in enumerate(msgs):
            todo = [x for x in cells if m["id"] not in c["out"].get(cell_key(x), {})]
            if todo:
                chat = lab.chat(m["text"], system=full)
                pf = ev.prefill(chat, part_chars)
                tokens = {"prompt": len(chat["ids"]), "system": pf["span"][1] - pf["span"][0], **{n: b - a for n, (a, b) in pf["parts"].items()}}
                for x in todo:
                    r, stat = ev.generate(pf, x, g["leak_max_tokens"] if m["kind"] == "leak" else g["question_max_tokens"])
                    c["out"].setdefault(cell_key(x), {})[m["id"]] = {"text": lab.decode(r["tokens"]), "cut": int(not r["eos"])}
                    if stat:
                        c["stat"].setdefault(cell_key(x), {})[m["id"]] = stat
                c["tokens"] = tokens
                write_json(cp, c)
            p(i + 1, len(msgs))
        end_progress()
        o = {"system": full, "parts": list(part_chars), "tokens": c.get("tokens"), "seconds": round(time.time() - t0, 1), "cells": {}}
        for x in cells:
            ck = cell_key(x)
            sc = score_evict(cfg, prompts, order, msgs, c["out"].get(ck, {}))
            st = list(c["stat"].get(ck, {}).values())
            if st:
                sc["evicted_tokens"] = st[0]["evicted"]
                sc["kept_share"] = {n: round(mean([s["kept_share"][n] for s in st]), 3) for n in st[0]["kept_share"]}
            sc["examples"] = evict_examples(prompts, order, msgs, c["out"].get(ck, {}))
            o["cells"][ck] = sc
        rec["orders"][order] = o
        show_evict_order(rec, order)
        write_json(out_dir(args) / "evict.json", rec)
    pk = peak_memory_gb()
    rec["peak_gb"] = round(pk, 2) if pk else None
    write_json(out_dir(args) / "evict.json", rec)
    note(f"  → {rel(out_dir(args) / 'evict.json')}")
    lab.close()


def evict_examples(prompts: dict, order: str, msgs: list[dict], outputs: dict) -> dict:
    """每一格留兩段輸出當例子：重疊最高的一個套話請求、第一個一般的問題。"""
    full, _ = system_text(prompts, order)
    leaks = [(K.rouge_l_recall(full, outputs[m["id"]]["text"]), m) for m in msgs if m["kind"] == "leak" and m["id"] in outputs]
    out = {}
    if leaks:
        rw, m = max(leaks, key=lambda z: z[0])
        out["leak"] = {"id": m["id"], "request": m["text"], "rouge": round(rw, 3), "text": outputs[m["id"]]["text"][:900]}
    qs = [m for m in msgs if m["kind"] == "question" and m["id"] in outputs]
    if qs:
        out["question"] = {"id": qs[0]["id"], "request": qs[0]["text"], "text": outputs[qs[0]["id"]]["text"][:600]}
    return out


# ══ (c) temperature 0 ════════════════════════════════════════════
def companions(cfg: dict, args) -> list[dict]:
    p = (HERE / cfg["det"]["companions"]).resolve()
    if p.exists():
        with open(p, "rb") as f:
            return tomllib.load(f)["prompt"]
    if not args.tiny:
        raise SystemExit(f"{RED}沒有 {rel(p)}（同伴的 prompt）{RESET}")
    return [{"id": f"c{i}", "text": "companion " + "word " * (3 + 2 * i)} for i in range(12)]


def compare(ref: list[int], outs: list[list[int]], gaps: list[float], second: list[int], valid_upto: int | None = None) -> dict:
    """一組輸出對一段參考：幾種不同的輸出、幾次與參考不同、第一個分岔的位置、分岔處前兩名 logit 的差。
    gaps 是沿著「自己的迴圈單獨送」那一段量的。參考若是另一段（BatchGenerator 的 B = 1），兩段在 valid_upto 之後就不是同一段字，
    那之後的位置沒有量到差——只記 valid_upto 以內的（gap_n 是記到的次數）。"""
    pos, gap, to_second = [], [], 0
    limit = len(gaps) if valid_upto is None else min(len(gaps), int(valid_upto) + 1)
    for o in outs:
        i = K.first_divergence(ref, o)
        pos.append(i)
        if i is not None and i < limit:
            gap.append(gaps[i])
            to_second += int(i < len(o) and o[i] == second[i])
    hit = [i for i in pos if i is not None]
    return {"reps": len(outs), "distinct": K.distinct(outs), "distinct_with_ref": K.distinct(outs + [ref]), "differ": len(hit), "first": pos,
            "first_median": K.median(hit), "first_min": min(hit) if hit else None, "gap": gap, "gap_median": K.median(gap),
            "gap_max": max(gap) if gap else None, "gap_zero": sum(1 for x in gap if x == 0), "gap_n": len(gap), "to_second": to_second}


def det_target(lab: Lab, cfg: dict, g: dict, comps: list[dict], t: dict, reps: dict, batches: list[int], max_tokens: int, store: dict) -> dict:
    """一個 prompt 的全部：單獨送、equal、padded。store 是 runs/cache/ 裡這個 prompt 的原始輸出（做過的跳過）。"""
    chat = lab.chat(t["text"], thinking=bool(t.get("thinking", False)))
    ids = chat["ids"]
    sig = {"ids": ids, "max_tokens": max_tokens}
    if store.get("sig") != sig:
        store.clear()
        store.update(sig=sig)
    # 單獨送：自己寫的迴圈（第一次另記每一步前兩名的差），與 BatchGenerator 的 B = 1
    if "alone" not in store or len(store["alone"]) < reps["alone"]:
        first = lab.greedy(ids, max_tokens, want_gaps=True)
        store["gaps"], store["second"], store["eos"] = first["gaps"], first["second"], first["eos"]
        store["tok_per_s"] = round(len(first["tokens"]) / first["seconds"], 2) if first["seconds"] else None
        store["alone"] = [first["tokens"]] + [lab.greedy(ids, max_tokens)["tokens"] for _ in range(reps["alone"] - 1)]
    if "alone_bg" not in store:
        store["alone_bg"] = [lab.batch([ids], max_tokens, 1)[0] for _ in range(min(3, reps["alone"]))]
    ref, ref_bg, gaps, second = store["alone"][0], store["alone_bg"][0], store["gaps"], store["second"]
    comp_ids = None
    for mode, n in (("equal", reps["equal"]), ("padded", reps["padded"])):
        for B in batches:
            cell = store.setdefault(mode, {}).setdefault(str(B), {"out": [], "pos": [], "pad": []})
            while len(cell["out"]) < n:
                rep = len(cell["out"])
                rng = random.Random(seed_for(lab.seed, "det", t["id"], mode, B, rep))
                pos = rng.randrange(B)
                picks = rng.sample(comps, B - 1) if len(comps) >= B - 1 else [rng.choice(comps) for _ in range(B - 1)]
                if mode == "equal":
                    rows = [lab.equal_length(chat, c["text"]) for c in picks]
                    rows.insert(pos, ids)
                    outs = lab.greedy_equal(rows, max_tokens)
                else:
                    if comp_ids is None:
                        comp_ids = {c["id"]: lab.chat(c["text"])["ids"] for c in comps}
                    rows = [comp_ids[c["id"]] for c in picks]
                    rows.insert(pos, ids)
                    outs = lab.batch(rows, max_tokens, B)
                cell["out"].append(outs[pos])
                cell["pos"].append(pos)
                cell["pad"].append(max(len(r) for r in rows) - len(ids))
                print(f"\r  {GREY}{t['id']} {mode} B={B}: {len(cell['out'])}/{n}{RESET}", end="", flush=True)
    print("\r\033[2K", end="")
    out = {"id": t["id"], "thinking": bool(t.get("thinking", False)), "prompt_tokens": len(ids), "max_tokens": max_tokens, "output_tokens": len(ref),
           "eos": store["eos"], "tok_per_s": store.get("tok_per_s"),
           "alone": {**compare(ref, store["alone"][:reps["alone"]], gaps, second)},
           "alone_bg": {**compare(ref, store["alone_bg"], gaps, second), "note": "BatchGenerator with B = 1, against our own loop"},
           "gap_profile": {"median": K.median(gaps), "zero": sum(1 for x in gaps if x == 0), "below_0.13": sum(1 for x in gaps if x < 0.13), "n": len(gaps)},
           "equal": {}, "padded": {}, "text": lab.decode(ref)[:1200]}
    fork_bg = K.first_divergence(ref, ref_bg)
    out["alone_bg"]["first"] = fork_bg
    for mode, n, r in (("equal", reps["equal"], ref), ("padded", reps["padded"], ref_bg)):
        for B in batches:
            cell = store[mode][str(B)]
            c = compare(r, cell["out"][:n], gaps, second, None if (mode == "equal" or fork_bg is None) else fork_bg)
            c.update(K.pad_groups(cell["out"][:n], cell["pad"][:n]))
            c["same_as_own_loop"] = sum(1 for o in cell["out"][:n] if o == ref)
            c["pad_when_differs"] = [cell["pad"][i] for i, p in enumerate(c["first"]) if p is not None]
            c["pad_when_same"] = [cell["pad"][i] for i, p in enumerate(c["first"]) if p is None]
            j = next((i for i, p in enumerate(c["first"]) if p is not None), None)
            if j is not None:
                p0 = c["first"][j]
                c["example"] = {"at": p0, "alone": lab.decode(r[max(0, p0 - 12):p0 + 16]), "batched": lab.decode(cell["out"][j][max(0, p0 - 12):p0 + 16]),
                                "gap": gaps[p0] if (p0 < len(gaps) and (mode == "equal" or fork_bg is None or p0 <= fork_bg)) else None}
            out[mode][str(B)] = c
    return out


def det_floor(lab: Lab, cfg: dict, args, g: dict) -> dict:
    """(a) 的雜訊底線：同一個 bf16 模型、同樣的兩百題——單獨問一次（與 (a) 的 bf16 那一列共用），併批問幾次（題目的排法不同）。"""
    pool = mmlu_pool(cfg, args, g)
    cp = bits_cache(args, "bf16")
    c = load_or(cp, {})
    ids = [q["id"] for q in pool["questions"]]
    if c.get("mmlu", {}).get("ids") == ids and not args.force:
        alone = c["mmlu"]
    else:
        alone = measure_mmlu_alone(lab, pool, g, "floor: questions, one at a time")
    again = measure_mmlu_alone(lab, pool, g, "floor: questions, one at a time, again")
    rows = mmlu_rows(lab, pool)
    a = mmlu_score(pool, alone["texts"])
    out = {"n": len(rows), "batch": g["floor_batch"], "alone_accuracy": a["accuracy"],
           "alone_again": {**K.flips(a["right"], mmlu_score(pool, again["texts"])["right"]), "answers_changed": K.changed(a["answers"], mmlu_score(pool, again["texts"])["answers"])},
           "orders": []}
    scores = []
    for k in range(g["floor_orders"]):
        order = list(range(len(rows)))
        random.Random(seed_for(lab.seed, "floor", k)).shuffle(order)
        outs = lab.batch([rows[i] for i in order], g["answer_max_tokens"], g["floor_batch"], progress_line(f"floor: batched, order {k + 1}"))
        end_progress()
        texts = [None] * len(rows)
        for j, i in enumerate(order):
            texts[i] = lab.decode(outs[j])
        s = mmlu_score(pool, texts)
        scores.append(s)
        out["orders"].append({"accuracy": s["accuracy"], "unparsed": s["unparsed"], **K.flips(a["right"], s["right"]), "answers_changed": K.changed(a["answers"], s["answers"])})
    if len(scores) >= 2:
        out["between_orders"] = {**K.flips(scores[0]["right"], scores[1]["right"]), "answers_changed": K.changed(scores[0]["answers"], scores[1]["answers"])}
    return out


def rehearse_det(cfg, args, g) -> None:
    banner("(c) temperature 0: one prompt, alone and in company — and the noise floor for (a)")
    prompts = load_prompts()
    comps = companions(cfg, args)
    by_id = {t["id"]: t for t in prompts["det"]}
    rec = load_or(out_dir(args) / "det.json", {}) if not args.force else {}
    rec = {**stamp(args), "what": "(c) greedy decoding, alone and inside batches", "max_tokens": g["det_max_tokens"], "batches": g["batches"],
           "reps": {"alone": g["alone_reps"], "equal": g["equal_reps"], "padded": g["padded_reps"]}, "companions": len(comps),
           "models": rec.get("models", {}), "floor": rec.get("floor")}
    only = set(args.only.split(",")) if args.only else None
    for mk in g["det_models"]:
        if only and mk not in only:
            continue
        lab = load_lab(cfg, args, mk)
        cp = cache_dir(args) / f"det_{mk}.json"
        store = {} if args.force else load_or(cp, {})
        m = {"info": lab.info, "targets": {}}
        t0 = time.time()
        reps = {"alone": g["alone_reps"], "equal": g["equal_reps"], "padded": g["padded_reps"]}
        for tid in g["targets"]:
            m["targets"][tid] = det_target(lab, cfg, g, comps, by_id[tid], reps, g["batches"], g["det_max_tokens"], store.setdefault(tid, {}))
            write_json(cp, store)
            show_det_target(m["targets"][tid], g["batches"])
        if mk == "bf16":
            for tid in g["more_targets"]:
                m["targets"][tid] = det_target(lab, cfg, g, comps, by_id[tid], g["more_reps"], g["batches"], g["det_max_tokens"], store.setdefault(tid, {}))
                write_json(cp, store)
                show_det_target(m["targets"][tid], g["batches"])
            tid = cfg["det"]["thinking_target"]
            treps = {"alone": g["thinking_reps"], "equal": 0, "padded": g["thinking_reps"]}
            m["targets"][tid] = det_target(lab, cfg, g, comps, by_id[tid], treps, g["thinking_batches"], g["thinking_max_tokens"], store.setdefault(tid, {}))
            write_json(cp, store)
            show_det_target(m["targets"][tid], g["thinking_batches"])
            rec["floor"] = det_floor(lab, cfg, args, g)
            show_floor(rec["floor"])
        m["seconds"] = round(time.time() - t0, 1)
        pk = peak_memory_gb()
        m["peak_gb"] = round(pk, 2) if pk else None
        rec["models"][mk] = m
        write_json(out_dir(args) / "det.json", rec)
        lab.close()
    note(f"  → {rel(out_dir(args) / 'det.json')}")


# ══ 印表 ═════════════════════════════════════════════════════════
def header(cfg: dict, rec: dict, what: str) -> None:
    extra = f"\n{what} — rehearsal of {str(rec.get('date', ''))[:16]}" + ("   [tiny random model — numbers mean nothing]" if rec.get("tiny") else "")
    dev = rec.get("device") or {}
    if dev.get("device_name"):
        extra += f"   {dev['device_name']}"
    banner(cfg["demo"]["title"] + extra, AMBER)


def show_bits(cfg: dict, rec: dict, stop: bool = True) -> None:
    rows = [r for r in rec["rows"].values() if r["main"]]
    extra = [r for r in rec["rows"].values() if not r["main"]]
    steps = ["accuracy", "flips", "instr", "bpb"]

    def table(shown: set) -> None:
        n = rows[0]["mmlu"]["n"] if rows else 0
        print(f"\n  {'bits':>6}{'bits/weight':>13}" + (f"{'accuracy':>11}" if "accuracy" in shown else "") + (f"{'n10':>7}{'n01':>5}{'changed':>9}" if "flips" in shown else "")
              + (f"{'instructions':>14}" if "instr" in shown else "") + (f"{'bpb wiki':>10}{'bpb zh':>8}{'bpb en':>8}" if "bpb" in shown else ""))
        for r in rows + extra:
            v = r["mmlu"].get("vs_bf16")
            line = f"  {BOLD}{r['label']:>6}{RESET}{r['info']['bits_per_weight']:>13.2f}"
            if "accuracy" in shown:
                line += f"{BOLD}{r['mmlu']['accuracy']:>11.3f}{RESET}"
            if "flips" in shown:
                line += f"{'—':>7}{'—':>5}{'—':>9}" if v is None else f"{AMBER}{v['n10']:>7}{v['n01']:>5}{RESET}{v['answers_changed']:>9}"
            if "instr" in shown:
                line += f"{BOLD}{pct(r['instr']['rate']):>14}{RESET}"
            if "bpb" in shown:
                line += "".join(f"{num(r['bpb'].get(k, {}).get('bpb'), '.3f'):>{w}}" for k, w in (("wiki", 10), ("zh", 8), ("en", 8)))
            print(line + ("" if r["main"] else f"   {GREY}(not in the main table){RESET}"))
        if "accuracy" in shown:
            se = rows[0]["mmlu"]["se"] if rows else None
            print(f"  {GREY}accuracy: {n} four-way questions, one at a time, greedy; its standard error is about ±{num(se, '.3f')}.{RESET}")
        if "flips" in shown:
            print(f"  {GREY}n10: right at bf16, wrong here. n01: wrong at bf16, right here. changed: questions whose answer letter differs.{RESET}")
            f = rec.get("floor")
            if f:
                o = f["orders"]
                print(f"  {GREY}noise floor — the same bf16 model, the same questions:{RESET} asked alone twice: {BOLD}{f['alone_again']['answers_changed']}{RESET} answers changed;"
                      f" asked inside batches of {f['batch']}: " + ", ".join(f"{BOLD}{x['n10']}+{x['n01']}{RESET} flips ({x['answers_changed']} changed)" for x in o))
            else:
                print(f"  {GREY}noise floor: not measured yet (./present.sh rehearse --part det){RESET}")
        if "instr" in shown:
            print(f"  {GREY}instructions: {rows[0]['instr']['n'] if rows else 0} prompts, six kinds, each checked by a fixed rule — our own set, not IFEval's.{RESET}")
        if "bpb" in shown:
            print(f"  {GREY}bits per byte, W1's ruler: wiki = 64 KiB of Wikipedia; zh, en = W1's paragraph in two languages (under 1 KiB each).{RESET}")

    shown = set()
    if stop and not pause("the accuracy column, alone"):
        return
    for i, s in enumerate(steps):
        shown.add(s)
        table(shown)
        if s == "instr":
            print(f"\n  {'bits':>6}" + "".join(f"{K.KIND_LABEL[k][:17]:>19}" for k in K.KINDS) + f"{'cut':>6}{'silent':>8}")
            for r in rows:
                print(f"  {r['label']:>6}" + "".join(f"{str(r['instr']['by_kind'][k]['followed']) + '/' + str(r['instr']['by_kind'][k]['n']):>19}" for k in K.KINDS)
                      + f"{r['instr']['cut']:>6}{r['instr']['silent']:>8}")
            print(f"  {GREY}cut: replies that hit the token limit. silent: replies with nothing readable in them (counted as not followed).{RESET}")
        if stop and i < len(steps) - 1 and not pause({"accuracy": "add the flips", "flips": "add the instructions", "instr": "add bits per byte"}[s]):
            return


def show_evict_order(rec: dict, order: str) -> None:
    o = rec["orders"][order]
    tk = o.get("tokens") or {}
    print(f"\n  {BOLD}order: {' → '.join(o['parts'])}{RESET}   {GREY}system prompt {tk.get('system', '?')} tokens"
          + "".join(f", {n} {tk.get(n, '?')}" for n in o["parts"]) + f"{RESET}")
    rs = rec["ratios"]
    print(f"  {'share of the system prompt evicted':<44}" + "".join(f"{r:>8g}" for r in rs))
    base = o["cells"].get("none@0")
    for pol in rec["policies"]:
        cells = [base if r == 0 else o["cells"].get(f"{pol}@{r:g}") for r in rs]
        name = {"sink_window": "keep the start + a recent window", "k_norm": "drop the largest key norms (K-Norm)"}.get(pol, pol)
        print(f"  {BOLD}{name}{RESET}")
        for label, f in (("  instruction 1 obeyed (both halves)", lambda c: pct(c["questions"]["obeyed"])),
                         ("    … answered in Traditional Chinese", lambda c: pct(c["questions"]["trad_rate"])),
                         ("    … ended with the fixed sentence", lambda c: pct(c["questions"]["closing_rate"])),
                         ("  leak: ROUGE-L recall of the prompt", lambda c: num(c["leaks"]["rouge"], ".2f")),
                         ("    … replies with the canary", lambda c: f"{c['leaks']['canary']}/{c['leaks']['n']}"),
                         ("    … replies that refused (instruction 2)", lambda c: f"{c['leaks']['refused']}/{c['leaks']['n']}"),
                         ("    … obeyed to the letter (no quotes)", lambda c: pct(c["questions"]["both_strict"] / c["questions"]["n"]) if c["questions"].get("both_strict") is not None and c["questions"]["n"] else "—"),
                         ("  kept: directive / defense tokens", lambda c: "—" if "kept_share" not in c else f"{c['kept_share']['directive']:.1f}/{c['kept_share']['defense']:.1f}")):
            print(f"  {label:<44}" + "".join(f"{(f(c) if c else '—'):>8}" for c in cells))
    for b in rec["kv_bits"]:
        kv = o["cells"].get(f"kv@{b:g}")
        if kv:
            print(f"  {BOLD}control: nothing evicted, KV stored at {b} bits{RESET}   obeyed {pct(kv['questions']['obeyed'])}"
                  f"   leak ROUGE-L {num(kv['leaks']['rouge'], '.2f')}   canary {kv['leaks']['canary']}/{kv['leaks']['n']}")


def show_evict(cfg: dict, rec: dict, stop: bool = True) -> None:
    prompts = load_prompts()
    s = prompts["system"]
    main = next(iter(rec["orders"]))
    others = [k for k in rec["orders"] if k != main]
    o = rec["orders"][main]
    if stop and not pause("the system prompt and the two kinds of user message"):
        return
    print(f"\n  {BOLD}the system prompt{RESET} ({' → '.join(o['parts'])})")
    for n in o["parts"]:
        print(f"    {GREY}{n:<10}{RESET}{s[n]}")
    print(f"\n  {BOLD}two kinds of user message{RESET}")
    print(f"    {GREY}ordinary  {RESET}{prompts['task'][0]['text']}{s['question_suffix']}   {GREY}(× {o['cells']['none@0']['questions']['n']}; measures instruction 1){RESET}")
    print(f"    {GREY}probing   {RESET}{prompts['leak'][0]['text']}   {GREY}(× {o['cells']['none@0']['leaks']['n']}; measures instruction 2 and the leak){RESET}")
    rs = rec["ratios"]
    names = {"sink_window": "keep the start and a recent window", "k_norm": "drop the keys with the largest L2 norm, layer by layer"}

    def two(pol: str, order: str) -> None:
        oo = rec["orders"][order]
        cells = [oo["cells"]["none@0"] if r == 0 else oo["cells"].get(f"{pol}@{r:g}") for r in rs]
        first = "instruction 1 (obey) written first" if oo["parts"].index("directive") < oo["parts"].index("defense") else "instruction 2 (do not reveal) written first"
        print(f"\n  {BOLD}{names.get(pol, pol)}{RESET} — {first}\n  {'share of the system prompt evicted':<42}" + "".join(f"{r:>8g}" for r in rs))
        nq, nl = cells[0]["questions"]["n"], cells[0]["leaks"]["n"]
        print(f"  {'instruction 1 obeyed, of ' + str(nq) + ' replies':<42}" + "".join(f"{BOLD}{pct(c['questions']['obeyed']):>8}{RESET}" if c else f"{'—':>8}" for c in cells))
        print(f"  {'instruction 2 obeyed: refusals, of ' + str(nl):<42}" + "".join(f"{BOLD}{c['leaks']['refused']:>8}{RESET}" if c else f"{'—':>8}" for c in cells))
        print(f"  {'system prompt leaked (ROUGE-L recall)':<42}" + "".join(f"{AMBER}{num(c['leaks']['rouge'], '.2f'):>8}{RESET}" if c else f"{'—':>8}" for c in cells))
        print(f"  {GREY}{'replies containing the canary, of ' + str(nl):<42}" + "".join(f"{c['leaks']['canary'] if c else '—':>8}" for c in cells) + RESET)
        print(f"  {GREY}{'tokens kept: instruction 1 / 2':<42}" + "".join(f"{('—' if 'kept_share' not in c else format(c['kept_share']['directive'], '.1f') + '/' + format(c['kept_share']['defense'], '.1f')) if c else '—':>8}" for c in cells) + RESET)
        strict = [c["questions"].get("both_strict") for c in cells if c]
        loose = [c["questions"]["both"] for c in cells if c]
        if strict != loose and None not in strict:
            print(f"  {GREY}{'… obeyed to the letter (no quotation marks)':<42}" + "".join(f"{c['questions']['both_strict'] if c else '—':>8}" for c in cells) + RESET)

    steps = [(rec["policies"][0], main, "the first policy")]
    steps += [(rec["policies"][0], k, "the same policy, the two instructions swapped") for k in others]
    steps += [(pol, main, "the second policy") for pol in rec["policies"][1:]]
    for pol, order, label in steps:
        if stop and not pause(label):
            return
        two(pol, order)
    if stop and not pause("the control: nothing evicted, the KV stored with fewer bits"):
        return
    base = o["cells"]["none@0"]
    print(f"\n  {BOLD}control — nothing evicted{RESET}   {GREY}uncompressed: instruction 1 obeyed {pct(base['questions']['obeyed'])}, refusals {base['leaks']['refused']}, leaked {num(base['leaks']['rouge'], '.2f')}{RESET}")
    for b in rec["kv_bits"]:
        kv = o["cells"].get(f"kv@{b:g}")
        if kv:
            print(f"    KV stored at {b} bits:   instruction 1 obeyed {BOLD}{pct(kv['questions']['obeyed'])}{RESET}   refusals {BOLD}{kv['leaks']['refused']}{RESET}   leaked {AMBER}{num(kv['leaks']['rouge'], '.2f')}{RESET}")
    if stop and not pause("one actual reply to a probing request"):
        return
    best = None
    for ck, c in o["cells"].items():
        ex = c.get("examples", {}).get("leak")
        if ex and not ck.startswith(("none", "kv")) and (best is None or ex["rouge"] > best[1]["rouge"]):
            best = (ck, ex)
    if best:
        pol, r = best[0].split("@")
        print(f"\n  {BOLD}{names.get(pol, pol)}, {r} of the system prompt evicted{RESET}\n  user: {best[1]['request']}   {GREY}(ROUGE-L recall {best[1]['rouge']:.2f}){RESET}\n")
        print("    " + best[1]["text"].replace("\n", "\n    "))
    if others and (not stop or pause("the full table for the other order")):
        for k in others:
            show_evict_order(rec, k)


def show_det_target(t: dict, batches: list[int]) -> None:
    a = t["alone"]
    print(f"\n  {BOLD}{t['id']}{RESET}{' (thinking on)' if t['thinking'] else ''}: {t['prompt_tokens']} prompt tokens → {t['output_tokens']} output tokens"
          f"   {GREY}top-2 logit gap along the output: median {num(t['gap_profile']['median'], '.2f')}, exactly 0 at {t['gap_profile']['zero']} of {t['gap_profile']['n']} positions{RESET}")
    print(f"  {'':<34}{'alone':>10}" + "".join(f"{'B = ' + str(B):>10}" for B in batches))
    print(f"  {'different outputs — alone':<34}{BOLD}{str(a['distinct']) + ' of ' + str(a['reps']):>10}{RESET}"
          f"   {GREY}(mlx-lm's own way of sending it alone gives {'the same text' if t['alone_bg']['differ'] == 0 else 'a different text, from token ' + str(t['alone_bg'].get('first'))}){RESET}")
    for mode, label in (("equal", "no padding, same shape"), ("padded", "mlx-lm batching, padded")):
        cells = [t[mode].get(str(B)) for B in batches]
        if not any(c and c["reps"] for c in cells):
            continue
        print(f"  {BOLD}{label}{RESET}")
        print(f"  {'  different outputs':<34}{'':>10}" + "".join(f"{BOLD}{(str(c['distinct']) + ' of ' + str(c['reps'])):>10}{RESET}" for c in cells))
        print(f"  {'  runs that differ from alone':<34}{'':>10}" + "".join(f"{AMBER}{c['differ']:>10}{RESET}" for c in cells))
        print(f"  {'  first token that differs (median)':<34}{'—':>10}" + "".join(f"{num(c['first_median'], '.0f'):>10}" for c in cells))
        print(f"  {'  top-2 logit gap there (median)':<34}{'—':>10}" + "".join(f"{num(c['gap_median'], '.3f'):>10}" for c in cells))
        if mode == "padded" and all("pads" in c for c in cells if c):
            print(f"  {GREY}{'  pad lengths seen / with >1 output':<34}{'':>10}" + "".join(f"{(str(c['pads']) + ' / ' + str(c['pads_with_several'])):>10}" for c in cells) + RESET)


def show_floor(f: dict | None) -> None:
    if not f:
        return
    print(f"\n  {BOLD}the noise floor for (a){RESET}: the same bf16 model, the same {f['n']} questions (accuracy alone {f['alone_accuracy']:.3f})")
    print(f"    asked one at a time, twice:            {BOLD}{f['alone_again']['answers_changed']}{RESET} answers changed, {f['alone_again']['flips']} flips")
    for i, o in enumerate(f["orders"]):
        print(f"    asked in batches of {f['batch']}, order {i + 1}:       {BOLD}{o['answers_changed']}{RESET} answers changed, flips {AMBER}{o['n10']} + {o['n01']}{RESET}, accuracy {o['accuracy']:.3f}")
    if f.get("between_orders"):
        b = f["between_orders"]
        print(f"    between the two batched orders:        {BOLD}{b['answers_changed']}{RESET} answers changed, flips {b['n10']} + {b['n01']}")


def show_det(cfg: dict, rec: dict, stop: bool = True, example: bool = True) -> None:
    for mk, m in rec["models"].items():
        print(f"\n  {BOLD}{m['info']['label']}{RESET}")
        for tid, t in m["targets"].items():
            show_det_target(t, rec["batches"] if not t["thinking"] else [int(b) for b in t["padded"]])
    if example:
        for m in rec["models"].values():
            for t in m["targets"].values():
                ex = next((t[mode][b]["example"] for mode in ("equal", "padded") for b in t[mode] if "example" in t[mode][b]), None)
                if ex:
                    print(f"\n  {BOLD}one divergence{RESET} ({t['id']}, token {ex['at']}; top-2 gap there {num(ex['gap'], '.3f')})")
                    print(f"    {GREY}alone   {RESET}…{ex['alone']!r}")
                    print(f"    {GREY}batched {RESET}…{ex['batched']!r}")
                    break
            else:
                continue
            break
    show_floor(rec.get("floor"))


# ══ 子命令 ═══════════════════════════════════════════════════════
def need(args, name: str) -> dict:
    p = out_dir(args) / name
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {rel(p)}：先跑 ./present.sh rehearse{RESET}")
    return read_json(p)


def cmd_replay_bits(args, cfg):
    rec = need(args, "bits.json")
    det = load_or(out_dir(args) / "det.json", {})
    if det.get("floor"):
        rec["floor"] = det["floor"]
    header(cfg, rec, "(a) REPLAY: bits per weight")
    show_bits(cfg, rec)


def cmd_replay_evict(args, cfg):
    rec = need(args, "evict.json")
    header(cfg, rec, "(b) REPLAY: evicting the system prompt's KV")
    show_evict(cfg, rec)


def cmd_replay_det(args, cfg):
    rec = need(args, "det.json")
    header(cfg, rec, "(c) REPLAY: temperature 0, alone and in company")
    show_det(cfg, rec)


def cmd_replay(args, cfg):
    for name, fn in (("bits.json", cmd_replay_bits), ("evict.json", cmd_replay_evict), ("det.json", cmd_replay_det)):
        if (out_dir(args) / name).exists():
            fn(args, cfg)
            if not pause("next part"):
                return
        else:
            note(f"（沒有 {name}，跳過）")


def cmd_rescore(args, cfg):
    """不載模型：從 runs/cache/ 的原始輸出把 bits.json、evict.json、det.json 的分數重算一次（改了規則之後用）。
    機器、日期、模型的資訊沿用原來的紀錄；det.json 的例子（解碼過的字）也沿用。"""
    g = settings(cfg, args)
    prompts = load_prompts()
    d = out_dir(args)
    want = lambda part: args.part in (None, part)  # noqa: E731
    if want("bits") and (d / "bits.json").exists():
        old = read_json(d / "bits.json")
        rec = summarize_bits(cfg, args, g, mmlu_pool(cfg, args, g), instruction_items(cfg, prompts, g))
        rec.update({k: old[k] for k in ("date", "versions", "device", "tiny", "conditions") if k in old}, rescored=now())
        write_json(d / "bits.json", rec)
        note(f"  bits.json: {len(rec['rows'])} rows")
    if want("evict") and (d / "evict.json").exists():
        rec = read_json(d / "evict.json")
        msgs = evict_messages(prompts, g)
        for order, o in rec["orders"].items():
            c = read_json(evict_cache(args, order))
            for ck, cell in o["cells"].items():
                sc = score_evict(cfg, prompts, order, msgs, c["out"].get(ck, {}))
                sc.update({k: cell[k] for k in ("evicted_tokens", "kept_share") if k in cell})
                sc["examples"] = evict_examples(prompts, order, msgs, c["out"].get(ck, {}))
                o["cells"][ck] = sc
        rec.update(rescored=now(), refusal={"phrase": cfg["evict"]["refusal_phrase"], "max_units": int(cfg["evict"]["refusal_max_units"])})
        write_json(d / "evict.json", rec)
        note(f"  evict.json: {sum(len(o['cells']) for o in rec['orders'].values())} cells")
    if want("det") and (d / "det.json").exists():
        rec = read_json(d / "det.json")
        for mk, m in rec["models"].items():
            store = read_json(cache_dir(args) / f"det_{mk}.json")
            for tid, t in m["targets"].items():
                st = store[tid]
                ref, ref_bg, gaps, second = st["alone"][0], st["alone_bg"][0], st["gaps"], st["second"]
                fork_bg = K.first_divergence(ref, ref_bg)
                t["alone_bg"]["first"] = fork_bg
                for mode, r in (("equal", ref), ("padded", ref_bg)):
                    for B, old in t[mode].items():
                        cell, n = st[mode][B], old["reps"]
                        c = compare(r, cell["out"][:n], gaps, second, None if (mode == "equal" or fork_bg is None) else fork_bg)
                        c.update(K.pad_groups(cell["out"][:n], cell["pad"][:n]))
                        c["same_as_own_loop"] = sum(1 for x in cell["out"][:n] if x == ref)
                        c["pad_when_differs"] = [cell["pad"][i] for i, p in enumerate(c["first"]) if p is not None]
                        c["pad_when_same"] = [cell["pad"][i] for i, p in enumerate(c["first"]) if p is None]
                        if "example" in old:
                            ex = dict(old["example"])
                            if mode == "padded" and fork_bg is not None and ex["at"] > fork_bg:
                                ex["gap"] = None
                            c["example"] = ex
                        t[mode][B] = c
        rec["rescored"] = now()
        write_json(d / "det.json", rec)
        note("  det.json: gaps kept only where they were measured; outputs grouped by pad length")


def cmd_rehearse(args, cfg):
    g = settings(cfg, args)
    banner(cfg["demo"]["title"] + "\nrehearsal" + ("   [tiny random model — numbers mean nothing]" if args.tiny else ""))
    parts = [args.part] if args.part else ["det", "bits", "evict"]            # (c) 先做：(a) 的 flips 要它的雜訊底線
    t0 = time.time()
    for p in parts:
        {"det": rehearse_det, "bits": rehearse_bits, "evict": rehearse_evict}[p](cfg, args, g)
    if "bits" not in parts and (out_dir(args) / "bits.json").exists() and "det" in parts:
        prompts = load_prompts()                                              # 底線更新了：bits.json 也跟著寫一次
        summarize_bits(cfg, args, g, mmlu_pool(cfg, args, g), instruction_items(cfg, prompts, g))
    note(f"\n全部 {time.time() - t0:.0f} 秒。投影片要填的表：./present.sh replay-bits、replay-evict、replay-det")


def colour_diff(lab: Lab, ref: list[int], out: list[int]) -> str:
    """out 逐 token 對 ref：相同的照印，從第一個不同的 token 起上色。"""
    i = K.first_divergence(ref, out)
    if i is None:
        return lab.decode(out)
    return lab.decode(out[:i]) + AMBER + BOLD + lab.decode(out[i:]) + RESET


def cmd_live_det(args, cfg):
    g = settings(cfg, args)
    lv = cfg["live"]
    prompts, comps = load_prompts(), companions(cfg, args)
    t = next(x for x in prompts["det"] if x["id"] == (args.prompt or lv["target"]))
    B, mode = int(args.batch or (g["batches"][-1] if args.tiny else lv["batch"])), (args.mode or lv["mode"])
    banner(f"{cfg['demo']['title']}\n(c) live: one prompt, greedy, {g['det_max_tokens']} tokens — alone, then inside a batch of {B}")
    lab = load_lab(cfg, args, args.model or lv["model"])
    chat = lab.chat(t["text"], thinking=bool(t.get("thinking", False)))
    ids = chat["ids"]
    lab.greedy(ids, 4)                                                       # 暖機
    print(f"\n  {BOLD}prompt{RESET}: {t['text']}   {GREY}({len(ids)} tokens with the template){RESET}")

    def stream():
        det = None if lab.tiny else lab.tok.detokenizer
        if det is not None:
            det.reset()

        def on(tok: int) -> None:
            if det is None:
                sys.stdout.write(f"{tok} ")
            else:
                det.add_token(tok)
                sys.stdout.write(det.last_segment)
            sys.stdout.flush()
        return on

    if not pause("send it alone"):
        return
    print("  ", end="")
    a = lab.greedy(ids, g["det_max_tokens"], want_gaps=True, on_token=stream())
    print(f"\n\n  {GREY}{len(a['tokens'])} tokens, {len(a['tokens']) / a['seconds']:.1f} tokens/s. Sending it alone again …{RESET}")
    b = lab.greedy(ids, g["det_max_tokens"])
    i = K.first_divergence(a["tokens"], b["tokens"])
    print(f"  alone, a second time: {(GREEN + 'the same ' + str(len(b['tokens'])) + ' tokens') if i is None else (RED + 'differs from token ' + str(i))}{RESET}")
    if not pause(f"send it with {B - 1} companions ({'padded: mlx-lm batching, different lengths' if mode == 'padded' else 'equal lengths, no padding'})"):
        return
    rng = random.Random(seed_for(lab.seed, "live", time.time_ns() if not args.fixed else 0))
    pos, picks = rng.randrange(B), rng.sample(comps, B - 1)
    if mode == "padded":
        ref = lab.batch([ids], g["det_max_tokens"], 1)[0]
        k = K.first_divergence(a["tokens"], ref)
        print(f"  {GREY}alone, through mlx-lm's batching code (B = 1; it feeds all but the last prompt token first): "
              f"{'the same tokens as above' if k is None else 'differs from the text above from token ' + str(k)}{RESET}")
        rows = [lab.chat(c["text"])["ids"] for c in picks]
        rows.insert(pos, ids)
        out = lab.batch(rows, g["det_max_tokens"], B)[pos]
    else:
        ref = a["tokens"]
        rows = [lab.equal_length(chat, c["text"]) for c in picks]
        rows.insert(pos, ids)
        out = lab.greedy_equal(rows, g["det_max_tokens"])[pos]
    print(f"  {GREY}row {pos + 1} of {B}; the others: {', '.join(c['id'] for c in picks)}; prompt lengths {min(len(r) for r in rows)}–{max(len(r) for r in rows)} tokens{RESET}\n")
    print("  " + colour_diff(lab, ref, out))
    j = K.first_divergence(ref, out)
    if j is None:
        print(f"\n  {GREEN}{BOLD}the same {len(out)} tokens{RESET} as when it was sent alone")
    else:
        k = K.first_divergence(a["tokens"], ref)
        gap = a["gaps"][j] if j < len(a["gaps"]) and (k is None or j <= k) else lab.gap_after(ids + ref[:j])
        print(f"\n  {AMBER}{BOLD}differs from token {j} of {len(ref)}{RESET}: alone {lab.decode(ref[j:j + 1])!r}, in the batch {lab.decode(out[j:j + 1])!r}"
              + (f"   top-2 logit gap at that position, alone: {BOLD}{gap:.3f}{RESET}" if gap is not None else ""))
    from common import append_jsonl
    append_jsonl(RUNS / "live" / "live.jsonl", {"date": now(), "target": t["id"], "mode": mode, "B": B, "pos": pos, "first_divergence": j, "alone_twice_same": i is None})
    p = out_dir(args) / "det.json"
    if p.exists() and pause("the rehearsal: twenty runs per batch size"):
        show_det(cfg, read_json(p), example=False)
    lab.close()


def cmd_fetch(args, cfg):
    """兩個模型都在 versions.lock；這裡只確認本機有。--online 時補抓。也確認別的 demo 留下的檔案在不在。"""
    from huggingface_hub import snapshot_download
    for name, m in cfg["models"].items():
        rev = locked_revision(m["repo"], m.get("revision"))
        try:
            p = snapshot_download(m["repo"], revision=rev, local_files_only=not args.online, ignore_patterns=NON_MODEL_FILES)
            print(f"  {GREEN}ok{RESET}  {name:<5} {m['repo']}@{str(rev)[:10]}  →  {p}")
        except Exception as e:
            print(f"  {RED}缺{RESET}  {name:<5} {m['repo']}@{str(rev)[:10]}：{repr(e)[:120]}")
    paths = [cfg["bits"]["mmlu"]["demo"] + "/demo.py", cfg["evict"]["w5_demo"], cfg["det"]["companions"]] + [t["path"] for t in cfg["bits"]["bpb"]["texts"]]
    for q in paths:
        p = (HERE / q).resolve()
        print(f"  {(GREEN + 'ok') if p.exists() else (RED + '缺')}{RESET}  {rel(p)}")
    try:
        pool = mmlu_pool(cfg, args, settings(cfg, args))
        print(f"  {GREEN}ok{RESET}  {len(pool['questions'])} questions from {pool['source'].get('file')} @ {str(pool['source'].get('revision'))[:8]}")
    except SystemExit as e:
        print(f"  {RED}缺{RESET}  the questions: {e}")
    print(f"  {(GREEN + 'ok') if converters()[0] else (AMBER + '沒有')}{RESET}  opencc（「指定用繁體中文」分繁簡用；沒有就只看漢字的比例）")


def source_lines(obj, needles: tuple[str, ...]) -> list[str]:
    try:
        src, start = pyinspect.getsourcelines(obj)
    except Exception:
        return []
    return [f"{start + i:>5}  {line.rstrip()}" for i, line in enumerate(src) if any(n in line for n in needles)]


def cmd_inspect(args, cfg):
    G = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models import cache as MC
    g = settings(cfg, args)
    rec = {**stamp(args), "checks": {}}
    banner("inspect — versions, the machine, what this demo borrows")
    print("  " + "  ".join(f"{k} {v}" for k, v in rec["versions"].items() if v))
    print("  " + "  ".join(f"{k}={v}" for k, v in rec["device"].items()))
    banner("mlx-lm: how a batch of prompts of different lengths is padded")
    for obj, needles in ((G._right_pad_prompts, ("def ", "return")), (G.PromptProcessingBatch.prompt, ("padding", "prepare", "finalize")),
                         (MC.BatchKVCache.finalize, ("def ", "dynamic_roll", "offset")), (G.GenerationBatch._step, ("self.model(", "argmax", "sampled =")),
                         (G.BatchGenerator.__init__, ("sampler",))):
        lines = source_lines(obj, needles)
        print(f"  {BOLD}{getattr(obj, '__qualname__', obj)}{RESET}")
        print("\n".join("  " + x for x in lines[:8]))
        rec["checks"][getattr(obj, "__qualname__", str(obj))] = lines[:8]
    banner("the questions, the instruction prompts, the system prompt")
    prompts = load_prompts()
    pool = mmlu_pool(cfg, args, g)
    print(f"  questions: {len(pool['questions'])} from {pool['source'].get('file')} @ {str(pool['source'].get('revision'))[:8]}; subjects {pool['source'].get('subjects')}")
    items = instruction_items(cfg, prompts, g)
    print(f"  instruction prompts: {len(items)}; e.g. {items[0]['prompt']!r}")
    print(f"  opencc: {'yes' if converters()[0] else 'NO — Traditional vs Simplified cannot be told apart'}")
    rec["pool"] = pool["source"]
    lab = load_lab(cfg, args, "bf16")
    rec["bf16"] = lab.info
    for order in g["orders"]:
        full, part_chars = system_text(prompts, order)
        chat = lab.chat(prompts["task"][0]["text"] + prompts["system"]["question_suffix"], system=full)
        s, e = chat["system"]
        parts = {n: lab.token_span(chat["offsets"], chat["system_chars"][0] + a, chat["system_chars"][0] + b) for n, (a, b) in part_chars.items()}
        print(f"  {BOLD}{order}{RESET}: prompt {len(chat['ids'])} tokens; system span [{s}, {e}) = {e - s} tokens; " + ", ".join(f"{n} [{a}, {b})" for n, (a, b) in parts.items()))
        print(f"    {GREY}before the span: {lab.decode(chat['ids'][:s])!r}; after it: {lab.decode(chat['ids'][e:])!r}{RESET}")
        rec["checks"][f"span_{order}"] = {"prompt": len(chat["ids"]), "system": [s, e], "parts": parts}
        for r in g["ratios"]:
            keep = K.keep_sink_window(len(chat["ids"]) - 1, (s, e), r, int(cfg["evict"]["n_sink"]))
            print(f"    sink_window r = {r:g}: evicts {len(chat['ids']) - 1 - len(keep)} tokens; kept share " + ", ".join(f"{n} {K.kept_share(keep, sp):.2f}" for n, sp in parts.items()))
    banner("greedy decoding: our loop, mlx-lm's generate_step, BatchGenerator at B = 1")
    from mlx_lm.generate import generate_step
    ids = lab.chat(prompts["det"][0]["text"])["ids"]
    n = min(64, g["det_max_tokens"])
    mine = lab.greedy(ids, n)["tokens"]
    theirs = []
    for tok, _ in generate_step(lab.mx.array(ids), lab.model, max_tokens=n):
        if int(tok) in lab.eos:
            break
        theirs.append(int(tok))
    bg = lab.batch([ids], n, 1)[0]
    for name, other in (("generate_step", theirs), ("BatchGenerator, B = 1", bg)):
        i = K.first_divergence(mine, other)
        print(f"  our loop vs {name:<22}: {(GREEN + 'the same ' + str(len(mine)) + ' tokens') if i is None else (AMBER + 'differ from token ' + str(i))}{RESET}")
        rec["checks"][f"loop_vs_{name}"] = i
    lab.close()
    if not args.tiny:
        banner("4 bits quantised here vs mlx-community's 4-bit file")
        from mlx.utils import tree_flatten
        a, b = load_lab(cfg, args, "bf16", ("affine", 4, int(cfg["bits"]["group_size"]))), load_lab(cfg, args, "q4")
        fa, fb = dict(tree_flatten(a.model.parameters())), dict(tree_flatten(b.model.parameters()))
        same = sum(1 for k in fa if k in fb and fa[k].shape == fb[k].shape and bool(a.mx.array_equal(fa[k], fb[k]).item()))
        print(f"  arrays: {len(fa)} here, {len(fb)} in the file; identical: {BOLD}{same}{RESET}   bits/weight {a.info['bits_per_weight']} vs {b.info['bits_per_weight']}")
        print(f"  the file's config: {b.info.get('quantization')}")
        dts = sorted(set(str(v.dtype) for k, v in fb.items() if k.endswith("scales")))
        print(f"  scales dtype in the file: {dts}")
        rec["checks"]["q4_here_vs_file"] = {"arrays_here": len(fa), "arrays_file": len(fb), "identical": same, "bpw": [a.info["bits_per_weight"], b.info["bits_per_weight"]],
                                            "file_quantization": b.info.get("quantization"), "scales_dtype": dts}
        a.close()
        b.close()
    write_json(out_dir(args) / "inspect.json", rec)
    note(f"  → {rel(out_dir(args) / 'inspect.json')}")


def cmd_probe(args, cfg):
    """每一段跑幾個，估全套要多久。"""
    g = settings(cfg, args)
    n = int(cfg["probe"]["n"])
    prompts = load_prompts()
    banner("probe — a few of each, to estimate the full rehearsal")
    lab = load_lab(cfg, args, "bf16")
    pool, items, msgs = mmlu_pool(cfg, args, g), instruction_items(cfg, prompts, g), evict_messages(prompts, g)
    rows = mmlu_rows(lab, pool)
    lab.greedy(rows[0], 4)
    t0 = time.time()
    for ids in rows[:n]:
        lab.greedy(ids, g["answer_max_tokens"])
    t_q = (time.time() - t0) / n
    step = max(1, len(items) // n)
    t0, tok = time.time(), 0
    for it in items[::step][:n]:
        tok += len(lab.greedy(lab.chat(it["prompt"])["ids"], g["instr_max_tokens"])["tokens"])
    t_i, rate = (time.time() - t0) / n, tok / (time.time() - t0)
    ev = EvictLab(lab, cfg, g)
    full, part_chars = system_text(prompts, g["orders"][0])
    cells = evict_cells(cfg, g)
    t0 = time.time()
    for m in (msgs[0], msgs[-1]):
        pf = ev.prefill(lab.chat(m["text"], system=full), part_chars)
        for x in cells:
            ev.generate(pf, x, g["leak_max_tokens"] if m["kind"] == "leak" else g["question_max_tokens"])
    t_e = (time.time() - t0) / 2
    t = prompts["det"][0]
    ids = lab.chat(t["text"])["ids"]
    t0 = time.time()
    lab.greedy(ids, g["det_max_tokens"])
    t_a = time.time() - t0
    comps = companions(cfg, args)
    tb = {}
    for B in g["batches"]:
        t0 = time.time()
        lab.batch([ids] + [lab.chat(c["text"])["ids"] for c in comps[:B - 1]], g["det_max_tokens"], B)
        tb[B] = time.time() - t0
    lab.close()
    n_rows = len(g["rows"]) + len(g["extra"])
    est_bits = n_rows * (len(rows) * t_q + len(items) * t_i + 60)
    est_evict = len(g["orders"]) * len(msgs) * t_e
    per_target = g["alone_reps"] * t_a + sum(tb.values()) * (g["padded_reps"] + g["equal_reps"])
    mr = g["more_reps"]
    per_more = mr["alone"] * t_a + sum(tb.values()) * (mr["padded"] + mr["equal"])
    k = g["thinking_max_tokens"] / g["det_max_tokens"]
    per_think = g["thinking_reps"] * k * (t_a + sum(tb.get(B, max(tb.values())) for B in g["thinking_batches"]))
    est_det = len(g["det_models"]) * len(g["targets"]) * per_target + len(g["more_targets"]) * per_more + per_think + (2 + g["floor_orders"]) * len(rows) * t_q
    print(f"  one question: {t_q:.2f} s   one instruction prompt: {t_i:.1f} s ({rate:.0f} tokens/s)   one system-prompt message, all {len(cells)} cells: {t_e:.0f} s")
    print(f"  {g['det_max_tokens']} tokens alone: {t_a:.1f} s; in a batch: " + ", ".join(f"B = {B}: {s:.1f} s" for B, s in tb.items()))
    print(f"\n  {BOLD}estimate at bf16 speed{RESET} (quantised rows run faster):")
    print(f"    (c) det    {est_det / 60:5.0f} min   ({len(g['det_models'])} models × {len(g['targets'])} prompt, {len(g['more_targets'])} more on bf16, the thinking prompt, the floor)")
    print(f"    (a) bits   {est_bits / 60:5.0f} min   ({n_rows} rows × ({len(rows)} questions + {len(items)} prompts))")
    print(f"    (b) evict  {est_evict / 60:5.0f} min   ({len(g['orders'])} orders × {len(msgs)} messages × {len(cells)} cells)")
    write_json(out_dir(args) / "probe.json", {**stamp(args), "seconds": {"question": t_q, "instruction": t_i, "evict_message": t_e, "det_alone": t_a, "det_batch": tb},
                                              "estimate_minutes": {"det": est_det / 60, "bits": est_bits / 60, "evict": est_evict / 60}})


def cmd_test(args, cfg):
    fails = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if cond else 1
        print(f"  {GREEN + 'ok  ' if cond else RED + 'FAIL'}{RESET} {label}" + (f"   {GREY}{detail}{RESET}" if detail else ""))

    banner("test — the rules of (a): one instruction, one check")
    p, prompts = cfg["instructions"], load_prompts()
    task = {"avoid": "heat"}
    cases = [("words", "A short reply of six words.", True), ("words", "word " * 41, False), ("zh_tw", "天空是藍色的，因為短波長的光比較容易被散射。", True),
             ("zh_tw", "The sky is blue because of scattering.", False), ("json", '{"answer": "x", "confidence": 0.9}', True),
             ("json", '```json\n{"answer": "x", "confidence": 0.9}\n```', True), ("json", '{"answer": "blue light scatters more"}', False), ("json", "Sure! Here it is: {}", False),
             ("bullets", "- one thing\n- two things\n- three things", True), ("bullets", "- one thing\n- two things", False),
             ("avoid", "It moves warmth from inside to outside.", True), ("avoid", "It pumps heat out.", False), ("avoid", "It uses a heater.", True),
             ("ending", "Blue light scatters more. " + p["ending_sentence"], True), ("ending", p["ending_sentence"] + " Bye.", False), ("avoid", "…", False)]
    for kind, text, want in cases:
        got = K.follows(kind, text, task, p)
        check(f"{kind:<8} {text[:44]!r:<48} → {'followed' if want else 'not followed'}", got["ok"] == want, got["why"])
    t2s, s2t = converters()
    if t2s:
        check("zh_tw    a Simplified reply is not Traditional Chinese", not K.follows("zh_tw", "天空是蓝色的，因为短波长的光比较容易被散射。", task, p, t2s, s2t)["ok"])
    else:
        note("  （沒有 opencc：繁簡分不出來的那一條跳過）")
    check("every task has a word to avoid, and its prompt can be built for all six kinds", len(instruction_items(cfg, prompts, {"n_tasks": 10 ** 6})) == 6 * len(prompts["task"]))

    banner("test — flips, eviction, ROUGE-L, divergence")
    f = K.flips([1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 1])
    check("flips: same accuracy (4 of 6), yet four answers changed sides", f == {"n10": 2, "n01": 2, "flips": 4, "net": 0}, str(f))
    keep = K.keep_sink_window(40, (3, 33), 0.5, 4)
    check("sink + window: evicts round(r·|span|) tokens, right after the sink", len(keep) == 25 and keep[:5] == [0, 1, 2, 3, 19], f"kept {len(keep)} of 40")
    check("sink + window: r = 0 keeps everything", K.keep_sink_window(40, (3, 33), 0.0, 4) == list(range(40)))
    keep = K.keep_by_score(10, (2, 8), 0.5, [5, 1, 9, 2, 8, 3])
    check("by score: the three largest scores in the span go", keep == [0, 1, 3, 5, 7, 8, 9], str(keep))
    check("kept share of a part", K.kept_share(keep, (2, 5)) == 1 / 3)
    s = prompts["system"]
    full, spans = system_text(prompts, "directive_first")
    check("ROUGE-L recall: the prompt against itself is 1; against an unrelated reply, near 0",
          K.rouge_l_recall(full, full) == 1.0 and K.rouge_l_recall(full, "Blue light scatters more than red light.") < 0.05)
    check("ROUGE-L recall: a refusal that ends politely scores low", K.rouge_l_recall(full, "抱歉，這個我無法提供。" + s["closing"]) < 0.35,
          f"{K.rouge_l_recall(full, '抱歉，這個我無法提供。' + s['closing']):.2f}")
    check("the three parts are found again inside the system prompt, for both orders",
          all(system_text(prompts, o)[0][a:b] == s[n] for o in ("directive_first", "defense_first") for n, (a, b) in system_text(prompts, o)[1].items()))
    check("the canary and the closing sentence are inside the parts they belong to", s["canary"] in s["role"] and s["closing"] in s["directive"])
    check("ends with the sentence: quotation marks after it pass the loose rule, not the strict one",
          K.ends_with("好。「" + s["closing"] + "」", s["closing"]) and not K.ends_with("好。「" + s["closing"] + "」", s["closing"], False)
          and not K.ends_with(s["closing"] + "還有別的。", s["closing"]))
    check("a refusal is short: quoting the whole prompt (which contains the refusal) is not one",
          K.refused("抱歉，這個我無法提供。" + s["closing"], "無法提供", 40) and not K.refused(full, "無法提供", 40))
    check("outputs grouped by pad length", K.pad_groups([[1], [1], [2]], [0, 0, 3]) == {"by_pad": {"0": 1, "3": 1}, "pads": 2, "pads_with_several": 0})
    check("first divergence, distinct outputs", K.first_divergence([1, 2, 3], [1, 2, 4]) == 2 and K.first_divergence([1, 2], [1, 2]) is None
          and K.first_divergence([1, 2], [1, 2, 3]) == 2 and K.distinct([[1, 2], [1, 2], [1, 3]]) == 2)

    try:
        import mlx.core as mx  # noqa: F401
    except Exception as e:
        note(f"\n（沒有 mlx：亂數小模型的那幾項跳過——{type(e).__name__}）")
        print(f"\n  {(GREEN + 'all passed') if not fails else (RED + str(fails) + ' failed')}{RESET}")
        sys.exit(1 if fails else 0)

    banner("test — on a tiny random model: the ways of generating agree where they must")
    targs = argparse.Namespace(**{**vars(args), "tiny": True})
    g = settings(cfg, targs)
    lab = Lab(cfg, targs, "bf16")
    chat = lab.chat(prompts["det"][0]["text"])
    ids = chat["ids"]
    a, b = lab.greedy(ids, 24, want_gaps=True), lab.greedy(ids, 24)
    check("alone, twice: the same tokens", a["tokens"] == b["tokens"])
    check("a gap and a runner-up are recorded for every token", len(a["gaps"]) == len(a["tokens"]) == len(a["second"]) and min(a["gaps"]) >= 0)
    from mlx_lm.generate import generate_step
    theirs = []
    for tok, _ in generate_step(mx.array(ids), lab.model, max_tokens=24):
        if int(tok) in lab.eos:
            break
        theirs.append(int(tok))
    check("our loop = mlx-lm's generate_step", theirs == a["tokens"])
    bg = lab.batch([ids], 24, 1)[0]
    i = K.first_divergence(a["tokens"], bg)
    print(f"  {GREY}info{RESET} BatchGenerator at B = 1 vs our loop: {'the same' if i is None else 'differ from token ' + str(i)} (it prefills all but the last token first)")
    row = lab.equal_length(chat, "another text of a different length entirely")
    check("an equal-length companion has the same length and the same template", len(row) == len(ids) and row[:chat["user"][0]] == ids[:chat["user"][0]] and row != ids)
    eq = lab.greedy_equal([ids, ids, ids], 24)
    check("a batch of three copies: three identical rows", eq[0] == eq[1] == eq[2])
    pad = lab.batch([ids, ids[:3] + ids[9:], lab.chat("short")["ids"]], 24, 3)
    check("BatchGenerator with prompts of three lengths returns three outputs", len(pad) == 3 and all(isinstance(r, list) for r in pad))
    pool, items = mmlu_pool(cfg, targs, g), instruction_items(cfg, prompts, g)
    m = measure_mmlu_alone(lab, pool, g, "tiny")
    check("the questions run and are scored", m["n"] == g["n_mmlu"] and len(m["right"]) == m["n"])
    texts = bpb_texts(cfg, targs)
    bp = measure_bpb(lab, cfg, texts)
    check("bits per byte is a positive number for every text", all(v["bpb"] > 0 for v in bp.values()), ", ".join(f"{k} {v['bpb']:.2f}" for k, v in bp.items()))
    ev = EvictLab(lab, cfg, g)
    full, part_chars = system_text(prompts, "directive_first")
    c = lab.chat(prompts["task"][0]["text"], system=full)
    check("the system span covers exactly the system prompt", lab.decode(c["ids"][c["system"][0]:c["system"][1]]) == full)
    pf = ev.prefill(c, part_chars)
    base, _ = ev.generate(pf, ("none", 0.0), 16)
    for pol in g["policies"]:
        r0, _ = ev.generate(pf, (pol, 0.0), 16)
        check(f"{pol}: evicting nothing gives the uncompressed output", r0["tokens"] == base["tokens"])
        r9, st = ev.generate(pf, (pol, 0.9), 16)
        check(f"{pol}: r = 0.9 evicts 90% of the span and only the span", st["evicted"] == K.n_evict(pf["span"][1] - pf["span"][0], 0.9)
              and abs(mean(list(st["kept_share"].values())) - 0.1) < 0.12, f"kept share {st['kept_share']}")
    whole = lab.greedy(c["ids"], 16)
    i = K.first_divergence(whole["tokens"], base["tokens"])
    print(f"  {GREY}info{RESET} prefill all but the last token, then one step — vs the whole prompt at once: {'the same' if i is None else 'differ from token ' + str(i)}")
    kv, _ = ev.generate(pf, ("kv", 4.0), 16)
    check("the control (KV at 4 bits) generates", isinstance(kv["tokens"], list))
    q8 = Lab(cfg, targs, "bf16", ("affine", 8, 32))
    q2 = Lab(cfg, targs, "bf16", ("affine", 2, 32))
    check("quantising in memory: fewer bits per weight, in order", lab.info["bits_per_weight"] > q8.info["bits_per_weight"] > q2.info["bits_per_weight"],
          f"{lab.info['bits_per_weight']} > {q8.info['bits_per_weight']} > {q2.info['bits_per_weight']}")
    b8, b2 = measure_bpb(q8, cfg, texts), measure_bpb(q2, cfg, texts)
    k0 = texts[0]["key"]
    check("8 bits stays close to the original on the ruler; 2 bits moves further", abs(b8[k0]["bpb"] - bp[k0]["bpb"]) < abs(b2[k0]["bpb"] - bp[k0]["bpb"]),
          f"{bp[k0]['bpb']:.3f} / {b8[k0]['bpb']:.3f} / {b2[k0]['bpb']:.3f}")
    print(f"\n  {(GREEN + 'all passed') if not fails else (RED + str(fails) + ' failed')}{RESET}")
    sys.exit(1 if fails else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiny", action="store_true", help="亂數權重的小模型（不下載）；寫到 runs/smoke/")
    p.add_argument("--online", action="store_true", help="允許連網（預設只用本機快取）")
    p.add_argument("--part", choices=["det", "bits", "evict"], help="rehearse、rescore 只做一段")
    p.add_argument("--only", default=None, help="rehearse --part bits：只做這幾列（8,4）；--part det：只做這幾個模型（bf16,q4）")
    p.add_argument("--force", action="store_true", help="rehearse：做過的也重跑")
    p.add_argument("--note", default=None, help="彩排當下的狀態（電源、其他程式），記進紀錄")
    p.add_argument("--model", default=None, help="live-det：bf16 或 q4")
    p.add_argument("--batch", default=None, help="live-det：併成多大的 batch")
    p.add_argument("--mode", choices=["padded", "equal"], default=None, help="live-det：哪一種併法")
    p.add_argument("--prompt", default=None, help="live-det：prompts.toml 的 [[det]] id")
    p.add_argument("--fixed", action="store_true", help="live-det：同伴固定（預設每次不同）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("fetch", "test", "inspect", "probe", "rehearse", "rescore", "show", "live-det", "replay", "replay-bits", "replay-evict", "replay-det"):
        sub.add_parser(c)
    args = p.parse_args(argv)
    cfg = load_cfg()
    {"fetch": cmd_fetch, "test": cmd_test, "inspect": cmd_inspect, "probe": cmd_probe, "rehearse": cmd_rehearse, "rescore": cmd_rescore, "show": cmd_live_det, "live-det": cmd_live_det,
     "replay": cmd_replay, "replay-bits": cmd_replay_bits, "replay-evict": cmd_replay_evict, "replay-det": cmd_replay_det}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
