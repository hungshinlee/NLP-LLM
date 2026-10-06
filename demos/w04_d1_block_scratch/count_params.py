#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 1 後半（投影片 p20「Live: invert config.json — three columns, two models」）：讀一個模型的 config.json，印三欄。

    ./present.sh count llama              # 三欄：近似式 12·n_layer·d² + V·d（或 2V·d）／逐項精算／safetensors 實際
    ./present.sh count qwen
    ./present.sh count llama --config-only   # 只印七個鍵，不印答案（學生先用近似式算第一欄、寫白板）
    python3 count_params.py /path/to/snapshot   # 任何有 config.json 與 *.safetensors 的資料夾

三欄的意思：
  approx  = 黑板上的近似式。embedding 共享時 + V·d，不共享時 + 2V·d（看 tie_word_embeddings）
  exact   = 逐項精算：embedding + Σ_layers [W^Q, W^O 各 d·n_head·d_head；W^K, W^V 各 d·n_kv·d_head；FFN 3·d·d_ff；兩個 Norm 2d；
            （若有）attention bias、q_norm／k_norm] + 最後一個 Norm + （不共享時）unembedding。全部只看 config.json
  actual  = 把每個 *.safetensors 的 header 讀出來（只讀 header，不載權重），把形狀乘開加總。
            MLX 的 4-bit 檔把 32/bits 個權重包成一個 uint32：weight 的元素數要乘回 32/bits；.scales 與 .biases 是量化的附屬張量，
            另外列成「quantization overhead」，不算進參數量（它們不是模型的參數，是壓縮格式的 metadata）。
exact 應該等於 actual——差一個都是我們漏了什麼；差了就逐類別印出來（embedding／attention／ffn／norm／head／other），
看漏在哪一類。近似式與精算之間的缺口才是下一張（p21）的四個修正。

純標準函式庫（safetensors 的 header 自己讀：前 8 bytes 是 header 長度，接著是 JSON）；找本機快取用 huggingface_hub（共用 .venv 有）。
"""
from __future__ import annotations

import json
import re
import struct
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CONFIG = HERE / "demo_config.toml"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m"
KEYS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "tie_word_embeddings")


class NotLocal(Exception):
    pass


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


# ── 找到模型資料夾 ───────────────────────────────────────────────
def resolve(name_or_path: str) -> tuple[Path, str]:
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return p, str(p)
    models = load_cfg()["models"]
    if name_or_path not in models:
        raise NotLocal(f"不認得 {name_or_path!r}：給 demo_config.toml 裡的名字（{', '.join(models)}）或一個資料夾")
    repo = models[name_or_path]["repo"]
    from common import locked_revision                      # noqa: E402
    from huggingface_hub import snapshot_download
    rev = locked_revision(repo)
    try:
        path = snapshot_download(repo, revision=rev or None, local_files_only=True,
                                 ignore_patterns=["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "*.gguf", "*.onnx", "*.msgpack", "*.h5"])
    except Exception as exc:                                 # 離線找不到：不是這個 demo 的事，W3 demo 2／3 負責 fetch
        raise NotLocal(f"{repo} 不在本機快取（{type(exc).__name__}）；W3 demo 2／demo 3 的 fetch 會下載它，並確認 HF_HOME 與 ../README.md 一致") from exc
    return Path(path), repo


# ── config.json 的七個鍵（+ 幾個決定「精算」要加什麼的旗標）──────────
def read_config(path: Path) -> dict:
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tc = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}

    def get(*keys, default=None):
        for k in keys:
            for src in (raw, tc):
                if k in src and src[k] is not None:
                    return src[k]
        return default

    d = get("hidden_size", "d_model")
    n_head = get("num_attention_heads", "n_head")
    c = {"model_type": get("model_type"),
         "hidden_size": d, "intermediate_size": get("intermediate_size", "d_ff"),
         "num_hidden_layers": get("num_hidden_layers", "n_layer"), "num_attention_heads": n_head,
         "num_key_value_heads": get("num_key_value_heads", default=n_head),
         "head_dim": get("head_dim") or (d // n_head), "head_dim_in_config": get("head_dim") is not None,
         "vocab_size": get("vocab_size"), "tie_word_embeddings": bool(get("tie_word_embeddings", default=False)),
         "attention_bias": bool(get("attention_bias", default=False)),
         "mlp_bias": bool(get("mlp_bias", default=False)),
         "quantization": raw.get("quantization")}
    # 精算要加的「這個模型多出來的小東西」：只從 config 判斷，判斷不到的會在 exact ≠ actual 時以類別差額出現
    c["qk_norm"] = c["model_type"] in ("qwen3", "qwen3_moe", "olmo2")        # 每層 q_norm、k_norm 各 d_head 個 γ
    return c


# ── 逐項精算（只看 config）───────────────────────────────────────
def exact_ledger(c: dict) -> dict:
    d, dff, n, nh, nkv, dh, V = (c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"],
                                 c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"])
    rows = [("embedding", "V·d", V * d)]
    per = [("attention", "W^Q, W^O: d·n_head·d_head each", 2 * d * nh * dh),
           ("attention", "W^K, W^V: d·n_kv·d_head each", 2 * d * nkv * dh),
           ("ffn", "gate, up, down: 3·d·d_ff", 3 * d * dff),
           ("norm", "two RMSNorm: 2d", 2 * d)]
    if c["attention_bias"]:
        per.append(("attention", "q, k, v bias: n_head·d_head + 2·n_kv·d_head", nh * dh + 2 * nkv * dh))
    if c["mlp_bias"]:
        per.append(("ffn", "ffn bias: 2·d_ff + d", 2 * dff + d))
    if c["qk_norm"]:
        per.append(("norm", "q_norm, k_norm: 2·d_head", 2 * dh))
    per_layer = sum(v for _, _, v in per)
    rows += [(cat, f"per layer  {desc}", v) for cat, desc, v in per]
    rows.append(("layers", f"× {n} layers", per_layer * n))
    rows.append(("norm", "final RMSNorm: d", d))
    if not c["tie_word_embeddings"]:
        rows.append(("head", "unembedding: V·d (untied)", V * d))
    by_cat = {"embedding": V * d, "attention": n * sum(v for cat, _, v in per if cat == "attention"),
              "ffn": n * sum(v for cat, _, v in per if cat == "ffn"),
              "norm": n * sum(v for cat, _, v in per if cat == "norm") + d,
              "head": 0 if c["tie_word_embeddings"] else V * d, "other": 0}
    exact = sum(by_cat.values())
    approx = 12 * n * d * d + (V * d if c["tie_word_embeddings"] else 2 * V * d)
    return {"rows": rows, "per_layer": per_layer, "by_cat": by_cat, "exact": exact, "approx": approx,
            "approx_rel_err": (approx - exact) / exact,
            "approx_formula": f"12·{n}·{d}² + {'V·d' if c['tie_word_embeddings'] else '2V·d'}"}


# ── safetensors：只讀 header ─────────────────────────────────────
def read_headers(path: Path) -> dict[str, dict]:
    tensors = {}
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{path} 裡沒有 *.safetensors")
    for f in files:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n).decode("utf-8"))
        for k, v in header.items():
            if k != "__metadata__":
                tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": f.name}
    return tensors


def _bits_for(name: str, quant) -> int | None:
    """MLX 的 config.quantization：{"group_size", "bits"} 之外可能有逐模組的覆蓋（鍵是模組路徑）。"""
    if not isinstance(quant, dict):
        return None
    mod = name.rsplit(".", 1)[0]
    ov = quant.get(mod)
    if isinstance(ov, dict) and "bits" in ov:
        return int(ov["bits"])
    if ov is False:
        return None
    return int(quant.get("bits", 4))


CAT = [("embedding", re.compile(r"embed_tokens|wte|tok_embeddings")),
       ("head", re.compile(r"lm_head|output\.weight$")),
       ("norm", re.compile(r"norm")),                      # 先於 attention：post_attention_layernorm、self_attn.q_norm 都歸 norm（與 exact_ledger 的分類一致）
       ("attention", re.compile(r"self_attn|attention|attn")),
       ("ffn", re.compile(r"mlp|feed_forward|ffn"))]


def categorize(name: str) -> str:
    for cat, rx in CAT:
        if rx.search(name):
            return cat
    return "other"


def actual_ledger(tensors: dict, quant, tied: bool = False) -> dict:
    by_cat = {"embedding": 0, "attention": 0, "ffn": 0, "norm": 0, "head": 0, "other": 0}
    overhead, overhead_elems, n_quant, uncategorized, tied_dup = 0, 0, 0, [], 0
    for name, t in tensors.items():
        elems = 1
        for s in t["shape"]:
            elems *= s
        if tied and categorize(name) == "head":
            # tie_word_embeddings 為 true 但檔裡仍存了 lm_head（Qwen3-0.6B 就是這樣，2026-10-06 實測）：
            # 同一個矩陣存了兩份，不是兩組參數。另外記下來、不算進 actual。
            tied_dup += elems
            continue
        if name.endswith(".scales") or name.endswith(".biases"):
            base = name.rsplit(".", 1)[0] + ".weight"
            if base in tensors and tensors[base]["dtype"] in ("U32", "I32"):
                overhead += elems                                        # 量化的附屬張量：不是參數
                continue
        bits = _bits_for(name, quant) if t["dtype"] in ("U32", "I32") and (name.rsplit(".", 1)[0] + ".scales") in tensors else None
        if bits:
            elems *= 32 // bits                                           # 一個 uint32 裝 32/bits 個權重
            n_quant += 1
        cat = categorize(name)
        if cat == "other":
            uncategorized.append(name)
        by_cat[cat] += elems
    return {"by_cat": by_cat, "actual": sum(by_cat.values()), "quant_overhead_elems": overhead,
            "n_tensors": len(tensors), "n_quantized_tensors": n_quant, "uncategorized": uncategorized,
            "has_lm_head": any(categorize(n) == "head" for n in tensors), "tied_head_stored_elems": tied_dup}


# ── 報告 ─────────────────────────────────────────────────────────
def report(name_or_path: str, config_only: bool = False, quiet: bool = False) -> dict:
    path, repo = resolve(name_or_path)
    c = read_config(path)
    out = {"model": repo, "path": str(path), "config": {k: c[k] for k in KEYS}, "model_type": c["model_type"],
           "head_dim_in_config": c["head_dim_in_config"], "attention_bias": c["attention_bias"], "qk_norm_assumed": c["qk_norm"],
           "quantization": c["quantization"]}
    if not quiet:
        print(f"{BOLD}{repo}{RESET}  {DIM}{path}{RESET}")
        print(f"  config.json ({c['model_type']}):")
        for k in KEYS:
            extra = "" if k != "head_dim" or c["head_dim_in_config"] else "  (not in config; hidden_size / num_attention_heads)"
            print(f"    {k:<22} {c[k]!s:>8}{extra}")
        nh_dh = c["num_attention_heads"] * c["head_dim"]
        print(f"    {'n_head × d_head':<22} {nh_dh:>8}  {'= d' if nh_dh == c['hidden_size'] else f'≠ d ({nh_dh / c['hidden_size']:.2g}·d)'}"
              f"   d_ff / d = {c['intermediate_size'] / c['hidden_size']:.3g}   n_kv / n_head = {c['num_key_value_heads'] / c['num_attention_heads']:.3g}")
        if c["quantization"]:
            q = c["quantization"]
            print(f"    {'quantization':<22} {q.get('bits', '?')}-bit, group {q.get('group_size', '?')}  (MLX; shapes are the original model's)")
    if config_only:
        return out
    ex = exact_ledger(c)
    ac = actual_ledger(read_headers(path), c["quantization"], tied=c["tie_word_embeddings"])
    out.update({"approx": ex["approx"], "approx_formula": ex["approx_formula"], "approx_rel_err": ex["approx_rel_err"],
                "exact": ex["exact"], "actual": ac["actual"], "exact_minus_actual": ex["exact"] - ac["actual"],
                "exact_by_cat": ex["by_cat"], "actual_by_cat": ac["by_cat"], "per_layer_exact": ex["per_layer"],
                "quant_overhead_elems": ac["quant_overhead_elems"], "n_tensors": ac["n_tensors"],
                "n_quantized_tensors": ac["n_quantized_tensors"], "has_lm_head_tensor": ac["has_lm_head"],
                "tied_head_stored_elems": ac["tied_head_stored_elems"],
                "uncategorized_tensors": ac["uncategorized"]})
    if not quiet:
        print(f"\n  {BOLD}{'approx  ' + ex['approx_formula']:<40}{ex['approx']:>16,}{RESET}   {DIM}≈ {ex['approx'] / 1e9:.3f} B{RESET}")
        print(f"  {BOLD}{'exact   term by term':<40}{ex['exact']:>16,}{RESET}   {DIM}≈ {ex['exact'] / 1e9:.3f} B{RESET}")
        print(f"  {BOLD}{'actual  safetensors':<40}{ac['actual']:>16,}{RESET}   {DIM}≈ {ac['actual'] / 1e9:.3f} B  ({ac['n_tensors']} tensors"
              f"{f', {ac['n_quantized_tensors']} quantized, +{ac['quant_overhead_elems']:,} scale/bias values not counted' if ac['n_quantized_tensors'] else ''}){RESET}")
        same = ex["exact"] == ac["actual"]
        print(f"  approx − exact = {ex['approx'] - ex['exact']:+,} ({ex['approx_rel_err']:+.2%})     "
              f"exact − actual = {GREEN + '0 — 差零個' + RESET if same else RED + f'{ex['exact'] - ac['actual']:+,}' + RESET}")
        print(f"\n  {DIM}exact, term by term:{RESET}")
        for _, desc, v in ex["rows"]:
            print(f"    {desc:<46} {v:>16,}")
        print(f"\n  {DIM}by category          exact          actual{RESET}")
        for cat in ("embedding", "attention", "ffn", "norm", "head", "other"):
            e, a = ex["by_cat"][cat], ac["by_cat"][cat]
            flag = "" if e == a else f"   {RED}← {e - a:+,}{RESET}"
            print(f"    {cat:<12} {e:>16,} {a:>16,}{flag}")
        if ac["uncategorized"]:
            print(f"  {AMBER}tensors not in any category: {', '.join(ac['uncategorized'][:8])}{'…' if len(ac['uncategorized']) > 8 else ''}{RESET}")
        if ac["tied_head_stored_elems"]:
            print(f"  {AMBER}tie_word_embeddings is true, yet the file also stores lm_head ({ac['tied_head_stored_elems']:,} values = V·d): "
                  f"the same matrix saved twice, counted once — the file is {ac['tied_head_stored_elems'] / (ac['actual'] + ac['tied_head_stored_elems']):.0%} larger than the parameter count{RESET}")
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    try:
        report(args[0], config_only="--config-only" in argv)
    except NotLocal as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
