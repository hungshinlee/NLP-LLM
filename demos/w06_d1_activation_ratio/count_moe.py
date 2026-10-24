#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 1（投影片 p12「Live: demo 1 — N_total, N_active, ρ from the config」）：從 config.json 算 MoE 的 activation ratio。

    ./present.sh keys qwen30          # 印 config 裡所有含 expert／moe／shared／router 的鍵（含 text_config 底下的），再說這支程式怎麼解讀它們
    ./present.sh count                # 四列（demo_config.toml 的順序）：逐項記帳 → N_total、N_active、ρ；config 的式子對 safetensors 的實際形狀
    ./present.sh count qwen30 kimi    # 只算幾個
    ./present.sh floor qwen30         # 地板：attention（含 linear／SSM 層）+ norm + embedding 佔 N_active 的比例——接 p14 的式子
    ./present.sh rehearse             # count 全部 + 寫 runs/rehearsal/ratios.json
    ./present.sh replay               # 印彩排紀錄（畫面標明非現場）

沿 W4 demo 1 的 count_params.py（三欄：近似式／逐項精算／safetensors 實際）加 MoE 分支。這裡的兩欄是：
  config  = 只看 config.json 的逐項記帳（attention 側沿 W4 的四個投影；expert 側 (n_E + n_S)·3·d·d_E + router n_E·d；embedding；norm）
  file    = 把每個 safetensors 的 header 讀出來（只讀 header，不載權重；遠端的由 fetch_configs.py 用 range 請求抓回來），依張量名稱分類後把形狀乘開
兩欄相等是「config 說的就是檔案裡的」；不等的地方是 config 沒說的事（Gemma 4 的 dense FFN 與 MoE 並存、Kimi 的 KDA 低秩投影、
Qwen3-Next checkpoint 裡的 MTP 模組）——那些差額就是這張投影片要學生看的東西。

N_active 不能從檔案直接讀（檔案不知道每個 token 選幾個 expert）：N_active = N_total − experts_total + experts_total · k / n_E，
k 與 n_E 來自 config——**而且各家鍵名不同**（Qwen `num_experts_per_tok`、Kimi `num_experts_per_token`、Gemma 4 `top_k_experts`）。
moe_spec() 列出它認得的鍵；認不得的印出來給人看，不猜。

純標準函式庫。本機模型（demo_config.toml 的 local = true）要找 HF cache：有 huggingface_hub 就用它問快取位置，沒有就看 HF_HOME。
"""
from __future__ import annotations

import gzip
import json
import os
import re
import struct
import sys
import tomllib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                      # demos/common.py（鎖定檔）
CONFIG = HERE / "demo_config.toml"
SNAPSHOTS = HERE.parent.parent / "slides" / "assets" / "w06" / "configs"      # 投影片用的快照（刪過鍵，只當最後的退路）
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY, BLUE = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[34m"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "U32": 4, "I32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
               "U8": 1, "I8": 1, "BOOL": 1, "U16": 2, "I16": 2}


def load_cfg() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def fmt(n: int | float | None, b: bool = False) -> str:
    if n is None:
        return "—"
    return f"{n / 1e9:.3f} B" if b else f"{int(n):,}"


# ══════════════════════════════════════════════════════════════
# 1. config 的來源與讀法
# ══════════════════════════════════════════════════════════════
class Source:
    def __init__(self, name: str, repo: str, kind: str, path: Path, raw: dict, tensors: dict | None, quant, index_total: int | None, note: str):
        self.name, self.repo, self.kind, self.path, self.raw, self.tensors, self.quant, self.index_total, self.note = (
            name, repo, kind, path, raw, tensors, quant, index_total, note)


def flatten(raw: dict, prefix: str = "") -> dict:
    """巢狀 dict 攤平成 'text_config.num_experts' 這種 dotted key；list 原樣保留。"""
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and k != "quantization":
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def get(flat: dict, *keys, default=None):
    """先找根層，再找 text_config 底下（Gemma 4 這類多模態 config 把文字模型包在 text_config）。回 (值, 用到的鍵)。"""
    for k in keys:
        for cand in (k, "text_config." + k):
            if cand in flat and flat[cand] is not None:
                return flat[cand], cand
    return default, None


def hf_cache_dir(repo: str) -> Path | None:
    """本機 HF cache 裡這個 repo 的 snapshot 資料夾。revision 取 ../versions.lock；沒鎖就取唯一的那個 snapshot。"""
    try:
        from common import locked_revision                      # noqa: E402
        rev = locked_revision(repo)
    except Exception:                                           # noqa: BLE001
        rev = None
    try:
        from huggingface_hub import constants
        hub = Path(constants.HF_HUB_CACHE)
    except Exception:                                           # noqa: BLE001
        hub = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    snaps = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    if rev and (snaps / rev / "config.json").exists():
        return snaps / rev
    cands = [p for p in snaps.iterdir() if (p / "config.json").exists()]
    return cands[0] if len(cands) == 1 else (sorted(cands)[-1] if cands else None)


def read_local_headers(path: Path) -> dict[str, dict]:
    """safetensors：前 8 bytes 是 header 長度，接著是 JSON。只讀 header，不載權重（與 W4 count_params.py 相同）。"""
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


def resolve(name: str, cfg: dict) -> Source:
    models = cfg["models"]
    if name not in models:
        raise KeyError(f"不認得 {name!r}：demo_config.toml 裡的名字是 {', '.join(models)}")
    m = models[name]
    repo = m["repo"]
    if m.get("local"):
        d = hf_cache_dir(repo)
        if d is None:
            raise FileNotFoundError(f"{repo} 不在本機 HF cache（HF_HOME={os.environ.get('HF_HOME', '未設')}）。它是 W1 demo 1 的模型；"
                                    f"在同一種 shell 裡跑 ../setup.sh --only w01_d1_live_model，見 ../README.md")
        raw = json.loads((d / "config.json").read_text(encoding="utf-8"))
        tensors = read_local_headers(d)
        return Source(name, repo, "local", d, raw, tensors, raw.get("quantization"), None, m.get("note", ""))
    fetched = HERE / "configs" / repo.replace("/", "--")
    if (fetched / "config.json").exists():
        raw = json.loads((fetched / "config.json").read_text(encoding="utf-8"))
        tensors, total = None, None
        tj = fetched / "tensors.json.gz"
        if tj.exists():
            with gzip.open(tj, "rt", encoding="utf-8") as f:
                tensors = json.load(f)
            meta = tensors.pop("__meta__", {})
            total = meta.get("index_total_size_bytes")
        return Source(name, repo, "fetched", fetched, raw, tensors, None, total, m.get("note", ""))
    snap = SNAPSHOTS / (repo.replace("/", "--") + ".json")
    if snap.exists():
        raw = json.loads(snap.read_text(encoding="utf-8"))
        total = raw.get("_safetensors_total_size")
        return Source(name, repo, "snapshot", snap, raw, None, None, total, m.get("note", ""))
    raise FileNotFoundError(f"{repo}：沒有 configs/{fetched.name}/（先 ./present.sh fetch），投影片的快照 {snap.name} 也不在")


# ══════════════════════════════════════════════════════════════
# 2. 解讀 config：基本形狀、MoE 的鍵、層型
# ══════════════════════════════════════════════════════════════
N_E_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "num_routed_experts", "moe_num_experts")
K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_topk", "moe_top_k", "num_selected_experts", "experts_per_token")
D_E_KEYS = ("moe_intermediate_size", "expert_intermediate_size", "moe_ffn_hidden_size")
N_S_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts")
D_S_KEYS = ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size")


def base_spec(flat: dict) -> dict:
    d, _ = get(flat, "hidden_size", "d_model")
    nh, _ = get(flat, "num_attention_heads", "n_head")
    nkv, _ = get(flat, "num_key_value_heads", default=nh)
    dh, dh_key = get(flat, "head_dim")
    mt, _ = get(flat, "model_type")
    return {"model_type": mt, "d": d, "d_ff": get(flat, "intermediate_size", "d_ff")[0], "n_layer": get(flat, "num_hidden_layers", "n_layer")[0],
            "n_head": nh, "n_kv": nkv, "d_head": dh if dh else (d // nh if d and nh else None), "head_dim_in_config": dh_key is not None,
            "V": get(flat, "vocab_size")[0], "tie": bool(get(flat, "tie_word_embeddings", default=False)[0])}


def moe_spec(flat: dict) -> dict:
    """MoE 的五個量與它們各自來自哪個鍵。鍵名沒有標準——這個函數就是「各家怎麼叫」的清單。"""
    used, s = {}, {}
    for field, keys in (("n_E", N_E_KEYS), ("k", K_KEYS), ("d_E", D_E_KEYS)):
        v, key = get(flat, *keys)
        s[field], used[field] = v, key
    nS, nS_key = get(flat, *N_S_KEYS)
    dS, dS_key = get(flat, *D_S_KEYS)
    if dS and not nS:                                      # Qwen：只給 shared 的寬，數量就是 1
        nS, nS_key = 1, dS_key + " (width given → one shared expert)"
    if nS and not dS:                                      # DeepSeek／Kimi：n 個 shared 合成一個寬 n·d_E 的 FFN
        dS, dS_key = (s["d_E"] or 0) * nS, f"{nS_key} × {used['d_E']} (DeepSeek-style: n_S shared experts of width d_E)"
    s["n_S"], s["d_S_total"] = (nS or 0), (dS or 0)
    used["n_S"], used["d_S_total"] = nS_key, dS_key
    fd, fd_key = get(flat, "first_k_dense_replace", default=0)
    mlp_only, mo_key = get(flat, "mlp_only_layers", default=[])
    step, step_key = get(flat, "decoder_sparse_step", default=1)
    s["first_k_dense"], used["first_k_dense"] = (fd or 0), fd_key
    s["mlp_only_layers"], used["mlp_only_layers"] = list(mlp_only or []), mo_key
    s["sparse_step"], used["sparse_step"] = (step or 1), step_key
    s["moe_enabled"], used["moe_enabled"] = get(flat, "enable_moe_block", default=None)
    s["keys_used"] = used
    s["is_moe"] = bool(s["n_E"] and s["k"])
    return s


def layer_kinds(flat: dict, base: dict) -> list[str]:
    """每層是 attn／linear／attn+ssm。沿 make_demo_w06.py：Qwen3-Next 用 full_attention_interval，Kimi 用 linear_attn_config.full_attn_layers（1-indexed），
    其他有 layer_types 清單的照清單；都沒有就是全部 attention。"""
    nl = base["n_layer"]
    iv, _ = get(flat, "full_attention_interval")
    fa = flat.get("linear_attn_config.full_attn_layers")
    lt, _ = get(flat, "layer_types")
    if iv:
        return ["attn" if (i + 1) % iv == 0 else "linear" for i in range(nl)]
    if isinstance(fa, list):
        return ["attn" if (i + 1) in set(fa) else "linear" for i in range(nl)]
    if isinstance(lt, list) and len(lt) == nl:
        return ["attn" if "attention" in str(t) else ("attn+ssm" if "mamba" in str(t) and "attention" in str(t) else
                                                       "linear" if any(w in str(t) for w in ("linear", "mamba", "ssm", "recurrent")) else "attn") for t in lt]
    if base["model_type"] == "falcon_h1":
        return ["attn+ssm"] * nl
    return ["attn"] * nl


def moe_layers(spec: dict, nl: int) -> list[bool]:
    """第 i 層（0-indexed）是不是 MoE 層。"""
    if not spec["is_moe"]:
        return [False] * nl
    if spec["moe_enabled"] is False:
        return [False] * nl
    out = []
    for i in range(nl):
        is_moe = i >= spec["first_k_dense"] and i not in spec["mlp_only_layers"] and ((i + 1) % spec["sparse_step"] == 0)
        out.append(is_moe)
    return out


# ══════════════════════════════════════════════════════════════
# 3. config 那一欄：逐項記帳（只看 config）
# ══════════════════════════════════════════════════════════════
def attn_formula(flat: dict, base: dict) -> tuple[int | None, str]:
    """一層 attention 的投影。GQA／MHA：W^Q、W^O 各 d·n_head·d_head，W^K、W^V 各 d·n_kv·d_head（W4 p19 的式子）；
    MLA（DeepSeek-V3、Kimi 的 full attention 層）：低秩 q／kv 加 W^O。Gemma 4 的 sliding／global 兩種 head 形狀與 attention_k_eq_v 不試算——由檔案說。"""
    mt, d = base["model_type"], base["d"]
    if mt in ("deepseek_v3", "kimi_linear"):
        nh = base["n_head"]; qr = get(flat, "q_lora_rank")[0]; kr = get(flat, "kv_lora_rank")[0]
        dn, dr, dv = get(flat, "qk_nope_head_dim")[0], get(flat, "qk_rope_head_dim")[0], get(flat, "v_head_dim")[0]
        if None in (kr, dn, dr, dv):
            return None, "MLA keys missing"
        q = (d * qr + qr * nh * (dn + dr)) if qr else d * nh * (dn + dr)
        kv = d * (kr + dr) + kr * nh * (dn + dv)
        return q + kv + nh * dv * d, "MLA: q %s, kv low-rank (kv_lora_rank %d + rope %d), W^O" % ("low-rank" if qr else "full-rank", kr, dr)
    if mt == "gemma4" or mt == "gemma4_text" or "gemma" in str(mt):
        return None, "not attempted from config (sliding/global heads differ; attention_k_eq_v) — see the file column"
    if None in (base["n_head"], base["n_kv"], base["d_head"]):
        return None, "attention keys missing"
    v = 2 * d * base["n_head"] * base["d_head"] + 2 * d * base["n_kv"] * base["d_head"]
    kind = "GQA" if base["n_kv"] != base["n_head"] else "MHA"
    if mt == "qwen3_next":
        v += d * base["n_head"] * base["d_head"]        # gated attention：q_proj 另外輸出一份 gate（Qwen3-Next 的 output gate）
        kind += " + output gate (q_proj doubled)"
    return v, "%s: W^Q, W^O d·n_head·d_head; W^K, W^V d·n_kv·d_head" % kind


def linear_formula(flat: dict, base: dict) -> tuple[int | None, str]:
    """hybrid 的 linear／SSM 層。Qwen3-Next 的 Gated DeltaNet 照 HF 的張量：in_proj_qkvz、in_proj_ba、conv1d、A_log、dt_bias、norm、out_proj。
    Kimi 的 KDA 有低秩的 f_a／f_b、g_a／g_b 投影，config 沒給秩——不試算，由檔案說。"""
    mt, d = base["model_type"], base["d"]
    if mt == "qwen3_next":
        hv, hk = get(flat, "linear_num_value_heads")[0], get(flat, "linear_num_key_heads")[0]
        dv, dk = get(flat, "linear_value_head_dim")[0], get(flat, "linear_key_head_dim")[0]
        conv = get(flat, "linear_conv_kernel_dim", default=4)[0]
        if None in (hv, hk, dv, dk):
            return None, "GDN keys missing"
        qkvz = d * (2 * hk * dk + 2 * hv * dv)
        ba = d * 2 * hv
        conv1d = (2 * hk * dk + hv * dv) * conv
        small = hv + hv + dv                              # A_log、dt_bias、norm
        out = hv * dv * d
        return qkvz + ba + conv1d + small + out, "Gated DeltaNet: in_proj_qkvz, in_proj_ba, conv1d(%d), A_log, dt_bias, norm, out_proj" % conv
    if mt == "kimi_linear":
        return None, "KDA: low-rank f/g projections, rank not in config — see the file column"
    return None, "no formula for %s linear layers" % mt


def config_ledger(flat: dict, base: dict, spec: dict, kinds: list[str]) -> dict:
    d, V, nl = base["d"], base["V"], base["n_layer"]
    is_moe_layer = moe_layers(spec, nl)
    n_moe = sum(is_moe_layer)
    per_attn, attn_how = attn_formula(flat, base)
    per_lin, lin_how = linear_formula(flat, base)
    n_attn = sum(1 for k in kinds if "attn" in k)
    n_lin = sum(1 for k in kinds if k == "linear")
    attention = per_attn * n_attn if per_attn is not None else None
    linear = (per_lin * n_lin if per_lin is not None else None) if n_lin else 0
    qk_norm = base["model_type"] in ("qwen3", "qwen3_moe", "qwen3_next", "olmo2")
    norms = nl * 2 * d + d + (n_attn * 2 * base["d_head"] if qk_norm and base["d_head"] else 0)
    emb = V * d * (1 if base["tie"] else 2)
    if spec["is_moe"]:
        per_expert = 3 * d * spec["d_E"]
        experts_total = n_moe * spec["n_E"] * per_expert
        experts_active = n_moe * spec["k"] * per_expert
        shared = n_moe * 3 * d * spec["d_S_total"]
        router = n_moe * spec["n_E"] * d
        dense = (nl - n_moe) * 3 * d * base["d_ff"] if base["d_ff"] else 0
        dense_how = "%d dense layer(s): 3·d·d_ff" % (nl - n_moe) if nl - n_moe else "no dense FFN layer"
        if spec["moe_enabled"] and base["d_ff"]:
            dense_how += "; config also carries intermediate_size = %d — dense FFN beside the MoE, or not? the config cannot say" % base["d_ff"]
    else:
        per_expert = experts_total = experts_active = shared = router = 0
        dense = nl * 3 * d * base["d_ff"]; dense_how = "dense FFN every layer"
    known = [x for x in (attention, linear) if x is not None]
    total = sum(known) + norms + emb + experts_total + shared + router + dense
    active = sum(known) + norms + emb + experts_active + shared + router + dense
    incomplete = attention is None or linear is None
    return {"attention": attention, "attention_how": attn_how, "n_attn_layers": n_attn, "per_attn": per_attn,
            "linear": linear, "linear_how": lin_how, "n_linear_layers": n_lin, "per_linear": per_lin,
            "experts_total": experts_total, "experts_active": experts_active, "per_expert": per_expert, "n_moe_layers": n_moe,
            "shared": shared, "router": router, "dense": dense, "dense_how": dense_how, "norms": norms, "qk_norm_assumed": qk_norm,
            "embedding": emb, "N_total": None if incomplete else total, "N_active": None if incomplete else active,
            "incomplete": incomplete}


# ══════════════════════════════════════════════════════════════
# 4. file 那一欄：張量分類（名稱 + 層型），形狀乘開
# ══════════════════════════════════════════════════════════════
CATS = ("attention", "linear", "experts", "shared", "router", "dense", "norms", "embedding", "head", "other")
EXCLUDED = ("mtp", "towers")                              # 不算進 N：多 token 預測頭、視覺／音訊塔
RX = {
    "mtp": re.compile(r"(^|\.)mtp(\.|_)"),
    "towers": re.compile(r"vision|audio|image|siglip|multi_modal|mm_projector|embed_vision|embed_audio|visual\."),
    "embedding": re.compile(r"embed_tokens|wte|tok_embeddings|word_embeddings"),
    "head": re.compile(r"lm_head|(^|\.)output\.weight$"),
    "router": re.compile(r"(mlp|moe|block_sparse_moe|experts)\.gate\.(weight|bias)$|router|e_score_correction_bias|shared_expert_gate|expert_bias"),
    "shared": re.compile(r"shared_expert"),
    "experts": re.compile(r"expert|switch_mlp"),           # 剩下含 expert 的都是 routed（HF 逐個 experts.N.*；MLX 疊成 switch_mlp.*）
    "norm": re.compile(r"norm"),
    "linear_marker": re.compile(r"linear_attn|mamba|ssm|A_log|dt_bias|conv1d|f_a_proj|f_b_proj|g_a_proj|g_b_proj|\.b_proj|o_norm"),
    "attention": re.compile(r"self_attn|attention|\.attn\.|linear_attn"),
    "dense": re.compile(r"mlp\.(gate_proj|up_proj|down_proj|fc\d?|w\d)|feed_forward\.(w\d|gate_proj|up_proj|down_proj)"),
}
LAYER_RX = re.compile(r"layers\.(\d+)\.")


def categorize(name: str, kinds: list[str]) -> str:
    if RX["mtp"].search(name):
        return "mtp"
    if RX["towers"].search(name):
        return "towers"
    if RX["embedding"].search(name):
        return "embedding"
    if RX["head"].search(name):
        return "head"
    if RX["router"].search(name):
        return "router"
    if RX["shared"].search(name):
        return "shared"
    if RX["experts"].search(name):
        return "experts"
    m = LAYER_RX.search(name)
    kind = kinds[int(m.group(1))] if m and int(m.group(1)) < len(kinds) else "attn"
    if RX["linear_marker"].search(name):
        return "linear"                                    # GDN／KDA／Mamba 自己的張量（含它們的 norm）
    if RX["norm"].search(name):
        return "norms"                                     # input/post_attention/pre_feedforward…layernorm、q_norm、k_norm、kv_a_layernorm、final norm
    if RX["attention"].search(name):
        return "linear" if kind == "linear" else "attention"   # Kimi 的 KDA 層也叫 self_attn：靠層型分
    if RX["dense"].search(name):
        return "dense"
    return "other"


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


def file_ledger(tensors: dict, quant, kinds: list[str], tied: bool) -> dict:
    by = {c: 0 for c in CATS + EXCLUDED}
    n_by = {c: 0 for c in CATS + EXCLUDED}
    bytes_total, overhead, n_quant, tied_dup, others, per_layer_dense = 0, 0, 0, 0, [], {}
    for name, t in tensors.items():
        elems = 1
        for s in t["shape"]:
            elems *= s
        bytes_total += elems * DTYPE_BYTES.get(t["dtype"], 0)
        if name.endswith(".scales") or name.endswith(".biases"):
            base = name.rsplit(".", 1)[0] + ".weight"
            if base in tensors and tensors[base]["dtype"] in ("U32", "I32"):
                overhead += elems                          # 量化的附屬張量：不是參數
                continue
        bits = _bits_for(name, quant) if t["dtype"] in ("U32", "I32") and (name.rsplit(".", 1)[0] + ".scales") in tensors else None
        if bits:
            elems *= 32 // bits
            n_quant += 1
        cat = categorize(name, kinds)
        if tied and cat == "head":
            tied_dup += elems                              # tie_word_embeddings 卻仍存了 lm_head：同一矩陣兩份，算一次
            continue
        if cat == "other":
            others.append(name)
        if cat == "dense":
            m = LAYER_RX.search(name)
            if m:
                per_layer_dense[int(m.group(1))] = per_layer_dense.get(int(m.group(1)), 0) + elems
        by[cat] += elems
        n_by[cat] += 1
    N_total = sum(by[c] for c in CATS)
    return {"by": by, "n_by": n_by, "N_total": N_total, "bytes_total": bytes_total, "quant_overhead_elems": overhead,
            "n_quantized": n_quant, "tied_head_dup": tied_dup, "uncategorized": others, "dense_layers": sorted(per_layer_dense),
            "excluded": {c: by[c] for c in EXCLUDED if by[c]}}


# ══════════════════════════════════════════════════════════════
# 5. 一個模型的完整報告
# ══════════════════════════════════════════════════════════════
def analyse(src: Source) -> dict:
    flat = flatten({k: v for k, v in src.raw.items() if not k.startswith("_")})
    base = base_spec(flat)
    spec = moe_spec(flat)
    kinds = layer_kinds(flat, base)
    cl = config_ledger(flat, base, spec, kinds)
    out = {"name": src.name, "repo": src.repo, "source": src.kind, "source_path": str(src.path), "base": base,
           "moe": {k: v for k, v in spec.items()}, "layer_kinds_summary": {k: kinds.count(k) for k in sorted(set(kinds))},
           "config_ledger": cl, "file_ledger": None, "index_total_size_bytes": src.index_total}
    fl = None
    if src.tensors:
        fl = file_ledger(src.tensors, src.quant, kinds, base["tie"])
        out["file_ledger"] = fl
    # 選用哪個 N_total 當答案：有檔案就用檔案（它是真的），沒有就用 config 的式子
    N_total = fl["N_total"] if fl else cl["N_total"]
    experts_total = fl["by"]["experts"] if fl else cl["experts_total"]
    if spec["is_moe"] and N_total:
        experts_active = experts_total * spec["k"] / spec["n_E"]
        N_active = N_total - experts_total + experts_active
    else:
        experts_active, N_active = experts_total, N_total
    kdep = experts_total                                   # 隨 k 變的只有 routed expert 這一塊
    expert_side = (fl["by"]["shared"] + fl["by"]["router"] + fl["by"]["dense"] if fl else cl["shared"] + cl["router"] + cl["dense"]) + experts_active
    floor = (N_active - expert_side) if N_active else None   # attention + linear/SSM + norms + embedding (+ head)：不管 k 都要付
    out["answer"] = {
        "N_total": N_total, "N_active": N_active, "rho": (N_active / N_total) if N_total else None,
        "experts_total": experts_total, "experts_active": experts_active,
        "k_over_nE": (spec["k"] / spec["n_E"]) if spec["is_moe"] else None,
        "expert_side_ratio": ((spec["k"] + spec["n_S"]) / (spec["n_E"] + spec["n_S"])) if spec["is_moe"] else None,
        "floor": floor, "floor_over_active": (floor / N_active) if floor and N_active else None,
        "floor_over_total": (floor / N_total) if floor and N_total else None,
        "from": "safetensors headers" if fl else ("config formula" + (" (incomplete: some layers have no formula)" if cl["incomplete"] else "")),
    }
    if src.index_total and fl:
        out["answer"]["bytes_check"] = {"index_total_size": src.index_total, "sum_of_shapes_bytes": fl["bytes_total"],
                                        "equal": src.index_total == fl["bytes_total"]}
    return out


# ══════════════════════════════════════════════════════════════
# 6. 印
# ══════════════════════════════════════════════════════════════
def print_keys(src: Source) -> None:
    flat = flatten({k: v for k, v in src.raw.items() if not k.startswith("_")})
    print(f"{BOLD}{src.repo}{RESET}  {DIM}{src.kind}: {src.path}{RESET}")
    if src.kind == "snapshot":
        print(f"  {AMBER}這是投影片用的快照，抓的時候刪過與記帳無關的鍵——鍵名清單不完整，課堂上要用 ./present.sh fetch 抓的原檔{RESET}")
    base = base_spec(flat)
    print(f"\n  {DIM}W4 的七個鍵{RESET}")
    for k in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings"):
        v, key = get(flat, k)
        print(f"    {(key or k):<44} {v!s}")
    rx = re.compile(r"expert|moe|shared|rout|topk|top_k|sparse|dense|aux|layer_types|full_attention|linear_attn|kda|mamba|ssm|interval", re.I)
    hits = sorted(k for k in flat if rx.search(k))
    print(f"\n  {DIM}名字裡有 expert／moe／shared／router／top-k／layer type 的鍵（{len(hits)} 個）{RESET}")
    for k in hits:
        v = flat[k]
        s = json.dumps(v) if not isinstance(v, list) or len(v) <= 8 else f"[list of {len(v)}] {v[:6]}…"
        print(f"    {k:<44} {s}")
    spec = moe_spec(flat)
    print(f"\n  {DIM}這支程式的解讀（moe_spec）{RESET}")
    for field in ("n_E", "k", "d_E", "n_S", "d_S_total", "first_k_dense", "mlp_only_layers", "sparse_step", "moe_enabled"):
        key = spec["keys_used"].get(field)
        val = spec[field]
        tag = f"{GREEN}← {key}{RESET}" if key else f"{GREY}(no key; default {val!r}){RESET}"
        print(f"    {field:<14} {val!s:<12} {tag}")
    kinds = layer_kinds(flat, base)
    print(f"    {'layers':<14} {base['n_layer']!s:<12} " + ", ".join(f"{kinds.count(k)} {k}" for k in sorted(set(kinds)))
          + f";  MoE in {sum(moe_layers(spec, base['n_layer']))} layer(s)")
    if not spec["is_moe"]:
        print(f"  {AMBER}沒有找到 n_E 與 k 的鍵：這支程式會把它當 dense 模型。若 config 裡其實有（另一個名字），加進 count_moe.py 的 N_E_KEYS／K_KEYS{RESET}")
    if spec["moe_enabled"] and base["d_ff"]:
        print(f"  {AMBER}enable_moe_block 與 intermediate_size = {base['d_ff']} 並存：每層是 dense FFN + MoE，還是只有 MoE？config 說不出來——看檔案（count 的 dense 那列）{RESET}")


def print_count(r: dict, src: Source) -> None:
    b, m, cl, fl, a = r["base"], r["moe"], r["config_ledger"], r["file_ledger"], r["answer"]
    print(f"\n{BOLD}{src.repo}{RESET}  {DIM}{src.kind}: {src.path}{RESET}")
    if src.kind == "snapshot":
        print(f"  {AMBER}來源是投影片的快照（無形狀表）：只有 config 那一欄。課前 ./present.sh fetch 才有 file 欄{RESET}")
    lk = ", ".join(f"{v} {k}" for k, v in r["layer_kinds_summary"].items())
    print(f"  n_layer {b['n_layer']}  d {b['d']}  V {b['V']:,}  n_head {b['n_head']}  n_kv {b['n_kv']}  d_head {b['d_head']}  "
          f"{'tied' if b['tie'] else 'untied'} embeddings   layers: {lk}")
    if m["is_moe"]:
        print(f"  n_E {m['n_E']}  k {m['k']}  d_E {m['d_E']}  n_S {m['n_S']}" + (f" (width {m['d_S_total']})" if m['n_S'] else "")
              + f"   MoE in {cl['n_moe_layers']} of {b['n_layer']} layers"
              + f"   {DIM}keys: {m['keys_used']['n_E']}, {m['keys_used']['k']}, {m['keys_used']['d_E']}"
              + (f", {m['keys_used']['n_S']}" if m['n_S'] else ", no shared-expert key") + f"{RESET}")
    else:
        print(f"  {AMBER}no MoE keys recognised — treated as dense{RESET}")
    W = 44
    print(f"\n  {DIM}{'':<{W}}{'config (formula)':>20}{'file (safetensors)':>22}{RESET}")

    def row(label, cv, fv, note=""):
        cs = fmt(cv) if cv is not None else "—"
        fs = fmt(fv) if fv is not None else "—"
        flag = ""
        if cv is not None and fv is not None and cv != fv:
            flag = f"   {RED}← {cv - fv:+,}{RESET}"
        elif cv is not None and fv is not None:
            flag = f"   {GREEN}={RESET}"
        print(f"  {label:<{W}}{cs:>20}{fs:>22}{flag}{('   ' + GREY + note + RESET) if note else ''}")

    g = (lambda c: fl["by"][c]) if fl else (lambda c: None)
    row(f"attention   {cl['n_attn_layers']} layers" + (f" × {fmt(cl['per_attn'])}" if cl["per_attn"] else ""), cl["attention"], g("attention"), cl["attention_how"])
    if cl["n_linear_layers"] or (fl and fl["by"]["linear"]):
        row(f"linear/SSM  {cl['n_linear_layers']} layers" + (f" × {fmt(cl['per_linear'])}" if cl["per_linear"] else ""), cl["linear"], g("linear"), cl["linear_how"])
    if m["is_moe"]:
        row(f"experts     {cl['n_moe_layers']} × {m['n_E']} × {fmt(cl['per_expert'])}", cl["experts_total"], g("experts"), "routed, all n_E — this is the k-dependent block")
        row(f"shared      {cl['n_moe_layers']} × 3·d·{m['d_S_total']}" if m["n_S"] else "shared      none", cl["shared"], g("shared"))
        row(f"router      {cl['n_moe_layers']} × n_E·d", cl["router"], g("router"), "every token scores all n_E experts")
    row("dense FFN", cl["dense"], g("dense"), cl["dense_how"])
    row("norms", cl["norms"], g("norms"), "2 RMSNorm per layer + final" + (" + q_norm/k_norm" if cl["qk_norm_assumed"] else ""))
    row("embedding" + (" (+ untied head)" if not b["tie"] else " (tied)"), cl["embedding"], (g("embedding") + g("head")) if fl else None)
    if fl and fl["by"]["other"]:
        row("other (uncategorised)", None, fl["by"]["other"], "; ".join(fl["uncategorized"][:4]) + ("…" if len(fl["uncategorized"]) > 4 else ""))
    print(f"  {'─' * (W + 42)}")
    row(f"{BOLD}N_total{RESET}" + " " * (W - 7), cl["N_total"], fl["N_total"] if fl else None)
    if fl and fl["excluded"]:
        for c, v in fl["excluded"].items():
            what = {"mtp": "multi-token-prediction module stored in the checkpoint", "towers": "vision/audio tower(s)"}[c]
            print(f"  {GREY}{'  not counted in N: ' + c:<{W}}{'':>20}{fmt(v):>22}   {what}{RESET}")
    if fl and fl["quant_overhead_elems"]:
        print(f"  {GREY}{'  not counted: quantization scales/biases':<{W}}{'':>20}{fmt(fl['quant_overhead_elems']):>22}   MLX {src.quant.get('bits', '?')}-bit, group {src.quant.get('group_size', '?')}; weights unpacked ×{32 // int(src.quant.get('bits', 4))}{RESET}")
    if fl and fl["tied_head_dup"]:
        print(f"  {GREY}{'  not counted: lm_head stored despite tying':<{W}}{'':>20}{fmt(fl['tied_head_dup']):>22}   same matrix as the embedding{RESET}")
    if a.get("bytes_check"):
        bc = a["bytes_check"]
        ok = f"{GREEN}= index metadata.total_size ✓{RESET}" if bc["equal"] else f"{RED}≠ index metadata.total_size {bc['index_total_size']:,}{RESET}"
        print(f"  {GREY}{'  every shape × dtype bytes':<{W}}{'':>20}{fmt(bc['sum_of_shapes_bytes']):>22}   {ok}")
    elif src.index_total and not fl:
        est = src.index_total / 2
        print(f"  {GREY}{'  index total_size / 2 (bf16)':<{W}}{'':>20}{fmt(est):>22}   from the snapshot; includes anything the checkpoint stores beyond N{RESET}")
    if fl and fl["dense_layers"] and m["is_moe"]:
        n_dense = len(fl["dense_layers"])
        if n_dense == cl["n_moe_layers"] + (b["n_layer"] - cl["n_moe_layers"]) and cl["n_moe_layers"] == b["n_layer"]:
            print(f"  {AMBER}decision: dense FFN tensors exist in all {n_dense} layers *alongside* the experts → each layer runs dense FFN + MoE; "
                  f"the dense block is counted in N_total and N_active alike{RESET}")
        elif n_dense:
            print(f"  {AMBER}dense FFN tensors in {n_dense} layer(s): {fl['dense_layers'][:8]}{'…' if n_dense > 8 else ''}{RESET}")
    # 答案
    print()
    if a["N_total"] is None:
        print(f"  {AMBER}N_total 算不出來：config 對這個家族沒有完整的式子，而且沒有形狀表——./present.sh fetch 之後再來{RESET}")
        return
    print(f"  {BOLD}N_total   {fmt(a['N_total']):>18}   ≈ {a['N_total'] / 1e9:6.2f} B{RESET}   {DIM}({a['from']}){RESET}")
    print(f"  {BOLD}N_active  {fmt(round(a['N_active'])):>18}   ≈ {a['N_active'] / 1e9:6.2f} B{RESET}   {DIM}= N_total − experts + experts·k/n_E{RESET}" if m["is_moe"]
          else f"  {BOLD}N_active  {fmt(a['N_active']):>18}   = N_total (dense){RESET}")
    if m["is_moe"]:
        print(f"  {BOLD}ρ = N_active / N_total = {100 * a['rho']:.2f} %{RESET}      k/n_E = {100 * a['k_over_nE']:.2f} %"
              f"      expert side (k + n_S)/(n_E + n_S) = {100 * a['expert_side_ratio']:.2f} %")


def print_floor(r: dict, src: Source) -> None:
    a, cl, fl, m, b = r["answer"], r["config_ledger"], r["file_ledger"], r["moe"], r["base"]
    print(f"\n{BOLD}{src.repo}{RESET}  — the floor (p14: attention and embeddings are paid whatever k is)")
    g = (lambda c: fl["by"][c]) if fl else (lambda c: {"attention": cl["attention"], "linear": cl["linear"], "norms": cl["norms"],
                                                     "embedding": cl["embedding"], "head": 0, "shared": cl["shared"], "router": cl["router"],
                                                     "dense": cl["dense"]}[c])
    rows = [("attention (all attention layers)", g("attention")), ("linear / SSM layers", g("linear")), ("norms", g("norms")),
            ("embedding + head", (g("embedding") or 0) + (g("head") or 0))]
    print(f"  {DIM}independent of k{RESET}")
    for lab, v in rows:
        if v:
            print(f"    {lab:<36}{fmt(v):>20}")
    print(f"    {BOLD}{'floor':<36}{fmt(round(a['floor'])):>20}{RESET}")
    print(f"  {DIM}the expert side{RESET}")
    print(f"    {'routed experts, k of n_E active':<36}{fmt(round(a['experts_active'])):>20}   (of {fmt(a['experts_total'])} present)")
    for lab, c in (("shared expert(s)", "shared"), ("router", "router"), ("dense FFN", "dense")):
        v = g(c)
        if v:
            print(f"    {lab:<36}{fmt(v):>20}")
    print(f"    {BOLD}{'N_active':<36}{fmt(round(a['N_active'])):>20}{RESET}")
    print(f"\n  {BOLD}floor / N_active = {100 * a['floor_over_active']:.1f} %{RESET}      floor / N_total = {100 * a['floor_over_total']:.2f} %  (ρ cannot go below this)"
          f"      ρ = {100 * a['rho']:.2f} %")


def final_table(results: list[dict]) -> None:
    print(f"\n{BLUE}{'─' * 100}{RESET}")
    print(f"{BOLD}{'model':<34}{'n_E / k / n_S':>16}{'N_total':>14}{'N_active':>12}{'ρ':>9}{'floor/active':>14}{RESET}")
    for r in results:
        a, m = r["answer"], r["moe"]
        spec = f"{m['n_E']} / {m['k']} / {m['n_S']}" if m["is_moe"] else "dense"
        short = r["repo"].split("/")[-1]
        if a["N_total"] is None:
            print(f"{short:<34}{spec:>16}{'—':>14}{'—':>12}{'—':>9}{'—':>14}   {AMBER}no shapes and no full formula: run fetch{RESET}")
        elif m["is_moe"]:
            print(f"{short:<34}{spec:>16}{a['N_total'] / 1e9:>12.2f} B{a['N_active'] / 1e9:>10.2f} B"
                  f"{100 * a['rho']:>8.1f}%{(100 * a['floor_over_active']):>13.0f}%")
        else:
            print(f"{short:<34}{spec:>16}{a['N_total'] / 1e9:>12.2f} B{a['N_active'] / 1e9:>10.2f} B{'100.0%':>9}{'—':>14}")
    print(f"{BLUE}{'─' * 100}{RESET}")


# ══════════════════════════════════════════════════════════════
# 7. 入口
# ══════════════════════════════════════════════════════════════
def run_count(names: list[str], cfg: dict, quiet: bool = False) -> list[dict]:
    results = []
    for n in names:
        try:
            src = resolve(n, cfg)
        except (KeyError, FileNotFoundError) as exc:
            print(f"\n{RED}{n}: {exc}{RESET}")
            continue
        r = analyse(src)
        results.append(r)
        if not quiet:
            print_count(r, src)
    if not quiet and len(results) > 1:
        final_table(results)
    return results


def rehearse(cfg: dict) -> int:
    names = list(cfg["models"])
    results = run_count(names, cfg)
    out = HERE / cfg["rehearse"]["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from common import versions                             # noqa: E402
        ver = versions()
    except Exception:                                           # noqa: BLE001
        import platform
        ver = {"python": platform.python_version(), "platform": platform.platform()}
    rec = {"when": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "versions": ver, "models": {r["name"]: r for r in results}}
    for r in rec["models"].values():                            # 檔案裡不需要整張分類表以外的大東西；uncategorized 留前 20 個
        if r["file_ledger"]:
            r["file_ledger"]["uncategorized"] = r["file_ledger"]["uncategorized"][:20]
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{GREEN}彩排紀錄 → {out.relative_to(HERE)}{RESET}  （{len(results)}/{len(names)} 個模型）")
    return 0 if len(results) == len(names) else 1


def replay(cfg: dict) -> int:
    out = HERE / cfg["rehearse"]["out"]
    if not out.exists():
        print(f"{RED}沒有 {out.relative_to(HERE)}：先 ./present.sh rehearse{RESET}")
        return 1
    rec = json.loads(out.read_text(encoding="utf-8"))
    print(f"{AMBER}{BOLD}REPLAY — 彩排紀錄 {rec['when']}，不是現場計算{RESET}  {DIM}{rec['versions'].get('platform', '')}{RESET}")
    results = list(rec["models"].values())
    for r in results:
        a, m, b, cl = r["answer"], r["moe"], r["base"], r["config_ledger"]
        print(f"\n{BOLD}{r['repo']}{RESET}  {DIM}{r['source']}{RESET}")
        if m["is_moe"]:
            print(f"  n_E {m['n_E']}  k {m['k']}  d_E {m['d_E']}  n_S {m['n_S']}   MoE in {cl['n_moe_layers']} of {b['n_layer']} layers   "
                  f"{DIM}keys: {m['keys_used']['n_E']}, {m['keys_used']['k']}, {m['keys_used']['d_E']}{RESET}")
        if a["N_total"] is None:
            print(f"  {AMBER}彩排時算不出 N_total（沒有形狀表）{RESET}")
            continue
        print(f"  N_total {a['N_total'] / 1e9:6.2f} B   N_active {a['N_active'] / 1e9:5.2f} B   ρ {100 * a['rho']:5.2f} %"
              + (f"   k/n_E {100 * a['k_over_nE']:.2f} %   floor/active {100 * a['floor_over_active']:.0f} %" if m["is_moe"] else ""))
    final_table(results)
    return 0


def main(argv: list[str]) -> int:
    cfg = load_cfg()
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "keys":
        if not args:
            print(f"{RED}keys 要一個模型名：{', '.join(cfg['models'])}{RESET}")
            return 2
        for n in args:
            print_keys(resolve(n, cfg))
        return 0
    if cmd == "count":
        names = args or list(cfg["models"])
        res = run_count(names, cfg)
        return 0 if len(res) == len(names) else 1
    if cmd == "floor":
        names = args or ["qwen30"]
        for n in names:
            src = resolve(n, cfg)
            print_floor(analyse(src), src)
        return 0
    if cmd == "rehearse":
        return rehearse(cfg)
    if cmd == "replay":
        return replay(cfg)
    if cmd == "fetch":
        import fetch_configs
        return fetch_configs.main(args)
    print(f"{RED}不認得的子命令 {cmd!r}{RESET}\n{__doc__}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (KeyError, FileNotFoundError) as exc:
        print(f"{RED}{exc}{RESET}")
        sys.exit(1)
