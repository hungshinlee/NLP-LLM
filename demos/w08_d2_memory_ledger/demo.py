#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 2 (b)：實測的峰值記憶體對公式（投影片 p41 replay）。(a) 的算術在 ledger.py。

    ./present.sh inspect      課前：版本、機器的記憶體、三個模型在不在本機、權重檔依 dtype 的大小、每個設定的公式值、要跑哪幾個 run
    ./present.sh fetch        下載缺的模型並把 revision 寫進 ../versions.lock（Qwen/Qwen3-8B 的 bf16 約 16.4 GB；需要網路）
    ./present.sh adam         現場可跑（不載模型，一秒）：一個 bf16 的權重做一步 Adam——m、v 是什麼 dtype、權重動了沒有；float32 對照
    ./present.sh rehearse     彩排：adam → 每個設定 × 兩種序列長度（＋長的那個再開 --grad-checkpoint）各一個子行程 → peaks.json；
                              已完成的 run 會跳過（--redo 重跑）；--setting lora_8b 只做一個設定
    ./present.sh run ...      一個 run（rehearse 在子行程裡呼叫的就是它）：--setting qlora_8b --seq 1024 [--grad-checkpoint]
    ./present.sh aggregate    只重算 peaks.json
    ./present.sh replay       課堂：Enter 一列，印 p41 的表（公式兩種、三個峰值），再印拆解；讀彩排紀錄，不載模型
    ./present.sh tiny [...]   不下載任何東西：亂數權重的小 Qwen3 走一遍真的 mlx 訓練流程（檢查程式用，數字沒有意義）→ runs/tiny/
    ./present.sh fake [...]   連 mlx 都沒有也能演練流程（數字是假的，畫面會標明）→ runs/fake/

每個 run：載入模型 → 凍住 → 接 LoRA（或 unfreeze 全部 block）→ mlx-lm 的 train() 跑幾十步、batch 1、固定長度的視窗 → 記
mx.get_peak_memory()。與 `mlx_lm.lora` 走同一個 train()，差別只有 optimizer 由這支程式建立——訓完才能把 Adam 的狀態拿出來看
（每個張量的 dtype 與 bytes）。峰值是**整個行程**的：載入的權重、活化、梯度、Adam 狀態都在裡面，所以每個 run 一個子行程。

相依：mlx、mlx-lm（git commit 釘在 ../versions.lock）、numpy、huggingface_hub、transformers（tokenizer）。
replay 與 aggregate 只用標準函式庫。Python 3.11+（tomllib；3.10 退回 tomli）。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import (AMBER, BOLD, DIM, GREEN, GREY, RED, RESET, banner, device_info, locked_revision, note, now,  # noqa: E402
                    revision_key, versions, write_lock)
import ledger as LG  # noqa: E402

RUNS = HERE / "runs"
WEIGHTS = RUNS / "weights"        # 暫存（train() 結束一定會存一次權重），每個 run 做完就刪；不進版控
NON_MODEL_FILES = ["*.md", ".gitattributes", "*.jpg", "*.png", "*.pdf", "LICENSE"]
GB = 1e9


def out_dir(args) -> Path:
    """彩排紀錄進 runs/rehearsal/（進版控）；--tiny 與 --fake 各寫各的資料夾，不會混進彩排紀錄。"""
    return RUNS / ("fake" if args.fake else "tiny" if args.tiny else "rehearsal")


def mode_tag(args) -> str:
    return "  [FAKE numbers]" if args.fake else "  [TINY random model — numbers mean nothing]" if args.tiny else ""


def pause(prompt: str = "") -> str:
    try:
        return input(f"{DIM}{prompt}{RESET}")
    except EOFError:
        return ""


def write_json(p: Path, obj) -> None:
    LG.write_json(p, obj)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def measure_cfg(cfg: dict, args) -> dict:
    """[measure]，--tiny 時用 [tiny] 的長度與步數蓋過去。"""
    m = dict(cfg["measure"])
    if args.tiny:
        for k in ("seq_lengths", "grad_checkpoint_at", "iters", "steps_per_report"):
            m[k] = cfg["tiny"][k]
    return m


def run_name(setting: str, L: int, gc: bool) -> str:
    return f"{setting}_L{L}{'_gc' if gc else ''}"


def planned_runs(cfg: dict, args) -> list[tuple[str, int, bool]]:
    m = measure_cfg(cfg, args)
    out = []
    for name, spec in cfg["measure"]["settings"].items():
        if not spec.get("run", True) or (args.setting and name != args.setting):
            continue
        out += [(name, int(L), False) for L in m["seq_lengths"]]
        out += [(name, int(L), True) for L in m["grad_checkpoint_at"]]
    return out


def formula(cfg: dict, setting: str) -> dict:
    spec = cfg["measure"]["settings"][setting]
    return LG.mlx_ledger(LG.load_shape(cfg, spec["model"]), spec["kind"], int(cfg["measure"]["lora"]["rank"]))


# ── mlx 的小工具（函式裡才 import，replay 不需要 mlx）────────────────
def mem() -> dict:
    import mlx.core as mx
    return {"active_gb": mx.get_active_memory() / GB, "cache_gb": mx.get_cache_memory() / GB, "peak_gb": mx.get_peak_memory() / GB}


def dtype_name(a) -> str:
    return str(a.dtype).split(".")[-1]


def bytes_by_dtype(flat) -> dict:
    """[(名稱, 陣列)] → {dtype: {bytes, elements, tensors}}，另加 total_bytes。"""
    import mlx.core as mx
    by, total = {}, 0
    for _, v in flat:
        if not isinstance(v, mx.array):
            continue
        e = by.setdefault(dtype_name(v), {"bytes": 0, "elements": 0, "tensors": 0})
        e["bytes"] += v.nbytes
        e["elements"] += v.size
        e["tensors"] += 1
        total += v.nbytes
    return {"by_dtype": by, "total_bytes": total, "total_gb": total / GB}


def model_path(cfg: dict, key: str, offline: bool) -> str:
    from huggingface_hub import snapshot_download
    m = cfg["models"][key]
    rev = locked_revision(m["repo"], m.get("revision"))
    return snapshot_download(m["repo"], revision=rev, local_files_only=offline, ignore_patterns=NON_MODEL_FILES)


def tiny_model(cfg: dict, kind: str):
    """亂數權重的小 Qwen3（bf16；qlora 再量化成 4-bit）。只為了不下載也能走一遍真的流程。"""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3
    t = cfg["tiny"]
    a = qwen3.ModelArgs(model_type="qwen3", hidden_size=t["hidden_size"], num_hidden_layers=t["num_hidden_layers"],
                        intermediate_size=t["intermediate_size"], num_attention_heads=t["num_attention_heads"], rms_norm_eps=1e-6,
                        vocab_size=t["vocab_size"], num_key_value_heads=t["num_key_value_heads"], max_position_embeddings=4096,
                        rope_theta=1e6, head_dim=t["head_dim"], tie_word_embeddings=(kind == "full"))
    model = qwen3.Model(a)
    model.set_dtype(mx.bfloat16)
    if kind == "qlora":
        nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


class Windows:
    """固定長度的 token 視窗；介面照 mlx-lm 的 dataset（process 回傳 (tokens, offset)，offset 0 = 每個 token 都算 loss）。"""

    def __init__(self, windows: list[list[int]]):
        self._data = windows

    def process(self, d):
        return (d, 0)

    def __getitem__(self, i: int):
        return self._data[i]

    def __len__(self):
        return len(self._data)


def make_windows(cfg: dict, tokenizer, L: int, n: int, vocab: int, seed: int) -> tuple[list[list[int]], str]:
    """n 個長度剛好 L 的視窗。有 demo 1 的訓練資料就把它接成一長串再切；沒有就用固定 seed 的亂數 token（記憶體只看長度）。"""
    import random
    p = (HERE / cfg["measure"]["text_from"]).resolve()
    if tokenizer is not None and p.exists():
        ids: list[int] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = "\n\n".join(m["content"] for m in json.loads(line)["messages"])
            ids += tokenizer.encode(text + "\n\n")
            if len(ids) >= n * L:
                break
        if len(ids) >= n * L:
            return [ids[i * L:(i + 1) * L] for i in range(n)], cfg["measure"]["text_from"]
    rng = random.Random(seed)
    return [[rng.randrange(vocab) for _ in range(L)] for _ in range(n)], "random token ids (fixed seed)"


class Recorder:
    """mlx-lm 的 TrainingCallback：每次 report 印 peak memory（mlx-lm 的畫面沒有這一欄）並記下來。"""

    def __init__(self, label: str):
        self.label, self.train, self.t0 = label, [], time.time()

    def on_train_loss_report(self, info: dict):
        rec = {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in info.items()}
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        self.train.append(rec)
        print(f"  {GREY}[{self.label}]{RESET} step {info['iteration']:>3}   train loss {info['train_loss']:.3f}   "
              f"peak memory {BOLD}{info['peak_memory']:.2f} GB{RESET}   {info['tokens_per_second']:,.0f} tok/s")

    def on_val_loss_report(self, info: dict):
        pass


def count_unchanged(model, before) -> dict:
    """可訓練的元素裡，有幾個跟訓練前逐位元相同。before(key) 回傳訓練前的陣列（或 None）。依形狀分：矩陣（Linear）與向量（norm）。"""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    out = {"matrix": [0, 0], "vector": [0, 0]}
    for k, v in tree_flatten(model.trainable_parameters()):
        b = before(k)
        if b is None:
            continue
        same = (v == b.astype(v.dtype)).sum()
        mx.eval(same)
        g = out["matrix" if v.ndim >= 2 else "vector"]
        g[0] += int(same.item())
        g[1] += int(v.size)
    tot = [out["matrix"][0] + out["vector"][0], out["matrix"][1] + out["vector"][1]]
    return {"same": tot[0], "total": tot[1], "fraction": tot[0] / tot[1] if tot[1] else None,
            "matrix_fraction": out["matrix"][0] / out["matrix"][1] if out["matrix"][1] else None,
            "vector_fraction": out["vector"][0] / out["vector"][1] if out["vector"][1] else None,
            "matrix_elements": out["matrix"][1], "vector_elements": out["vector"][1]}


# ── 一個 run（子行程）────────────────────────────────────────────
def cmd_run(args, cfg):
    if not args.setting or not args.seq:
        raise SystemExit(f"{RED}run 要給 --setting 與 --seq{RESET}")
    m = measure_cfg(cfg, args)
    spec = cfg["measure"]["settings"][args.setting]
    kind, key, L, gc = spec["kind"], spec["model"], int(args.seq), bool(args.grad_checkpoint)
    iters, bs, lr = int(args.iters or m["iters"]), int(m["batch_size"]), float(spec["learning_rate"])
    name = run_name(args.setting, L, gc)
    out_json = out_dir(args) / "runs" / f"{name}.json"
    rec = {"name": name, "setting": args.setting, "label": spec["label"], "kind": kind, "model": cfg["models"][key]["repo"],
           "seq_length": L, "grad_checkpoint": gc, "iters": iters, "batch_size": bs, "learning_rate": lr,
           "seed": int(cfg["demo"]["seed"]), "date": now(), "versions": versions(), "formula": formula(cfg, args.setting)}
    banner(f"{spec['label']} · {cfg['models'][key]['short']} · {L} tokens{' · --grad-checkpoint' if gc else ''} · {iters} steps{mode_tag(args)}", AMBER)
    if args.fake:
        rec.update(fake_run(cfg, args.setting, L, gc, iters), fake=True)
        write_json(out_json, rec)
        print(f"  {GREEN}{name} done [FAKE]{RESET}: peak {rec['peak_gb']:.2f} GB")
        return
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    rec["device"] = device_info()
    mx.random.seed(rec["seed"])
    np.random.seed(rec["seed"])
    mx.reset_peak_memory()
    t0 = time.time()
    if args.tiny:
        model, tokenizer, path, vocab = tiny_model(cfg, kind), None, None, int(cfg["tiny"]["vocab_size"])
        rec["tiny"] = True
        rec["formula"] = None          # 公式是真模型的，對亂數小模型沒有意義
    else:
        from mlx_lm.utils import load
        path = model_path(cfg, key, offline=not args.online)
        model, tokenizer = load(path)
        vocab = int(LG.load_shape(cfg, key)["V"])
        rec["model_snapshot"] = Path(path).name
    mx.eval(model.parameters())
    rec["load_seconds"] = round(time.time() - t0, 1)
    rec["after_load"] = mem()
    rec["parameters"] = bytes_by_dtype(tree_flatten(model.parameters()))
    by = ", ".join("%s %.2f" % (k, v["bytes"] / GB) for k, v in rec["parameters"]["by_dtype"].items())
    print(f"  loaded: {rec['parameters']['total_gb']:.2f} GB of parameters ({by}); active memory {rec['after_load']['active_gb']:.2f} GB")

    model.freeze()
    if kind == "full":            # mlx-lm 的 --fine-tune-type full --num-layers -1：只有 block 變成可訓練
        for layer in model.layers:
            layer.unfreeze()
    else:                         # --fine-tune-type lora --num-layers -1：每個 block 的七個 Linear（量化過的也一樣接）
        lo = cfg["measure"]["lora"]
        linear_to_lora_layers(model, len(model.layers), {"rank": int(lo["rank"]), "scale": float(lo["scale"]),
                                                         "dropout": float(lo["dropout"]), "keys": list(lo["keys"])})
    trainable = tree_flatten(model.trainable_parameters())
    rec["trainable"] = dict(bytes_by_dtype(trainable), n=int(sum(v.size for _, v in trainable)))
    print(f"  trainable: {rec['trainable']['n']:,} parameters, dtype {', '.join(rec['trainable']['by_dtype'])}")
    before_tiny = dict(trainable) if (args.tiny and kind == "full") else None

    windows, rec["tokens_source"] = make_windows(cfg, tokenizer, L, max(iters, bs), vocab, rec["seed"])
    tmp = WEIGHTS / f"tmp_{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    opt = optim.Adam(learning_rate=lr)
    targs = TrainingArgs(batch_size=bs, iters=iters, val_batches=0, steps_per_report=int(m["steps_per_report"]), steps_per_eval=10 ** 9,
                         steps_per_save=10 ** 9, adapter_file=str(tmp / "adapters.safetensors"), max_seq_length=L, grad_checkpoint=gc)
    r = Recorder(name)
    t1 = time.time()
    train(model=model, args=targs, optimizer=opt, train_dataset=CacheDataset(Windows(windows)), val_dataset=None, training_callback=r)
    rec["train_seconds"] = round(time.time() - t1, 1)
    rec["seconds_per_step"] = round(rec["train_seconds"] / iters, 3)
    rec["peak_gb"] = mx.get_peak_memory() / GB
    rec["after_train"] = mem()
    rec["train_log"] = r.train
    state = [(k, v) for k, v in tree_flatten(opt.state) if isinstance(v, mx.array) and v.size > 1]
    rec["optimizer_state"] = bytes_by_dtype(state)
    rec["over_formula_gb"] = (rec["peak_gb"] - rec["formula"]["GB"]) if rec["formula"] else None
    shutil.rmtree(tmp, ignore_errors=True)

    if kind == "full" and cfg["measure"].get("count_unchanged", True):
        if before_tiny is not None:
            rec["unchanged"] = count_unchanged(model, before_tiny.get)
        else:                      # 訓練前的值從權重檔再讀一次（峰值已經記下來了，這一步不影響它）
            base = {}
            for f in sorted(Path(path).glob("*.safetensors")):
                base.update(mx.load(str(f)))
            rec["unchanged"] = count_unchanged(model, base.get)
        u = rec["unchanged"]
        print(f"  after {iters} steps at lr {lr:g}: {100 * u['fraction']:.1f} % of the trainable elements are bit-for-bit unchanged "
              f"(matrices {100 * (u['matrix_fraction'] or 0):.1f} %, norm vectors {100 * (u['vector_fraction'] or 0):.1f} %)")
    st = rec["optimizer_state"]
    print(f"  Adam state: {st['total_gb']:.2f} GB, dtype {', '.join(st['by_dtype'])}")
    over = f", formula {rec['formula']['GB']:.1f} GB → {rec['over_formula_gb']:+.1f} GB" if rec["formula"] else ""
    print(f"\n  {GREEN}{name} done{RESET}: peak {BOLD}{rec['peak_gb']:.2f} GB{RESET}{over}; {rec['seconds_per_step']} s/step")
    write_json(out_json, rec)
    note(f"→ {out_json.relative_to(HERE)}")


def fake_run(cfg: dict, setting: str, L: int, gc: bool, iters: int) -> dict:
    """--fake：公式 + 一個隨長度線性的假活化項。只為了演練流程。"""
    f = formula(cfg, setting)
    s = LG.load_shape(cfg, cfg["measure"]["settings"][setting]["model"])
    act = 1.0 + 9e-9 * s["n_layer"] * s["d"] * L * (0.35 if gc else 1.0) * 2
    kind = cfg["measure"]["settings"][setting]["kind"]
    dt = "bfloat16" if kind == "full" else "float32"
    out = {"peak_gb": f["GB"] + act, "over_formula_gb": act, "load_seconds": 0.1, "train_seconds": 1.0, "seconds_per_step": 0.05 * (1.3 if gc else 1.0),
           "after_load": {"active_gb": f["frozen_GB"] if kind != "full" else f["GB"] / 4, "cache_gb": 0.0, "peak_gb": f["frozen_GB"]},
           "parameters": {"by_dtype": {}, "total_bytes": 0, "total_gb": f["frozen_GB"] if kind != "full" else 2 * f["n_trainable"] / GB + f["frozen_GB"]},
           "trainable": {"n": f["n_trainable"], "by_dtype": {dt: {}}, "total_gb": 0.0},
           "optimizer_state": {"by_dtype": {dt: {}}, "total_gb": f["n_trainable"] * (4 if kind == "full" else 8) / GB},
           "train_log": [{"iteration": i, "train_loss": 3.0, "peak_memory": f["GB"] + act, "tokens_per_second": 500.0} for i in range(5, iters + 1, 5)],
           "tokens_source": "(fake)"}
    if kind == "full":
        out["unchanged"] = {"same": 0, "total": 0, "fraction": 0.9, "matrix_fraction": 0.9, "vector_fraction": 1.0}
    return out


def spawn_run(args, setting: str, L: int, gc: bool) -> None:
    cmd = [sys.executable, str(HERE / "demo.py")]
    cmd += ["--fake"] if args.fake else []
    cmd += ["--tiny"] if args.tiny else []
    cmd += ["--online"] if args.online else []
    cmd += ["run", "--setting", setting, "--seq", str(L)] + (["--grad-checkpoint"] if gc else [])
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit(f"{RED}子行程失敗（{run_name(setting, L, gc)}）：exit {r.returncode}；修好之後重跑同一個指令，已完成的 run 會跳過{RESET}")


# ── adam：一個權重、一步 ─────────────────────────────────────────
def ulp_bf16(x: float) -> float:
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)      # bfloat16：8 個有效 bit


def cmd_adam(args, cfg) -> dict:
    lr = float(args.lr or cfg["measure"]["settings"]["full_1p7b"]["learning_rate"])
    banner(cfg["demo"]["title"] + f"  ·  one weight, one Adam step at lr {lr:g}" + ("  [FAKE]" if args.fake else ""))
    rows = []
    if args.fake:
        rows = [{"dtype": d, "weight": w, "state_dtype": d, "after": w - (0.0 if (d == "bfloat16" and w > 0.004) else 3.16 * lr),
                 "moved": not (d == "bfloat16" and w > 0.004), "half_ulp": ulp_bf16(w) / 2 if d == "bfloat16" else None}
                for d in ("bfloat16", "float32") for w in (0.02, 0.002, 0.0002)]
    else:
        import mlx.core as mx
        import mlx.optimizers as optim
        for dt in (mx.bfloat16, mx.float32):
            for w0 in (0.02, 0.002, 0.0002):
                params = {"w": mx.full((4,), w0, dtype=dt)}
                opt = optim.Adam(learning_rate=lr)
                new = opt.apply_gradients({"w": mx.ones((4,), dtype=dt)}, params)
                mx.eval(new, opt.state)
                before, after = float(params["w"][0].item()), float(new["w"][0].item())
                rows.append({"dtype": dtype_name(params["w"]), "weight": before, "state_dtype": f"{dtype_name(opt.state['w']['m'])}, {dtype_name(opt.state['w']['v'])}",
                             "after": after, "moved": after != before, "step_taken": before - after,
                             "half_ulp": ulp_bf16(before) / 2 if dt == mx.bfloat16 else None})
    print(f"  {GREY}mlx's Adam: m and v are zeros_like(parameter); the update is parameter − lr·m/(√v + ε), computed in the parameter's dtype.")
    print(f"  No bias correction by default, so the very first step is lr·0.1/√0.001 ≈ 3.16·lr = {3.16 * lr:.2e}.{RESET}\n")
    print(f"  {'weight dtype':<14}{'weight':>10}   {'Adam m, v':<20}{'after one step':>16}   {'moved?':<8}{'half a bf16 step at this size':>30}")
    for r in rows:
        hu = f"{r['half_ulp']:.1e}" if r["half_ulp"] else ""
        mv = f"{GREEN}yes{RESET}     " if r["moved"] else f"{RED}no{RESET}      "
        print(f"  {r['dtype']:<14}{r['weight']:>10.4g}   {r['state_dtype']:<20}{r['after']:>16.6g}   {mv}{hu:>30}")
    print(f"\n  {GREY}bfloat16 keeps 8 significant bits: a weight of 0.02 can only move in steps of {ulp_bf16(0.02):.1e}. An update smaller than half a step is rounded away.")
    print(f"  That is what the fp32 master copy in the 16-byte ledger is for — and mlx does not keep one.{RESET}")
    rec = {"date": now(), "lr": lr, "rows": rows}
    if not args.fake:
        rec["versions"] = versions()
    return rec


# ── inspect／fetch ───────────────────────────────────────────────
def cmd_inspect(args, cfg) -> None:
    banner(cfg["demo"]["title"] + "  ·  inspect" + mode_tag(args))
    v = versions()
    rec = {"date": now(), "versions": v}
    print(f"  python {v['python']} · mlx {v.get('mlx')} · mlx-lm {v.get('mlx-lm')} @ {str(v.get('mlx_lm_commit'))[:7]} · transformers {v.get('transformers')}")
    if not args.fake:
        di = device_info()
        rec["device"] = di
        ms, ws = di.get("memory_size"), di.get("max_recommended_working_set_size")
        print(f"  device: {di.get('device_name', '?')}" + (f" · memory {ms / GB:.1f} GB" if isinstance(ms, (int, float)) else "")
              + (f" · recommended working set {ws / GB:.1f} GB (mlx-lm sets the wired limit to this)" if isinstance(ws, (int, float)) else ""))
        try:
            import inspect as pyinspect
            import mlx.optimizers as optim
            src = pyinspect.getsource(optim.Adam.init_single).strip().splitlines()
            print(f"\n  {GREY}mlx.optimizers.Adam.init_single (this installation):{RESET}")
            for line in src:
                print(f"    {DIM}{line}{RESET}")
            rec["adam_init_single"] = src
        except Exception as e:      # noqa: BLE001
            note(f"  (could not read the Adam source: {e})")
    print()
    rec["models"] = LG.cmd_files(args, cfg)
    print()
    rec["formulas"] = LG.cmd_mlx(args, cfg)
    m = measure_cfg(cfg, args)
    runs = planned_runs(cfg, args)
    rec["planned_runs"] = [run_name(*r) for r in runs]
    p = (HERE / cfg["measure"]["text_from"]).resolve()
    print(f"\n  {len(runs)} runs: {m['iters']} steps each, batch {m['batch_size']}, sequence lengths {m['seq_lengths']}, --grad-checkpoint at {m['grad_checkpoint_at']}")
    print(f"  windows from {cfg['measure']['text_from']}: {'found' if p.exists() else 'NOT found → random token ids (memory depends on the length only)'}")
    done = [n for n in rec["planned_runs"] if (out_dir(args) / "runs" / f"{n}.json").exists()]
    print(f"  already recorded in {out_dir(args).relative_to(HERE)}/runs/: {len(done)}/{len(runs)}")
    write_json(out_dir(args) / "inspect.json", rec)
    note(f"→ {(out_dir(args) / 'inspect.json').relative_to(HERE)}")


def cmd_fetch(args, cfg) -> None:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    need = {s["model"] for s in cfg["measure"]["settings"].values() if s.get("run", True)}
    for key in cfg["models"]:
        if key not in need:
            continue
        mdl = cfg["models"][key]
        rev = locked_revision(mdl["repo"], mdl.get("revision")) or api.model_info(mdl["repo"]).sha
        banner(f"下載 {mdl['repo']} @ {rev[:10]}")
        snapshot_download(mdl["repo"], revision=rev, ignore_patterns=NON_MODEL_FILES)
        write_lock({revision_key(mdl["repo"]): rev})
    print(f"{GREEN}完成，revision 已寫進 ../versions.lock；接著 ./present.sh inspect{RESET}")


# ── rehearse／aggregate／replay ──────────────────────────────────
def cmd_rehearse(args, cfg) -> None:
    out = out_dir(args)
    banner(cfg["demo"]["title"] + "  ·  rehearse (b)" + mode_tag(args))
    write_json(out / "adam.json", cmd_adam(args, cfg))
    runs = planned_runs(cfg, args)
    for i, (setting, L, gc) in enumerate(runs, 1):
        p = out / "runs" / f"{run_name(setting, L, gc)}.json"
        if p.exists() and not args.redo:
            note(f"  [{i}/{len(runs)}] {p.stem}: already recorded, skipped (--redo to run it again)")
            continue
        print(f"\n{DIM}[{i}/{len(runs)}]{RESET}")
        spawn_run(args, setting, L, gc)
    aggregate(args, cfg, echo=True)
    print(f"\n{GREEN}done{RESET} → {out.relative_to(HERE)}/peaks.json · next: ./present.sh replay")


def aggregate(args, cfg, echo: bool = False) -> dict:
    out = out_dir(args)
    m = measure_cfg(cfg, args)
    rows, missing, dates = [], [], []
    for name, spec in cfg["measure"]["settings"].items():
        f = formula(cfg, name)
        row = {"setting": name, "label": spec["label"], "kind": spec["kind"], "model": cfg["models"][spec["model"]]["repo"],
               "short": cfg["models"][spec["model"]]["short"], "run": bool(spec.get("run", True)),
               "textbook_gb": f["textbook_GB"], "mlx_gb": f["GB"], "mlx_frozen_gb": f["frozen_GB"], "mlx_trained_gb": f["train_GB"],
               "n_trainable_formula": f["n_trainable"], "peaks": {}, "seconds_per_step": {}}
        if row["run"]:
            for L in m["seq_lengths"]:
                for gc in ([False, True] if L in m["grad_checkpoint_at"] else [False]):
                    p = out / "runs" / f"{run_name(name, int(L), gc)}.json"
                    if not p.exists():
                        missing.append(p.stem)
                        continue
                    r = read_json(p)
                    k = f"{L}{'_gc' if gc else ''}"
                    row["peaks"][k] = r["peak_gb"]
                    dates.append(r["date"])
                    row.setdefault("after_train_active_gb", {})[k] = r["after_train"]["active_gb"]
                    row["seconds_per_step"][k] = r["seconds_per_step"]
                    if not gc and "parameters_gb" not in row:      # 與長度無關的量，取第一個 run 的
                        row.update(parameters_gb=r["parameters"]["total_gb"], after_load_active_gb=r["after_load"]["active_gb"],
                                   n_trainable=r["trainable"]["n"], trainable_dtypes=sorted(r["trainable"]["by_dtype"]),
                                   adam_state_gb=r["optimizer_state"]["total_gb"], adam_state_dtypes=sorted(r["optimizer_state"]["by_dtype"]))
                    if r.get("unchanged") and not gc:
                        row.setdefault("unchanged", {})[str(L)] = r["unchanged"]
                    row["iters"], row["batch_size"], row["learning_rate"] = r["iters"], r["batch_size"], r["learning_rate"]
                    row["tokens_source"] = r.get("tokens_source")
            Ls = [int(x) for x in m["seq_lengths"]]
            lo, hi = str(min(Ls)), str(max(Ls))
            if lo in row["peaks"] and hi in row["peaks"] and lo != hi:
                row["gb_per_1k_tokens"] = (row["peaks"][hi] - row["peaks"][lo]) / (int(hi) - int(lo)) * 1000
            if hi in row["peaks"]:
                row["over_formula_long_gb"] = row["peaks"][hi] - (0.0 if args.tiny else row["mlx_gb"])
            if lo in row["peaks"]:
                row["over_formula_short_gb"] = row["peaks"][lo] - (0.0 if args.tiny else row["mlx_gb"])
            if hi in row["peaks"] and f"{hi}_gc" in row["peaks"]:
                row["grad_checkpoint_saves_gb"] = row["peaks"][hi] - row["peaks"][f"{hi}_gc"]
                a, b = row["seconds_per_step"].get(hi), row["seconds_per_step"].get(f"{hi}_gc")
                row["grad_checkpoint_time_ratio"] = (b / a) if a and b else None
        rows.append(row)
    rec = {"date": max(dates) if dates else now(), "seq_lengths": [int(x) for x in m["seq_lengths"]], "grad_checkpoint_at": [int(x) for x in m["grad_checkpoint_at"]],
           "iters": int(m["iters"]), "batch_size": int(m["batch_size"]), "rows": rows, "missing": missing,
           "note": "GB = 1e9 bytes. peaks: mx.get_peak_memory() of the whole process, one process per run. textbook_gb: 16 bytes per trainable "
                   "parameter and 2 per frozen one. mlx_gb: what mlx stores before any activation (Adam state in the parameter's dtype, no fp32 master)."}
    if args.fake:
        rec["fake"] = True
    if args.tiny:
        rec["tiny"] = True
    insp = out / "inspect.json"
    if insp.exists():
        rec["device"] = read_json(insp).get("device")
    write_json(out / "peaks.json", rec)
    if echo and missing:
        print(f"  {AMBER}missing runs: {', '.join(missing)}{RESET}")
    return rec


def cmd_aggregate(args, cfg) -> None:
    aggregate(args, cfg, echo=True)
    note(f"→ {(out_dir(args) / 'peaks.json').relative_to(HERE)}")


def gbs(x, w=9) -> str:
    return f"{x:>{w}.1f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"


def cmd_replay(args, cfg) -> None:
    out = out_dir(args)
    p = out / "peaks.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p.relative_to(HERE)}：先 ./present.sh rehearse{RESET}")
    d = read_json(p)
    lo, hi = min(d["seq_lengths"]), max(d["seq_lengths"])
    banner(cfg["demo"]["title"] + f"  ·  replay of the rehearsal of {d['date'][:10]}" + mode_tag(args))
    print(f"  {GREY}every run: batch {d['batch_size']}, {d['iters']} steps, one process each; peak = mx.get_peak_memory() of the whole process; GB = 10^9 bytes")
    print(f"  both formula columns are bytes before a single activation{RESET}\n")
    print(f"  {'run':<19}{'model':<24}{'ledger,':>9}{'as mlx':>11}   {'peak,':>8}{'peak,':>10}{'':>2}{'checkpointed,':>14}")
    print(f"  {'':<19}{'':<24}{'16 bytes':>9}{'stores it':>11}   {str(lo) + ' tok':>8}{str(hi) + ' tok':>10}{'':>2}{str(hi) + ' tok':>14}")
    for row in d["rows"]:
        if not args.all:
            pause("")
        if not row["run"]:
            print(f"  {row['label']:<19}{row['short']:<24}{gbs(row['textbook_gb'])}{BOLD}{gbs(row['mlx_gb'], 11)}{RESET}   {AMBER}not run{RESET}")
            continue
        pk = row["peaks"]
        print(f"  {row['label']:<19}{row['short']:<24}{gbs(row['textbook_gb'])}{BOLD}{gbs(row['mlx_gb'], 11)}{RESET}   "
              f"{BOLD}{gbs(pk.get(str(lo)), 8)}{gbs(pk.get(str(hi)), 10)}{RESET}{'':>2}{BOLD}{gbs(pk.get(f'{hi}_gc'), 14)}{RESET}")
    if not args.all:
        pause("\n  Enter → where the difference goes ")
    print()
    for row in d["rows"]:
        if not row["run"] or not row["peaks"]:
            continue
        print(f"  {BOLD}{row['label']} · {row['short']}{RESET}")
        if "parameters_gb" in row:
            print(f"    loaded parameters {row['parameters_gb']:.2f} GB (ledger's frozen copy: {row['mlx_frozen_gb'] if row['kind'] != 'full' else row['mlx_frozen_gb'] + row['n_trainable_formula'] * 2 / GB:.2f})"
                  f" · trainable {row['n_trainable']:,} ({', '.join(row['trainable_dtypes'])})"
                  f" · Adam state {row['adam_state_gb']:.2f} GB ({', '.join(row['adam_state_dtypes'])})")
        at = row.get("after_train_active_gb") or {}
        if at:
            print(f"    still held after the last step: {', '.join(f'{v:.2f} GB' for v in at.values())}   (as mlx stores it: {row['mlx_gb']:.2f})")
        if "over_formula_short_gb" in row and "over_formula_long_gb" in row:
            print(f"    peak − formula: {row['over_formula_short_gb']:+.1f} GB at {lo} tokens, {row['over_formula_long_gb']:+.1f} GB at {hi}"
                  + (f"   → grows by {row['gb_per_1k_tokens']:.1f} GB per 1,000 tokens" if "gb_per_1k_tokens" in row else ""))
        if "grad_checkpoint_saves_gb" in row:
            tr = row.get("grad_checkpoint_time_ratio")
            print(f"    --grad-checkpoint at {hi} tokens: saves {row['grad_checkpoint_saves_gb']:.1f} GB" + (f", at {tr:.2f}× the time per step" if tr else ""))
        for L, u in (row.get("unchanged") or {}).items():
            if u.get("fraction") is not None:
                print(f"    after {row['iters']} steps at lr {row['learning_rate']:g} ({L} tokens): {100 * u['fraction']:.1f} % of the trainable elements bit-for-bit unchanged"
                      f" (matrices {100 * (u.get('matrix_fraction') or 0):.1f} %, norm vectors {100 * (u.get('vector_fraction') or 0):.1f} %)")
    for row in d["rows"]:
        if not row["run"]:
            dev = d.get("device") or {}
            ms, ws = dev.get("memory_size"), dev.get("max_recommended_working_set_size")
            mem_s = f"{ms / 2 ** 30:.0f} GiB = {ms / GB:.1f} GB" if isinstance(ms, (int, float)) else "64 GiB"
            ws_s = f"; recommended working set (mlx-lm's wired limit) {ws / GB:.1f} GB" if isinstance(ws, (int, float)) else ""
            print(f"\n  {AMBER}not run{RESET} — {row['label']} · {row['short']}: {row['textbook_gb']:.0f} GB by the 16-byte ledger, {row['mlx_gb']:.1f} GB the way mlx stores it,"
                  f"\n  both before a single activation. This machine: {mem_s}{ws_s}; nowhere to offload to.")
    if d.get("missing"):
        print(f"\n  {AMBER}missing runs: {', '.join(d['missing'])}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="W8 demo 2 (b): measured peak memory against the formula")
    ap.add_argument("--fake", action="store_true", help="不需要 mlx：假數字，只演練流程")
    ap.add_argument("--tiny", action="store_true", help="亂數權重的小模型走真的流程（不下載）")
    ap.add_argument("--online", action="store_true", help="允許連 Hugging Face（預設只讀本機快取）")
    ap.add_argument("--setting", help="[measure.settings.*] 的鍵")
    ap.add_argument("--seq", type=int, help="run：序列長度")
    ap.add_argument("--grad-checkpoint", action="store_true", help="run：開 gradient checkpointing")
    ap.add_argument("--iters", type=int, help="run：步數（預設 [measure] iters）")
    ap.add_argument("--lr", type=float, help="adam：learning rate")
    ap.add_argument("--redo", action="store_true", help="rehearse：已有紀錄的 run 也重跑")
    ap.add_argument("--all", action="store_true", help="replay／table 不停下來等 Enter")
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--modules", help=argparse.SUPPRESS)
    ap.add_argument("--layers", type=int, help=argparse.SUPPRESS)
    ap.add_argument("cmd", choices=["inspect", "fetch", "adam", "run", "rehearse", "aggregate", "replay"])
    args = ap.parse_args()
    cfg = LG.load_cfg()
    if args.cmd == "adam":
        cmd_adam(args, cfg)
        return
    {"inspect": cmd_inspect, "fetch": cmd_fetch, "run": cmd_run, "rehearse": cmd_rehearse, "aggregate": cmd_aggregate, "replay": cmd_replay}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
