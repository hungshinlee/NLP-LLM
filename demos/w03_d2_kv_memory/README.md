<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W3 · Demo 2 — predict the KV-cache curve from `config.json`, then measure it (slide p25)

The demo reads three numbers from a model's `config.json` — `num_hidden_layers`, `num_key_value_heads`, `head_dim` — plugs them into the ledger formula

    M_KV = 2 · n_layer · n_kv · d_head · L · b        (b = bytes per element of the cache)

to **predict** the KV-cache size per token and at several context lengths, and then **measures** it: for each context length `L` it prefills `L` random tokens and records three numbers against the one prediction:

| Measured | Meaning |
|---|---|
| cache tensor bytes | `nbytes` of the K and V tensors actually allocated |
| Δactive | memory still allocated when the prefill ends, minus the post-load baseline ≈ the KV cache |
| Δpeak | peak memory during the prefill, minus the baseline = KV cache + temporaries of the computation |

The point is that the formula and the measurement agree byte for byte, so that in later weeks (W5, W6, W11) you can trust the formula without measuring.

Two things the demo makes visible that the formula hides: a **4-bit** model still keeps its KV cache in **16-bit** (`inspect` prints the cache dtype), and the model used here has 8 KV heads (GQA), so the number is 128 KiB per token rather than the 512 KiB of an MHA layout with 32 heads — this is the "same 16 GiB, reached by a different route" of the week's opening: 32 KV heads × 32k tokens and 8 KV heads × 128k tokens both land on 16 GiB.

## Requirements

| | |
|---|---|
| Platform | **Apple silicon only.** The demo uses `mlx` and `mlx-lm` (pinned in `../versions.lock`) and MLX's own memory counters (`get_active_memory` / `get_peak_memory`). There is no CUDA path; writing one with `torch.cuda.max_memory_allocated()` is straightforward but has **not** been done or tested. |
| Model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (≈ 4.5 GB on disk, revision pinned in `../versions.lock`). Backup: `mlx-community/Qwen3-8B-4bit` (`--model backup`). |
| Memory | weights ≈ 4.5 GB, plus the cache you are measuring: 128 KiB × L. The 32k row needs ≈ 10 GB in total (fits a 16 GB Mac); the 128k row needs ≈ 23 GB (needs 32 GB or more). On a smaller machine set `max_context` in `demo_config.toml` (e.g. `32768`) or let `live_limit_s` skip the slow row. |
| Time | prefill time grows faster than linearly: 1k 0.3 s · 4k 1.1 s · 8k 2.4 s · 32k 17 s · 128k 195 s on the M5 Max. `show` does not run rows whose rehearsal time exceeded `live_limit_s` (45 s by default); it shows the recorded value and says so on screen. |
| Tested on | Apple M5 Max, 64 GB, macOS 26.6, Python 3.12.14, mlx 0.32.2, mlx-lm at commit `872ae88`, transformers 5.17.0 (2026-09-29) |

## Run it

```bash
cd demos && ./setup.sh --no-fetch      # once: shared .venv with the pinned mlx / mlx-lm (see ../README.md)
cd w03_d2_kv_memory
./present.sh fetch                     # once, needs network: downloads the model into the Hugging Face cache and pins its revision
./present.sh inspect                   # the three config numbers, the cache dtype, bytes per token, predicted size at each context length
./present.sh                           # the demo: Enter measures the next row (1k → 4k → 8k → 32k → maximum); q quits
./present.sh rehearse                  # measures every row without waiting and writes runs/rehearsal/curve.json
./present.sh replay                    # prints the instructor's rehearsal record, not a live run
./present.sh fake                      # walks through the screens with made-up numbers (no mlx needed; the screen says so)
```

`./present.sh rehearse` overwrites `runs/rehearsal/curve.json` with *your* measurements; `git checkout` the file to get the instructor's record back. If `fetch` returns HTTP 401, the model repository requires a Hugging Face login (`hf auth login`) before downloading.

## What you should see

From `runs/rehearsal/curve.json` (2026-09-29, Llama 3.1 8B Instruct 4-bit):

- `inspect`: 32 layers, 8 KV heads, head_dim 128 (not in the config file; derived as 4096 / 32), cache dtype `float16` → **131,072 bytes = 128 KiB per token**; maximum context 131,072.
- Every row's Δactive equals the formula **exactly**: 1k → 128 MiB, 4k → 512 MiB, 8k → 1 GiB, 32k → 4 GiB, 128k → **16 GiB** (17,179,869,184 bytes). Fitted slope 128.0 KiB/token.
- Δpeak exceeds Δactive by a roughly constant 0.65–0.76 GiB up to 32k, and by 1.35 GiB at 128k. That transient has two parts. One is the working set of the chunk being computed: the prefill runs 2,048 tokens at a time, so this part follows the chunk size, not `L`. The other is the cache being copied as it grows: `mlx-lm`'s `KVCache` is one contiguous buffer per layer, lengthened by concatenation, so the old and the new copy briefly coexist. That second part is small up to 32k and about 0.86 GiB at 128k; with the cache allocated at full length up front, the 128k transient drops to about 0.5 GiB (measured 2026-10-07).
- Timing as in the table above; the 128k row took 194.8 s, which is why `show` replays it instead of running it live.

On your machine Δactive should still equal the formula to the byte if you use the same model; Δpeak and the timings will differ.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `inspect` / `rehearse` / `show` / `replay`, plus `--fake` |
| `demo_config.toml` | context lengths, prefill chunk size, `live_limit_s`, `max_context`, the two models |
| `present.sh` | entry point; always `HF_HUB_OFFLINE=1`, uses `../.venv/bin/python` |
| `runs/rehearsal/curve.json`, `inspect.json` | the rehearsal record quoted above |
| `../common.py` | version lock, memory probes (`_mx_fn` finds the MLX memory API under either its current or its older `mx.metal.*` name) |
