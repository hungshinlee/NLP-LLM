<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 4 — A parallel hybrid against full attention: cache against $L$, and recall (slides p36–p37)

Two models of comparable size, both 4-bit MLX: **Qwen3-8B** (full attention in every layer, 36 layers, 8 KV heads) and **Falcon-H1-7B-Instruct** (a *parallel* hybrid: every one of its 44 layers runs a Mamba-2 block and an attention block side by side and adds their outputs; 2 KV heads). Two measurements:

- **(a) the cache against the prompt length.** W3 demo 2's method, unchanged: prefill a prompt of $L$ random tokens for $L$ from 1k to 32k and read off how many bytes the cache holds, how much memory the process actually grew by, and how long the prefill took. New here: the cache of a hybrid layer is not one tensor but three — the attention's keys and values, which grow with $L$, and the Mamba-2 block's SSM state and convolution state, which do not — and the program walks the cache and adds them up separately.
- **(b) key–value retrieval.** W5 demo 3's task, unchanged, with the same $K$ ($81$ and $326$ pairs) and the same seed, so both models are asked the same 200 questions that W5 asked Llama 3.1 8B: a JSON object of UUID pairs, the value of one key, the answer pair moved to five positions.

In class, two or three questions are put to both models live, and the rehearsed tables are replayed.

## What is being measured

For (a), the formula under the curves is the one on the slides. Attention pays $2 \, n_{\text{attn}} \, n_{\text{kv}} \, d_{\text{head}} \, b$ bytes **per token**: Qwen3-8B $2 \times 36 \times 8 \times 128 \times 2 = 144$ KiB, Falcon-H1-7B $2 \times 44 \times 2 \times 128 \times 2 = 44$ KiB (fp16 keys and values; $n_{\text{attn}}$ is counted from the cache the program actually builds, not read from the config). The Mamba-2 half pays a **fixed** state per layer — $n_{\text{head}} \times d_{\text{head}} \times d_{\text{state}} = 24 \times 128 \times 256$ elements, plus a few tokens of convolution state — whose size the program measures after an 8-token probe and then checks at every $L$. So the hybrid's line is not flat: it has attention in every layer. It has a slope about three times smaller than Qwen3-8B's, and an intercept above zero. A flat line needs a model with no attention at all (the optional third model, pure Mamba-2, if it loads).

For (b), a reply counts as **exact match** when, after stripping quotes and whitespace, it equals the value; the kinds of wrong answer (another pair's value and how far away, a key, a UUID not in the context, no UUID) are recorded as in W5. The question the slide asks is whether a model with 2 KV heads per layer and a fixed-size state beside them retrieves a literal key as reliably as one with 8 KV heads and nothing else — and if not, at which positions and which length it starts to lose. Either answer is informative; the slide does not predict one.

What this does and does not show: one model of each kind, one quantisation, one retrieval task. The curves have the shape of the cost accounting, not the ranking of any benchmark.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`; `falcon_h1.py` is in that commit), `huggingface-hub`; `transformers` for `inspect` (tokenizers only) |
| Models | `mlx-community/Qwen3-8B-4bit` (≈ 4.6 GB, shared with W3 demo 2 and W5 demo 3) and `mlx-community/Falcon-H1-7B-Instruct-4bit` (≈ 4.3 GB, `fetch` once; revisions pinned in `../versions.lock`). Optional: `mlx-community/Mamba-Codestral-7B-v0.1-4bit` for the flat line |
| Device | **Apple silicon only** (MLX). Both models loaded together for the live questions (≈ 9 GB) plus up to 4.5 GiB of cache at 32k |
| Time | (a) five prefills per model, minutes; (b) 400 generations — about an hour is the estimate; **not yet measured** (the rehearsal has not been run; `runs/rehearsal/` will hold the numbers) |
| Portability | to be tested on an Apple M5 Max (64 GB), macOS 27, Python 3.12 |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w06_d4_hybrid_vs_full
./present.sh fetch                    # once: download Falcon-H1-7B-Instruct-4bit and pin its revision
./present.sh inspect                  # per model: the cache tree, state shapes and dtypes, the KV formula, the fixed state, token counts
./present.sh rehearse --part memory   # (a): prompts of 1k … 32k tokens, both models → runs/rehearsal/memory.json
./present.sh rehearse --part recall --q 2   # (b) smoke test: 2 questions per cell, written to runs/smoke/
./present.sh rehearse --part recall   # (b): 2 sizes × 5 positions × 20 questions × 2 models → runs/rehearsal/recall_*.json
./present.sh replay                   # the rehearsed tables, no model loaded
./present.sh                          # the class flow: live questions to both models; m / r replay (a) / (b)
./present.sh fake                     # the same flow without mlx (fake numbers, marked as such)
```

Without `present.sh`: `python demo.py replay`, `python demo.py --part memory rehearse`, and so on, with `mlx_lm` importable and `HF_HUB_OFFLINE=1` set. `demo.py` imports `../w03_d2_kv_memory/demo.py` and `../w05_d3_lost_in_middle/demo.py` by file path, so the three folders have to stay side by side.

## What you should see

**Rehearsal pending.** The folder was built on 2026-10-19 and the flow was exercised only with `--fake`; the real numbers will be in `runs/rehearsal/` and on the slide once the rehearsal has run. Until then, the predictions from the formulas (fp16 KV; the SSM state assumed fp32, which `inspect` will confirm or correct):

| $L$ | Qwen3-8B, KV | Falcon-H1-7B, KV + fixed state |
|---|---|---|
| 1k | 144 MiB | 44 MiB + ≈ 133 MiB |
| 4k | 576 MiB | 176 MiB + ≈ 133 MiB |
| 16k | 2.25 GiB | 704 MiB + ≈ 133 MiB |
| 32k | 4.50 GiB | 1.375 GiB + ≈ 133 MiB |

The rehearsal will replace these with measured cache sizes, the memory growth of the process, prefill times, and the ten retrieval cells per model.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `inspect` / `rehearse` / `show` / `replay`; the cache walk and the two-model tables; everything else imported from W3 demo 2 and W5 demo 3 |
| `demo_config.toml` | the models and their order, the five prompt lengths, $Q$, seed, the fixed $K$ per size, the live questions, the prompt wording |
| `present.sh` | entry point; uses `../.venv/bin/python`, offline |
| `runs/rehearsal/` | `inspect.json`, `memory.json` (a), `recall_summary.json` and `recall_<model>.json` (b) |
