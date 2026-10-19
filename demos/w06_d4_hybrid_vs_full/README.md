<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 4 — A parallel hybrid against full attention: cache against $L$, and recall (slides p36–p37)

Two models of comparable size, both 4-bit MLX: **Qwen3-8B** (full attention in every layer, 36 layers, 8 KV heads) and **Falcon-H1-7B-Instruct** (a *parallel* hybrid: every one of its 44 layers runs a Mamba-2 block and an attention block side by side and adds their outputs; 2 KV heads). Two measurements:

- **(a) the cache against the prompt length.** W3 demo 2's method, unchanged: prefill a prompt of $L$ random tokens for $L$ from 1k to 32k and read off how many bytes the cache holds, how much memory the process actually grew by, and how long the prefill took. New here: the cache of a hybrid layer is not one tensor but three — the attention's keys and values, which grow with $L$, and the Mamba-2 block's SSM state and convolution state, which do not — and the program walks the cache and adds them up separately.
- **(b) key–value retrieval.** W5 demo 3's task, unchanged, with the same $K$ ($81$ and $326$ pairs) and the same seed, so both models are asked the same 200 questions that W5 asked Llama 3.1 8B: a JSON object of UUID pairs, the value of one key, the answer pair moved to five positions. Under these two tokenizers the same JSON is about 5.6k and 22k tokens (Qwen3: 5,576 / 22,163; Falcon-H1: 5,999 / 23,870), not the 4k and 16k it was under Llama 3's — hex strings split differently.

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
| Time | (a) five prefills per model, under a minute each (18–19 s for the 32k prefill); (b) 400 generations, 2,124 s for Qwen3-8B and 2,530 s for Falcon-H1 — about 4–5 s per $K = 81$ question and 17–20 s per $K = 326$ question; peak memory 8.5 / 9.0 GB |
| Portability | tested only on an Apple M5 Max (64 GB), macOS 27.0.1, Python 3.12, mlx 0.32.2, mlx-lm 0.32.0 |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w06_d4_hybrid_vs_full
./present.sh fetch                    # once: download Falcon-H1-7B-Instruct-4bit and pin its revision
./present.sh inspect                  # per model: the cache tree, state shapes and dtypes, the KV formula, the fixed state, token counts
./present.sh probe                    # per model: one plain question and a 5-pair / 20-pair version of the task, raw replies printed
./present.sh rehearse --part memory   # (a): prompts of 1k … 32k tokens, both models → runs/rehearsal/memory.json
./present.sh rehearse --part recall --q 2   # (b) smoke test: 2 questions per cell, written to runs/smoke/
./present.sh rehearse --part recall   # (b): 2 sizes × 5 positions × 20 questions × 2 models → runs/rehearsal/recall_*.json
./present.sh replay                   # the rehearsed tables, no model loaded
./present.sh                          # the class flow: live questions to both models; m / r replay (a) / (b)
./present.sh fake                     # the same flow without mlx (fake numbers, marked as such)
```

Without `present.sh`: `python demo.py replay`, `python demo.py --part memory rehearse`, and so on, with `mlx_lm` importable and `HF_HUB_OFFLINE=1` set. `demo.py` imports `../w03_d2_kv_memory/demo.py` and `../w05_d3_lost_in_middle/demo.py` by file path, so the three folders have to stay side by side.

## What you should see

From `runs/rehearsal/` (2026-10-19, Apple M5 Max, macOS 27.0.1, Python 3.12, mlx 0.32.2, mlx-lm 0.32.0 at the pinned commit).

**(a) The cache against $L$** (`memory.json`; bytes held by the cache tensors after the prefill):

| $L$ | Qwen3-8B (KV only) | Falcon-H1-7B: total | of which KV | fixed state | prefill, Qwen3 / Falcon |
|---|---|---|---|---|---|
| 1k | 144.0 MiB | 176.9 MiB | 44.0 MiB | 132.9 MiB | 0.46 s / 0.71 s |
| 4k | 576.0 MiB | 308.9 MiB | 176.0 MiB | 132.9 MiB | 1.1 s / 1.7 s |
| 8k | 1.125 GiB | 484.9 MiB | 352.0 MiB | 132.9 MiB | 2.5 s / 3.8 s |
| 16k | 2.250 GiB | 836.9 MiB | 704.0 MiB | 132.9 MiB | 6.9 s / 8.3 s |
| 32k | 4.500 GiB | 1.505 GiB | 1.375 GiB | 132.9 MiB | 18.5 s / 19.1 s |

Both lines sit on their formulas to the byte. Qwen3-8B holds 36 `KVCache` objects in bfloat16: $2 \times 36 \times 8 \times 128 \times 2 = 147{,}456$ bytes per token. Falcon-H1 holds, in every one of its 44 layers, a `KVCache` (2 KV heads, 45,056 bytes per token) *and* an `ArraysCache` with two tensors: the SSM state, shape $1 \times 24 \times 128 \times 256$ in **float32** — 3,145,728 bytes per layer, 138,412,032 bytes in all, exactly the config formula — and a small convolution state ($1 \times 3 \times 3584$, bfloat16). The fixed part is 132.9 MiB at every length. The measured slope ratio is 3.27, the formula's 3.27; the hybrid's line starts *above* the full-attention line (177 against 144 MiB at 1k) and crosses it near $L \approx 1{,}360$ tokens — after that it is cheaper, by a factor that tends to three. Prefill time did not follow the cache: 19.1 s against 18.5 s at 32k, because the SSM half of every layer is computed in 256-token chunks during the prefill, and that work does not go away when the cache does. (The process's resident memory grew by about 0.3–0.6 GiB more than the cache tensors for Falcon-H1, a constant that did not grow with $L$ and was not traced; the table reports the tensors.)

**(b) Key–value retrieval** (`recall_summary.json`; exact matches out of 20):

| answer position | Qwen3-8B, $K = 81$ (≈ 5.6k tok) | Qwen3-8B, $K = 326$ (≈ 22k tok) | Falcon-H1-7B, $K = 81$ | Falcon-H1-7B, $K = 326$ |
|---|---|---|---|---|
| $1$ | 15 | 16 | 0 | 0 |
| $\lfloor K/4 \rfloor$ | 16 | 5 | 0 | 0 |
| $\lfloor K/2 \rfloor$ | 17 | 3 | 0 | 0 |
| $\lfloor 3K/4 \rfloor$ | 17 | 3 | 0 | 0 |
| $K$ | 20 | 19 | 0 | 0 |

**Qwen3-8B shows the U-shape of Liu et al. at 22k tokens**, and a deep one: three correct out of twenty in the middle three positions, with the wrong answers split between another pair's value from tens of pairs away and a UUID that is not in the context at all. At 5.6k it is not perfect either — at the first position it answered with the key itself five times. These are the very questions that Llama 3.1 8B answered 200 of 200 in W5 demo 3 (at 4.1k and 16.3k tokens under its tokenizer); three things changed at once — the model, its quantisation, and the token count — so the comparison says that lost-in-the-middle is alive on a 2025 8B model at 22k, not which model is better.

**Falcon-H1-7B scored 0 of 200, and the zeros are not retrieval errors.** Not one reply contained a well-formed UUID: the model produced fragments of hex ("cfbf18-bf110c-7cf1-cfeb-…", the same few syllables repeating) or filler prose ("The answer originates from the fact the extraction originates from…"), usually running to the 48-token limit, on the 5.6k prompts as much as the 22k ones and in the smoke test as well. The chat template accepted the system message and the prompt is otherwise identical to Qwen3's. So the hybrid's recall was not measured here — only its failure to generate the kind of output the task asks for. Whether the cause is the 4-bit MLX quantisation, the port's handling of the model's muP scaling multipliers, or the model itself on this kind of prompt was not determined; `probe` (one plain question, then the task with 5 and 20 pairs) is the first thing to run before drawing a conclusion. The live question in class (answer at the first of 326 pairs): Qwen3 correct in 9.7 s, Falcon-H1 wrong in 13.0 s.

What this does and does not show: the cost accounting of the slides is confirmed exactly, down to the dtype of the state. The retrieval comparison the slide set out to make did not happen, because one of the two models did not answer; the result that did appear — a U on Qwen3-8B at 22k — is the W5 lesson in reverse.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `inspect` / `probe` / `rehearse` / `show` / `replay`; the cache walk and the two-model tables; everything else imported from W3 demo 2 and W5 demo 3 |
| `demo_config.toml` | the models and their order, the five prompt lengths, $Q$, seed, the fixed $K$ per size, the live questions, the prompt wording |
| `present.sh` | entry point; uses `../.venv/bin/python`, offline |
| `runs/rehearsal/` | `inspect.json`, `memory.json` (a), `recall_summary.json` (b, the tables) and `recall_<model>.json` (b, every question: prompt tokens, reply, verdict, seconds) |
