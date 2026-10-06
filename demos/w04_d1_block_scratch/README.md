<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W4 · Demo 1 — RMSNorm → SwiGLU → Block → GPT, and inverting `config.json` (slides p16, p20, p22)

Three things happen in this folder, matching three slides:

1. **p16 — assemble a Transformer block live.** In `block_live.py` the instructor writes `RMSNorm`, `SwiGLU`, `Block` and `GPT` from zero, importing last week's `scaled_dot_product_attention` and `causal_mask` from W3 demo 1. The multi-head reshape (`CausalSelfAttention`) is already in the file: it is written but not taught. Tests run after each piece.
2. **p20 — invert a real `config.json`.** `count_params.py` reads a model's `config.json` and prints three columns: the slide's approximation $12\,n_{\text{layer}} d^2 + Vd$, a term-by-term exact count, and the number of parameters actually stored in the safetensors files (read from the file headers only — no weights are loaded). The exact column must equal the actual column; the gap between the approximation and the exact count is what slide p21 explains.
3. **p22 — replay a training run.** `train.py` trains the tiny GPT on the same Shakespeare text as W1's n-gram models, with the same held-out split, so that its bits-per-byte can be put next to W1's bigram. Training is done in rehearsal (about ten minutes); in class only the record is replayed.

## The three tests

| | Checks | Passes when | Why it matters |
|---|---|---|---|
| (a) | `count_params(model)` on the tiny configuration against a term-by-term count that `tests.py` computes on its own from `demo_config.toml`: embedding $Vd$, per layer $4d^2 + 3\,d\,d_{\text{ff}} + 2d$, final norm $d$ | **exactly equal**; also prints the approximation $12\,n_{\text{layer}} d^2 + Vd$ and its relative error (about −0.6 % here, because $d_{\text{ff}} = 688 \approx \tfrac{8}{3}d$) | the blackboard accounting and the code agree. Needs `GPT` to exist, so during the live build it reads `skip` until the end |
| (b) | `count_params.py` on the two models in `demo_config.toml`: Llama 3.1 8B Instruct (MLX 4-bit) and Qwen3-0.6B | exact count = stored count, for both; a mismatch is printed by category (embedding / attention / ffn / norm / head / other). Models not in the local cache → `skip` | approximation vs. exact is the four corrections on p21; exact vs. stored should have no gap at all |
| (c) | Shapes step by step: `RMSNorm` (same shape, row RMS becomes 1) → `SwiGLU` (same shape) → `Block` ($(B, L, d)$ in and out, finite) → `GPT` ($(B, L, V)$, all logits finite) | all four `ok`; a half-written file reports `info` | a model that can be trained |

Two details of (b) that are easy to get wrong and are printed on screen: MLX 4-bit safetensors pack eight weights into one `uint32`, so element counts are multiplied back by $32/\text{bits}$ and the `.scales` / `.biases` tensors are listed separately (they are quantization metadata, not parameters); and Qwen3-0.6B has `tie_word_embeddings: true` but still stores `lm_head.weight` in the file, so the file holds $Vd = 155{,}582{,}464$ more values than the model has parameters — file size is not parameter count.

## Requirements

| | |
|---|---|
| Packages | `torch 2.14.0` only. `count` additionally reads files from the Hugging Face cache but imports nothing else. |
| Code dependency | imports `../w03_d1_attention_scratch/attention_final.py` by file path — keep the `demos/` folder structure as published. |
| Models (only for `count` / test (b)) | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` and `Qwen/Qwen3-0.6B` must already be in the local Hugging Face cache — W3 demo 2's and demo 3's `./present.sh fetch` put them there. Nothing is downloaded by this demo. Without them (b) is `skip` and `count` reports the model as not local. |
| Corpus (only for training) | `slides/_corpus/tinyshakespeare.txt`, i.e. [karpathy's `tinyshakespeare/input.txt`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (public-domain text, 1.1 MB). It is not in this repository; download it to `slides/_corpus/tinyshakespeare.txt` relative to the repository root (the path is set in `demo_config.toml`). The W1 n-gram numbers it is compared with are read from `slides/assets/w01/w01-data.json`, which is in the repository. |
| Device | tests and `count`: CPU, well under a second. Training: `mps` by default, falls back to CPU automatically (and says so); on CPU expect it to be many times slower than the 9 minutes below. `device = "cuda"` in `demo_config.toml` is accepted by the code but has never been run. |
| Memory | training: ≈ 4.8 M parameters, batch 64 × 256 — well under 2 GB. |
| Tested on | Apple M5 Max, macOS 27.0, Python 3.12.14, torch 2.14.0 (2026-10-06) |

## Run it

```bash
cd demos                          # see ../README.md for creating .venv
cd w04_d1_block_scratch
./present.sh test                 # three tests against block_final.py: a ok, b ok (or skip without the models), c ok
./present.sh show                 # shapes and parameter table printed stage by stage, then the three tests
./present.sh count llama          # the three columns for Llama 3.1 8B;  ./present.sh count qwen  for Qwen3-0.6B
./present.sh count llama --config-only   # only the seven config keys, if you want to compute the first column yourself first
./present.sh reset                # empties block_live.py below the "write from here" line
./present.sh live                 # the three tests against *your* block_live.py; unwritten parts show as skip / info
./present.sh probe                # is MPS available, does the built-in SDPA accept dropout on MPS, seconds per training step
./present.sh peek                 # a few hundred live training steps, just to see the loss start to fall (nothing saved)
./present.sh rehearse             # the three tests, then the full training run → runs/rehearsal/ (≈ 10 min on the M5 Max)
./present.sh replay train         # the recorded loss curve and the BPB table against W1's n-grams (no training)
./present.sh replay               # the recorded tests, including the three columns of count
```

`rehearse` overwrites `runs/rehearsal/*.json` and `loss.svg` with your run; `git checkout` the folder to get the instructor's back.

## What you should see

From `runs/rehearsal/` (2026-10-06):

- **(a)** `count_params = 4,763,136`, term by term `4,763,136` — difference zero. Approximation `4,735,232`, relative error −0.59 % ($V = 65$ characters).
- **(b)** Llama 3.1 8B: approximation 7,493,124,096 / exact **8,030,261,248** / stored **8,030,261,248** (approximation −6.7 %; 743 tensors, 226 of them quantized, 250,937,344 scale/bias values listed separately). Qwen3-0.6B: 507,904,000 / **596,049,920** / **596,049,920** (approximation −14.8 %; the duplicated `lm_head` is reported and excluded).
- **(c)** four stages `ok`; `GPT` output `(2, 16, 65)`, finite.
- **probe**: MPS available; the built-in `F.scaled_dot_product_attention` accepts `dropout_p = 0.1` on MPS with finite gradients on torch 2.14.0; 0.135 s per training step.
- **train**: 3000 steps in 557 s (88k tokens/s); held-out **1.514 nats/char = 2.184 bits/char**; `bpb_raw` 2.184, `bpb_w1_denominator` 2.103, against W1's bigram at 1.861. The loss curve flattens at 2.18–2.19 bits after about 2000 steps. The record also holds a 300-character sample from the trained model.

On the BPB comparison: the held-out *text* is the same as W1's, but the prediction *target* is not — W1's n-grams predict only the regex-tokenized words and punctuation, the character model predicts every character including spacing and case. The two BPB values use two denominators (raw UTF-8 bytes; W1's `' '.join(tokens)` bytes), but no denominator removes that difference. This is the W1 p27 point in practice: bits-per-byte is comparable only over the same byte stream.

## Files

| File | |
|---|---|
| `block_final.py` | reference `RMSNorm` / `SwiGLU` / `CausalSelfAttention` / `Block` / `GPT` / `count_params`; `--show`, `--rehearse`. Also imported by W4 demo 2 (`norm="post"`, `qk_norm`, `record_stats` exist for it) |
| `block_live.py` | the file written live in class; `./present.sh reset` empties the live part |
| `tests.py` | the three tests, usable against any module exposing the same classes |
| `count_params.py` | p20: `config.json` → three columns; `--config-only` |
| `train.py` | `probe` / `peek` / `rehearse` / `replay`; character-level data loading, W1-aligned held-out split, BPB with both denominators |
| `demo_config.toml` | tiny GPT size, training hyperparameters, corpus path, the two models to invert |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/tests.json`, `probe.json`, `train.json`, `loss.svg` | the rehearsal record quoted above |
