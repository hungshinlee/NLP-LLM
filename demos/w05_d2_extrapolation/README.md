<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 2 — Length extrapolation on last week's tiny GPT: NoPE vs. RoPE, then PI and NTK-aware without retraining (slides p25–p26)

Last week's tiny GPT (W4 demo 1: character-level, six layers, $d = 256$, trained on windows of $L = 256$ characters) has **no positional encoding** — position is inferred from the causal mask alone (NoPE). This demo trains a second copy that is identical in every respect except one: RoPE on $\mathbf{q}$ and $\mathbf{k}$, base $b = 10^4$. Same corpus, same held-out text, same hyperparameters, same 3000 steps, same random initialisation. Then both are run on held-out windows of 256, 512 and 1024 characters — one, two and four times the training length — and the **loss at each position** is recorded. Finally the RoPE model is run twice more **without any retraining**, with its $\theta_i$ changed: position interpolation ($s = 4$) and NTK-aware base scaling. Four curves, four columns in the table on the slide.

Nothing is run live in class; the rehearsal records in `runs/rehearsal/` are replayed.

## What is being measured

For each model and each $L_{\text{eval}} \in \{256, 512, 1024\}$, the held-out text is cut into non-overlapping windows of that length and the mean cross-entropy at every position (in nats per character) is recorded. The slide's table takes two numbers from the 1024-window pass: the mean over positions below 256 (inside the training length) and the mean over positions 256–1024 (beyond it).

One fact worth knowing before reading the curves: in a causal model the loss at position $p$ depends only on characters $0 \ldots p$, not on how long the window is. So the first 256 positions of the 1024-window curve *are* the 256-window curve (the windows start at different offsets, so the digits agree only to about three places). The *mean over the whole window* does change with $L_{\text{eval}}$ — because the range being averaged changes, not because the model behaves differently in a short window. That is why the per-position curve is a more honest picture than a single "perplexity at length $L$".

## The two repairs

Both change only the frequencies $\theta_i = b^{-2i/d_{\text{head}}}$ and leave the trained weights alone (`rope_gpt.set_rope`):

| | What changes | Here ($d_{\text{head}} = 32$, $s = 4$) |
|---|---|---|
| position interpolation (PI) | positions $m \to m/s$, equivalently $\theta_i \to \theta_i/s$ — every pair slowed down by the same factor | positions 0–1023 are presented as 0–255.75 |
| NTK-aware | the base: $b \to b\,s^{d_{\text{head}}/(d_{\text{head}} - 2)}$ — the lowest-frequency pair is slowed by exactly $s$, the highest almost not at all | $b = 10^4 \to 43{,}873$ |

With $d_{\text{head}} = 32$ there are 16 rotating pairs; at $b = 10^4$, 9 of them never complete a full turn within the training length of 256 (the program prints this). The slides' figure on p20 uses $d_{\text{head}} = 128$, where the count is 38 of 64.

## Requirements

| | |
|---|---|
| Packages | `torch` only (`2.14.0` pinned). No model download, no network. |
| Other folders | imports `../w04_d1_block_scratch/block_final.py` (the `GPT`) and `train.py` (corpus, device, optimiser groups, the held-out measure), and reads that folder's `demo_config.toml`; `block_final.py` in turn imports W3 demo 1's `attention_final.py`. Keep the folder layout. |
| Corpus | `slides/_corpus/tinyshakespeare.txt`, not in the repository — see W4 demo 1's README for where to download it (about 1.1 MB). |
| Device | training uses `mps` on Apple silicon and falls back to the CPU (slow); `cuda` is accepted by the code but has never been run. `replay` and `show` need no device at all. |
| Memory / time | the model has 4,763,136 parameters (RoPE adds none); about 20 minutes for the two training runs on the machine below, under a minute for the measurements. Two 19 MB weight files are written to `runs/weights/` (not committed). |
| Portability | nothing platform-specific; tested only on an Apple M5 Max (64 GB), macOS 27.0.1, Python 3.12, torch 2.14.0 |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv
cd w05_d2_extrapolation
./present.sh replay           # the rehearsal records, one curve per Enter, then the slide's table
./present.sh show             # the same without pauses
./present.sh estimate         # how long rehearse will take, from W4 demo 1's timing probe
./present.sh rehearse --steps 30   # smoke test: 30 steps per model + all measurements, written to runs/smoke/
./present.sh rehearse         # the real thing: train nope, train rope, measure four variants → runs/weights/, runs/rehearsal/
./present.sh measure          # re-measure from the saved weights (e.g. after changing s in demo_config.toml)
```

Without `present.sh`: `python demo.py replay`, `python demo.py rehearse`, and so on, with `torch` importable.

## What you should see

From `runs/rehearsal/summary.json` (2026-10-12, Apple M5 Max, macOS 27.0.1, Python 3.12, torch 2.14.0, `mps`; training took 629 s for NoPE and 664 s for RoPE). Mean loss in nats per character from the 1024-window pass:

| | NoPE | RoPE | RoPE + PI ($s = 4$) | RoPE + NTK-aware |
|---|---|---|---|---|
| positions $< 256$ | 1.50 | 1.62 | 4.69 | 1.74 |
| positions $256$–$1024$ | 2.71 | 3.84 | 4.80 | 2.36 |

The retrained NoPE model's held-out figure is 2.183 bits/char under W4's windowed measure, against 2.184 for the original W4 run — the consistency check `replay` prints. The RoPE model's is 2.371.

Things to notice, all specific to this setup: the NoPE model *also* degrades past 256 (the position it infers from the mask was never counted higher than that); the RoPE model is slightly worse than NoPE *inside* the training length (one run, one seed, hyperparameters tuned last week for NoPE — not a conclusion, just not a free improvement); position interpolation without fine-tuning is above $\ln 65 = 4.17$, the loss of guessing uniformly over the 65 characters — every pair's angular speed is divided by four, so adjacent characters sit at an angle the model only ever saw for a quarter of a character; NTK-aware scaling without fine-tuning stays flat to about position 600 and is the best of the four beyond 256, though it is still rising at 1024.

A caveat the program prints every time: this is a character-level model of under five million parameters, mostly learning spelling and formatting; its long-range dependencies are weak. The *shape* of each curve is the shape of a mechanism. None of the numbers is a threshold for any large model, and every conclusion holds for this setup only.

## Files

| File | |
|---|---|
| `rope_gpt.py` | the `pos` switch: `PosAttention` (rotate-half RoPE on $\mathbf{q}, \mathbf{k}$), `Demo1` (loads W4 demo 1), `set_rope`, `ntk_base` |
| `demo.py` | `rehearse` / `train` / `measure` / `replay` / `show` / `estimate` |
| `demo_config.toml` | RoPE base, evaluation lengths, bin size, the scale $s$; model size and training hyperparameters live in W4 demo 1's config |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/` | `train.json` (both training curves), one JSON per variant with the per-position losses at each length, `summary.json` (the slide's table), `curves.svg` |
