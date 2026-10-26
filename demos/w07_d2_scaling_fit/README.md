<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 2 — A small scaling law: five sizes on one ray, fit three, test two (slides p32–p33)

Five byte-level GPTs — W4 demo 1's block, with a vocabulary of the 256 byte values instead of characters — are trained on the same sample of FineWeb-Edu, with the same seed, each to **$D = 20N$ tokens** (the "Chinchilla ray"). The learning rate is scaled with width as muP prescribes ($\eta \propto 1/d$, from W4's $10^{-3}$ at $d = 256$). Then $\mathcal{L}(N) = E + A N^{-\alpha}$ is fitted to the **three smallest** models and extrapolated to the two largest; the error is reported. A fit to all five follows for comparison, and finally the five training curves are cut into an iso-FLOPs envelope (Hoffmann et al.'s Approach 1): at each compute budget $C = 6ND$, which size is lowest?

Nothing is run live in class; the rehearsal records in `runs/rehearsal/` are replayed. The demo is built to trip on the first two of the week's five methodological reasons a fit goes wrong — too few points, and a collinear design (every run on $D = 20N$, so the $D$ term is folded into $A$ and the fit cannot say anything off the ray). Whether the extrapolation error turns out large or small is the result; neither is a failure of the demo.

## What is being measured

| | |
|---|---|
| Corpus | FineWeb-Edu `sample-10BT` (ODC-By 1.0): about 200 MB of text read from the first parquet shard, row groups spread evenly over the shard so that several Common Crawl dumps are represented. Stored as JSONL in `../w07_corpus/` (not committed; `fetch` once). |
| Held-out | the last documents of the sample, about 1 MiB, never trained on. Loss in nats per byte; **bits/byte = nats ÷ ln 2**, because a token is a byte. |
| The five sizes | $d \in \{64, 96, 128, 192, 320\}$, 4–8 layers, $d_{\text{head}} = 32$, $d_{\text{ff}} \approx 8d/3$: 0.22 M, 0.58 M, 1.20 M, 3.15 M, 9.88 M parameters (total, embedding included — Chinchilla's count; the non-embedding count is also printed). Each trains for $20N$ tokens: 4.4 M to 198 M. |
| Training | AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1 on matrices, warmup 3 % then cosine to $0.1\,\eta$, clip 1.0, batch $32 \times 256$, no dropout; windows do not overlap and each epoch is a fresh permutation (the largest run sees the corpus slightly less than once). Only the learning rate follows muP; the initialisation and the output multiplier do not. |
| Recorded | every 50 steps, the loss on a fixed held-out subset (16,384 bytes) together with the tokens seen — the training curves; at the end, W4's windowed measure over the whole held-out (each byte predicted exactly once) — the number the fit uses. |
| Fit | $\alpha$ by a one-dimensional search, $E$ and $A$ by linear least squares at each $\alpha$ — deliberately the simplest version, not Hoffmann et al.'s (log space, Huber loss, many starting points). |

## Requirements

| | |
|---|---|
| Packages | `torch` (`2.14.0` pinned) for training; `huggingface_hub` and `pyarrow` for `fetch` only. `replay`, `show` and `fit` need neither. |
| Other folders | imports `../w04_d1_block_scratch/block_final.py` (the `GPT`) and `train.py` (device, optimiser groups, the learning-rate schedule, the held-out measure); `block_final.py` in turn imports W3 demo 1's `attention_final.py`. Keep the folder layout. |
| Network | only `fetch`, once (about 200 MB per corpus). Everything else runs offline. |
| Device | training uses `mps` on Apple silicon and falls back to the CPU (far slower); `cuda` is accepted by the code but has never been run. |
| Memory / time | the models are small (the largest 9.9 M parameters); the corpus is held in memory as one byte tensor (200 MB). Time is dominated by the largest run — see `estimate`, and the smoke test, which times each size. Weights (up to about 40 MB each) go to `runs/weights/` (not committed). |
| Portability | nothing platform-specific; tested only on an Apple M5 Max (64 GB), macOS 27, Python 3.12, torch 2.14.0 (the pipeline was also exercised once on Linux/CPU with 12 steps per size). |

## Run it

```bash
cd demos                       # see ../README.md for creating .venv
cd w07_d2_scaling_fit
./present.sh fetch             # once, with network: the FineWeb-Edu sample → ../w07_corpus/ (--corpus both also fetches FineWeb for demo 3)
./present.sh estimate          # the five sizes, their D and step counts, and a rough time estimate
./present.sh rehearse --steps 30   # smoke test: 30 steps per size → runs/smoke/, with a projected time for the full run
./present.sh rehearse          # the real thing: five runs → runs/weights/, runs/rehearsal/ (JSON, fit, summary, curves.svg)
./present.sh replay            # the rehearsal record, one stage per Enter: points → three-point fit → five-point fit → envelope → the slide's table
./present.sh show              # the same without pauses
```

Without `present.sh`: `python demo.py replay`, `python demo.py rehearse`, and so on, with `torch` importable.

## What you should see

The rehearsal has not been run yet (2026-10-26); the slide's table is blank until it has. When it has, `runs/rehearsal/summary.json` holds the table and `fit.json` the full fit: the five measured points, the three-point fit's $E, A, \alpha$ and its predictions at the two largest sizes against their measured loss, the five-point fit, and the envelope.

A caveat the program prints every time: these are models of a few million parameters on 200 MB of bytes, one seed, five points, one ray. What you see is the shape of a mechanism — few points, collinear design, extrapolation — not the numbers of any real scaling law. If the extrapolation error is small, the right conclusion is "within two decades a power law holds here", not "extrapolation is safe".

## Files

| File | |
|---|---|
| `train.py` | `fetch_corpus`, `load_corpus`, `Demo1` (loads W4 demo 1), `build_model`, `train_size` |
| `fit.py` | `fit_power`, `evaluate_fit`, `envelope` — standard library only |
| `demo.py` | `fetch` / `estimate` / `rehearse` / `train` / `fit` / `replay` / `show` |
| `demo_config.toml` | corpus sources and size, the split, the ray, training hyperparameters, the five sizes, the fit's settings |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/` | `s1.json`–`s5.json` (one per run: settings, curve, final held-out), `fit.json`, `summary.json` (the slide's table), `curves.svg` |
