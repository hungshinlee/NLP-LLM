<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 3 — Lost in the middle, scaled down: key-value retrieval with the answer at five positions (slide p29)

A scaled-down replication of the synthetic key–value retrieval experiment of Liu et al., *Lost in the Middle* (TACL 2024, §4), on the same Llama 3.1 8B Instruct (4-bit MLX) that W3 demo 2 measured. $K$ random UUID key–value pairs are written as one JSON object; the model is asked for the value of one key. The pair holding the answer is placed at position $1, \lfloor K/4 \rfloor, \lfloor K/2 \rfloor, \lfloor 3K/4 \rfloor, K$ — the same $K$ pairs, the same question, only the answer's position moves — and exact-match accuracy is recorded at each position, for two context sizes of about 4k and 16k tokens.

In class three questions are asked live (answer at the beginning, the middle, the end) and the rehearsed table is replayed.

## What is being measured

For each context size, $K$ is calibrated so that the whole prompt (instruction, JSON, question, chat template) is close to the target token count — the program prints the $K$ it chose and the actual token count. For each of $Q = 20$ questions, one set of $K$ pairs and one answer key are drawn from a seed; the question is then asked five times with the answer pair moved to each of the five positions and the other $K - 1$ pairs left in place. The model generates at temperature 0.

A reply counts as **exact match** when, after stripping quotes and whitespace, it equals the value. The program also records a lenient count (the first UUID in the reply is the value) and, for every wrong answer, what it was: another pair's value from the context (and how far from the answer), a key instead of a value, a UUID that is not in the context at all, or no UUID. The slide's table uses exact match; the kinds of error are what the live questions are there to show — "found the wrong one" is a different failure from "could not answer".

This is a retrieval task, not a reading task: a flat curve says the model can locate a literal string at this length, and no more. Whether the U-shape of Liu et al. appears, and how deep it is, depends on the model and the length (their experiments are on 2023 models); if a model is perfect at 16k here, that is the point made on the next slide — a needle test has saturated, and the evaluation has to get harder (more needles, aggregation, no literal overlap).

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `huggingface-hub`; `transformers` for `inspect` (tokenizer only) |
| Model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (≈ 4.5 GB, revision pinned in `../versions.lock`; shared with W2 demo 3 and W3 demo 2). Backup: `mlx-community/Qwen3-8B-4bit` |
| Device | **Apple silicon only** (MLX). About 4.5 GB for the weights plus 2 GiB of KV cache at 16k |
| Time | `rehearse` is 200 generations; prefill dominates — on the order of a second per 4k question and several seconds per 16k question (W3 demo 2's measured prefill rates). Not yet timed |
| Portability | tested only on an Apple M5 Max (64 GB), macOS 27.0.1, Python 3.12 |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv and downloading the model
cd w05_d3_lost_in_middle
./present.sh inspect          # calibrate K for both sizes, print one prompt and its token count
./present.sh rehearse --q 2   # smoke test: 2 questions per cell, written to runs/smoke/
./present.sh rehearse         # the real thing: 2 sizes × 5 positions × 20 questions → runs/rehearsal/
./present.sh replay           # the rehearsed table
./present.sh                  # the class flow: three live questions, then the table
./present.sh fake             # the same flow without mlx (fake replies, marked as such)
```

Without `present.sh`: `python demo.py replay`, `python demo.py rehearse`, and so on, with `mlx_lm` importable and `HF_HUB_OFFLINE=1` set.

## What you should see

**Not yet rehearsed (2026-10-12).** The rehearsal record (`runs/rehearsal/summary.json`) is checked in once it exists, and this section and the slide's table are filled from it. No expected numbers are written here on purpose: the shape of the curve at 16k is the result, not the premise.

## Files

| File | |
|---|---|
| `demo.py` | `inspect` / `rehearse` / `show` / `replay` / `fetch`; the task, the calibration of $K$, the judge |
| `demo_config.toml` | $Q$, seed, the two target sizes, which size and positions the live questions use, the prompt wording, the models |
| `present.sh` | entry point; uses `../.venv/bin/python`, offline |
| `runs/rehearsal/` | `inspect.json` (calibration), `summary.json` (the table and the kinds of error), `rehearsal.json` (every question: prompt tokens, reply, verdict, seconds) |
