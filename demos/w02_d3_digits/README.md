<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W2 · Demo 3 — one model, one set of sums, three ways of writing the numbers (slide p48)

The same model gets the same random multi-digit additions written three ways, and nothing else changes:

| | Example | What the pre-tokenizer does with it |
|---|---|---|
| (A) plain | `5591 + 3092 =` | Llama 3 / o200k-style tokenizers group digits in threes **from the left** (`\p{N}{1,3}`): `559 | 1` |
| (B) thousands separators | `5,591 + 3,092 =` | the commas force groups of three **from the right**, the direction the carry runs |
| (C) one digit per token | `5 5 9 1 + 3 0 9 2 =` | every digit is its own token |

Accuracy is reported per number length (4, 7, 10 digits). The main model is Llama 3.1 8B Instruct, whose digit rule is the left-to-right `\p{N}{1,3}`; the control is Qwen3 8B, which tokenizes digits one at a time, so for it (A) and (C) are already the same segmentation and the three columns should differ much less. If the effect follows the tokenizer rather than the model's size or "reasoning", that is misconception 6 of the course — "LLMs can't do arithmetic because they can't reason" — taken apart with one variable.

Prompt: system *"You are a calculator. Reply with the final answer only: digits, nothing else."*, user `{a} + {b} =`, the model's own chat template, thinking disabled where the template allows it (`enable_thinking=False`), greedy decoding, 48 max tokens. Scoring takes the first run of digits (commas and spaces allowed, then stripped) after the last `=` in the reply, or from the first line containing digits, and compares it with the exact sum — so an answer written with commas or spaces is still correct, and a reply that restates the question is still parsed.

Keys during `show`: **Enter** runs the next number length (three columns × `n_live` = 20 problems, printing ✓/✗ and the raw reply for each); **q** loads the control model and runs the 7-digit row; **f** shows the full rehearsal tables (50 problems per cell, dated) — the numbers quoted on the slide; **x** quits.

## Requirements

| | |
|---|---|
| Platform | **Apple silicon only**: both models are 4-bit MLX conversions run through `mlx-lm` (pinned in `../versions.lock`). |
| Models | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (≈ 4.5 GB) and `mlx-community/Qwen3-8B-4bit` (≈ 4.6 GB), downloaded by `./present.sh fetch --only llama31_8b` and `--only qwen3_8b`. The other four candidates in `demo_config.toml` were never downloaded. |
| Memory | one model at a time; rehearsal peaks were 4.8 GB. |
| Time | 8B models: a cell of 50 problems takes 10–20 s; a row of 3 × 20 live problems 30–60 s. |
| Tested on | Apple M5 Max, macOS 26.6, Python 3.12.14, mlx 0.32.2, mlx-lm at commit `872ae88` (2026-09-20) |

## Run it

```bash
cd demos && ./setup.sh --no-fetch          # once: shared .venv with the pinned mlx / mlx-lm (see ../README.md)
cd w02_d3_digits
./present.sh fetch --only llama31_8b       # once, needs network
./present.sh fetch --only qwen3_8b
./present.sh inspect                       # tokenizer only: how each model splits the three spellings; whether thinking can be turned off
./present.sh                               # the demo: Enter per row (4 / 7 / 10 digits); q = control model; f = rehearsal tables; x = quit
./present.sh scan                          # model selection: every downloaded candidate × 3 lengths × 3 spellings, 20 problems each
./present.sh rehearse                      # main and control, 50 problems per cell → runs/rehearsal/
./present.sh replay                        # the instructor's recorded tables (marked as not live)
./present.sh fake                          # the screens with made-up accuracies; no model needed
```

Problems are generated from `seed = 20260920` and the number length, so `scan`, `rehearse` and `show` use the same sums (the three example sums printed on the slide are a different draw, shown only to illustrate the spellings). `rehearse` overwrites `runs/rehearsal/main-*.json` and `control-*.json`; `git checkout` restores the instructor's.

## What you should see

From `runs/rehearsal/` (2026-09-20, 50 problems per cell, greedy):

| Llama 3.1 8B | (A) plain | (B) commas | (C) spaced digits | B − A |
|---|---:|---:|---:|---:|
| 4 digits | 86 % | 86 % | 8 % | +0 |
| 7 digits | 42 % | **88 %** | 2 % | **+46** |
| 10 digits | 20 % | **84 %** | 0 % | **+64** |

| Qwen3 8B (control) | (A) | (B) | (C) | B − A |
|---|---:|---:|---:|---:|
| 4 digits | 94 % | 96 % | 16 % | +2 |
| 7 digits | 86 % | 94 % | 2 % | +8 |
| 10 digits | 82 % | 80 % | 0 % | −2 |

Two things to read off:

- **The commas do the work on Llama and not on Qwen.** For Llama, 10-digit accuracy goes from 20 % to 84 % with no change but the separators; for Qwen, which already sees one digit per token, the (A)/(B) gap is within noise. The effect follows the tokenizer.
- **(C) collapses on both models — and that is real, not a scoring artefact.** Reading the raw replies, Llama reads `4 5 0 5 + 7 4 9 0` as 4390 and `9 4 1 1` as 9421; Qwen does the same kind of thing. One digit per token is in principle the cleanest representation, but neither model was trained on numbers written that way. So the lesson has two halves: the representation matters, *and* so does whether the model was trained on that representation.

With 50 problems per cell the 95 % interval is about ±14 points (±20 at the 20 problems of a live row), so read directions, not decimals. A cell marked `!` means `<think>` tags appeared in the reply and were stripped before scoring — the model reasoning its way around the tokenizer, which is where W10 picks this up.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `inspect` / `scan` / `rehearse` / `show` / `replay`; `--fake`, `--only <key>` |
| `demo_config.toml` | candidates (`main` / `control` chosen after `scan`), number lengths, problems per cell, seed, prompt |
| `present.sh` | entry point; `HF_HUB_OFFLINE=1` except for `fetch`; uses `../.venv/bin/python` |
| `runs/rehearsal/main-llama31_8b.json`, `control-qwen3_8b.json` | the tables above, with every problem, raw reply and verdict |
| `runs/rehearsal/scan.json` | the model-selection run (20 problems per cell) |
