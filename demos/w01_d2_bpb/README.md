<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W1 · Demo 2 — the same paragraph, two tokenizers (slide p28)

One Traditional Chinese paragraph (209 characters, 627 bytes, written for this course) is scored by two models whose tokenizers split it very differently, and the table on slide p28 is filled in step by step:

| Step | Screen | Fills in |
|---|---|---|
| 1 | the text | characters and UTF-8 bytes — identical for both models |
| 2 | how each tokenizer cuts the first sentence (coloured blocks; a subscript when several tokens make up one character) | tokens, tokens per character |
| 3 | one teacher-forced forward pass per model | perplexity |
| 4 | the same total negative log-likelihood divided by bytes instead of tokens, with the check $\text{BPB} = (T/B)\,\log_2 \text{PPL}$ | bits per byte |
| `l` | `decode(encode(text)) == text`? | lossless round trip — premise (a) of the next slide |

Scoring follows the lm-evaluation-harness convention: one start token (BOS if the model has one, else `<|endoftext|>`, else EOS), one forward pass, sum of the negative log-probabilities of the text's tokens in nats. $\text{PPL} = \exp(\text{NLL}/T)$ with $T$ the token count; $\text{BPB} = \text{NLL}/(\ln 2 \cdot B)$ with $B$ the byte count. Only the tokenizer's canonical segmentation is scored.

The point: **perplexity's denominator is the token count, so it moves with the tokenizer**; bits per byte keeps the same NLL and divides by something both models share, so the two land on one scale.

## Requirements

| | |
|---|---|
| Platform | **Apple silicon only**: 4-bit MLX models through `mlx-lm` (pinned in `../versions.lock`). |
| Models | default pair `mlx-community/Qwen3.8-27B-4bit` (≈ 16 GB; shared with W1 demo 1) and `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (≈ 4.1 GB). `gemma-4-26b-a4b-it-4bit` is configured but **not used for scoring** (see below). `../setup.sh --only w01_d2_bpb` downloads what is missing. |
| Memory | both models are loaded together; the rehearsal peaked at **19.7 GB** → a 32 GB Mac. |
| Time | a forward pass over 141–272 tokens: 0.25 s (Qwen3.8) and 0.5 s (Mistral); loading the two models takes longer than the demo. |
| Tested on | Apple M5 Max, 64 GB, macOS 26.6, Python 3.12.14, mlx 0.32.2, mlx-lm at commit `872ae88` (2026-09-20) |

## Run it

```bash
cd demos && ./setup.sh --no-fetch                  # once: shared .venv with the pinned mlx / mlx-lm (see ../README.md)
(cd demos && ./setup.sh --only w01_d2_bpb)         # once, needs network: downloads the models (skips the ones already present)
cd w01_d2_bpb
./present.sh inspect                               # tokenizers only, all configured models: token counts and lossless round trip (fast)
./present.sh                                       # the demo: Enter through steps 1–4, l = round trip, q = quit
./present.sh rehearse                              # the full computation for the default pair → runs/rehearsal/zh_city-qwen38-mistral7b.json
./present.sh --pair qwen38 gemma4 rehearse         # another pair (flags before the subcommand in this demo)
./present.sh --text taigi rehearse                 # the Taiwanese paragraph, with 𪜶 (U+2A736, four bytes)
./present.sh replay                                # the instructor's recorded table (marked as not live)
./present.sh diag                                  # when a score looks wrong: BOS duplicated? three forward paths agree? English sanity check?
./present.sh fake                                  # the screens with made-up numbers; no model needed
```

## What you should see

From `runs/rehearsal/zh_city-qwen38-mistral7b.json` (2026-09-20):

| | Qwen3.8 27B | Mistral 7B v0.3 |
|---|---:|---:|
| tokens (209 characters, 627 bytes) | 141 | 272 |
| tokens per character | 0.675 | 1.301 |
| start token | `<|endoftext|>` | `<s>` (BOS) |
| total NLL (nats) | 353.0 | 601.8 |
| **perplexity** | 12.23 | **9.14** |
| **bits per byte** | **0.812** | 1.385 |
| lossless round trip | yes | yes |

Mistral has the **lower perplexity** and the **worse** bits per byte. Its 32k vocabulary breaks most Han characters into three byte tokens, and once the first byte is seen the second and third are nearly certain — many easy tokens pull the per-token average down. Divide the same NLL by bytes instead and Qwen3.8 is better by 0.57 bit per byte. Judging by perplexity alone picks the wrong model; that is the slide.

On the Taiwanese paragraph both tokenizers are lossless and both cut 𪜶 into four byte tokens — byte-level fallback at work, the W2 story in advance.

**Gemma 4 is in the configuration but not in the default pair.** Scored with this `mlx-lm` commit it gives a perplexity of 3,435 on the paragraph (about 8 nats per token, close to uniform) and 5,843 on an English sentence, although it generates normally in W1 demo 1. `diag` ruled out a duplicated BOS and a forward-path bug (three paths agree); what has not been ruled out is the correctness of the Gemma 4 implementation for teacher-forced scoring, or the instruction tuning having crushed the likelihood of raw text. The record of that run is kept as `zh_city-qwen38-gemma4.json`; treat it as an open question, not a result.

## Files

| File | |
|---|---|
| `demo.py` | `inspect` / `rehearse` / `show` / `replay` / `diag`; `--pair`, `--text`, `--fake` |
| `demo_config.toml` | default text and pair, the two texts, the three models |
| `texts/zh_city.txt`, `texts/taigi.txt` | the two course-written paragraphs (also read by W2 demo 1 and W4 demo 3) |
| `present.sh` | entry point; `HF_HUB_OFFLINE=1`; uses `../.venv/bin/python` |
| `runs/rehearsal/zh_city-qwen38-mistral7b.json`, `zh_city-qwen38-gemma4.json` | the rehearsal records quoted above |
