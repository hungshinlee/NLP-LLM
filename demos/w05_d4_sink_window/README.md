<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 4 — A sliding window with and without the sink: per-token perplexity under four KV caches (slides p42–p43)

The same text, the same model, the same tokenizer — only the rule for what the KV cache keeps changes. Per-token negative log-likelihood is recorded under four caches: everything kept; only the last $W$ tokens kept; the last $W$ plus the first $n_{\text{sink}}$ tokens, at their original positions; and the last $W$ plus the first $n_{\text{sink}}$ with positions re-indexed inside the cache, which is what StreamingLLM (Xiao et al., ICLR 2024, §3) does. Everything is rehearsed beforehand and replayed in class, one line at a time.

## What is being measured

The text is the opening of *tinyshakespeare* (the first 20,000 characters, 5,455 tokens with Llama 3's tokenizer), fed as plain text without a chat template. The tokenizer of this checkpoint did not prepend a BOS token, so position 0 is the token of the word `First` — and the sink formed on it anyway. For each cache the model predicts every token given the cache it has at that moment, and the loss of that prediction is recorded. The slide's table reports perplexity, $\exp$ of the mean loss, over two ranges of token positions: before $W$ and from $W$ on. The curve in `runs/rehearsal/curve.svg` averages the loss over 256-token bins.

Why the first few tokens matter: softmax must put weight *somewhere*, and a head with nothing to attend to puts it on the first token, which every query can see (W3 p42; Xiao et al. §3). A plain sliding window drops that token at step $W$; the weight has to land on real tokens instead, and the prediction degrades at once. Keeping the first $n_{\text{sink}}$ tokens' KV under the window keeps the drain in place.

Why there are two "with sink" lines: in mlx-lm (0.32.0) the rotation for the position is applied to $\mathbf{q}$ and $\mathbf{k}$ *before* they enter the cache, at the token's absolute position in the stream, so the library's `RotatingKVCache(keep=n)` keeps the first $n$ tokens at their original positions — the sink's distance from the query grows without bound. StreamingLLM §3 assigns positions inside the cache instead, so the sinks stay a fixed distance of about $W + n_{\text{sink}}$ away. `SinkWindowCache` in `demo.py` does the re-indexing by hand: RoPE is a rotation, so a stored key can be re-rotated to any position at any time. At 5k tokens — far inside this model's nominal length of 131,072 — the two should behave similarly; the point of the fourth line is that a library argument named after the paper's idea need not implement it, and `inspect` prints the source lines that show where the positions are assigned. Read the source, not the README (course appendix C).

What this does not show: a cache that forgets gracefully is not a cache that remembers. Nothing that left the window can be retrieved; p44 is about that.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `huggingface-hub`; `transformers` for `inspect` (tokenizer only) |
| Model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (≈ 4.5 GB, revision pinned in `../versions.lock`; shared with W3 demo 2 and W5 demo 3). Backup: `mlx-community/Qwen3-8B-4bit` |
| Text | `slides/_corpus/tinyshakespeare.txt`, i.e. [karpathy's `tinyshakespeare/input.txt`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (public domain, 1.1 MB), not in this repository; download it to that path relative to the repository root, as W4 demo 1's README says |
| Device | **Apple silicon only** (MLX). About 4.5 GB for the weights; the caches are a few hundred tokens |
| Time | 804 s in all on the machine below: the full-cache line is 5 s of prefill; each windowed line is one forward pass per token (5,454 of them), 175–226 s at 25–31 tokens per second; peak memory 6.1 GB |
| Portability | tested only on an Apple M5 Max (64 GB), macOS 27.0.1, Python 3.12 |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv and downloading the model
cd w05_d4_sink_window
./present.sh inspect          # token count; the mlx-lm source lines; what RotatingKVCache(keep) keeps; three RoPE checks
./present.sh rehearse --chars 4000   # smoke test on ~1k tokens, written to runs/smoke/
./present.sh rehearse --check # the real thing: four caches (+ a cross-check of the hand-written cache) → runs/rehearsal/
./present.sh replay           # the class flow: one line per Enter, then the table
./present.sh fake             # the same flow without mlx (fake curves, marked as such)
```

Without `present.sh`: `python demo.py replay`, `python demo.py rehearse`, and so on, with `mlx_lm` importable and `HF_HUB_OFFLINE=1` set. Options go before the subcommand (`python demo.py --chars 4000 rehearse`).

## What you should see

From `runs/rehearsal/curve.json` (2026-10-12, Apple M5 Max, macOS 27.0.1, Python 3.12, mlx 0.32.2, mlx-lm 0.32.0; $W = 512$, $n_{\text{sink}} = 4$, 5,455 tokens):

| cache | perplexity, tokens $< W$ | perplexity, tokens $\geq W$ | worst 256-token bin | tokens in the cache |
|---|---|---|---|---|
| full | 5.64 | 6.46 | 13.8 | 5,455 |
| window, no sink | 5.64 | **619.82** | 1,270 | 512 |
| window + sink, original positions (the library's `keep`) | 5.64 | 14.85 | 33.5 | 516 |
| window + sink, re-indexed (StreamingLLM §3) | 5.64 | 14.91 | 34.0 | 516 |

Before the window first slides all four lines are the same computation — the largest difference from the full cache over those positions is 0.08 nats, floating-point noise — and the table's first column says so. Then, at position 513, the first prediction made after the first token has been evicted costs the no-sink window 18.9 nats where the full cache pays 0.5, and the line never comes back: every later bin sits between 200 and 1,270. The collapse is immediate, not gradual, and it happened although the evicted token was the word `First`, not a BOS token — the sink is a property of the first position, not of a special token.

Keeping the first four tokens removes the collapse: 14.9 instead of 620. It does not make the window free: 14.9 is 2.3 times the full cache's 6.46, and that is the cost of seeing only 512 tokens of a play whose names and scenes recur further apart than that. "Stable" means it does not collapse or drift; it does not mean it predicts as well. Whether the kept tokens sit at their original positions or are re-indexed inside the cache makes no visible difference here (14.85 against 14.91): 5,455 tokens are far inside this model's nominal length of 131,072, so both kinds of position are ones the model has seen. Separating the two would need a stream longer than the nominal length, which this demo does not attempt. One text, one model, one window: the 2.3 is a number for this setting, not a constant.

## Files

| File | |
|---|---|
| `demo.py` | `inspect` / `rehearse` / `replay` / `fetch`; `SinkWindowCache` (the hand-written sink + window cache with optional re-indexing); the per-token loss loops |
| `demo_config.toml` | $W$, $n_{\text{sink}}$, prefill chunk, bin size, text path and length, the models |
| `present.sh` | entry point; uses `../.venv/bin/python`, offline |
| `runs/rehearsal/` | `inspect.json`, `curve.json` (the table and the bins), `tokens.json` (every token's loss under every cache), `curve.svg` |
