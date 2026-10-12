<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 4 — A sliding window with and without the sink: per-token perplexity under four KV caches (slide p39)

The same text, the same model, the same tokenizer — only the rule for what the KV cache keeps changes. Per-token negative log-likelihood is recorded under four caches: everything kept; only the last $W$ tokens kept; the last $W$ plus the first $n_{\text{sink}}$ tokens, at their original positions; and the last $W$ plus the first $n_{\text{sink}}$ with positions re-indexed inside the cache, which is what StreamingLLM (Xiao et al., ICLR 2024, §3) does. Everything is rehearsed beforehand and replayed in class, one line at a time.

## What is being measured

The text is the opening of *tinyshakespeare* (the first 20,000 characters, about 5k tokens with Llama 3's tokenizer; the exact count is printed), fed after the BOS token without a chat template. For each cache the model predicts every token given the cache it has at that moment, and the loss of that prediction is recorded. The slide's table reports perplexity, $\exp$ of the mean loss, over two ranges of token positions: before $W$ and from $W$ on. The curve in `runs/rehearsal/curve.svg` averages the loss over 256-token bins.

Why the first few tokens matter: softmax must put weight *somewhere*, and a head with nothing to attend to puts it on the first token, which every query can see (W3 p40; Xiao et al. §3). A plain sliding window drops that token at step $W$; the weight has to land on real tokens instead, and the prediction degrades at once. Keeping the first $n_{\text{sink}}$ tokens' KV under the window keeps the drain in place.

Why there are two "with sink" lines: in mlx-lm (0.32.0) the rotation for the position is applied to $\mathbf{q}$ and $\mathbf{k}$ *before* they enter the cache, at the token's absolute position in the stream, so the library's `RotatingKVCache(keep=n)` keeps the first $n$ tokens at their original positions — the sink's distance from the query grows without bound. StreamingLLM §3 assigns positions inside the cache instead, so the sinks stay a fixed distance of about $W + n_{\text{sink}}$ away. `SinkWindowCache` in `demo.py` does the re-indexing by hand: RoPE is a rotation, so a stored key can be re-rotated to any position at any time. At 5k tokens — far inside this model's nominal length of 131,072 — the two should behave similarly; the point of the fourth line is that a library argument named after the paper's idea need not implement it, and `inspect` prints the source lines that show where the positions are assigned. Read the source, not the README (course appendix C).

What this does not show: a cache that forgets gracefully is not a cache that remembers. Nothing that left the window can be retrieved; p40 is about that.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `huggingface-hub`; `transformers` for `inspect` (tokenizer only) |
| Model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (≈ 4.5 GB, revision pinned in `../versions.lock`; shared with W3 demo 2 and W5 demo 3). Backup: `mlx-community/Qwen3-8B-4bit` |
| Text | `slides/_corpus/tinyshakespeare.txt`, i.e. [karpathy's `tinyshakespeare/input.txt`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (public domain, 1.1 MB), not in this repository; download it to that path relative to the repository root, as W4 demo 1's README says |
| Device | **Apple silicon only** (MLX). About 4.5 GB for the weights; the caches are a few hundred tokens |
| Time | not yet measured: the full-cache line is a few seconds of prefill; each windowed line is one forward pass per token (about 5k), a few minutes each at 8B 4-bit decode speed |
| Portability | tested only on an Apple M5 Max (64 GB), macOS 27.0.1, Python 3.12 — once rehearsed |

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

**Not yet rehearsed (2026-10-12).** No expected numbers are written here on purpose: the course rule is that a number on a slide comes from a checked-in record or a verified paper, and this folder's record does not exist yet. What the rehearsal looks at, in order: whether the no-sink window's perplexity jumps once the text passes $W$ and by how much (Xiao et al. Figure 1 is the shape being tested); whether the two with-sink lines stay flat and how much above the full cache they sit; how close the original-position and re-indexed variants are to each other at this length; and, before any eviction, that all four lines agree with the full cache to floating-point noise (`rehearse` prints this check).

## Files

| File | |
|---|---|
| `demo.py` | `inspect` / `rehearse` / `replay` / `fetch`; `SinkWindowCache` (the hand-written sink + window cache with optional re-indexing); the per-token loss loops |
| `demo_config.toml` | $W$, $n_{\text{sink}}$, prefill chunk, bin size, text path and length, the models |
| `present.sh` | entry point; uses `../.venv/bin/python`, offline |
| `runs/rehearsal/` | `inspect.json`, `curve.json` (the table and the bins), `tokens.json` (every token's loss under every cache), `curve.svg` |
