<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W4 · Demo 3 — skip one layer at a time, and measure the perplexity (slide p32)

Take a fixed text, measure the teacher-forced perplexity of Qwen3-0.6B on it (the baseline), then for each of its 28 decoder layers replace that layer's forward pass by the identity — **the parameters are not removed; the layer simply adds nothing to the residual stream this time** — and measure again. Twenty-eight points, printed one per line: perplexity, the multiple of the baseline, top-1 next-token accuracy, agreement with the baseline's predictions, and a bar of $\log_2(\text{ppl}/\text{baseline})$ (green below 1.15×, amber below 2×, red at 2× and above, capped at 16×). In class the model is loaded, the baseline is printed, and the run **pauses** so that predictions can be made before the 28 points appear.

| Choice | What | Why |
|---|---|---|
| Text | two paragraphs: a Traditional Chinese one (W1 demo 2's `zh_city.txt`, 627 bytes) and its English translation (W2 demo 1's `zh_city_en.txt`, 754 bytes), both written for this course | same content, two languages; each is run separately and the perplexity is pooled over tokens, with per-text numbers kept — the layer dependence differs between the two |
| Perplexity | summed NLL over both texts ÷ number of predicted tokens, `add_special_tokens=False`, first token not predicted | the W1 demo 2 definition. Comparable only with this demo's own baseline, not with W1's numbers (different tokenizer, different model) |
| "Skipping" | a `forward_hook` on `model.model.layers[i]` that returns the layer's input `hidden_states` in place of its output | the module is not swapped out (the Qwen3 forward reads `attention_type` from each layer to pick the mask, so an `nn.Identity` would break it); independent of whether the layer returns a tensor (transformers 5.x) or a tuple (4.x) — `inspect` prints which. The skipped layer still computes once and the result is dropped: under a second for 0.6B |
| Extra metrics | top-1 accuracy against the true next token; top-1 agreement with the baseline's prediction | a cheap stand-in for the "QA-style metrics fall slowly while loss falls fast" observation on p33 — same direction, not a benchmark |
| Windows | class: skip 1 layer; rehearsal also 2 and 4 consecutive layers | "skip $k$ consecutive layers" is the optional extension |
| Device | `mps` default; `--device cpu` | forward only, no gradients, no attention weights: 8–10 ms per forward, **0.22 s for all 28 points** on the M5 Max — so `show` deliberately pauses 0.4 s per point (`show_pace`), otherwise the curve appears all at once |

## Requirements

| | |
|---|---|
| Packages | `torch 2.14.0`, `transformers 5.17.0` (pinned in `../versions.lock`). Attention backend `sdpa` (no attention weights are needed, unlike W3 demo 3). |
| Model | `Qwen/Qwen3-0.6B` (float32 ≈ 2.4 GB in memory); W3 demo 3's `fetch` already downloaded it and pinned its revision — `./present.sh fetch` here only confirms. Backup `Qwen/Qwen3-1.7B` (`--model backup`, ≈ 6.8 GB). |
| Texts | `../w01_d2_bpb/texts/zh_city.txt` (the file is published although W1's demos are not yet) and `../w02_d1_fertility/texts/zh_city_en.txt`. Any two UTF-8 text files can be substituted in `demo_config.toml`. |
| Memory | ≈ 3 GB free. |
| Portability | should run anywhere `torch` and `transformers` install; `--device cpu` for machines without MPS (not timed). Tested only on an Apple M5 Max, macOS 27.0, Python 3.12.14 (2026-10-07). |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w04_d3_layer_removal
./present.sh fetch                    # only if Qwen3-0.6B is not in your Hugging Face cache yet (needs network)
./present.sh inspect                  # versions, device, the decoder layer's return type, whether the hook is an exact identity, seconds per forward
./present.sh                          # the demo: load → baseline → pause → Enter → 28 points → summary table (writes runs/live/)
./present.sh show --windows 1,2       # also skip two consecutive layers after the single-layer pass
./present.sh show --device cpu --no-pause   # without MPS, without waiting for Enter (flags go after the subcommand)
./present.sh rehearse                 # baseline + windows 1 / 2 / 4 → runs/rehearsal/curve.json and curve.svg
./present.sh replay                   # the instructor's record, same screens (marked as not live)
./present.sh fake                     # the screens with made-up numbers, no model needed
```

## What you should see

From `runs/rehearsal/curve.json` (2026-10-07, Qwen3-0.6B, MPS):

- `inspect`: the decoder layer returns a plain `Tensor` on transformers 5.17; the hook is an exact identity (removing it gives max |Δ| = 0.0 against the baseline, also on MPS); skipping layer 0 gives max |Δ| = 29.7.
- Text: 166 Chinese tokens (3.8 bytes/token) and 165 English tokens — nearly equal by coincidence, so the pooled number favours neither. **Baseline perplexity 29.63** (zh 31.67, en 27.71), top-1 accuracy 39.8 % over 329 positions.
- **The first three layers are catastrophic, not just the first**: skipping layer 0 → ×310,032 (perplexity 9.2 M), layer 1 → ×1,113, layer 2 → ×107; from layer 3 on it is ×1.47.
- **The middle is cheap**: layer 13 ×1.05 (the cheapest), 14 ×1.09, 8 ×1.12, 15 ×1.12, 12 ×1.15; the median over all 28 is ×1.33.
- **The last layer costs ×2.50 — visible, not catastrophic**: two orders of magnitude below layer 2, on a par with layer 17 (×2.44). The common prediction "first and last are catastrophic" is right about the first and wrong about the last.
- **Layers 16–18 are unexpectedly expensive (×1.97, ×2.44, ×1.63) and only for Chinese**: zh ×3.44 / ×5.40 / ×2.30 versus en ×1.12 / ×1.10 / ×1.15. The last layer is also Chinese-heavy (zh ×4.25, en ×1.47). Overall the Chinese text is more sensitive to every middle layer (×1.2–1.5 against ×1.05–1.2).
- **Accuracy falls more slowly than perplexity in the middle, not at the ends**: layer 17 costs ×2.44 in perplexity but accuracy only drops 39.8 % → 32.2 % (agreement 65 %); across the middle, accuracy stays at 32–39 % and agreement at 61–83 %. Skipping layers 0–2 destroys both (accuracy 0.3 % / 0 % / 9 %).
- **Windows 2 and 4**: medians ×1.96 and ×6.77. Two adjacent layers cost less than the product of their single multiples (16–17: ×3.07 against 1.97 × 2.44 = 4.8; 13–14: ×1.13 ≈ 1.05 × 1.09) — neighbouring layers partly overlap. Layers 16–19 together ×10.2 (zh ×30), 23–26 ×20.7, 0–3 ×307,120.

Your own run on the same model and texts should reproduce the multiples closely (the computation is deterministic up to floating point); timings will differ.

Extensions nobody has run: the same curve on Llama 3.1 8B (the MLX 4-bit copy from W3 demo 2 cannot be loaded by `transformers`; a bf16 copy is ≈ 16 GB); a short "healing" fine-tune after skipping, as in the layer-pruning literature; ranking layers by a block-influence score and comparing with this ranking.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `inspect` / `rehearse` / `show` / `replay`; `--fake`, `--model backup`, `--device`, `--windows`, `--pace`, `--no-pause` |
| `demo_config.toml` | model, the two text paths, device, attention backend, windows for show and rehearse, `show_pace` |
| `present.sh` | entry point; always `HF_HUB_OFFLINE=1` except for `fetch`, uses `../.venv/bin/python` |
| `runs/rehearsal/inspect.json`, `curve.json`, `curve.svg` | the rehearsal record quoted above; the SVG is the curve (log-scale perplexity multiple, accuracy on the right axis) |
