<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 4 — what the side road left behind: intruder dimensions and forgetting (slides p54–p55, p58)

Demo 3 compared two fine-tuning methods by their scores. This demo looks at the weights they produced, and at what they cost.

- **(a) Intruder dimensions, live.** Shuttleworth et al. (NeurIPS 2025, Definition 3.1): take the fine-tuned matrix $\mathbf{W}'$, its top $k = 10$ left singular vectors; for each one, the largest $\lvert\cos\rvert$ against *every* left singular vector of the pre-trained $\mathbf{W}_0$. If that maximum is below $\epsilon$, the direction is new — an *intruder dimension*. We count them for demo 3's two runs (LoRA and full fine-tuning, each at its own best learning rate) on one block's $\mathbf{W}^Q$ and down-projection. Five lines of `numpy`, written in class into `intruders_live.py`; `intruders_final.py` is the reference.
- **(b) Forgetting, replayed.** The base model, the LoRA run and the full fine-tuning run are scored on four held-out texts with the ruler from W1 demo 2 — bits per byte, which does not care about the tokenizer: W1's English paragraph, W1's Traditional Chinese paragraph, the Wikipedia sample and the FineWeb-Edu tail from W7 demo 3. The fine-tuning data were English instructions and answers.

Expectations stated before running: (a) LoRA should show intruders and full fine-tuning should not — but our scale (1.7B, 1,000 examples, 500 steps, small learning rates) is far below the paper's, and seeing none would be a statement about our run, not about the paper; (b) bits per byte on original text should rise after fine-tuning, more for full fine-tuning than for LoRA (Biderman et al.: *learns less, forgets less*); the Chinese row could move most or least.

## What is being measured

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B-Base`, bf16 (Apache-2.0; revision pinned in `../versions.lock`) — demo 1's model. |
| Fine-tuned weights | demo 3's best run of each method, as named in `../w08_d3_lr_sweep/runs/rehearsal/best.json`: full fine-tuning at $5 \times 10^{-6}$ (all 28 blocks, bf16, saved as `adapters.safetensors`) and LoRA $r = 16$, `scale 20`, at $2.4 \times 10^{-6}$ (`lora_a`, `lora_b` in float32). Both were trained on the same 1,000 Dolly examples for 500 steps. The weights themselves are not committed; run demo 3 first. |
| (a) The matrices | `[intruders].matrices` in `demo_config.toml`: block 14's `self_attn.q_proj` ($2048 \times 2048$) and `mlp.down_proj` ($2048 \times 6144$). `scan` ran both matrices on all 28 blocks first; no block differed, so the middle one is used. |
| (a) The computation | Weights are read straight from the safetensors files (bf16 → float32 is exact) into float64. LoRA is merged in float64: $\mathbf{W}' = \mathbf{W}_0 + s\,(\texttt{lora\_a}\,\texttt{lora\_b})^\top$ with $s$ = `scale`. Then `np.linalg.svd` (reduced), $\lvert\mathbf{U}'_{[:, :k]}{}^\top \mathbf{U}_0\rvert$, row-wise maximum, count below each $\epsilon$. CPU only. Also printed per matrix: $\lVert\Delta\rVert_F / \lVert\mathbf{W}_0\rVert_F$, $\sigma_1(\Delta)$ against $\sigma_k(\mathbf{W}_0)$ and $\sigma_{\min}(\mathbf{W}_0)$, the number of singular values of $\Delta$ above 1 % of its largest ("effective rank"), and how much of the LoRA update would survive a bf16 merge (the delta from a small learning rate can fall below bf16's resolution — `mlx_lm.fuse` casts it to bf16). |
| (b) The models | `mlx_lm.load(model, adapter_path=…)`: the LoRA run is evaluated **unmerged**, the same path demo 3 used for its test loss; the full run's tensors replace the block weights. |
| (b) The ruler | each text is tokenized without special tokens and cut into non-overlapping windows of 2,048 tokens; each window is preceded by `<|endoftext|>`; bits per byte $= \sum \text{NLL(nats)} / \ln 2 / \text{UTF-8 bytes}$. Same definition as W1 demo 2 and W7 demo 3. The first token of each window sees only the prefix — identical for all three models. |
| (b) The texts | `../w02_d1_fertility/texts/zh_city_en.txt` (754 B), `../w01_d2_bpb/texts/zh_city.txt` (627 B), `../w07_corpus/wikipedia_heldout.jsonl` (290 articles joined with blank lines, about 1 MiB; CC BY-SA 4.0), and the FineWeb-Edu held-out tail exactly as W7 demo 2's `load_corpus` builds it (about 1 MiB; ODC-By 1.0). The W7 corpus comes from W7 demo 3's `fetch`. |
| Optional: intervention (not yet run) | `intervene` scales the singular values of every intruder direction (all LoRA'd matrices, $\epsilon = 0.5$) by $\lambda$ and re-measures (b), with a $\lambda = 1$ control — the paper's v3 experiment. Merged weights are kept in float32 to avoid the bf16 issue above. Not yet run. |

## What came out (rehearsal of 2026-11-08, `runs/rehearsal/`)

**(a) Zero intruders, everywhere.** For both runs, on `q_proj` and `down_proj` of all 28 blocks, at every $\epsilon$ from 0.1 to 0.9, the count is 0; every one of the top-10 $\max\lvert\cos\rvert$ values is 1.000 to three decimals. The diagnostics for block 14 say why:

| | $\lVert\Delta\rVert_F / \lVert\mathbf{W}_0\rVert_F$ | $\sigma_1(\Delta)$ | $\sigma_1(\Delta) / \sigma_{10}(\mathbf{W}_0)$ | effective rank ($\ge 1\%\,\sigma_1$) |
|---|---|---|---|---|
| `q_proj`, full fine-tuning | 0.33 % | 0.038 | 0.007 | 1,937 of 2,048 |
| `q_proj`, LoRA $r = 16$ | 0.60 % | 0.30 | 0.055 | 16 |
| `down_proj`, full fine-tuning | 0.34 % | 0.054 | 0.009 | 2,048 of 2,048 |
| `down_proj`, LoRA $r = 16$ | 0.40 % | 0.25 | 0.042 | 16 |

($\mathbf{W}_0$: `q_proj` $\sigma_1 = 9.44$, $\sigma_{10} = 5.56$; `down_proj` $\sigma_{10} = 6.06$.) Neither update's largest singular value reaches 6 % of $\mathbf{W}_0$'s tenth, so neither can change which ten directions come first — 1,000 examples, 500 steps and the small learning rates demo 3 found leave the weights almost where they were. This is a statement about our run, not about the paper, whose fine-tuning runs are far longer. What *does* differ is the shape of the update: LoRA's is larger in norm but has rank 16; full fine-tuning's is spread over essentially all directions. A bf16 merge of the LoRA update would leave 28 % (q) and 39 % (down) of the entries unchanged, with the norm intact — rounding noise rather than shrinkage — which is why (b) evaluates the LoRA run unmerged. Timing: about 30 s for the two matrices in class; `scan` of 28 blocks about 8 min.

**(b) Forgetting is measurable, small, and the same for both methods** (bits per byte, base → LoRA → full fine-tuning):

| held-out text | bytes | base | after LoRA | after full fine-tuning |
|---|---|---|---|---|
| W1's English paragraph (1 window, 165 tokens) | 753 | 0.8479 | 0.8441 (−0.0038) | 0.8447 (−0.0032) |
| W1's Traditional Chinese paragraph (1 window, 166 tokens) | 627 | 1.0500 | 1.0514 (+0.0014) | 1.0523 (+0.0023) |
| Wikipedia sample (124 windows, 253k tokens) | 1,056,160 | 0.8051 | 0.8118 (+0.0067) | 0.8121 (+0.0070) |
| FineWeb-Edu tail (108 windows, 220k tokens) | 1,053,404 | 0.7154 | 0.7208 (+0.0054) | 0.7203 (+0.0049) |

The fine-tuning target itself (demo 3, test response loss, nats per token): 2.717 → 1.924 → 1.934. Of the three expectations stated above, the first holds on the two megabyte-sized texts (+0.005–0.007 bits per byte, under 1 %) while the two short paragraphs move within noise; the second — full fine-tuning forgetting more — does not show (differences ≤ 0.0005: both runs moved the weights very little, see (a)); the Chinese paragraph barely moves, and the row that rises most is the English Wikipedia sample. On this ruler, this SFT cost almost nothing; it is also only the first of three ways to measure forgetting (slide p60). Timing on an M5 Max: 45 / 63 / 53 s per model, 0.3 s to load, peak 5.9–6.2 GB.

## Requirements

| | |
|---|---|
| Packages | (a): `numpy` only (version pinned in `../versions.lock`); Python 3.11+ for `tomllib` (3.10 with `tomli`). (b): `mlx`, `mlx-lm` (git commit pinned), `transformers`, `huggingface_hub`. |
| Inputs | demo 1's `fetch` (the model), demo 3's `rehearse` (the two weight folders, about 2.9 GB), W7 demo 3's `fetch` (`../w07_corpus/`). Everything runs offline. |
| Device | (a) any CPU; (b) Apple silicon (MLX), about 4–7 GB of unified memory per model. |
| Time | (a) about 30 s for the two matrices (loading 5 + 12 s, four float64 SVDs 2 + 2 + 5 + 5 s); `scan` of 28 blocks × 2 matrices about 8 min. (b) 45–63 s per model on an M5 Max, peak 6.2 GB. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w08_d4_intruders_forgetting
./present.sh inspect                  # weights found, keys and shapes match, held-out sizes
./present.sh scan                     # all blocks: intruder counts for q_proj and down_proj → pick the block in demo_config.toml
./present.sh rehearse                 # inspect → intruders (reference) → forget → runs/rehearsal/*.json
./present.sh                          # class, part (a): runs intruders_live.py — write the five lines first
./present.sh intruders --ref          # same with the reference implementation
./present.sh replay                   # class, part (b): the bits-per-byte table, one row per Enter (no model is loaded)
./present.sh fake rehearse            # the whole flow without MLX (numbers are fake and labelled)
```

## Files

| | |
|---|---|
| `demo.py` | the program: `inspect`, `intruders`, `scan`, `forget`, `replay`, `rehearse`, `intervene` |
| `intruders_live.py` | the skeleton written in class (`count_intruders`) |
| `intruders_final.py` | the reference implementation, same signature |
| `demo_config.toml` | matrices, $k$, the $\epsilon$ grid, the four texts, the intervention's $\lambda$ |
| `present.sh` | entry point; sets `HF_HUB_OFFLINE=1`, uses the shared `../.venv` |
| `runs/rehearsal/inspect.json` | versions, weight folders, key/shape checks, held-out sizes |
| `runs/rehearsal/intruders.json` | (a) for the configured matrices: counts for every $\epsilon$, the ten $\max\lvert\cos\rvert$ values, the diagnostics |
| `runs/rehearsal/scan.json` | (a) for every block |
| `runs/rehearsal/forgetting.json` | (b): bits per byte, tokens, windows and seconds for 3 models × 4 texts — what `replay` shows |
| `runs/rehearsal/intervene.json` | optional: (b) after scaling the intruder directions |
