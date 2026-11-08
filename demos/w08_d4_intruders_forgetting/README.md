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
| (a) The matrices | `[intruders].matrices` in `demo_config.toml`: one block's `self_attn.q_proj` ($2048 \times 2048$) and `mlp.down_proj` ($2048 \times 6144$). The block is chosen after `scan` (all 28 blocks, both matrices). |
| (a) The computation | Weights are read straight from the safetensors files (bf16 → float32 is exact) into float64. LoRA is merged in float64: $\mathbf{W}' = \mathbf{W}_0 + s\,(\texttt{lora\_a}\,\texttt{lora\_b})^\top$ with $s$ = `scale`. Then `np.linalg.svd` (reduced), $\lvert\mathbf{U}'_{[:, :k]}{}^\top \mathbf{U}_0\rvert$, row-wise maximum, count below each $\epsilon$. CPU only. Also printed per matrix: $\lVert\Delta\rVert_F / \lVert\mathbf{W}_0\rVert_F$, $\sigma_1(\Delta)$ against $\sigma_k(\mathbf{W}_0)$ and $\sigma_{\min}(\mathbf{W}_0)$, the number of singular values of $\Delta$ above 1 % of its largest ("effective rank"), and how much of the LoRA update would survive a bf16 merge (the delta from a small learning rate can fall below bf16's resolution — `mlx_lm.fuse` casts it to bf16). |
| (b) The models | `mlx_lm.load(model, adapter_path=…)`: the LoRA run is evaluated **unmerged**, the same path demo 3 used for its test loss; the full run's tensors replace the block weights. |
| (b) The ruler | each text is tokenized without special tokens and cut into non-overlapping windows of 2,048 tokens; each window is preceded by `<|endoftext|>`; bits per byte $= \sum \text{NLL(nats)} / \ln 2 / \text{UTF-8 bytes}$. Same definition as W1 demo 2 and W7 demo 3. The first token of each window sees only the prefix — identical for all three models. |
| (b) The texts | `../w02_d1_fertility/texts/zh_city_en.txt` (754 B), `../w01_d2_bpb/texts/zh_city.txt` (627 B), `../w07_corpus/wikipedia_heldout.jsonl` (290 articles joined with blank lines, about 1 MiB; CC BY-SA 4.0), and the FineWeb-Edu held-out tail exactly as W7 demo 2's `load_corpus` builds it (about 1 MiB; ODC-By 1.0). The W7 corpus comes from W7 demo 3's `fetch`. |
| Optional: intervention | `intervene` scales the singular values of every intruder direction (all LoRA'd matrices, $\epsilon = 0.5$) by $\lambda$ and re-measures (b), with a $\lambda = 1$ control — the paper's v3 experiment. Merged weights are kept in float32 to avoid the bf16 issue above. Not yet run. |

## Requirements

| | |
|---|---|
| Packages | (a): `numpy` only (version pinned in `../versions.lock`); Python 3.11+ for `tomllib` (3.10 with `tomli`). (b): `mlx`, `mlx-lm` (git commit pinned), `transformers`, `huggingface_hub`. |
| Inputs | demo 1's `fetch` (the model), demo 3's `rehearse` (the two weight folders, about 2.9 GB), W7 demo 3's `fetch` (`../w07_corpus/`). Everything runs offline. |
| Device | (a) any CPU; (b) Apple silicon (MLX), about 4–7 GB of unified memory per model. |
| Time | (a) tens of seconds per matrix (float64 SVD of $2048 \times 6144$ is the slow one); `scan` of 28 blocks × 2 matrices: tens of minutes. (b) a minute or two per model on an M5 Max. Measured values are filled in after the rehearsal. |

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
