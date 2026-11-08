<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 3 — two learning-rate sweeps: LoRA and full fine-tuning (slides p48–p49)

The same model, the same data and the same seed as demo 1; two ways of fine-tuning it; each swept over its own learning rates. LoRA ($r = 16$, every linear layer of every block) and full fine-tuning are each trained at five learning rates with `mlx_lm.lora` — same 500 steps, same batch, same mask, same seed — and the held-out loss at the end is plotted against the learning rate. Two U-shaped curves come out, and two numbers are read off them: where each minimum sits, and what the ratio of the two best learning rates is. Then the comparison most papers actually make is run on purpose: both methods at *full fine-tuning's* best learning rate. How much worse does LoRA look when it is not given its own sweep?

Everything is rehearsed in advance (ten training runs plus two cross runs do not fit in a lecture) and **replayed** in class from `runs/rehearsal/curves.json`.

- **Figure 1.** Final held-out response loss against learning rate, two curves. Printed as a text plot and as a table; the two best learning rates and their ratio, in mlx units and converted to PEFT units (see *The ratio* below).
- **Figure 2.** Validation loss every 25 steps for four runs: full fine-tuning at its own best, LoRA at its own best, LoRA at full fine-tuning's best, and full fine-tuning at LoRA's best.
- **The table on p49.** Best learning rate per method; loss with each at its own best; loss with both at full fine-tuning's best.

## What is being measured

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B-Base`, bf16, 28 blocks, tied embeddings (Apache-2.0; revision pinned in `../versions.lock`) — demo 1's model. |
| Data | demo 1's `data/`: 1,000 / 100 / 200 train / valid / test examples filtered from `databricks/databricks-dolly-15k` (CC BY-SA 3.0), in mlx-lm's chat format, through the tokenizer's own Qwen3 chat template. |
| LoRA | `rank 16`, `scale 20`, `dropout 0`, on every `Linear` of every block (q, k, v, o, gate, up, down; 17,432,576 trainable parameters). Not on the embedding. mlx-lm computes $y = \mathbf{W}x + \text{scale} \cdot \mathbf{B}\mathbf{A}x$, so `scale` is $\alpha / r$ in the original LoRA notation: $\alpha = 320$ here. |
| Full fine-tuning | mlx-lm `--fine-tune-type full --num-layers -1`: all 28 blocks are trainable. The embedding matrix (and with it the tied LM head) and the final norm are outside `model.layers` and stay frozen — so both methods move the same set of matrices; the program prints the trainable-parameter count. |
| Training | Adam with a constant learning rate, batch 4, 500 steps = 2 epochs, `--mask-prompt` (loss on the assistant's turn only), `max_seq_length 1024`, seed 20261108 — identical across all runs; only the learning rate changes. mlx's Adam keeps its state in the parameters' dtype (bf16 for full fine-tuning, fp32 for the LoRA factors) and casts the learning rate to the gradient's dtype; each run's JSON records the learning rate as bf16 sees it. |
| The grids | Five learning rates per method, about 1.5 decades end to end (neighbours ×2.4): $\{2, 5, 12, 30, 70\} \times 10^{-6}$ for full fine-tuning and $\{1, 2.4, 6, 14, 35\} \times 10^{-6}$ for LoRA, in mlx units. A `probe` (30 steps per candidate) first finds where training diverges; a first sweep with LoRA's grid at $6 \times 10^{-6}$–$2 \times 10^{-4}$ put LoRA's minimum at the grid's lower edge, so the grid was moved down and three runs added — a sweep whose best point sits at an edge has not found the minimum. Both grids' off-grid points (the first round, the cross runs) are kept in `curves.json`. |
| Loss | **Curves**: mlx-lm's validation loss on the 100 held-out examples every 25 steps (measured before step $k$, i.e. after $k - 1$ updates). **Final**: mlx-lm's `evaluate()` on the 200 test examples after training, response tokens only, nats per token — the same measure as demo 1 (c), so the base model's value from demo 1 is the reference line. |
| The ratio | Adam's update to $\mathbf{B}\mathbf{A}$ scales with `scale` × learning rate. With `scale 20` the same effective step is reached at a learning rate 10× smaller than in a setup with $\alpha / r = 2$ (the common PEFT default, $\alpha = 2r$). So the ratio "best LoRA learning rate / best full fine-tuning learning rate" measured here in mlx units is multiplied by 10 to compare with claims stated in PEFT units; the program prints both. |
| The cross runs | LoRA at full fine-tuning's best learning rate, full fine-tuning at LoRA's best, and LoRA at full fine-tuning's best *copied in $\alpha = 2r$ units* — i.e. the same number handed to a LoRA with `scale 2`, which in this implementation is $\eta / 10$ (`[sweep] peft_copy`). If a value is already in the other grid, that run is reused; otherwise one more run is trained. |
| Peak memory | `mx.get_peak_memory()` as reported to mlx-lm's training callback, per run, in its own process. |

Every replay prints the caveat: one model, one dataset of a thousand examples, one seed. This is not a replication of any published number; a ratio of 3× or 30× is possible, and a curve too flat to show a minimum means the run was too short or the data too small.

## What came out (rehearsal of 2026-11-08, `runs/rehearsal/curves.json`)

Test response loss after 500 steps (nats/token; the base model: 2.717):

| learning rate (mlx units) | $10^{-6}$ | $2 \times 10^{-6}$ | $2.4 \times 10^{-6}$ | $5 \times 10^{-6}$ | $6 \times 10^{-6}$ | $1.2 \times 10^{-5}$ | $1.4 \times 10^{-5}$ | $3 \times 10^{-5}$ | $3.5 \times 10^{-5}$ | $7 \times 10^{-5}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| full fine-tuning | | 1.992 | *1.975* | **1.934** | *1.933* | 1.984 | | 2.257 | | 2.929 |
| LoRA, $r = 16$ | 1.941 | | **1.924** | *1.928* | 1.931 | | 1.970 | | 2.081 | |

Bold: each method's best within its grid; italics: cross runs outside the grid. LoRA at $5 \times 10^{-7}$ (full fine-tuning's best copied in $\alpha = 2r$ units): 1.980.

- Full fine-tuning's curve is a clear U; LoRA's is much flatter over the same range. The second is an observation from this one run, not a claim from the literature.
- Ratio of best learning rates: $0.48\times$ in mlx units, i.e. $4.8\times$ in $\alpha = 2r$ units — the blog post's figure is about $10\times$.
- Each at its own best: LoRA 1.924 vs. full fine-tuning 1.934. Both at full fine-tuning's best: LoRA 1.928 — no penalty, because in mlx units the two minima nearly coincide. The penalty appears when the number is copied across conventions ($5 \times 10^{-7}$: 1.980, still falling at step 500) or the other way round (full fine-tuning at LoRA's best: 1.975).
- Most runs reach their lowest validation loss around step 250–350 and drift up slightly afterwards (two epochs); the table uses the loss at step 500 for every run.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib` (3.10 works with `tomli`). `replay` needs none of the MLX packages. |
| Models and data | demo 1's `fetch` and `prepare` must have been run (`../w08_d1_sft_base_to_assistant`): the model (about 3.4 GB) and `data/`. Everything here runs offline. |
| Device | **Apple silicon only** (MLX). Full fine-tuning of a 1.7B bf16 model with mlx's Adam needs about 8 bytes per trainable parameter (≈ 11 GB) plus the frozen embedding and activations; the program prints the measured peak per run. |
| Disk | a full fine-tuning run saves about 2.8 GB of weights, a LoRA run about 80 MB. By default only the best run of each method is kept after the sweep (`[weights] keep`); weights are never committed. |
| Time | on an M5 Max: about 8 min per 500-step run for both methods (full fine-tuning 0.94 s/step, peak 21.6 GB; LoRA 0.89 s/step, peak 17.8 GB — LoRA's backward pass still runs through the whole network); 14 runs, about 1.6 h in all. `probe` prints seconds per step and an estimate before you commit to the sweep. |
| Portability | nothing platform-specific apart from MLX; rehearsed on an Apple M5 Max (64 GB), macOS 27.2, Python 3.12, mlx 0.32.2; the flow also in `--fake` mode on Linux. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w08_d3_lr_sweep
./present.sh inspect                  # versions, data, the two grids (with their bf16 values and PEFT-unit equivalents)
./present.sh probe                    # 30 steps at each candidate learning rate: divergence and seconds per step
#   → edit [sweep] in demo_config.toml so each grid straddles its minimum
./present.sh rehearse                 # base loss → 5 + 5 runs (one subprocess each) → cross runs → curves.json, best.json
./present.sh                          # replay: Enter → figure 1, Enter → figure 2, Enter → the table
./present.sh run --group full --lr 1.2e-5   # one run by hand; then ./present.sh aggregate
./present.sh fake rehearse            # the whole flow without MLX (numbers are fake and labelled)
```

`rehearse` skips runs whose JSON already exists, so an interrupted sweep continues where it stopped (`--redo` repeats them; `--group lora` does one method; `--no-cross` skips the cross runs).

## Files

| | |
|---|---|
| `demo.py` | the program: `inspect`, `probe`, `rehearse`, `run`, `aggregate`, `replay` |
| `demo_config.toml` | model, data location, shared training settings, LoRA and full-fine-tuning settings, the two grids, probe candidates, weight-keeping policy |
| `present.sh` | entry point; sets `HF_HUB_OFFLINE=1` and uses the shared `../.venv` |
| `lora.yaml` | the `lora_parameters` block passed to `mlx_lm.lora -c` (generated from `[lora]`) |
| `runs/rehearsal/runs/*.json` | one file per run: the equivalent `mlx_lm.lora` command line, train loss / peak memory / tokens per second every 25 steps, validation loss every 25 steps, the final test loss, timings |
| `runs/rehearsal/probe/*.json`, `probe.json` | the 30-step probes and their summary |
| `runs/rehearsal/curves.json` | everything the replay shows: both curves, the two best learning rates, the ratio in both units, the four cross-run numbers |
| `runs/rehearsal/best.json` | which run is best for each method and where its weights are (demo 4 reads this) |
| `runs/weights/` | trained weights (not committed) |
