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
| The grids | Five learning rates per method, about 1.5 decades apart end to end (neighbours ×2.4), set in `demo_config.toml` `[sweep]` after a `probe` that runs each candidate for 30 steps to find where training diverges. The two grids are offset because the two methods have different best learning rates; the values in the file are the ones used in the rehearsal. |
| Loss | **Curves**: mlx-lm's validation loss on the 100 held-out examples every 25 steps (measured before step $k$, i.e. after $k - 1$ updates). **Final**: mlx-lm's `evaluate()` on the 200 test examples after training, response tokens only, nats per token — the same measure as demo 1 (c), so the base model's value from demo 1 is the reference line. |
| The ratio | Adam's update to $\mathbf{B}\mathbf{A}$ scales with `scale` × learning rate. With `scale 20` the same effective step is reached at a learning rate 10× smaller than in a setup with $\alpha / r = 2$ (the common PEFT default, $\alpha = 2r$). So the ratio "best LoRA learning rate / best full fine-tuning learning rate" measured here in mlx units is multiplied by 10 to compare with claims stated in PEFT units; the program prints both. |
| The cross runs | LoRA at full fine-tuning's best learning rate (and full fine-tuning at LoRA's best). If the value is already in the other grid, that run is reused; otherwise one more run is trained. |
| Peak memory | `mx.get_peak_memory()` as reported to mlx-lm's training callback, per run, in its own process. |

Every replay prints the caveat: one model, one dataset of a thousand examples, one seed. This is not a replication of any published number; a ratio of 3× or 30× is possible, and a curve too flat to show a minimum means the run was too short or the data too small.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib` (3.10 works with `tomli`). `replay` needs none of the MLX packages. |
| Models and data | demo 1's `fetch` and `prepare` must have been run (`../w08_d1_sft_base_to_assistant`): the model (about 3.4 GB) and `data/`. Everything here runs offline. |
| Device | **Apple silicon only** (MLX). Full fine-tuning of a 1.7B bf16 model with mlx's Adam needs about 8 bytes per trainable parameter (≈ 11 GB) plus the frozen embedding and activations; the program prints the measured peak per run. |
| Disk | a full fine-tuning run saves about 2.8 GB of weights, a LoRA run about 80 MB. By default only the best run of each method is kept after the sweep (`[weights] keep`); weights are never committed. |
| Time | not yet measured on this machine. `probe` prints seconds per step and an estimate for the whole sweep; expect the LoRA runs to take about 6.5 min each (demo 1) and full fine-tuning longer. |
| Portability | nothing platform-specific apart from MLX; the flow was also run in `--fake` mode on Linux. |

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
