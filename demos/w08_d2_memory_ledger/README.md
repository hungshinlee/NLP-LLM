<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 2 — the memory ledger: formula against measured peak (slides p32–p33, p41)

How much memory does it take to fine-tune an 8B model — all of it, with LoRA, with QLoRA? Part (a) answers with arithmetic from one `config.json`. Part (b) runs the three settings on an Apple M5 Max (64 GB) and puts the measured peak next to the formula.

- **(a) Live, standard library only.** Reads `Qwen/Qwen3-8B`'s `config.json`, counts $N$ matrix by matrix (and checks it against the parameter total the Hub reports), counts $N_{\text{LoRA}}$ for $r = 16$ on all seven linear layers of every block, and prints four rows — full fine-tuning, LoRA over the bf16 weights, and QLoRA with two ways of storing 4 bits — against two lines: 64 GB (this machine) and 16 GB (a project GPU). Then the FLOPs per token: $6N$ against $4N + 6N_{\text{LoRA}}$.
- **(b) Replayed.** `mlx-lm`'s training loop, batch 1, 20 steps, on windows of exactly 256 and 1,024 tokens, and once more at 1,024 with `--grad-checkpoint`: LoRA on Qwen3-8B in bf16, the same LoRA on `mlx-community/Qwen3-8B-4bit` (that is what "QLoRA" means in mlx), and full fine-tuning of Qwen3-1.7B-Base. Each run is its own process and records `mx.get_peak_memory()`. Full fine-tuning of the 8B model is not run: it does not fit.
- **`adam`, live, one second.** One bf16 weight, one Adam step: prints the dtype of Adam's $m$ and $v$ and whether the weight moved. This is why part (b) has two formula columns.

## Two ledgers

The slides' ledger (W4, and p25 this week) is the usual mixed-precision one: a trainable parameter costs 16 bytes — the bf16 weight 2, its gradient 2, an fp32 master copy 4, and Adam's two moments 4 + 4 — and a frozen one costs 2.

mlx does not store it that way (read from `mlx/optimizers/optimizers.py` and `mlx_lm/tuner/lora.py` at the pinned versions, and measured by the program after every run):

| | bytes | why |
|---|---|---|
| a trainable bf16 weight | 2 + 2 + 2 + 2 = **8** | Adam's `m` and `v` are `zeros_like(parameter)`: bf16. There is no fp32 master copy. |
| a LoRA factor ($\mathbf{A}$, $\mathbf{B}$) | 4 + 4 + 4 + 4 = **16** | the factors are created in float32, so weight, gradient, `m`, `v` are float32 — 16 again, by a different split |
| a frozen bf16 weight | 2 | |
| a frozen weight in mlx's 4-bit format | 0.5625 | 4 bits + one 16-bit scale and one 16-bit bias per group of 64 |

And `mlx_lm.lora --fine-tune-type full` unfreezes the blocks only; the embedding, the LM head and the final norm stay frozen. So for full fine-tuning the two ledgers disagree by more than a factor of two (`./present.sh mlx`):

| run | model | 16-byte ledger | as mlx stores it |
|---|---|---|---|
| LoRA, $r = 16$ | Qwen3-8B, bf16 | 17.1 GB | 17.1 GB = 16.38 frozen + 0.70 |
| QLoRA, $r = 16$ | Qwen3-8B, mlx 4-bit | 5.3 GB | 5.3 GB = 4.61 frozen + 0.70 |
| full fine-tuning | Qwen3-1.7B-Base | 27.5 GB | 11.9 GB = 0.62 frozen + 11.28 |
| full fine-tuning | Qwen3-8B (not run) | 131.1 GB | 58.1 GB = 2.49 frozen + 55.57 |

All of these are bytes **before a single activation**. What the missing master copy costs is what `adam` shows: bf16 keeps 8 significant bits, so a weight of 0.02 can only move in steps of $1.2 \times 10^{-4}$, and an update smaller than half a step is rounded away. After each full-fine-tuning run the program counts how many trainable elements are bit-for-bit what they were before training.

## What is being measured in (b)

| | |
|---|---|
| Models | `Qwen/Qwen3-8B` (bf16, 8,190,735,360 parameters, Apache-2.0), `mlx-community/Qwen3-8B-4bit` (the same model in mlx's affine 4-bit format), `Qwen/Qwen3-1.7B-Base` (bf16, 1,720,574,976 parameters). Revisions are pinned in `../versions.lock`. |
| LoRA | `rank 16`, `scale 20`, `dropout 0`, on the seven `Linear` layers of every block (the settings of demos 1 and 3): 43,646,976 trainable parameters on the 8B model. On the 4-bit model the same call wraps `QuantizedLinear` layers. |
| Full fine-tuning | every block unfrozen, as `--fine-tune-type full --num-layers -1` does: 1,409,408,000 trainable parameters on the 1.7B model. |
| Data | windows of exactly $L$ tokens, cut from demo 1's training file if it is there (`../w08_d1_sft_base_to_assistant/data/train.jsonl`), otherwise random token ids with a fixed seed — the record says which. Memory depends on the length, not on the content. The loss is on every token. |
| Training | `mlx_lm.tuner.trainer.train` (the loop behind `mlx_lm.lora`), Adam, batch 1, 20 steps. The program builds the optimizer itself so that it can read Adam's state afterwards. |
| Peak | `mx.get_peak_memory()`: the peak of the whole process — loaded weights, activations, gradients, Adam's state. One process per run, so nothing is left over from the run before. |
| Also recorded | bytes of the loaded parameters by dtype; number and dtype of the trainable parameters; bytes and dtype of Adam's state; seconds per step; for full fine-tuning, the fraction of trainable elements unchanged after the 20 steps. |

Three caveats the numbers carry. They are one machine's, one framework's and one batch size's. A peak includes whatever the allocator had not yet released, so it is an upper bound on what was needed. And 20 steps of batch 1 measure memory, not learning.

## Requirements

| | |
|---|---|
| Packages | (a): none beyond Python 3.11+ (`tomllib`; 3.10 works with `tomli`). (b): `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `numpy`, `huggingface_hub`, `transformers`. `replay` needs none of them. |
| Models | (a) needs only a `config.json`: it looks in the local Hugging Face cache, then falls back to the copy of the relevant keys in `demo_config.toml`. (b): `./present.sh fetch` once (network) — the bf16 Qwen3-8B is about 16.4 GB, the 4-bit one 4.6 GB, the 1.7B base 3.4 GB. |
| Device | (b) is **Apple silicon only** in practice (MLX on the GPU). The LoRA run on the bf16 8B model needs the 16.4 GB of weights plus activations; do not try it on a 16 GB machine — that is the point of part (a). |
| Check without downloading | `./present.sh tiny` runs the whole of (b) on a tiny Qwen3 with random weights (a real MLX training loop, numbers meaningless, written to `runs/tiny/`); `./present.sh fake` needs no MLX at all. |
| Portability | only tested on an Apple M5 Max (64 GB); the flow also with `tiny` on Linux (mlx's CPU build). |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w08_d2_memory_ledger
./present.sh count                    # (a) where config.json came from; N, matrix by matrix
./present.sh table                    # (a) N and N_LoRA, Enter, then the four rows
./present.sh table --modules q,v      #     Hu et al.'s main setting; or --rank 8 --layers 16 (mlx-lm's defaults)
./present.sh flops                    # (a) 6N against 4N + 6·N_LoRA
./present.sh mlx                      # the two ledgers for the settings of (b)
./present.sh adam                     # one bf16 weight, one Adam step
./present.sh fetch                    # (b) once: the three models
./present.sh inspect                  # versions, device, weight files by dtype, the planned runs
./present.sh rehearse                 # (a) again, then nine runs → runs/rehearsal/
./present.sh                          # (b) in class: replay the table, then where the difference goes
./present.sh tiny                     # the whole of (b) on a tiny random model
```

Without `present.sh`: `python ledger.py table`, `python demo.py replay`, and so on.

## What you should see

**(a)** — arithmetic, so the same everywhere:

```
  N      =  8,190,735,360   counted matrix by matrix
  N_LoRA =     43,646,976   r = 16 on 7 gates (q, k, v, o, gate, up, down) of 36 blocks   = 0.533 % of N

                                    bytes per parameter    frozen + trained       GB   64 GB 16 GB
  full fine-tuning                  16                                         131.1   no    no
  LoRA, r = 16, bf16 base           2 + 16·N_LoRA/N            16.38 + 0.70     17.1   yes   no
  QLoRA, NF4 + double quantization  b_q + 16·N_LoRA/N           6.07 + 0.70      6.8   yes   yes
  QLoRA, mlx's 4-bit                b_q + 16·N_LoRA/N           4.61 + 0.70      5.3   yes   yes
```

The LoRA row is the one to read twice: the adapter and all of its optimizer state are 0.70 GB; the frozen bf16 weights are 16.38 GB. A bf16 LoRA of an 8B model does not fit in 16 GB, and the reason is the weights it does not train. The two QLoRA rows differ for two reasons: NF4 with double quantization costs 4.127 bits where mlx's affine format costs 4.5, and the usual bitsandbytes setup leaves the embedding and the LM head in bf16 where the mlx conversion quantizes them too. FLOPs per token: $4.91 \times 10^{10}$ against $3.30 \times 10^{10}$, a ratio of 0.672.

**`adam`** — at a learning rate of $5 \times 10^{-6}$ a bf16 weight of 0.02 does not move (half a bf16 step there is $6.1 \times 10^{-5}$); weights of 0.002 and 0.0002 do; in float32 all three move. Adam's `m` and `v` have the weight's dtype in both cases.

**(b)** — from `runs/rehearsal/` (2026-11-10, Apple M5 Max with 64 GiB, macOS 27.2, Python 3.12, mlx 0.32.2, mlx-lm 0.32.0 at the pinned commit; batch 1, 20 steps; the nine runs took about four minutes including loading).

| run | 16-byte ledger | as mlx stores it | peak, 256 tokens | peak, 1,024 tokens | 1,024, `--grad-checkpoint` |
|---|---|---|---|---|---|
| LoRA, $r = 16$ · Qwen3-8B, bf16 | 17.1 | 17.1 | 20.1 | 30.3 | 19.8 |
| QLoRA, $r = 16$ · Qwen3-8B, mlx 4-bit | 5.3 | 5.3 | 8.3 | 17.9 | 7.4 |
| full fine-tuning · Qwen3-1.7B-Base | 27.5 | 11.9 | 13.4 | 17.4 | 13.9 |
| full fine-tuning · Qwen3-8B | 131.1 | 58.1 | not run | | |

(GB = $10^9$ bytes.) What the breakdown shows:

- **The mlx column is measured, not only read.** After the last step each process still holds 17.08, 5.31 and 11.90 GB — the mlx column to two decimals. Adam's state is float32 for the two LoRA runs (0.349 GB = 8 bytes × 43,646,976) and bfloat16 for full fine-tuning (5.638 GB = 4 bytes × 1,409,408,000). All three full-fine-tuning peaks are *below* the 16-byte ledger's 27.5 GB.
- **The rest is activations and overhead, and it grows with the length.** Peak minus the mlx column: +3.0 and +13.2 GB for LoRA, +3.0 and +12.6 for QLoRA, +1.5 and +5.5 for the 1.7B model — about 13 GB per 1,000 tokens on the 8B model. The two 8B rows are nearly identical: storing the weights in 4 bits does not shrink the activations. At 1,024 tokens more than two thirds of the QLoRA peak is not weights.
- **`--grad-checkpoint`** gives back 10.5, 10.5 and 3.5 GB at 1.19–1.25× the time per step; at 1,024 tokens the checkpointed LoRA run peaks lower than the un-checkpointed one at 256.
- **QLoRA is slower here**: about 440–470 tokens/s against 970–1,000 for the bf16 LoRA, at both lengths. (We did not isolate the cause.)
- **Without a master copy, most weights do not move.** After 20 steps at $5 \times 10^{-6}$, 81.3 % of the trainable elements of the full-fine-tuning run are bit-for-bit what they were (matrices 81.3 %, norm vectors 99.7 %). That is 20 steps; what remains unchanged after a full run was not measured.
- **The row that was not run.** `mx.device_info()` reports a recommended working set of 55.7 GB on this machine (64 GiB = 68.7 GB). 58.1 GB is over it before any activation.
- The 4-bit weight files, from their headers: 4.095 GB of packed `uint32` weights plus 0.256 GB each of bf16 scales and biases = 4.608 GB, in 254 quantized tensors ($36 \times 7 + 2$: the embedding and the LM head are quantized too).

## Files

| | |
|---|---|
| `ledger.py` | part (a): `count`, `table`, `flops`, `mlx`, `files`, `check`. Standard library only. |
| `demo.py` | part (b): `inspect`, `fetch`, `adam`, `run`, `rehearse`, `aggregate`, `replay`; `--tiny` and `--fake` modes. |
| `demo_config.toml` | the model, rank and modules of (a); the three models' shapes (a saved copy of the `config.json` keys); sequence lengths, steps and the four settings of (b). |
| `present.sh` | the entry point; offline except for `fetch`. |
| `runs/rehearsal/` | `ledger.json` (part (a)), `inspect.json`, `adam.json`, `peaks.json`, `runs/*.json` — written by `rehearse`. |
