<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W11 · Demo 1 — two roofs: measuring the roofline on this machine (slides p15–p17, p25)

A language model does two different jobs with the same forward pass. *Prefill* reads a prompt: many tokens go through the weights at once. *Decode* writes: one token per step, and every step has to bring all the weights through memory again. The roofline model says a step takes at least

$$t \;\ge\; \max\!\left(\frac{\text{bytes moved}}{\text{BW}},\ \frac{\text{FLOPs}}{F_{\max}}\right),$$

so a machine has two limits — a bandwidth $\text{BW}$ in bytes per second and a peak compute rate $F_{\max}$ in FLOPs per second — and which one binds depends on the ratio of FLOPs to bytes. This demo measures both limits on one machine with one model, and then checks two predictions that follow from them.

## What is measured

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` in bf16 (16.4 GB) and `mlx-community/Qwen3-8B-4bit` (about 4.6 GB): the same network stored two ways. |
| Prompts | Random token ids of an exact length. Timing does not depend on the content; the generated text is meaningless and end-of-sequence tokens are ignored. |
| Prefill | The prompt goes through the model in chunks of 2,048 tokens; only the KV cache is evaluated (as in `mlx-lm`, no logits). Reported: tokens per second. |
| Decode | Greedy, one token per step, no logits processor, written directly against the model (`model(y[:, None], cache)` then `argmax`), with the next step submitted before waiting for the current one, as `mlx-lm`'s `generate_step` does. Reported: tokens per second summed over the batch, and milliseconds per step. |
| Each point | One untimed warm-up, then three timed runs from the same starting context; the median is reported and all three are stored. |
| (a) | Prefill at 64 to 16,384 tokens; decode of 64 tokens at batch size 1. |
| (b) | Decode of 128 tokens after a 32-token prompt at batch sizes 1 to 128. |
| (c) | Decode of 64 tokens after a prefill of 1k, 4k and 16k tokens at batch sizes 1, 4 and 16; and at batch size 1 with the KV cache stored at 8 and 4 bits. |

For a batch of $B$ sequences with a long context, only one sequence is prefilled and its KV cache is copied $B$ times (marked `tiled`); the bytes a decode step has to move do not depend on what the cache contains. Short contexts use $B$ different prompts (`distinct`). The KV buffers are allocated to their final size before timing starts, so no reallocation falls inside a timed run.

### From tokens per second to the two roofs

With $N$ parameters, hidden size $d$, $W$ bytes of weights in memory and $M_{\text{KV}}(L) = 2\,n_{\text{layer}}\,n_{\text{kv}}\,d_{\text{head}}\,b_{\text{kv}}\,L$ bytes of KV cache:

- prefill of $L_{\text{in}}$ tokens: $(2N + 2\,n_{\text{layer}}\,L_{\text{in}}\,d)$ FLOPs per token; the largest FLOPs per second over the prefill lengths is the measured $F_{\max}$;
- decode at batch size 1: $(W + M_{\text{KV}})$ bytes per token; times tokens per second, this is the measured effective bandwidth;
- $I^{*} = F_{\max}/\text{BW}$, and from the bound for one decode step, $t \ge \max\big((W + B\,M_{\text{KV}})/\text{BW},\ B\,(2N + 4\,n_{\text{layer}}\,L\,d)/F_{\max}\big)$, the batch size $B^{*}$ at which the two terms meet for a short context and the context length $L_c$ beyond which they never meet.

`roofs.py` computes two counts and stores both. `deck` is the bookkeeping above. `strict` leaves out what a step does not actually do: the input embedding is a table lookup (no multiplication, one row read per token), and a prefill that only fills the cache never applies the output layer. For this model the strict count is 8–15% smaller.

The bound in (b) uses the two numbers measured in (a) for the same model, so the batch-size-1 point of (a) lies on it by construction; what the sweep tests is the shape.

Every replay prints the caveat: one machine, one model in two formats, random prompts, greedy decoding; thermal state and background load are not controlled.

## What came out

Rehearsed on 2026-11-27 (MacBook Pro, M5 Max, 64 GB, on mains power; `runs/rehearsal/roofs.json`). The whole run took 13 minutes. Every number is the median of three timed runs.

| (a) | bf16 | 4-bit |
|---|---|---|
| prefill, tokens/s at 64 / 256 / 1,024 / 4,096 / 16,384 tokens | 1,264 / 3,600 / **4,100** / 2,897 / 2,388 | 840 / 1,656 / **2,223** / 2,053 / 1,591 |
| decode at batch size 1, tokens/s | 33.5 | 104 |
| $F_{\max}$, FLOPs/s (`deck` / `strict`) | $6.84 \times 10^{13}$ / $5.82 \times 10^{13}$ | $3.71 \times 10^{13}$ / $3.16 \times 10^{13}$ |
| effective bandwidth (`deck`) | 549 GB/s | 479 GB/s |

Per token, writing is 122 times slower than reading (bf16). The vendor specifies "up to 614 GB/s" for this chip; the bf16 decode reaches 89% of that. From the bf16 pair, $I^{*} \approx 125$ FLOPs per byte, $B^{*} \approx 125$ and $L_c \approx 920$ tokens. **The 4-bit model writes 3.1 times faster and reads at 0.54 times the speed**: its weights are decoded back to floating point in every forward pass, and that work is not in the $2N$ count.

| (b) tokens/s, all sequences | $B$ = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| bf16 | 33.0 | 70.0 | 136 | 245 | 293 | 615 | 1,034 | 1,539 |
| 4-bit | 81.2 | 133 | 207 | 226 | 347 | 700 | 847 | 1,461 |

The bf16 line follows the bound up to $B = 8$ and leaves it at $B = 16$, where the time per step jumps from 33 ms to 55 ms; at $B = 128$ it reaches 37% of the flat part of the bound. **It does not flatten at $B^{*}$.** The cause of the jump was not investigated. The 4-bit line starts higher and bends from $B = 2$; at $B = 128$ the two meet.

| (c) tokens/s | $B$ = 1 | 4 | 16 |
|---|---|---|---|
| bf16, context 1k / 4k / 16k | 32.8 / 32.7 / 23.1 | 132 / 95.7 / 57.7 | 256 / 164 / not run (memory) |
| 4-bit, context 1k / 4k / 16k | 97.3 / 84.5 / 53.8 | 203 / 159 / 85.2 | 314 / 221 / 108 |

The longer the context, the less a batch buys: sixteen sequences give 8.9 times the throughput of one at a short context (bf16), 7.8 times at 1k and 5.0 times at 4k. Storing the KV cache at 4 bits (16k tokens, batch size 1, 4-bit weights) cut its bytes to 28% and changed the speed from 53.8 to 60.8 tokens/s, which is inside the spread of the three repeats (47 to 70).

Two things were unstable and are not explained: the 4-bit decode at batch size 1 ranged from 81 to 110 tokens/s across the run, and at a 16k context the three repeats at batch size 1 fell from the first to the last. One machine, one run.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `huggingface_hub`. Python 3.11+ for `tomllib`. `replay`, `replay-batch` and the arithmetic half of `test` need only the standard library. |
| Models | `Qwen/Qwen3-8B` (about 16.4 GB) and `mlx-community/Qwen3-8B-4bit` (about 4.6 GB); revisions in `../versions.lock`. |
| Device | **Apple silicon** for the real models (MLX on Metal); intended for an M5 Max with 64 GB. The largest cell of (c), sixteen sequences of 16k tokens, needs about 44 GB; cells whose estimate exceeds `memory_limit_gb` in the config are skipped and marked. `./present.sh tiny` runs the whole flow on a small random model without downloading anything, also on Linux with the CPU build of MLX. |
| Time | about 13 minutes for `rehearse` on the machine above; peak memory 44 GB (the sixteen 16k-token sequences on 4-bit weights). |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w11_d1_two_roofs
./present.sh test                     # the arithmetic, then the measuring code on a tiny random model
./present.sh tiny                     # the whole flow on a tiny random model (numbers mean nothing)
./present.sh inspect                  # shapes, bytes, KV dtype; this demo's decode loop against mlx-lm's own
./present.sh rehearse                 # (a), (b), (c) for both models; resumes where it stopped
./present.sh replay                   # (a): the two points and the measured roofs
./present.sh replay-batch             # (b) and (c)
./present.sh live                     # one prefill, one decode, side by side
./present.sh live-batch 16            # one extra point of (b), measured now
```

## Files

| | |
|---|---|
| `roofs.py` | The arithmetic, standard library only: parameter count from the shape, FLOPs per token, bytes per step, the bound, $I^{*}$, $B^{*}$, $L_c$. |
| `demo.py` | The measurements (`Bench.measure_prefill`, `Bench.measure_decode`) and the commands above. |
| `demo_config.toml` | Models, grids, warm-up and repeats, the memory limit, and the two reference numbers the slides start from. |
| `present.sh` | Entry point; offline by default. |
| `runs/rehearsal/` | `roofs.json` (every measured point and the derived roofs) and `inspect.json`. |
