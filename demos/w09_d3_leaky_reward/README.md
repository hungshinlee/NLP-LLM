<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W9 · Demo 3 — a verifier with a hole in it: proxy and gold against $\sqrt{\mathrm{KL}}$ (slides p56–p57)

A program that checks answers cannot be flattered. It can still be fooled, because it checks only what it checks. This demo builds the smallest case we could: demo 2's task and training loop, unchanged, with a verifier that has a hole in it on purpose.

- The **strict** verifier is demo 2's: the answer scores 1 if its **last number** equals the result.
- The **leaky** verifier scores 1 if the result appears **anywhere** in the answer.

The model is trained on the leaky score. The strict score is computed on every answer but never enters an update: it is the gold. Everything is rehearsed in advance and replayed in class.

A second pair of runs replaces the verifier by **a coin flip**, once with one update per batch of rollouts and once with four. This is a very small version of the question in this week's reading (why can a random reward raise a benchmark score?) on a task where nothing can have been memorised.

## What is being measured

| | |
|---|---|
| Task, model, sampling, advantage | Exactly demo 2's (`../w09_d2_grpo_scratch`): integer arithmetic generated on the spot, `Qwen/Qwen3-1.7B-Base` with last week's adapter C as $\pi_{\text{ref}}$, a new LoRA bypass on top starting at $\mathbf{B} = \mathbf{0}$, $G = 8$ answers per prompt, 64 rollouts per step. The settings are read from demo 2's `demo_config.toml`, and the seeds are the same. |
| The runs | `leaky` (trained on the leaky verifier), `strict` (trained on the strict one: the honest comparison), `coin_1` and `coin_k` (a fair coin decides each reward; one update per batch, or four), and `count`, a wider hole kept in reserve: the reward is the number of numbers in the answer, divided by 8 and capped at 1. |
| Proxy and gold | Proxy is the score the run is trained on. Gold is always the strict verifier. Both are measured on demo 2's fixed evaluation problems (128 problems, two sampled answers each, a fixed sampling seed) before training, every 25 steps and at the end. One such number has a standard error of about 0.03. |
| The hole itself | The share of answers that pass the leaky verifier and fail the strict one. |
| Clues | Tokens per answer, and how many numbers an answer contains. The leaky verifier compares whole numbers, not substrings: 1275 does not count as 275. |
| Distance from $\pi_{\text{ref}}$ | A sampled estimate of the KL divergence, in nats per answer: the mean of $\log\pi_{\boldsymbol{\theta}}(y \mid x) - \log\pi_{\text{ref}}(y \mid x)$ over answers $y$ sampled from the current policy. $\pi_{\text{ref}}$ is the same weights with the bypass switched off, evaluated on the same padded batch, so the estimate is exactly 0 before the first update. With a finite sample it can come out negative; the tables show $\sqrt{\max(\mathrm{KL}, 0)}$. |
| Several updates per batch | The first update is demo 2's (the ratio is 1, so the clip cannot act). Each further update on the same rollouts uses the clipped surrogate $\min(\rho_t w,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,w)$ with $\epsilon = 0.2$, where $\rho_t$ is the per-token probability ratio against the first forward pass and $w$ carries the advantage. The share of tokens on which the clip is acting is recorded. This run takes four times as many optimizer steps as the others from the same number of rollouts. |

Why one update per batch is the control for the coin: when the reward does not depend on the answer, the expected update is zero (the baseline argument on the slides). That row therefore shows how large the noise is, and the four-update row has to differ from it by clearly more than the noise before anything can be said.

Every replay prints the caveat: one 1.7B model, one kind of problem, one seed, a few hundred steps, and a hole that was put there on purpose. **The random-reward rows cannot decide between the two explanations in this week's reading.**

## What came out

Not rehearsed yet (built 2026-11-15). This section will be filled in from `runs/rehearsal/` after the rehearsal, whichever way it turns out. If the model does not find the hole within the run, that is reported as the result, together with the `count` run.

## Requirements

| | |
|---|---|
| Packages | The same as demo 2: `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `numpy`, `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib`. `replay` needs only the standard library. |
| Other demos | `../w09_d2_grpo_scratch` must be present (its `grpo.py`, `demo.py` and `demo_config.toml` are loaded from there), and W8 demo 1 must have been run first for the model and adapter C. |
| Device | **Apple silicon** for the real model. `./present.sh tiny` runs the whole flow on a small random model without downloading anything, also on Linux with the CPU build of MLX; its answers are three characters long, too short for the hole to be reachable. |
| Time and memory | not rehearsed yet. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w09_d3_leaky_reward
./present.sh test                     # no downloads: the verifiers, the coin, the clipped objective, gradients on a tiny random model
./present.sh tiny                     # the whole flow on a tiny random model (numbers mean nothing)
./present.sh inspect                  # settings inherited from demo 2; both scores of the starting model; KL = 0 at the start
./present.sh rehearse                 # leaky, strict, coin_1, coin_k, then summary.json
./present.sh rehearse --only count    # the wider hole
./present.sh replay                   # print the tables from runs/rehearsal/summary.json
```

## Files

| | |
|---|---|
| `rewards.py` | The verifiers: `reward_leaky`, `reward_count`, `make_coin`; `reward_strict` is imported from demo 2. |
| `demo.py` | The commands above; `train_run` is demo 2's loop with the extra bookkeeping and the option of several updates per batch. |
| `demo_config.toml` | The runs, the number of updates per batch, the clip width, which run fills which row. |
| `present.sh` | Entry point; offline by default. |
| `runs/rehearsal/` | The rehearsal: one JSON per run with per-step scores under both verifiers, the KL estimate, answer lengths, numbers per answer and timings; `summary.json` with the evaluation points and the answers to the first six evaluation problems before and after training. |
