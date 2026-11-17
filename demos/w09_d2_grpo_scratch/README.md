<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W9 · Demo 2 — GRPO by hand: the advantage, the identity, and how large $G$ should be (slides p51–p53)

GRPO replaces the critic with the other $G - 1$ answers to the same prompt. This demo writes that out in full, with no RL library: the pinned version of `mlx-lm` has no reinforcement-learning command, and the few lines that make up the method are the point.

- **(a), live.** Five lines compute the advantage of each of $G$ answers to one prompt: the group mean, the group standard deviation (dividing by $G$), subtract, divide. Then the update direction is computed in two ways from the same per-answer gradients — the definition, $\frac{1}{G}\sum_i \hat{A}_i\,\nabla\log\pi_{\boldsymbol{\theta}}(y_i \mid x)$, and the identity derived on the slide for binary rewards, $s_r\,(\overline{\nabla}_{+} - \overline{\nabla}_{-})$ — and the norm of the difference is printed.
- **(b), rehearsed and replayed.** A hand-written loop — sample, verify, advantage, **one** update per batch of rollouts — is run at $G \in \{2, 4, 8, 16\}$ with the number of rollouts per step held fixed at 64, and then once more for each of four switches.

## What is being measured

| | |
|---|---|
| Task | Integer arithmetic generated on the spot (`grpo.make_problem`): a few two- or three-digit numbers joined by $+$, $-$, $\times$. Every step draws new problems from a seed derived from the step number, so nothing can have been memorised; a fixed set of 128 evaluation problems is excluded from training. The difficulty level is chosen so that the starting model is right three to five times in ten. |
| Verifier | `reward_strict`: 1 if the **last number** in the answer equals the result, otherwise 0. |
| Model | `Qwen/Qwen3-1.7B-Base` (bf16, Apache-2.0) with last week's adapter C from demo 1 — the adapter that taught the base model the chat format and to stop. This is $\pi_{\text{ref}}$. |
| What is trained | A new LoRA bypass (`grpo.Bypass`, rank 16, `scale 20`) stacked on the seven linear layers of every block. Its $\mathbf{B}$ starts at zero, so the policy at step 0 is exactly $\pi_{\text{ref}}$; everything else is frozen. |
| Sampling | $G$ answers per prompt at temperature 1, at most 48 new tokens, 64 sequences at a time through `mlx-lm`'s `BatchGenerator`. Each of the $G$ copies of a prompt is prefilled separately. |
| The update | $\hat{A}_i = (r_i - \bar{r})/s_r$ with $s_r$ the standard deviation dividing by $G$. Loss $= -\frac{1}{P}\sum_{\text{prompts}}\frac{1}{G}\sum_i \frac{\hat{A}_i}{\lvert y_i \rvert}\log\pi_{\boldsymbol{\theta}}(y_i \mid x)$, one forward and backward pass over the batch, one Adam step, constant learning rate. One update per batch of rollouts means the ratio is 1 and the clip never acts; there is no KL term ($\beta = 0$) and no gradient clipping. |
| The $G$ sweep | 64 rollouts per step in every run, so $G = 2, 4, 8, 16$ means 32, 16, 8, 4 prompts per step. |
| The switches | Each once, at $G = 8$: do not divide by $s_r$; divide each answer's log-probability by a constant instead of by its own length; drop silent groups and draw new prompts to replace them; LoRA rank 1 instead of 16. |
| Silent groups | A group is silent when all $G$ answers are right or all are wrong: its advantages are all zero. Measured over every group sampled, next to $p^{G} + (1-p)^{G}$ with $p$ each step's measured training accuracy, averaged over steps. The formula assumes every prompt has the same $p$. |
| Accuracy | On the fixed evaluation problems, two sampled answers each (256 rollouts, fixed sampling seed), before training, every 25 steps and at the end. The standard error of one such number is about 0.03. |
| Time | Seconds per step spent sampling and spent updating, measured separately. |

Every replay prints the caveat: one 1.7B model, one kind of problem, one seed, a few hundred steps. The demo shows what the switches are, how often a group is silent and where the time goes. **It cannot confirm or refute published results on how to choose $G$**, which were obtained with far larger models and budgets.

## What came out

Not rehearsed yet (built 2026-11-14). This section will be filled in from `runs/rehearsal/` after the rehearsal.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `numpy`, `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib` (3.10 works with `tomli`). `replay` and `replay-a` need none of the MLX packages. |
| Model | W8 demo 1 must have been run first (`../w08_d1_sft_base_to_assistant`): its `fetch` downloads the model (about 3.4 GB) and its rehearsal trains adapter C. The adapter weights are not distributed. |
| Device | **Apple silicon** for the real model (MLX on Metal). `./present.sh tiny` runs the whole flow on a small random model without downloading anything, also on Linux with the CPU build of MLX. |
| Time and memory | measured by `./present.sh probe` before the sweep; not rehearsed yet. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w09_d2_grpo_scratch
./present.sh test                     # no model: the advantage, the identity on random vectors, the verifier
./present.sh tiny                     # the whole flow on a tiny random model (numbers mean nothing)
./present.sh inspect                  # versions, chat template, stop tokens, bypass parameters
./present.sh probe                    # accuracy of the starting model per difficulty level, seconds per step, a small learning-rate probe
./present.sh rehearse                 # (a) with the reference implementation, then the eight runs, then summary.json
./present.sh replay                   # print the tables of (b) from runs/rehearsal/summary.json
./present.sh live --ref               # (a) with the reference implementation
```

To do (a) yourself, fill in the five lines of `advantage()` in `advantage_live.py` and run `./present.sh live`.

## Files

| | |
|---|---|
| `grpo.py` | The hand-written parts: problem generator, verifier, `Bypass`, sampling, `advantages`, `rollout_weights` (four ways of aggregating over tokens), `grpo_loss`. |
| `advantage_live.py` | The skeleton for (a): `advantage()` is empty. |
| `advantage_final.py` | The reference implementation of `advantage()`. |
| `demo.py` | The commands above; `train_run` is the loop. |
| `demo_config.toml` | Model, bypass, difficulty levels, the eight runs, which run fills which table cell. |
| `present.sh` | Entry point; offline by default. |
| `runs/rehearsal/` | The rehearsal: `identity.json` for (a); one JSON per run with per-step accuracy, the number of right answers in every group, timings and evaluations; `summary.json` with the two tables. |
