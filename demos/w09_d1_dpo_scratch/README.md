<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W9 · Demo 1 — DPO: a few lines of loss, five tests, one real run (slides p33–p36)

The derivation of DPO takes four boards. The result is a function of five lines. This demo writes that function, checks it, and then uses it to train a real model.

- **(a), live.** `dpo_loss(logp_w, logp_l, ref_w, ref_l, beta)` is written from scratch. Its inputs are four log-probabilities of whole answers: the model being trained and the frozen reference, each on the preferred answer $y_w$ and on the other answer $y_l$. Five tests follow.
- **(b), rehearsed and replayed.** Last week's model writes several answers to each prompt, a hard-coded rule decides which answer of a pair wins, and the model is trained with DPO at $\beta \in \{0.05, 0.1, 0.5\}$. The log-probabilities of $y_w$ and of $y_l$ are recorded **separately**, because the loss only sees their difference.

## (a) The five tests

The inputs are six made-up pairs of log-probabilities (no model is loaded). Every number is a multiple of 1/8, so adding an integer is exact in float32.

| test | what it checks | expected |
|---|---|---|
| (i) | the policy equals the reference | the loss is $\log 2 \approx 0.693$, and every pair has weight $\tfrac{1}{2}$ in the gradient |
| (ii) | add 1 to every `logp_w` | the loss falls |
| (iii) | add the same constant to `logp_w` **and** `logp_l` ($+5$, then $-50$) | the loss does not change |
| (iv) | $\beta = 10^{-4}$, and $\beta = 50$ | $\log 2$ with a vanishing gradient; then a hard verdict: about 0 for a pair ranked right, $\beta \times \lvert\text{gap}\rvert$ for a pair ranked wrong |
| (v) | the formulation in Appendix B of the DPO paper (Rafailov et al., 2023), retyped in PyTorch, on the same inputs | equal to floating-point error |

`paper_reference.py` follows the paper's interface (one tensor of log-probabilities plus two index tensors) and its grouping (the policy's $y_w - y_l$ first, then the reference's). `dpo_live.py` groups the other way (policy minus reference for $y_w$, then for $y_l$). The two are the same number.

A practical point that test (iv) catches: writing the fourth line as `-log(sigmoid(z))` gives `inf` when $z$ is very negative, because the sigmoid underflows to 0. `nn.log_sigmoid` computes the same quantity stably.

## (b) What is being measured

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B-Base` (bf16, Apache-2.0) with last week's adapter C from W8 demo 1. This is $\pi_{\text{ref}}$. |
| What is trained | A new LoRA bypass (demo 2's `grpo.Bypass`, rank 16, `scale 20`) on the seven linear layers of every block. Its $\mathbf{B}$ starts at zero, so the policy at step 0 is exactly $\pi_{\text{ref}}$. Switching the bypass off gives $\pi_{\text{ref}}$ back at any time: **there is one copy of the frozen weights in memory, not two.** |
| Prompts | The user turns of last week's split of databricks-dolly-15k (CC BY-SA 3.0): `valid.jsonl` for the training pairs; from `test.jsonl`, the first 40 prompts for held-out pairs and the next 48 for sampling only. The human-written answers in the dataset are not used. |
| Answers | $\pi_{\text{ref}}$ samples 6 answers per prompt at temperature 1, at most 160 new tokens. |
| The rule | An answer wins if it **ends properly** (it reaches a stop token instead of the token limit) **and is shorter**. If both end properly, the winner must be at least 8 tokens shorter. For each prompt the answers are sorted by the rule; the best is paired with the worst and the second best with the second worst, when the rule separates them. |
| Why a hard-coded rule | The "true preference" is then something that can be measured directly: the length of an answer and whether it ends properly. No human labels are needed. **These are not human preferences.** |
| Reference log-probabilities | $\log\pi_{\text{ref}}(y_w \mid x)$ and $\log\pi_{\text{ref}}(y_l \mid x)$ are computed once, with the bypass off, and stored. (`ref = "online"` in the config recomputes them at every step on the same padded batch instead.) |
| Training | 4 pairs per step (8 sequences, one forward and one backward pass), Adam, constant learning rate, 200 steps, the pairs reshuffled every epoch. The loss is `dpo_final.dpo_loss`. One run per $\beta$, same seed. |
| On the training pairs | Every 25 steps, over all of them: the loss, the share of pairs ranked right, the mean of $\log\pi_{\boldsymbol{\theta}}(y_w \mid x)$ and of $\log\pi_{\boldsymbol{\theta}}(y_l \mid x)$ in nats per whole answer, and the paired change of each since step 0 with its standard error. A log-probability "rose" or "fell" only if the mean change exceeds two standard errors. |
| On held-out pairs | The same quantities on pairs built from prompts that were never trained on. |
| The rule's gold | On 48 unseen prompts, two sampled answers each with a fixed sampling seed: the mean length in tokens and the share that end properly. |
| Distance from $\pi_{\text{ref}}$ | A sampled estimate of the KL divergence on those answers, in nats per answer (as in demo 3). |
| What was forgotten | Bits per byte on the four held-out texts of W8 demo 4 (b), measured the same way: before training, and after each run. |

What would count as likelihood displacement here: the loss falls **and** the mean of $\log\pi_{\boldsymbol{\theta}}(y_w \mid x)$ falls.

Every replay prints the caveat: one 1.7B model, one hard-coded rule, one seed, a few hundred steps. This is not a replication of any paper.

## What came out

(a): the five tests pass with the reference implementation: 0.693147 and six weights of 0.500; 0.799502 → 0.746316; no change under either shift; 0.693237 at $\beta = 10^{-4}$, then 0.000000 and 187.5 at $\beta = 50$; a difference of 0 from the appendix formulation.

(b): not rehearsed yet (built 2026-11-15). This section will be filled in from `runs/rehearsal/` after the rehearsal, whichever way it turns out.

## Requirements

| | |
|---|---|
| Packages | (a): `mlx`, and `torch` for test (v) (without it the test is skipped). (b): also `mlx-lm` (git commit pinned in `../versions.lock`), `numpy`, `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib`. `replay` and `replay-a` need only the standard library. |
| Other demos | `../w09_d2_grpo_scratch` must be present (its `grpo.py`, `demo.py` and `demo_config.toml` are loaded from there). For (b): W8 demo 1 must have been run first (the model, adapter C and the Dolly split), and the bits-per-byte texts come from W8 demo 4 and W7 demos 2–3. |
| Device | (a) runs anywhere MLX runs, including Linux with the CPU build. (b) with the real model needs **Apple silicon**. `./present.sh tiny` runs the whole flow of (b) on a small random model without downloading anything; its numbers mean nothing. |
| Time and memory | (a): under a second. (b): not rehearsed yet. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w09_d1_dpo_scratch
./present.sh live --ref               # (a) with the reference implementation; without --ref it runs your dpo_live.py
./present.sh test                     # no downloads: the five tests, the rule, gradients on a tiny random model
./present.sh tiny                     # the whole flow of (b) on a tiny random model
./present.sh inspect                  # the template, the stop tokens, bypass on = bypass off at the start, bytes of weights in memory
./present.sh pairs                    # sample the answers, apply the rule, store the reference log-probabilities
./present.sh probe                    # seconds per step and a small learning-rate scan
./present.sh rehearse                 # (a), pairs if missing, one run per beta, then summary.json
./present.sh replay                   # print the tables from runs/rehearsal/summary.json
```

## Files

| | |
|---|---|
| `dpo_live.py` | The skeleton filled in during class: the signature and five comment lines. |
| `dpo_final.py` | The reference implementation; also the loss used for training in (b). |
| `paper_reference.py` | Test (v): the Appendix B formulation, retyped in PyTorch. |
| `dpo.py` | The rule and the pairing, log-probabilities of a batch of pairs, the metrics, the KL estimate, bits per byte. |
| `demo.py` | The commands above. |
| `demo_config.toml` | The inputs of (a), where the prompts come from, sampling and rule thresholds, the training settings, the three values of $\beta$. |
| `present.sh` | Entry point; offline by default. |
| `runs/rehearsal/` | The rehearsal: `tests.json` for (a); `pairs.json` (how many pairs, their lengths and reference log-probabilities, a few pairs as text); one JSON per run with every step and every evaluation; `summary.json`. The token ids of the pairs are kept in `runs/cache/`, which is not in the repository; `./present.sh pairs` recreates them. |
