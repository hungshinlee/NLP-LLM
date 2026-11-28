<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W10 · Demo 1 — no thinking, dots, thinking against the length of the chain; then sweep the thinking budget (slides p17, p19, p28)

Writing the steps out before answering helps a language model on some problems. Two accounts of why make different predictions about one measurable thing: accuracy as a problem needs more steps in a row. This demo measures it, and then asks how much reasoning is enough.

- **(a) Three ways of answering the same problem.** *No thinking*: the answer comes straight away. *Thinking*: the model reasons first. *Dots*: the model's own reasoning is replaced by the same number of "." tokens, so the model gets the extra positions but none of the content. Accuracy is plotted against the number of steps $m$.
- **(b) Budget forcing.** The two moves of Muennighoff et al. (*s1*, EMNLP 2025, §3.1), neither of which needs retraining: *force a stop* by appending the end-of-thinking token once a token limit is reached, and *refuse a stop* by replacing the end-of-thinking token with the word "Wait" when the model tries to finish.

## What is being measured

| | |
|---|---|
| Task | Integer arithmetic generated from a seed (`make_problem` in `../w09_d2_grpo_scratch/grpo.py`, the generator of W9 demo 2): two-digit operands joined by $+$, $-$, $\times$ (the right-hand factor is 2–9), usual precedence. "$m$ steps" means $m$ operators; $m \in \{1, 2, 3, 4, 6, 8, 12\}$. |
| Verifier | 1 if the **last number** in the answer equals the result, otherwise 0. |
| Model | `Qwen/Qwen3-8B` (bf16, Apache-2.0). Thinking and non-thinking are the same weights; the chat template's `enable_thinking` flag switches between them. |
| Prompt | `Compute {expr}. Reply with only the final number.` — the same sentence for all three ways. |
| Sampling | temperature 0.6, top-p 0.95, top-k 20, for everything. One sampled answer per problem and condition. |
| Answer prefix | After `</think>` the text `The final number is $\boxed` is forced (without the opening brace: in this tokenizer `{-` is a single token, so forcing `{` would decide the sign for the model), in all three ways and in every budget column, whether the model stopped thinking by itself or was stopped. Without it the model rewrites the whole solution in its answer. (*s1* optionally appends "Final Answer:" in the same place.) |
| Answer limit | 16 tokens after the prefix: room for one number, not for a worked solution. Answers that run into the limit are judged on their last number; how often that happens is recorded. |
| No thinking | The template with `enable_thinking=False`, which closes an empty `<think>` block for the model. |
| Thinking | The template with `enable_thinking=True`. A safety cap (4,096 thinking tokens) forces `</think>`; how often it is reached is recorded. |
| Dots | The thinking template, then `<think>`, then $n$ copies of the token ".", then `</think>` and the answer prefix, where $n$ is the number of thinking tokens the model wrote for that same problem. |
| $q^{m}$ | One number $q$ fitted by maximum likelihood to the thinking column under the assumption that every step succeeds independently with probability $q$. The per-row value $\text{accuracy}^{1/m}$ is printed next to it; if it is not constant, the assumption does not hold. |

### How the budget columns are produced

Every column of (b) starts from the reasoning the model wrote in (a) for that problem.

| Column | |
|---|---|
| `0` | The no-thinking answers of (a). |
| `32` … `1024` | If the reasoning is longer than the limit: keep its first $c$ tokens, append `</think>`, draw the answer again. Otherwise the limit never acts and the result is the one from (a). Truncating a finished piece of reasoning has the same distribution as generating under the limit, and it makes the columns paired. |
| `none` | The thinking answers of (a). |
| `+1`, `+2`, `+4` *Wait* | Where the model wrote `</think>`, "Wait" is inserted instead and generation continues; the model's next three attempts to stop are replaced in the same way. `+4` is the result of that run; `+1` and `+2` are the same run cut at its second and third attempt to stop. |
| flips | Problems that were right in the column to the left and are wrong here (and the reverse). |
| control | The reasoning left untouched and only the answer drawn again: the number of flips that sampling the answer alone produces. |

`Forcer` in `forcing.py` is the state machine, attached to `mlx-lm`'s `BatchGenerator` as a per-sequence logits processor. To refuse a stop it draws a token with the same sampler; any token other than `</think>` is passed through unchanged, so the distribution of everything else is untouched.

Every replay prints the caveat: one 8B model, one kind of problem, one seed, one sampled answer per cell. A cell of $n$ problems has a standard error of up to $0.5/\sqrt{n}$.

## What came out

Rehearsed on 2026-11-25 (M5 Max, 64 GB; `runs/rehearsal/`). Fifty problems per cell, one sampled answer each, so a cell has a standard error of up to 0.07. Part (a) took 54 minutes, part (b) 22; peak memory 52.7 GB.

| steps $m$ | no thinking | dots | thinking | thinking tokens, mean |
|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 497 |
| 2 | 0.64 | 0.64 | 1.00 | 743 |
| 3 | 0.18 | 0.24 | 1.00 | 974 |
| 4 | 0.06 | 0.02 | 1.00 | 1,283 |
| 6 | 0.00 | 0.00 | 1.00 | 1,788 |
| 8 | 0.00 | 0.00 | 1.00 | 2,177 |
| 12 | 0.00 | 0.00 | 1.00 | 2,732 |

Without thinking, accuracy is gone by four steps. Dots follow the no-thinking column cell by cell (the two agree on 343 of 350 problems). With thinking, all 350 problems were answered correctly, so **no decline of the form $q^{m}$ was observed up to $m = 12$**; the fitted $q$ is 1.

| limit on thinking tokens | 0 | 32 | 64 | 128 | 256 | 512 | 1024 | none | +1 *Wait* | +2 | +4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $m = 2$ | 0.64 | 0.70 | 0.68 | 0.64 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| $m = 8$ | 0.00 | 0.02 | 0.00 | 0.00 | 0.02 | 0.04 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |

Accuracy does not rise gradually with the budget: reasoning cut off halfway is worth almost nothing. Refusing a stop one, two or four times changed no answer in either direction (99 problems), and neither did the control. **No fall was observed.** One model, one kind of problem, one seed.

An earlier rehearsal was discarded: with the forced prefix ending in `{`, answers to problems with a negative result lost their minus sign.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib` (3.10 works with `tomli`). The three `replay` commands need none of the MLX packages. |
| Model | `Qwen/Qwen3-8B`, about 16.4 GB; downloaded by `fetch` in `../w08_d2_memory_ledger`. |
| Device | **Apple silicon** for the real model (MLX on Metal); only an M5 Max with 64 GB has been tried. `./present.sh tiny` runs the whole flow on a small random model without downloading anything, also on Linux with the CPU build of MLX. |
| Time and memory | about 76 minutes for `rehearse` on the machine above — (a) 54, (b) 22 — with a peak of 52.7 GB; `./present.sh probe` prints an estimate first. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w10_d1_depth_and_budget
./present.sh test                     # no model: the state machine, the prefixes, the fit
./present.sh tiny                     # the whole flow on a tiny random model (numbers mean nothing)
./present.sh inspect                  # templates, token ids, one real piece of reasoning, the two moves
./present.sh probe                    # tokens per second by batch size; a small sample at each m; a time estimate
./present.sh rehearse                 # (a), then (b), then the one live problem
./present.sh replay                   # the table of (a)
./present.sh replay-budget            # the table of (b)
./present.sh live                     # one problem, answered three ways
```

## Files

| | |
|---|---|
| `forcing.py` | The hand-written parts: `Forcer` (budget forcing as a logits processor), `split_think`, the three prefix builders (`prefix_dots`, `prefix_capped`, `prefix_wait`), batched generation, the fit of $q$, the count of flips. |
| `demo.py` | The commands above. |
| `demo_config.toml` | Model, sampler, problem settings, the grid of $m$, the budgets, the live problem. |
| `present.sh` | Entry point; offline by default. |
| `runs/rehearsal/` | The rehearsal: `chain.json` for (a), `budget.json` for (b), `live.json` for the live problem, `inspect.json`, `probe.json`. |
