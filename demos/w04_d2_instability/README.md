<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W4 · Demo 2 — training instability on a tiny GPT: pre-LN vs. post-LN, learning-rate sweep, QK-norm (slides p29, p44)

This demo is **replay only**: every run was done in rehearsal (38 minutes in total) and in class only the records in `runs/rehearsal/` are shown. The model is the tiny GPT assembled in W4 demo 1 (`../w04_d1_block_scratch/block_final.py`), trained on the same Shakespeare text; this folder only holds the three experimental settings.

Every 25 steps the run records four statistics inside the training forward pass (no extra pass):

| Statistic | How | Slide |
|---|---|---|
| loss | cross-entropy of the batch, nats/char | p43 left: loss spikes |
| $\max\lvert s_{ij}\rvert$ | largest $\lvert \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_{\text{head}}}\rvert$ over all layers and heads, masked positions excluded; with QK-norm, after the norm | p43 middle: attention-logit growth; the bound on p27 |
| $\log Z$ | $\log \sum_v e^{z_v}$ of the output logits, averaged over positions | p43 right: output-logit divergence; what z-loss watches |
| gradient norm | $\lVert \mathbf{g} \rVert_2$ of the whole model before clipping, plus the norm per block | p29 third column; "which layer goes wrong first" |

A **spike** is defined operationally by this demo as the first record where the loss is 0.5 nats above its running minimum, or NaN. There is no standard definition in the literature.

## The three settings

| | Model | Variable | Slide |
|---|---|---|---|
| (i) | the tiny GPT deepened to **16 layers** | pre-LN vs. post-LN, **no warmup**, constant LR $10^{-3}$, 1000 steps | p29 |
| (ii) | the 6-layer tiny GPT of demo 1 | LR $\{1, 3, 10\} \times 10^{-3}$, 100 warmup steps then constant, 2000 steps | p44, rows 1–3 |
| (iii) | same | the $10\times$ LR **with QK-norm** (an RMSNorm on $\mathbf{q}$ and on $\mathbf{k}$, $2 d_{\text{head}}$ extra $\gamma$ per layer); optionally a z-loss run | p44, row 4 |

Three deliberate differences from demo 1's training recipe, all meant to let symptoms show rather than to reach convergence: dropout **0** (noise would hide the trends), gradient clipping **off** (clipping is one of the fixes on p46 — it is turned off so that the symptom is visible; `--clip 1.0` runs a clipped comparison), and a **warmup-then-constant** LR schedule instead of cosine (a decaying LR makes spikes disappear). Everything else follows demo 1: AdamW, $\beta = (0.9, 0.95)$, decoupled weight decay 0.1 on matrices only, batch 64 × 256, seed 1.

**An honest limit, printed in every replay banner**: this is a proxy of a proxy. The published proxy studies use tens to hundreds of millions of parameters; here it is 4.8 M (6 layers) and 12.7 M (16 layers). What transfers is the *shape* of the mechanisms — none of the numbers (LR multipliers, step counts, logit magnitudes) is a threshold for a real model.

## Requirements

| | |
|---|---|
| Packages | `torch 2.14.0` only; nothing pre-trained is loaded. |
| Code dependency | imports `GPT` from `../w04_d1_block_scratch/block_final.py` and the data / device / optimizer helpers from `../w04_d1_block_scratch/train.py` by file path; reads that folder's `demo_config.toml` for the tiny GPT and the corpus. Keep the published folder structure. |
| Corpus (only for `rehearse`) | the same `slides/_corpus/tinyshakespeare.txt` as demo 1 — see that README for the download. `replay` and `show` need nothing but the JSON records. |
| Device | `mps` by default, automatic fallback to CPU (printed); `cuda` is accepted by the shared helper but was never run. |
| Time | rehearsal on the M5 Max: (i) 401 s + 470 s, (ii) 331 / 338 / 333 s, (iii) 391 s — **38 minutes** for all three. `./present.sh estimate` extrapolates from demo 1's `probe.json` (it estimated 30 minutes against 38 actual, because it ignores the statistics recording). |
| Tested on | Apple M5 Max, macOS 27.0, Python 3.12.14, torch 2.14.0 (2026-10-06) |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w04_d2_instability
./present.sh replay i                 # p29: table, the two loss / gradient-norm curves, per-layer gradient norm at step 0
./present.sh replay ii                # p44 rows 1–3: table and the four statistics over steps
./present.sh replay iii               # p44 row 4: 10× + QK-norm next to the plain 10× run
./present.sh show ii                  # table only
open runs/rehearsal/ii.svg            # the four-panel static figure (one per setting)
./present.sh baselines                # one second: unigram and bigram cross-entropy of the held-out text — the "stuck at" yardstick
./present.sh estimate                 # how long the three settings would take on this machine, from demo 1's probe
./present.sh rehearse i               # re-run a setting yourself (≈ 15 min); ii ≈ 17 min; iii ≈ 7 min. Overrides: --depth 24, --lr 3e-3, --steps N, --mult 1,3,10,30, --z-loss 1e-4, --clip 1.0
```

`rehearse` overwrites that setting's JSON and SVG in `runs/rehearsal/`; the other two settings are untouched. `git checkout` restores the instructor's record.

## What you should see

From `runs/rehearsal/{i,ii,iii}.json` (2026-10-06):

| run | final loss | stuck at | peak $\max\lvert s_{ij}\rvert$ | final $\log Z$ | grad norm, first 200 steps (mean / max) | per-layer grad norm at step 0 | spike / divergence |
|---|---:|---|---:|---:|---|---|---|
| (i) pre-LN, 16 L, no warmup | 1.245 | below bigram | 109 | 9.0 | 1.33 / 7.49 | 4.00 at layer 0, decreasing to 0.73 at layer 15 | none / none |
| (i) post-LN, 16 L, no warmup | **3.321** | **≈ unigram (3.35)** | 26 | 4.6 | 0.62 / 3.33 | all sixteen layers 0.16–0.18 | none / none — **flat from step 100 on; it does not learn** |
| (ii) 1× | 0.914 | below bigram | 97 | 10.9 | 1.64 / 7.79 | 4.61 → 1.36 | none |
| (ii) 3× | 1.134 | below bigram | 353 | 9.8 | | | none |
| (ii) 10× | **2.475** | **≈ bigram (2.48)** | **11,577** | 6.0 | | | none — **flat from step 100 on** |
| (iii) 10× + QK-norm | 1.257 | below bigram | 58 | 8.2 | 1.25 / 7.33 | 4.18 → 1.33 | none |

(Yardsticks from `baselines.json`: character unigram 3.348, bigram 2.479 nats/char on the held-out text.)

How to read it:

- **No run spiked** by the definition above. What the sweep shows instead is **entropy collapse**: the peak attention logit grows monotonically with the LR (97 → 353 → 11,577), and at $10\times$ the loss freezes at the character-bigram level from the end of warmup — the softmax has saturated to one-hot rows, gradients stop flowing through attention (this run has the lowest gradient norm), and the model only learns "look at the previous character". $\log Z$ does not diverge; at $10\times$ it is the lowest.
- **QK-norm turns the same LR from not-learning into learning**: peak 11,577 → 58, loss 2.49 → 1.26. Note 58 is far above the initial bound $\sqrt{32} \approx 5.66$ — the $\gamma$ parameters grow during training, so the bound exists by construction but is not a constant.
- **Post-LN without warmup stalls rather than blows up** at this scale: it does not diverge, it sits at the unigram level. Its per-layer gradient norms at step 0 are uniform and small (0.17, twenty times smaller than pre-LN's layer 0); the "gradients largest in the last layers" picture from the post-LN analysis in the literature does **not** appear in this setting — that analysis is about LayerNorm with Xavier initialization, this is RMSNorm with std-0.02 initialization and Adam, and the difference has not been isolated here.
- "≈ unigram / bigram" compares a training-batch loss to a held-out yardstick; the 0.01–0.03 gaps are within noise.

Suggested follow-ups that have **not** been run: a finer LR sweep between $3\times$ and $10\times$ to see whether a spike exists; `--clip 1.0` at $10\times$; `--z-loss 1e-4`; setting (i) with LayerNorm and Xavier initialization.

## Files

| File | |
|---|---|
| `demo.py` | `rehearse [i\|ii\|iii]` / `replay <setting>` / `show <setting>` / `estimate` / `baselines` |
| `demo_config.toml` | the three settings and the shared training hyperparameters |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/i.json`, `ii.json`, `iii.json` + `.svg` | per-run statistics curves, per-layer gradient norms, spike detection, summaries; four-panel figures |
| `runs/rehearsal/baselines.json` | unigram / bigram yardsticks |
