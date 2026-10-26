<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 1 — Compute-optimal training, numerically (slides p8–p9 and p22)

Every number in the first half of the week comes from one fitted equation — Hoffmann et al. 2022 ("Chinchilla"), Appendix D.2, Eq. (10):

$$\mathcal{L}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad E = 1.69,\; A = 406.4,\; B = 410.7,\; \alpha = 0.34,\; \beta = 0.28, \qquad C = 6ND.$$

This demo does nothing but arithmetic on it, so that three questions get numbers before the derivation on the slides: does the optimal $D^{*}/N^{*}$ stay at 20 as $C$ grows; how far from the valley floor are three real models, and what does the gap cost in loss; and what happens to the optimum once inference tokens are charged for.

## What it computes

| Sub-command | What | Slide |
|---|---|---|
| `table` | The closed form (the paper's Eq. 4): $N^{*} = G\,(C/6)^{a}$, $D^{*} = G^{-1}(C/6)^{b}$ with $a = \beta/(\alpha+\beta)$, $b = \alpha/(\alpha+\beta)$, $G = (\alpha A/\beta B)^{1/(\alpha+\beta)}$. Prints $a$, $b$, $G$, $G^{-2}$ and the exponent of $D^{*}/N^{*} = G^{-2}(C/6)^{(\alpha-\beta)/(\alpha+\beta)}$, then one row per decade of $C$ from $10^{18}$ to $10^{26}$: $N^{*}$, $D^{*}$, $D^{*}/N^{*}$, $\mathcal{L}^{*}$. The five starred rows are the ones filled in on the slide | p9, p15 |
| `points` | Three real $(N, D)$ pairs — W1's exercise (7B / 140B), Chinchilla itself (70B / 1.4T), Llama 3 8B (8B / 15T): $C = 6ND$, the optimum at that $C$, and $\mathcal{L} - \mathcal{L}^{*}$ under the fit | p9, p17 |
| `inference` | Hold the loss fixed at $\ell = \mathcal{L}(7\text{B}, 140\text{B})$ and, for $D_{\text{inf}} \in \{0, 10^{12}, 10^{13}, 10^{14}\}$ inference tokens, minimise $6ND + 2N D_{\text{inf}}$ over the level set $\mathcal{L} = \ell$ (golden-section search on $\log N$; $D$ is solved from $\ell$). One row per Enter; `--all` prints them at once. "Saving" compares with keeping the $D_{\text{inf}} = 0$ shape and paying for inference anyway. Ends with the threshold $2ND_{\text{inf}} > 6ND \Leftrightarrow D_{\text{inf}} > 3D$ | p22 |
| `tiny` | W4 demo 1's character-level tiny GPT as a point on the same table: $N$ from its parameter count, $D$ from the tokens it saw, and the fit's optimum at that $C$ | p22 |
| `check` | Compares every number above with the slides' data file `slides/assets/w07/w07-data.json`, if it is present (it is in the course repository, not in this one) | — |

The cost function with an inference term is from Sardana et al. 2024 (*Beyond Chinchilla-Optimal*); the numerical solution is this demo's.

## Requirements

| | |
|---|---|
| Packages | **none** — standard library only (`tomllib` on Python ≥ 3.11; a small fallback reader covers older interpreters) |
| Network | none |
| Memory / time | negligible — a few hundred evaluations of one formula |
| Inputs | `tiny` reads `../w04_d1_block_scratch/runs/rehearsal/{train,tests}.json` (checked in with W4 demo 1) |
| Portability | nothing platform-specific; the instructor runs it on an Apple M5 Max (macOS 27, Python 3.12) |

## Run it

```bash
cd demos                      # see ../README.md for the shared environment (optional here)
cd w07_d1_compute_optimal
./present.sh table            # N*, D*, D*/N*, L* for C = 10^18 … 10^26
./present.sh points           # three real models: C, the optimum at that C, the loss gap
./present.sh inference        # Enter per row; or:  ./present.sh inference --all
./present.sh tiny             # W4's tiny GPT on the table
./present.sh replay           # the instructor's rehearsal record (runs/rehearsal/*.json), not a live run
```

Without `present.sh`: `python3 optimal.py table`, `python3 optimal.py inference --all`, and so on.

## What you should see

`table`: $a = 0.4516$, $b = 0.5484$, $G = 1.345$, $G^{-2} = 0.553$; the exponent of $D^{*}/N^{*}$ is $+0.097$, so the ratio is multiplied by $1.25$ per decade of $C$ — it is not a constant under this fit.

| $C$ (FLOPs) | $N^{*}$ | $D^{*}$ | $D^{*}/N^{*}$ | $\mathcal{L}^{*}$ |
|---|---|---|---|---|
| $10^{18}$ | 80.6 M | 2.07 B | 25.7 | 3.535 |
| $10^{20}$ | 645 M | 25.8 B | 40.1 | 2.600 |
| $10^{22}$ | 5.16 B | 323 B | 62.6 | 2.139 |
| $10^{24}$ | 41.3 B | 4.04 T | 97.7 | 1.911 |
| $10^{26}$ | 330 B | 50.4 T | 152.6 | 1.799 |

`points`:

| model | $N$ | $D$ | $D/N$ | $C$ | optimum at that $C$ | $D^{*}/N^{*}$ | $\mathcal{L} - \mathcal{L}^{*}$ |
|---|---|---|---|---|---|---|---|
| W1 p18 | 7 B | 140 B | 20 | $5.88 \times 10^{21}$ | 4.06 B / 241 B | 59.4 | 0.0068 nats |
| Chinchilla | 70 B | 1.4 T | 20 | $5.88 \times 10^{23}$ | 32.5 B / 3.02 T | 92.8 | 0.0067 nats |
| Llama 3 8B | 8 B | 15 T | 1875 | $7.20 \times 10^{23}$ | 35.6 B / 3.37 T | 94.7 | 0.0259 nats |

Neither 7B / 140B nor Chinchilla's own 70B / 1.4T sits on the floor of the valley that Eq. (10) defines — and the floor is flat: trading $N$ down by 40 % and $D$ up by 70 % changes the loss by 0.007 nats.

`inference` (fixed loss $\ell = 2.1835$):

| $D_{\text{inf}}$ | $N^{*}$ | $D^{*}$ | $D^{*}/N^{*}$ | saving vs. keeping the $D_{\text{inf}} = 0$ shape |
|---|---|---|---|---|
| 0 | 3.90 B (×1.00) | 230 B (×1.00) | 59 | — |
| $10^{12}$ | 1.97 B (×0.50) | 547 B (×2.38) | 278 | 21.1 % |
| $10^{13}$ | 1.09 B (×0.28) | 1.88 T (×8.18) | 1717 | 58.9 % |
| $10^{14}$ | 723 M (×0.19) | 8.58 T (×37.3) | 11,869 | 76.9 % |

The inference term charges $N$ and not $D$, so the optimum moves to a smaller model trained on more tokens. Note that the $D_{\text{inf}} = 0$ row (3.90 B) differs from the W1 optimum in `points` (4.06 B): one minimises cost at a fixed loss, the other minimises loss at a fixed cost; on a flat floor the two optima need not coincide.

`tiny`: $N = 4{,}763{,}136$, $D = 49{,}152{,}000$ (3000 steps × 64 × 256), $D/N = 10.3$, $C = 1.40 \times 10^{15}$ FLOPs; the fit's optimum at that $C$ would be 4.15 M / 56.4 M. The fit was made on subword-tokenised MassiveText with models from 70 M to over 16 B parameters; a 65-symbol character model on one book is outside its range in every variable, so the point sits on the table but not on the curve.

Every number above is arithmetic on the published constants, not a measurement; `check` confirms they equal the values the slides were built from.

## Files

| File | |
|---|---|
| `optimal.py` | `table` / `points` / `inference` / `tiny` / `check` / `rehearse` / `replay`; the equation, the closed form and the golden-section search are the first hundred lines |
| `demo_config.toml` | the five constants, the $C$ grid, the three real points, the $D_{\text{inf}}$ values, the W4 record paths |
| `present.sh` | entry point |
| `runs/rehearsal/*.json` | the rehearsal records `replay` prints |
