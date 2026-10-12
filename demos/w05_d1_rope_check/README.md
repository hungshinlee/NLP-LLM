<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 1 — RoPE's relativity, checked to float64 (slide p18; the wavelength table is slide p19)

In class the instructor writes three functions live in `rope_live.py`, running the tests after each one:

| Function | Signature | What it does |
|---|---|---|
| `rope_angles(d_head, base=10000.0)` | → `(d_head/2,)` float64 | $\theta_i = b^{-2i/d_{\text{head}}}$, $i = 0, \dots, d_{\text{head}}/2 - 1$ |
| `rotate(x, m, theta)` | `x: (..., d_head)`, `m`: a position, `theta`: the output of `rope_angles` → same shape as `x` | treats coordinates $(2i, 2i+1)$ as a point in the plane and rotates it by $m\theta_i$ — the matrix $\mathbf{R}_m$ applied to $\mathbf{x}$. Adjacent pairing is used here; Hugging Face's Llama code pairs $(i, i + d_{\text{head}}/2)$ ("rotate-half"). Same $\mathbf{R}_m$, different ordering of the coordinates. |
| `score(q, k, m, n, theta)` | → scalar | $(\mathbf{R}_m\mathbf{q})^\top(\mathbf{R}_n\mathbf{k})$ |

`rope_final.py` is the finished reference implementation; `tests.py` holds the three tests and is shared by both files, so you can write your own version in `rope_live.py` and test it step by step — functions you have not written yet are reported as `skip`, not as failures. `wavelengths.py` is the second half of the demo: the table behind slide p19.

## The three tests

All three use the same random $\mathbf{q}, \mathbf{k}, \mathbf{v}$ (float64, $d_{\text{head}} = 128$, $b = 10^4$), keep $m - n = 7$ fixed and slide $m$ over ten values from 0 to 900.

| | Checks | Passes when | Why it matters |
|---|---|---|---|
| (a) | the RoPE score $(\mathbf{R}_m\mathbf{q})^\top(\mathbf{R}_n\mathbf{k})$ over the ten values of $m$; the same inputs are run again in float32 for comparison | the float64 spread is $< 10^{-10}$ (expect $\sim 10^{-13}$: the angles reach 900 rad and `cos` rounds; float32 gives $\sim 10^{-4}$) | the blackboard proof that the score depends only on $m - n$, run on a machine. This is why the demo is pinned to the CPU in float64 — Apple's Metal backend has no double type |
| (b) | the *additive* sinusoidal stamp under the same shift: $(\mathbf{q} + \mathbf{p}_m)^\top(\mathbf{k} + \mathbf{p}_n)$ with $\mathbf{p}_m = (\sin m\theta_i, \cos m\theta_i)_i$, the same frequencies as RoPE; the three terms are printed separately | the score **moves** (spread $> 10^{-3}$; about 24 here), and all of the motion is in the cross terms $\mathbf{q}^\top\mathbf{p}_n + \mathbf{p}_m^\top\mathbf{k}$ — $\mathbf{q}^\top\mathbf{k}$ and $\mathbf{p}_m^\top\mathbf{p}_n$ are constant to $10^{-13}$ | adding a position vector gives cross terms that depend on absolute position. Note that $\mathbf{p}_m^\top\mathbf{p}_n$ by itself *is* relative: the additive scheme fails because of the cross terms, not because of $\mathbf{p}$ |
| (c) | rotate $\mathbf{v}$ as well: two keys at positions $n$ and $n + 3$ with fixed weights $\tfrac12, \tfrac12$; the output $\sum_j w_j \mathbf{R}_{n_j}\mathbf{v}$ as $n$ slides | its first component moves (spread $> 10^{-3}$; about 0.10 here) while its norm is constant to $10^{-9}$ | the output equals $\mathbf{R}_n(w_0\mathbf{v} + w_1\mathbf{R}_3\mathbf{v})$: the whole vector is rotated by the *absolute* position $n$, and nothing downstream undoes it. That is why only $\mathbf{q}$ and $\mathbf{k}$ are rotated, never $\mathbf{v}$ |

## The wavelength table (`wavelengths.py`)

For $d_{\text{head}} = 128$ and two bases, $b = 10^4$ and $b = 5 \times 10^5$ (Llama 3), it prints each pair's wavelength $\lambda_i = 2\pi/\theta_i$ and how many turns it completes within a training length $L_{\text{train}}$ of 256 (last week's tiny GPT) and 8192; pairs that never complete one turn are coloured. The summary at the top gives the counts the slides quote (14 and 29 pairs at $L = 8192$; 38 at $L = 256$) and the last line checks them against `slides/assets/w05/w05-data.json`, which the slides were built from. It needs only the standard library.

## Requirements

| | |
|---|---|
| Packages | `torch` only (`2.14.0` is pinned; nothing else is imported). `wavelengths.py` needs no package at all. No model, no download, no network. |
| Device | **CPU, deliberately, in float64.** `rope_final.py` sets `torch.set_default_device("cpu")` and `torch.set_default_dtype(torch.float64)`. If you write your own, make `rope_angles` return float64 explicitly (`torch.arange(0, d_head, 2, dtype=torch.float64)`); a float32 $\theta$ silently drops test (a) to $\sim 10^{-5}$ and it fails. |
| Memory / time | negligible — a few vectors of length 128 |
| Portability | should run on any machine where `torch` installs (nothing platform-specific is used); tested only on the instructor's Apple M5 Max — see "What you should see" |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv
cd w05_d1_rope_check
./present.sh                  # three tests against the reference implementation: a ok, b ok, c ok
./present.sh show             # a d_head = 4 example printed step by step, then the three tests
./present.sh wavelengths      # the wavelength table; add --no-table for the summary only, --d-head 64 for another head size
./present.sh reset            # empties rope_live.py down to its header comment and `import torch`
./present.sh live             # tests against *your* rope_live.py; unwritten functions show as skip
./present.sh replay           # prints the instructor's rehearsal record (runs/rehearsal/tests.json), not a live run
```

Without `present.sh` (any shell, your own interpreter): `python rope_final.py`, `python rope_final.py --show`, `python tests.py rope_live`, `python wavelengths.py`.

Suggested order if you write it yourself, matching the class: `rope_angles` → `rotate` → look at (b) and (c), which need only these two → `score` → look at (a).

## What you should see

The rehearsal record `runs/rehearsal/tests.json` will be checked in once the instructor has run `./present.sh rehearse` on the Apple M5 Max; until then the expected magnitudes above come from a NumPy mirror of the same tests with the same constants (2026-10-12): (a) $2.6 \times 10^{-13}$ in float64, $9.7 \times 10^{-5}$ in float32; (b) a spread of about 24, entirely in the cross terms; (c) a spread of about 0.10 in the first component, norm constant to $10^{-14}$. Your figures will agree with these in magnitude, not digit for digit — the random vectors come from a `torch.Generator`.

## Files

| File | |
|---|---|
| `rope_final.py` | reference implementation + `--show` / `--rehearse` |
| `rope_live.py` | the file written live in class; `./present.sh reset` empties it |
| `tests.py` | the three tests; `python tests.py <module>` runs them against any module exposing the three functions |
| `wavelengths.py` | the wavelength / turns table, checked against the slides' data file |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/tests.json` | the rehearsal record (once written) |
