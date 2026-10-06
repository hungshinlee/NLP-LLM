<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W3 · Demo 1 — attention and a KV cache, from scratch (slide p21)

In class the instructor writes three functions live in `attention_live.py`, running the tests after each one:

| Function | Signature | What it does |
|---|---|---|
| `scaled_dot_product_attention(Q, K, V, mask=None)` | `Q: (L_q, d_k)`, `K: (L, d_k)`, `V: (L, d_v)`, `mask: (L_q, L)` of `0`/`-inf` added to the scores → `(out, A)` | `A = softmax(QKᵀ/√d_k + mask)`, `out = A V`. Leading batch dimensions are allowed. |
| `causal_mask(L)` | → `(L, L)` | `-inf` above the diagonal, `0` elsewhere |
| `decode_step(x_t, cache, W)` | `x_t: (1, d)`, `cache = {"K": (t-1, d_k), "V": (t-1, d_v)}`, `W = {"q","k","v"}` → `(1, d_v)` | Compute `k_t, v_t` once, append them to the cache, attend one query over the whole cache. No mask needed: the cache only holds the past. |

`attention_final.py` is the finished reference implementation; `tests.py` holds the five tests (a)–(e) and is shared by both files, so you can write your own version in `attention_live.py` and test it step by step — functions you have not written yet are reported as `skip`, not as failures.

## The five tests

| | Checks | Passes when | Why it matters |
|---|---|---|---|
| (a) | Output against `torch.nn.functional.scaled_dot_product_attention` (no mask, then causal vs. `is_causal=True`) | max abs error `< 1e-5` in float32 | the whole operation is five lines |
| (b) | Jacobian `∂out_t / ∂x_s` via `torch.autograd.functional.jacobian` | the block `s > t` is **exactly** 0 (not small: `e^{-inf} = 0`, and so is the softmax gradient there); the block `s ≤ t` is non-zero | causality proven by autograd, not by looking at a heat-map |
| (c) | One causal forward pass over a sequence vs. token-by-token `decode_step` with a cache | max abs error `< 1e-5`; cache has exactly `L` rows; cached `K` equals `X W_k` | causality is what makes the KV cache *correct*, not just cheaper |
| (d) | Mean row entropy of the attention weights, scaled vs. unscaled, for `d_k ∈ {16, 128, 1024}` (`q, k ~ N(0, I)`, 64 keys, 200 trials) | scaled: all three `> 3.3` nats; unscaled: decreasing, `< 0.4` at `d_k = 1024` | the `1/√d_k` variance argument, measured. You do not have to comment the scaling out — the test pre-multiplies `Q` by `√d_k`, which cancels it. |
| (e) | A fully masked row (causal mask plus a "padding" row set to `-inf`) | first version returns `NaN` (reported as `info`: this is expected), the fixed version returns finite values (`ok`). The same line prints what the built-in `F.scaled_dot_product_attention` returns on the same input. | the most common first bug in a hand-written attention: it does not raise, it silently spreads `NaN` through the model |

Two fixes for (e) are both acceptable and both shown by `--show`: replace `NaN` with `0` (the row's output becomes zero), or replace `-inf` with `torch.finfo(dtype).min` (the row becomes a uniform distribution).

## Requirements

| | |
|---|---|
| Packages | `torch` only (`2.14.0` is pinned; nothing else is imported). No model, no download, no network. |
| Device | **CPU, deliberately.** Test (b) uses autograd in float32, which is enough; the finite-difference alternative (`torch.autograd.gradcheck`) needs float64, and Apple's Metal backend has no double type at all. `attention_final.py` sets `torch.set_default_device("cpu")`. |
| Memory / time | negligible — all five tests ran in 0.12 s on the instructor's machine |
| Portability | should run on any machine where `torch` installs (the code uses nothing platform-specific); tested only on an Apple M5 Max, macOS 26.6, Python 3.12.14, torch 2.14.0 |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv
cd w03_d1_attention_scratch
./present.sh                  # five tests against the reference implementation: a ok, b ok, c ok, d ok, e ok
./present.sh show             # a small worked example printed step by step, then the five tests
./present.sh reset            # empties attention_live.py down to its header comment
./present.sh live             # tests against *your* attention_live.py; unwritten functions show as skip
./present.sh replay           # prints the instructor's rehearsal record (runs/rehearsal/tests.json), not a live run
```

Without `present.sh` (any shell, your own interpreter): `python attention_final.py`, `python attention_final.py --show`, `python tests.py attention_live`.

Suggested order if you write it yourself, matching the class: `scaled_dot_product_attention` → look at (a) and the `NaN` in (e) → `causal_mask` → look at (b) and the causal half of (a) → fix (e) → `decode_step` → look at (c) → finally read (d).

## What you should see

From `runs/rehearsal/tests.json` (2026-09-28, torch 2.14.0, CPU):

- (a) max error `0.0` with and without mask — the reference matches `F.scaled_dot_product_attention` bit for bit in float32 here.
- (b) `max |∂out_t/∂x_s|` for `s > t` is `0.0e+00`; for `s ≤ t` it is `8.2e-01`.
- (c) step-by-step vs. full forward: `1.19e-07`; cached `K` vs. `X W_k`: `0.0`.
- (d) entropy in nats, scaled vs. unscaled (maximum possible `ln 64 = 4.16`): `d_k=16`: 3.70 vs 1.40; `d_k=128`: 3.68 vs 0.32; `d_k=1024`: 3.68 vs 0.10.
- (e) after the fix the fully masked row is `[0, 0, 0, 0, 0]`; the built-in `F.scaled_dot_product_attention` on the same input returned finite values on this torch version (it fixes the row for you — your own implementation does not).

Test (d) draws its random numbers with a `torch.Generator`, so your figures will agree with these in magnitude, not digit for digit.

## Files

| File | |
|---|---|
| `attention_final.py` | reference implementation + `--show` / `--rehearse` |
| `attention_live.py` | the file written live in class; `./present.sh reset` empties it |
| `tests.py` | the five tests; `python tests.py <module>` runs them against any module exposing the three functions |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/tests.json` | the rehearsal record quoted above |
