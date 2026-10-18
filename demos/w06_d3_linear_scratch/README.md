<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 3 — Linear attention written from scratch: two algorithms, a recurrence, and a notebook that fills up (slides p23–p24)

In class the instructor writes these functions live in `linear_live.py`, running the tests after each one:

| Function | Signature | What it does |
|---|---|---|
| `phi(x)` | → same shape | the feature map $\phi(x) = \mathrm{elu}(x) + 1$, element-wise and positive (Katharopoulos et al. 2020) |
| `attn_quadratic(Q, K, V)` | `Q, K, V: (L, d_head)` → `(L, d_head)` | $(\phi(\mathbf{Q})\phi(\mathbf{K})^\top)\mathbf{V}$ — the $L \times L$ score matrix first, then $\mathbf{V}$; $O(L^2 d_{\text{head}})$ |
| `attn_linear(Q, K, V)` | same | $\phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{V})$ — the $d_{\text{head}} \times d_{\text{head}}$ state first; $O(L\,d_{\text{head}}^2)$. Mathematically equal to the line above, computationally not |
| `step(S, z, k, v, q)` | `S: (d, d)`, `z, k, v, q: (d,)` → `(S, z, o)` | one token of the causal recurrence: $\mathbf{S} \mathrel{+}= \phi(\mathbf{k})\mathbf{v}^\top$, $\mathbf{z} \mathrel{+}= \phi(\mathbf{k})$, $\mathbf{o} = \mathbf{S}^\top\phi(\mathbf{q}) \,/\, \mathbf{z}^\top\phi(\mathbf{q})$ |
| `store(keys, vals)` | `(m, d), (m, d)` → `(d, d)` | the additive notebook $\mathbf{S} = \sum_s \mathbf{k}_s\mathbf{v}_s^\top$ (the keys are stored as given — no $\phi$) |
| `recall(S, k)` | → `(d,)` | read a value back with its key: $\mathbf{S}^\top\mathbf{k}$ |
| `delta_write(S, k, v, beta=1.0)` | → `(d, d)` | one delta-rule write: $\mathbf{S} + \beta\,\mathbf{k}(\mathbf{v} - \mathbf{S}^\top\mathbf{k})^\top$ — erase what the key already points to, then write |

`linear_final.py` is the finished reference implementation; `tests.py` holds the four tests and is shared by both files, so you can write your own version in `linear_live.py` and test it step by step — functions you have not written yet are reported as `skip`, not as failures.

## The four tests

All four run in float64 on the CPU with $d_{\text{head}} = 128$; the random tensors come from `torch.Generator`s with fixed seeds.

| | Checks | Passes when | Why it matters |
|---|---|---|---|
| (a) | the two orderings on the same random $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ with $L = 256$: the largest difference, divided by the largest output | relative difference $< 10^{-12}$ (expect $\sim 10^{-15}$) | associativity, on a machine: the two brackets agree to the last digit of a double. There is no softmax between them — that is the whole point; with softmax the bracket cannot move |
| (b) | timing of both functions at $L \in \{32, 128, 512, 2048, 8192\}$ (median of repeated calls, one warm-up); prints where the two lines cross and the speed-up at 8192 | both run (timing has no threshold) | the two costs $O(L^2 d_{\text{head}})$ and $O(L\,d_{\text{head}}^2)$ cross at $L = d_{\text{head}}$ in theory. Where they cross on your machine depends on the BLAS and the thread count — read the order of magnitude and the fact that they cross, not the digits. This is not a benchmark |
| (c) | `step` run token by token for $L = 64$ steps (with the normaliser $\mathbf{z}$), against the same output computed in one shot with `cumsum` | relative difference $< 10^{-10}$ | the recurrence and the closed form agree — the same check W3 ran on online softmax |
| (d) | write $m$ key–value pairs into a $128 \times 128$ state, read each back with its key, average the relative error $\|\text{recalled} - \mathbf{v}\|/\|\mathbf{v}\|$ over the pairs and over 5 trials, for $m$ from $d_{\text{head}}/4$ to $4\,d_{\text{head}}$; three curves — **orthogonal keys**, **random unit keys**, **random keys with the delta rule** ($\beta = 1$, each pair written once) — and one extra line: the same key written twice | orthogonal keys: floating-point zero up to $m = d_{\text{head}}$ and $> 0.5$ at $m = 2\,d_{\text{head}}$; random keys: $> 0.1$ already at $m = d_{\text{head}}/4$ | the capacity of a fixed state (slide p25). With orthogonal keys the notebook holds exactly $d_{\text{head}}$ pairs without error and the $(d_{\text{head}} + 1)$-th breaks it — a cliff. Random keys are never orthogonal, so interference is there from the first pair and grows like $\sqrt{(m-1)/d_{\text{head}}}$ — no knee. The delta rule lowers every point but keeps the bound. Writing the same key twice: additive recall gives $\mathbf{v}_1 + \mathbf{v}_2$, delta recall gives $\mathbf{v}_2$ — overwritten, not stacked |

Why (a) and (b) have no denominator: the normaliser is the same bracket applied a second time ($\phi(\mathbf{Q})\phi(\mathbf{K})^\top\mathbf{1}$ versus $\phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{1})$); it would double the code without adding anything to the point. It appears in `step`, and (c) checks the normalised output.
Why `store` applies no $\phi$: the keys in (d) *are* the vectors being stored (the slide's $\phi(\mathbf{k})$), already unit or orthogonal; passing them through $\mathrm{elu} + 1$ would make orthogonal keys non-orthogonal and the cliff would disappear.

## Requirements

| | |
|---|---|
| Packages | `torch` only (`2.14.0` is pinned in `demos/versions.lock`; nothing else is imported). No model, no download, no network. |
| Device | **CPU, deliberately, in float64.** `linear_final.py` sets `torch.set_default_device("cpu")` and `torch.set_default_dtype(torch.float64)`. Apple's Metal backend has no double type, and on the GPU the small-$L$ timings in (b) would be dominated by launch overhead. |
| Memory / time | (b) at $L = 8192$ allocates an $8192 \times 8192$ float64 matrix (512 MB) for the quadratic form; the whole run takes a few seconds |
| Portability | should run on any machine where `torch` installs (nothing platform-specific is used); tested on an Apple M5 Max, macOS 27.0.1, Python 3.12.14, torch 2.14.0 (the rehearsal record below), and once on an x86-64 Linux VM with torch 2.14.1 during the build — (a), (c) and (d) agreed to the digit, (b) differed only in the speed-up |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv
cd w06_d3_linear_scratch
./present.sh                  # four tests against the reference implementation: a ok, b ok, c ok, d ok
./present.sh show             # an L = 3, d_head = 2 example printed step by step, then the four tests
./present.sh reset            # empties linear_live.py down to its header comment and the two imports
./present.sh live             # tests against *your* linear_live.py; unwritten functions show as skip
./present.sh replay           # prints the instructor's rehearsal record (runs/rehearsal/tests.json), not a live run
```

Without `present.sh` (any shell, your own interpreter): `python linear_final.py`, `python linear_final.py --show`, `python tests.py linear_live`.

Suggested order if you write it yourself, matching the class: `phi` → `attn_quadratic`, `attn_linear` → look at (a) and (b) → `step` → (c) → `store`, `recall` → (d) with the delta column empty → `delta_write` → (d) complete.

## What you should see

From `runs/rehearsal/tests.json` (2026-10-19, Apple M5 Max, macOS 27.0.1, Python 3.12.14, torch 2.14.0, CPU, float64, 6 threads; all four tests in 1.06 s):

- (a) relative difference $1.6 \times 10^{-15}$ (absolute $1.8 \times 10^{-11}$ on outputs of order $10^4$ — there is no denominator in (a)).
- (b) quadratic: 44 µs, 129 µs, 0.61 ms, 5.6 ms, 86 ms; linear: 46 µs, 126 µs, 0.38 ms, 0.70 ms, 3.0 ms for $L = 32, 128, 512, 2048, 8192$. The lines cross between $L = 32$ and $L = 128$ and are level at $L = 128$; linear is $28\times$ faster at $L = 8192$. On a 2-thread x86-64 VM the crossover was in the same interval and the speed-up at 8192 was $16$–$21\times$ — expect the ratio to move with the machine.
- (c) relative difference $1.2 \times 10^{-15}$.
- (d) orthogonal keys: $\sim 10^{-16}$ for $m \le 128$, then 0.62, 0.88, 1.20, 1.63, 1.94 at $m = 160, 192, 256, 384, 512$. Random keys: 0.50, 0.71, 0.86, 1.00, 1.12, 1.22, 1.42, 1.73, 2.01 for the nine values of $m$ — the same as $\sqrt{(m-1)/128}$ to two decimals. Random keys with the delta rule: 0.35, 0.51, 0.64, 0.74, 0.82, 0.89, 0.99, 1.12, 1.20. Same key twice: additive $= \mathbf{v}_1 + \mathbf{v}_2$, delta $= \mathbf{v}_2$, both to $10^{-16}$.

Same seeds, same torch version: you should see these digits for (a), (c) and (d); for (b) expect the same magnitudes only.

## Files

| File | |
|---|---|
| `linear_final.py` | reference implementation + `--show` / `--rehearse` |
| `linear_live.py` | the file written live in class; `./present.sh reset` empties it |
| `tests.py` | the four tests; `python tests.py <module>` runs them against any module exposing the functions above |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/tests.json` | the rehearsal record quoted above |
