<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 2 — Three ways to balance a mixture of experts, on last week's tiny GPT (slides p16–p17)

W4 demo 1's tiny GPT (character-level, six layers, $d = 256$, windows of $L = 256$ characters) has a dense feed-forward block in every layer. This demo replaces each of those blocks with a **mixture-of-experts layer**: a router $\mathbf{W}^R \in \mathbb{R}^{8 \times 256}$, eight experts (each is W4's `SwiGLU`, half as wide: $d_E = 344$), and every token is sent to the **two** experts with the largest router logits, weighted by the softmax of the logits. Nothing else changes: same corpus, same held-out text, same hyperparameters, same 3000 steps, same random initialisation. The model is trained three times, once per balancing rule:

| | Rule | Source |
|---|---|---|
| (i) `none` | nothing | — |
| (ii) `switch` | add $\mathcal{L}_{\text{aux}} = \alpha\,n_E\sum_i f_i P_i$ to the loss, one term per layer, $\alpha = 10^{-2}$ | Fedus et al. 2022 §2.2 |
| (iii) `bias` | select with $\mathrm{TopK}_2(\mathbf{r}_t + \mathbf{u})$ but gate with $\mathbf{r}_t$; after every step $u_i \leftarrow u_i + \gamma\,\mathrm{sign}(\bar{c} - c_i)$, $\gamma = 0.001$; nothing enters the loss | Wang et al. 2024 §3; DeepSeek-V3 §2.1.2 |

Nothing is run live in class; the rehearsal records in `runs/rehearsal/` are replayed.

## What is being measured

Every 25 steps the model is run (in eval mode) on the **same** 32 held-out windows — 8,192 tokens, hence 16,384 expert slots — and three things are recorded per layer: the held-out loss, the share of slots each expert received, $f_i = \mathrm{count}_i / (T\,k)$, and the mean router probability $P_i$. Two summary numbers come out of the shares: the largest share $\max_i f_i$ (perfect balance is $1/n_E = 0.125$; because every token must pick $k = 2$ *different* experts, one expert can hold at most $1/k = 0.5$ of the slots, which is what "collapse" looks like here) and the number of experts holding less than 1% of the slots ("starved"). The slide's table reports the mean over the six layers, with the worst layer in parentheses. Every 500 steps the full held-out measure of W4 (windows of $L$, stride 128) is also run, so the three runs can be compared with W4's dense model (2.184 bits/char).

A fact worth knowing before reading the numbers: with $d_E = d_{\text{ff}}/2$ and $k = 2$, the expert parameters a token actually uses, $k \cdot 3\,d\,d_E = 528{,}384$, equal W4's dense FFN exactly. The whole model has 14.3 M parameters against W4's 4.76 M — the same compute per token, three times the parameters. That is what a mixture of experts is.

## Requirements

| | |
|---|---|
| Packages | `torch` only (`2.14.0` pinned). No model download, no network. |
| Other folders | imports `../w04_d1_block_scratch/block_final.py` (the `GPT` and `SwiGLU`) and `train.py` (corpus, device, optimiser groups, the held-out measure), and reads that folder's `demo_config.toml`; `block_final.py` in turn imports W3 demo 1's `attention_final.py`. Keep the folder layout. |
| Corpus | `slides/_corpus/tinyshakespeare.txt`, not in the repository — see W4 demo 1's README for where to download it (about 1.1 MB). |
| Device | training uses `mps` on Apple silicon and falls back to the CPU (slow); `cuda` is accepted by the code but has never been run. `replay`, `show` and `summarize` need no device at all. |
| Memory / time | 14,286,336 parameters, 4,775,424 active per token. On the machine below the three runs took 60, 36 and 43 minutes (the collapsed run is slowest: its two surviving experts each process every token). `replay` is instant. |
| Portability | nothing platform-specific; tested only on an Apple M5 Max (64 GB), macOS 27.0.1, Python 3.12, torch 2.14.0 |

## Run it

```bash
cd demos                      # see ../README.md for creating .venv
cd w06_d2_moe_balance
./present.sh replay           # the rehearsal records, one setting per Enter, then the slide's table
./present.sh show             # the same without pauses
./present.sh estimate         # a lower bound on how long rehearse takes, from W4 demo 1's timing probe
./present.sh rehearse --steps 30   # smoke test: 30 steps per setting + all records, written to runs/smoke/
./present.sh rehearse         # the real thing: three runs of 3000 steps → runs/rehearsal/
./present.sh train bias       # one setting only (none | switch | bias), then ./present.sh summarize
```

Without `present.sh`: `python demo.py replay`, `python demo.py rehearse`, and so on, with `torch` importable.

## What you should see

From `runs/rehearsal/summary.json` (2026-10-18, Apple M5 Max, macOS 27.0.1, Python 3.12, torch 2.14.0, `mps`). Final values after 3000 steps, mean over the six layers with the worst layer in parentheses:

| | (i) none | (ii) Switch $\mathcal{L}_{\text{aux}}$ | (iii) loss-free bias |
|---|---|---|---|
| largest expert share (ideal $1/8 = 0.125$; at most $1/k = 0.5$) | 0.397 (0.500) | 0.151 (0.159) | 0.147 (0.163) |
| experts below 1% of slots, per layer | 2.5 (6 of 8) | 0 | 0 |
| held-out loss, fixed subset (nats/char) | 1.392 | 1.447 | 1.421 |
| held-out, W4's full measure (bits/char) | 2.280 | 2.367 | 2.306 |

What happens along the way is more telling than the end points. **Without a balancing term, every layer collapses within 25 steps** — inside the learning-rate warmup, with the learning rate still below $3 \times 10^{-4}$: all tokens go to the same two experts (share 0.499 each), the other six receive nothing. The router keeps learning and some layers pick other experts back up; at the end two layers are still fully collapsed, two sit at 0.46–0.48, two at about 0.2, and 15 of the 48 experts are starved. **Switch's auxiliary loss** balances the layers within 100 steps (0.164) and holds them there; the price is the held-out loss, 0.055 nats/char above run (i). **The auxiliary-loss-free bias collapses exactly like run (i) first** — a step of $\gamma = 0.001$ per update cannot do anything in 25 steps — and then pulls the layers back: 0.285 at step 500, 0.193 at 1500, level by the end, no expert starved, at a cost of 0.029 nats/char. Its bias vector ends between $-0.84$ and $+1.15$. The `replay` command prints the two curves as sparklines and each layer's final shares as a row of bars.

One thing the table on the slide does not show: all three MoE models are *worse* on W4's full measure than W4's dense model (2.184 bits/char), and their held-out loss reaches its minimum around step 2000 and rises afterwards. Same compute per token, three times the parameters, the same one megabyte of text: the extra parameters overfit. That is a statement about this corpus size, not about mixtures of experts, and it is where next week's scaling laws begin.

A caveat the program prints every time: this is a character-level model of fourteen million parameters with eight experts, one seed, and $\alpha$ and $\gamma$ taken from the papers without tuning. The *shape* of each curve is the shape of a mechanism. None of the numbers is a threshold for any model with 256 experts.

## Files

| File | |
|---|---|
| `moe_gpt.py` | `MoEFFN` (router, eight narrow `SwiGLU` experts, the three balancing rules), `Demo1` (loads W4 demo 1), `load_stats`, `count_params_split` |
| `demo.py` | `rehearse` / `train` / `summarize` / `replay` / `show` / `estimate` |
| `demo_config.toml` | the MoE shape, $\alpha$, $\gamma$, recording frequency; model size and training hyperparameters live in W4 demo 1's config |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/` | one JSON per setting (every 25 steps: loss, per-layer per-expert shares and probabilities, the bias vector), `summary.json` (the slide's table), `curves.svg` |
