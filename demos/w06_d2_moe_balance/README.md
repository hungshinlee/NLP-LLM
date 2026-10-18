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
| Memory / time | 14,286,336 parameters, 4,775,424 active per token. Training time on the machine below: to be measured (`rehearse --steps 30` prints the time for 30 steps of each setting; multiply by 100). A 30-step smoke test on a 2-core CPU container with batch 8 took about 50 s per setting. |
| Portability | nothing platform-specific; tested only on an Apple M5 Max (64 GB), macOS 27, Python 3.12, torch 2.14.0 |

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

The rehearsal has not been run yet; this section is filled in from `runs/rehearsal/summary.json` once it has. Until then, the only numbers are from the 30-step smoke test on a CPU container with batch 8: within those 30 steps — inside the learning-rate warmup — every layer of run (i) had already collapsed onto two experts ($\max f = 0.499$, six of eight experts below 1%), run (ii) was already pulling the largest share down (0.388, two starved per layer on average), and run (iii) looked like run (i) because the bias had only moved by $\pm 0.03$. What 3000 steps do to each of them is the question the demo asks.

A caveat the program prints every time: this is a character-level model of fourteen million parameters with eight experts, one seed, and $\alpha$ and $\gamma$ taken from the papers without tuning. The *shape* of each curve is the shape of a mechanism. None of the numbers is a threshold for any model with 256 experts, and if the three runs do not separate at this scale, the conclusion is "this scale cannot see the difference", not "the three methods are the same".

## Files

| File | |
|---|---|
| `moe_gpt.py` | `MoEFFN` (router, eight narrow `SwiGLU` experts, the three balancing rules), `Demo1` (loads W4 demo 1), `load_stats`, `count_params_split` |
| `demo.py` | `rehearse` / `train` / `summarize` / `replay` / `show` / `estimate` |
| `demo_config.toml` | the MoE shape, $\alpha$, $\gamma$, recording frequency; model size and training hyperparameters live in W4 demo 1's config |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/` | one JSON per setting (every 25 steps: loss, per-layer per-expert shares and probabilities, the bias vector), `summary.json` (the slide's table), `curves.svg` |
