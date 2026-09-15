<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## The Compute Ledger: The Course's Shared Coordinate System

You get this in week 1, and each week adds the layer it is responsible for. It is what lets an abstract argument fall back onto a number at any point.

::: {.ledger}

| Where the cost falls | Formula | What actually binds it |
|:---------------------|:-------:|:-----------------------|
| **Training** — one run | $C_{\text{train}} \approx 6ND$ FLOPs | **Compute.** The $6$ is $2$ forward $+\ 4$ backward, per parameter per token |
| **Prefill** — once per request | $C_{\text{pre}} \approx 2NL_{\text{in}}$ FLOPs | **Compute.** The whole prompt goes through together, so the matrices are large enough to keep the GPU busy |
| **Decode** — per generated token | $C_{\text{dec}} \approx 2N$ FLOPs | **Memory bandwidth.** One token's worth of arithmetic against a full read of all $N$ weights out of HBM |
| **KV cache** — state carried, not FLOPs | $M_{\text{KV}} = 2\,n_{\text{layer}}\,n_{\text{kv}}\,d_{\text{head}}\,L\,b$ bytes | **Capacity first, then bandwidth.** The leading $2$ is key $+$ value; the whole thing is re-read at every decode step |
| **MoE** — the two columns come apart | memory $\propto N_{\text{total}}$, FLOPs $\propto N_{\text{active}}$ | Both, but separately — which is the entire point of the architecture (W6) |

:::

$N$ parameters · $D$ training tokens · $L_{\text{in}}$ prompt length · $L$ context length so far · $b$ bytes per element ($2$ at FP16) · $n_{\text{kv}}$ key/value heads ($=n_{\text{head}}$ for MHA, $1$ for MQA).

**The line worth memorising.** The first two rows are arithmetic problems; the last two are traffic problems. Decode is not short of FLOPs — it is short of bandwidth, and the KV cache is what fills the road. Most of Part III is an attempt to make those two rows cheaper.

Three numbers to work out for yourself in the first session, and to revisit as the course goes:

1. The training FLOPs for a 7B dense model at Chinchilla-optimal ($D \approx 20N$), and how many years that is on a single T4.
2. The KV cache for that model at batch 1, 32k context, FP16 — in GB, against the T4's 16 GB. **That number is the shared motivation for W5, W6 and W11.**
3. Working backwards from the activation ratio: why a 2026 flagship is rational at roughly a trillion total parameters and tens of billions active, and where that design moves the cost *to*.
