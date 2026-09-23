<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## Derivations and Hand-Built Components

One board derivation and one component per week. Everything is demonstrated in class. The last column says **what each component needs to run on**, not how long it takes: the demo machine has no CUDA, and nothing here has been timed on it yet. Timings will be added once they are measured rather than estimated.

| Week | At the board | Built by hand | Runs on |
|:--|:--------------------------------|:----------------------------|:------|
| [W1](weeks/w01.qmd) | Cross-entropy as compression; the power-law decay of unseen mass under Zipf–Heaps; the BPTT product and its singular-value bound | — (demos only) | CPU |
| [W2](weeks/w02.qmd) | Vocabulary size inside the scaling law | **BPE training, encoding, decoding** | CPU — tokenizers are hardware-independent |
| W3 | The variance of $q^\top k$ → $1/\sqrt{d_k}$; the KV-cache formula | **Scaled dot-product attention + KV cache** | **CPU** — the gradient check needs `float64`, which Metal does not have |
| W4 | Parameter accounting → $N \approx 12 n_{\text{layer}}d^2$; $C \approx 6ND$ | **A full GPT block, and a tiny model trained** | MPS or CPU. **SDPA on MPS refuses attention dropout** |
| W5 | **RoPE's relative property** in complex form; the phase argument for extrapolation | RoPE and its extrapolation variants | MPS; the long-context demo uses a quantized mid-size model |
| W6 | **The linear-attention associativity identity** → recurrent form; the SSD skeleton | Both algorithms for linear attention | MPS. SSM and hybrid models run on MLX's own Metal kernels |
| W7 | **Chinchilla's first-order condition** (one of the course's three peaks) | A scaling law fitted at small scale | MPS — four or five tiny transformers |
| W8 | LoRA's parameter and memory accounting; the SVD behind intruder dimensions | **LoRA SFT with `mlx_lm.lora`** | MLX. Pointing it at a quantized model *is* QLoRA |
| W9 | **DPO in four steps** (one of the three peaks); GRPO's advantage normalization | **The DPO loss, with four tests** | CPU |
| W10 | The two crossing pass@k curves; the binomial bound behind self-consistency | Budget forcing | MLX, a mid-size model with a thinking mode |
| W11 | **Roofline and the batch threshold for decode**; the three-line losslessness proof for speculative decoding | A roofline measured | MLX (`mlx_lm.benchmark`, `--draft-model`) |
| W12 | InfoNCE, and how the negatives define "similar"; RAG's three-term decomposition | **A minimal RAG, and three deliberate failures** | CPU |
| W13 | The compound error rate $p^n$, and what a verifier does to it | **An agent broken by prompt injection** | A small local model is enough |
| W14 | The binomial standard error and paired bootstrap; the sparsity–reconstruction Pareto | An evaluation workshop (no new model) | CPU — TransformerLens avoids MPS by default |
