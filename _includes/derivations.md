<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## Derivations and Hand-Built Components

One board derivation and one component per week. The compute estimates assume free-tier Colab (T4, 16 GB).

| Week | At the board | Built by hand | Compute |
|:--|:--------------------------------|:----------------------------|:------|
| [W1](weeks/w01.qmd) | Cross-entropy as compression; the BPTT product | — (demos only) | CPU |
| [W2](weeks/w02.qmd) | Vocabulary size inside the scaling law | **BPE training, encoding, decoding** | CPU, 10 min |
| [W3](weeks/w03.qmd) | The variance of $q^\top k$ → $1/\sqrt{d_k}$; the KV-cache formula | **Scaled dot-product attention + KV cache** | CPU, 5 min |
| [W4](weeks/w04.qmd) | Parameter accounting → $N \approx 12 n_{\text{layer}}d^2$; $C \approx 6ND$ | **A full GPT block, and a tiny model trained** | T4, 20 min |
| [W5](weeks/w05.qmd) | **RoPE's relative property** in complex form; the phase argument for extrapolation | RoPE and its extrapolation variants | T4, 15 min |
| [W6](weeks/w06.qmd) | **The linear-attention associativity identity** → recurrent form; the SSD skeleton | Both algorithms for linear attention | T4, 10 min |
| [W7](weeks/w07.qmd) | **Chinchilla's first-order condition** (one of the course's three peaks) | A scaling law fitted at small scale | T4, 30 min |
| [W8](weeks/w08.qmd) | LoRA's parameter and memory accounting; the SVD behind intruder dimensions | **LoRA SFT on Qwen3-0.6B/1.7B** | T4, 25 min |
| [W9](weeks/w09.qmd) | **DPO in four steps** (one of the three peaks); GRPO's advantage normalization | **The DPO loss, with four tests** | CPU, 10 min |
| [W10](weeks/w10.qmd) | The two crossing pass@k curves; the binomial bound behind self-consistency | Budget forcing | T4, 30 min |
| [W11](weeks/w11.qmd) | **Roofline and the batch threshold for decode**; the three-line losslessness proof for speculative decoding | A roofline measured | T4, 20 min |
| [W12](weeks/w12.qmd) | InfoNCE, and how the negatives define "similar"; RAG's three-term decomposition | **A minimal RAG, and three deliberate failures** | CPU, 40 min |
| [W13](weeks/w13.qmd) | The compound error rate $p^n$, and what a verifier does to it | **An agent broken by prompt injection** | CPU/T4, 30 min |
| [W14](weeks/w14.qmd) | The binomial standard error and paired bootstrap; the sparsity–reconstruction Pareto | An evaluation workshop (no new model) | CPU |
