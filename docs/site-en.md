<!--
網站英文片段的來源檔（手寫）。scripts/build_weeks.py 依 `<!-- file: X -->` 切段，
寫成 _includes/X，供 syllabus.qmd 與 resources.qmd 引用。

中文原文仍留在 docs/course-outline.md（離線閱讀用的完整大綱，不上網站）；兩邊各自維護。
改了大綱的對應區塊（使用說明、算力帳本、主要教科書、附錄 A、附錄 B、附錄 C、附錄 D）
記得回來同步這裡。首頁的英文課程地圖另存 docs/course-map-en.md（ASCII 對齊敏感，單獨一檔）。
-->

<!-- file: ledger.md -->
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

Two numbers to work out for yourself in the first session, and to revisit as the course goes:

1. The training FLOPs for a 7B dense model at Chinchilla-optimal ($D \approx 20N$), and how many years that is on a single consumer GPU — an RTX 5070 Ti, say, which is the compute a term project actually has.
2. The KV cache for that model at batch 1, 32k context, FP16 — in GB, against two real ceilings: the 16 GB on that same card, which cannot hold it, and the 64 GB of unified memory the classroom demos run on, which can, at a fraction of a datacenter GPU's bandwidth. **Capacity and bandwidth are two separate limits, and 64 GB relieves only the first — which is the first hint of the roofline in W11.** That number is the shared motivation for W5, W6 and W11.

A third number, working backwards from the MoE activation ratio to why a 2026 flagship is rational at roughly a trillion total parameters and tens of billions active, opens W6 instead — it needs the idea of an active parameter, which arrives that week.

<!-- file: reading-questions.md -->
## Four Questions for Reading a Paper

Handed out in week 1 and used every week after. They are short on purpose — they have to be usable in the ten minutes before a seminar.

1. **Was the baseline actually tuned?** The most common source of illusory progress in this field. W8 (LoRA variants) and W9 (GRPO variants) both have concrete cases where the improvement disappears once the baseline gets a learning-rate sweep.
2. **Is the comparison at equal compute?** Iso-FLOPs, iso-parameter and iso-latency are three different comparisons, and they frequently give opposite answers. W6 is built around this.
3. **What is the metric rewarding?** The gap between a metric and the capability it stands for is the subject of W14, but it comes up every week — most sharply in W11, where KV-cache compression leaves perplexity almost untouched and quietly breaks instruction following.
4. **Is the claim mechanistic or correlational?** "The attention weight is high", "the chain of thought wrote this step", "the reward went up" are all correlational. W3, W10 and W14 are the same error three times over, and the answer is the same each time: ask for an intervention, and ask what the strong baseline does.

<!-- file: textbooks.md -->
## Core Textbooks

| Code | Source | Used for |
|:--|:------------------------------|:------------------|
| **JM3** | Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. **online draft, 2026-08-19 release** (<https://web.stanford.edu/~jurafsky/slp3/>) | The teaching baseline for the whole course. **This release is reorganized into three volumes and 26 chapters**, with Volume I devoted to LLMs |
| **RAS** | Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024 (a Traditional Chinese edition exists) | **The primary text for building components by hand.** The labs in W3, W4 and W8 take their skeletons from it |
| **RAS-R** | Raschka, *Build a Reasoning Model (From Scratch)*, Manning | Hand-built material for W9 and W10 — the DPO loss, verifiers, budget forcing |
| **XZ** | Xiao & Zhu, *Foundations of Large Language Models*, arXiv:2501.09223 (2025-01, last revised 2025-06) | **An arXiv monograph, not a published book.** Treat it as free lecture notes rather than as a textbook |
| **LLMBook** | 趙鑫、李軍毅、周昆、唐天一、文繼榮, *大語言模型*, 高等教育出版社, 2024 (13 chapters) | The Chinese companion. Its site carries lecture PDFs, code and the original slides on request |
| **BB** | Bishop & Bishop, *Deep Learning: Foundations and Concepts*, Springer, 2024 | Prerequisite mathematics and deep-learning background; free online edition |
| **UDL** | Prince, *Understanding Deep Learning*, MIT Press; GitHub **v5.0.3, 2026-02-09**, ch 1–21, CC BY-NC-ND | **The best catch-up text for a class with no prerequisites**: free PDF, notebooks, slides and an answer booklet, and still maintained |
| **HOLLM** | Alammar & Grootendorst, *Hands-On Large Language Models*, O'Reilly, 2024 (12 chapters) | The lab manual for Colab. The visualizations in ch 2 (tokens and embeddings) and ch 8 (RAG) are unusually good |
| **ZZM** | Zong, Zhao & Ma, *Natural Language Processing and Large Language Models: Theory, Hands-on Codes, and Case Studies*, Springer Nature Singapore / Tsinghua University Press, **2026** (ISBN 978-981-92-0681-0; **open access**, 400 pages) | **The English companion for W1–W3.** Ch 5 (n-grams, smoothing, perplexity, FNN- and LSTM-based LMs), ch 3 (word2vec, ELMo) and ch 4 (seq2seq through Transformer) line up almost section by section with W1 and W3; ch 2 is the shortest catch-up chapter for anyone arriving without the prerequisites; ch 7 is the only treatment of Chinese word segmentation on this list. **Read it, do not run it** — see the note below |
| **LLMSurvey** | Zhao et al., *A Survey of Large Language Models*, arXiv:2303.18223 (**last revised 2026-03-18**, 144 pages, 1081 references, marked ongoing) | A reference, not a textbook |

### Three things to know about JM3

Checked directly against the Stanford page.

1. **It is still a draft with no print edition** (the site's own answer to "when will the book be finished?" is "don't ask"). **Download and keep the `2026-08-19` release**, because chapter numbers move between releases and this course cites that one.
2. **The structure of the 2026-08-19 release** — the largest improvement since the 2024 version, since pretraining and post-training finally have separate chapters:
   - **Vol I — Large Language Models**: 1 Introduction (rewritten for this release) / 2 **Words and Tokens** / 3 N-gram LM / 4 Logistic Regression & Text Classification / 5 Embeddings / 6 Neural Networks / 7 **Transformers and Pretraining** / 8 **Post-training**
   - **Vol II — Advanced LLM Topics and Tools**: 9 Masked LM / 10 **Interpretability** (incomplete) / 11 **Information Retrieval and RAG** / 12 **Agents "[not written yet]"** / 13 MT / 14 RNNs & LSTMs / 15–17 speech
   - **Vol III — Annotating Linguistic Structure**: 18–26 (POS/NER, parsing, IE, SRL, coreference, discourse, conversation)
3. **W13 has no textbook** — ch 12 *Agents* is unwritten, so that week runs entirely on papers and lecture notes. Ch 10 *Interpretability* is also incomplete; use it with care in W14.

### Two limits on ZZM

1. **The code is on a different stack.** Chapters 7–15 are written entirely in PaddlePaddle / PaddleNLP against Baidu's ERNIE models and AI Studio: the book mentions `paddle` 474 times and `paddlenlp` 123 times, and `torch` and `pytorch` exactly zero times. This course runs on Hugging Face `transformers` with Qwen3, so **ZZM is reading, not lab material**.
2. **Coverage stops around 2023.** Ch 6 compresses instruction tuning and RLHF into a section each, and the substance of W5–W14 — RoPE and long context, MoE and state-space models, scaling laws, DPO and GRPO, reasoning and test-time compute, inference efficiency, RAG, agents, evaluation and interpretability — is simply not in the book.

The authors are candid about the application chapters themselves: ch 1.2 says they demonstrate "the implementation process of the task through a specific example, rather than getting a practical system with the best performance." Read the application chapters in that spirit.

### Deliberately not assigned

Goldberg, *Neural Network Methods for NLP* (2017) and Paaß & Giesselbach, *Foundation Models for NLP* (2023) have both dated badly, and appear only as historical context in W1. Raaijmakers, *Large Language Models* (MIT Press, 2025) is conceptual and social rather than technical — not deep enough for a graduate course. Burkov, *The Hundred-Page Language Models Book* (2025) is too thin to carry the course, but works as pre-reading before week 1.

<!-- file: reading-table.md -->
## One Paper per Week

| Week | Topic | If you read only one |
|:--|:---------------------|:---------------------------------------|
| W1 | From n-grams to seq2seq | Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015 |
| W2 | Tokenization | Limisiewicz et al., *Compute Optimal Tokenization*, arXiv:2605.01188 |
| W3 | Attention | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — with Jain & Wallace, *Attention is not Explanation* |
| W4 | The full block | Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, arXiv:2309.14322 |
| W5 | Position and long context | Liu et al., *Lost in the Middle*, TACL 2023 |
| W6 | Architecture routes | Du et al., *Kimi Linear*, arXiv:2510.26692 — with Dao & Gu, *Mamba-2* |
| W7 | Pre-training and scaling | Hoffmann et al., *Chinchilla*, arXiv:2203.15556 — with Besiroglu et al.'s replication |
| W8 | SFT and PEFT | Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, arXiv:2410.21228 |
| W9 | Alignment and RL | Rafailov et al., *DPO*, NeurIPS 2023 — with the GRPO section of Shao et al., *DeepSeekMath* |
| W10 | Reasoning and test-time compute | Yue et al., *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?*, arXiv:2504.13837 |
| W11 | Inference efficiency | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 |
| W12 | RAG | Li et al., *RAG or Long-Context LLMs?*, EMNLP 2024 industry track |
| W13 | Agentic systems | Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, arXiv:2503.13657 |
| W14 | Evaluation and interpretability | Korznikov et al., *Sanity Checks for Sparse Autoencoders*, arXiv:2602.14111 — with Miller, *Adding Error Bars to Evals* |

> **Why these.** The table leans deliberately towards critical work and negative results rather than the paper that first proposed each method. The originals are largely in the textbook — JM3 Vol I covers the spine of W1–W9 — while "here is where this method breaks" has no textbook and is worth far more to a graduate student. If you read one paper for a week, read the critical one.

<!-- file: derivations.md -->
## Derivations and Hand-Built Components

One board derivation and one component per week. Everything is demonstrated in class; the compute column is there so you can see what each component would cost to rerun yourself, against a free-tier Colab T4.

| Week | At the board | Built by hand | Compute |
|:--|:--------------------------------|:----------------------------|:------|
| W1 | Cross-entropy as compression; the BPTT product | — (demos only) | CPU |
| W2 | Vocabulary size inside the scaling law | **BPE training, encoding, decoding** | CPU, 10 min |
| W3 | The variance of $q^\top k$ → $1/\sqrt{d_k}$; the KV-cache formula | **Scaled dot-product attention + KV cache** | CPU, 5 min |
| W4 | Parameter accounting → $N \approx 12 n_{\text{layer}}d^2$; $C \approx 6ND$ | **A full GPT block, and a tiny model trained** | T4, 20 min |
| W5 | **RoPE's relative property** in complex form; the phase argument for extrapolation | RoPE and its extrapolation variants | T4, 15 min |
| W6 | **The linear-attention associativity identity** → recurrent form; the SSD skeleton | Both algorithms for linear attention | T4, 10 min |
| W7 | **Chinchilla's first-order condition** (one of the course's three peaks) | A scaling law fitted at small scale | T4, 30 min |
| W8 | LoRA's parameter and memory accounting; the SVD behind intruder dimensions | **LoRA SFT on Qwen3-0.6B/1.7B** | T4, 25 min |
| W9 | **DPO in four steps** (one of the three peaks); GRPO's advantage normalization | **The DPO loss, with four tests** | CPU, 10 min |
| W10 | The two crossing pass@k curves; the binomial bound behind self-consistency | Budget forcing | T4, 30 min |
| W11 | **Roofline and the batch threshold for decode**; the three-line losslessness proof for speculative decoding | A roofline measured | T4, 20 min |
| W12 | InfoNCE, and how the negatives define "similar"; RAG's three-term decomposition | **A minimal RAG, and three deliberate failures** | CPU, 40 min |
| W13 | The compound error rate $p^n$, and what a verifier does to it | **An agent broken by prompt injection** | CPU/T4, 30 min |
| W14 | The binomial standard error and paired bootstrap; the sparsity–reconstruction Pareto | An evaluation workshop (no new model) | CPU |

<!-- file: toolchain.md -->
## Toolchain

**Students need no compute for this course.** It is lectures throughout, and every demo is run live in class on an Apple-silicon laptop with 64 GB of unified memory and **no CUDA** — which is what decides most of this table.

| Purpose | First choice | Notes |
|:------------|:----------------|:--------------|
| Inference for in-class demos | **MLX (`mlx-lm`)**, or llama.cpp / Ollama (GGUF) | Native to Apple silicon. **Unified memory is the real advantage here**: models far larger than 16 GB load comfortably |
| Building components by hand | Plain `torch` (MPS backend, or CPU) | The hand-built labs in W3, W4, W6 and W9 are small enough that MPS — or even CPU — finishes them in minutes |
| Tokenizer experiments | `tokenizers`, `sentencepiece`, `tiktoken` | Pure CPU, hardware-independent. The W2 fertility table |
| Fine-tuning demos | **LoRA in `mlx-lm`** | W8. The `peft` + `bitsandbytes` QLoRA path does not run here — see below |
| Evaluation | `lm-evaluation-harness` | Pointed at an MLX or llama.cpp backend. The W14 workshop looks at per-item results, not just aggregate scores |
| RAG | `sentence-transformers` + `faiss-cpu`, or plain numpy | W12. Brute-force search on CPU is enough; no vector database needed |
| Interpretability | `transformer-lens` (MPS) | Activation patching in W14; fine at small model sizes |

### What cannot be demonstrated on this hardware

| Not available | Why | What stands in for it |
|:-----------------|:--------|:---------------------|
| `bitsandbytes` 4/8-bit, QLoRA | CUDA only | MLX's own quantization and LoRA. The concept is identical; only the backend differs |
| `vLLM`'s PagedAttention and continuous batching | Not usable on Apple silicon | W11 teaches the argument and cites the SOSP 2023 numbers. PagedAttention's memory argument was always clearest on a whiteboard |
| **FP8 / NVFP4 measurements** | Apple silicon has no such tensor formats | W7 and W11 cite the figures from the Quartet and NVFP4-pretraining reports |
| FlashAttention, custom Triton kernels | CUDA only | The IO argument in W3 is a board derivation, not an implementation exercise |
| `auto-gptq`, `autoawq` | CUDA | MLX or GGUF quantization for the "what quantization changes besides perplexity" comparison |

### Models used in the demos

The demos are not limited to models that fit a 16 GB card, so the sizes are chosen for what each point needs rather than for what is cheap to run:

| Used for | Suggested size | Why it is worth scaling up |
|:-------------|:-------------------|:-------------------------------|
| Hand-built components (W3, W4, W6, W9) | Qwen3-0.6B / 1.7B | Small is the right choice here — the run finishes while the class is still watching |
| **Long-context behaviour (W5)** | A quantized mid-size model | The lost-in-the-middle curve is far more convincing on a real model than on 0.6B |
| **MoE routing (W6)** | A genuine open-weight MoE | No need to simulate with a tiny model plus four experts |
| **Reasoning (W10)** | A mid-size model with a thinking mode | Budget forcing and overthinking only show up on a real reasoning model |
| The 2026 flagships | Read the paper's table | DeepSeek-V4, Kimi K3 and the rest are still out of reach — that has not changed |
