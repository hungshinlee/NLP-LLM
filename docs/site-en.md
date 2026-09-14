<!--
網站英文片段的來源檔（手寫）。scripts/build_weeks.py 依 `<!-- file: X -->` 切段，
寫成 _includes/X，供 syllabus.qmd 與 resources.qmd 引用。

中文原文仍留在 docs/course-outline.md（離線閱讀用的完整大綱，不上網站）；兩邊各自維護。
改了大綱的對應區塊（使用說明、算力帳本、主要教科書、附錄 A、附錄 B、附錄 C、附錄 D）
記得回來同步這裡。首頁的英文課程地圖另存 docs/course-map-en.md（ASCII 對齊敏感，單獨一檔）。
-->

<!-- file: disclaimer.md -->
## How to Read This Site

1. **Mathematical density.** Each week's "key mathematics" block gives the equations, the symbols to define, and the skeleton of the derivation — the critical steps and the point where students get stuck — not the full algebra. That is a deliberate trade-off: it is meant to transfer straight onto slides, with the algebra left for the board.
2. **Citation-confidence tags.** Every reference in the outline was checked one at a time, and the tag records how strong that check was.
   - **Untagged** — title and first author verified against `arxiv.org/abs/<id>`. Safe to put on a slide.
   - **`[驗]`** — confirmed only through a secondary mirror; the date or the author order may be wrong. **Verify before putting it on a slide.**
   - **`[題]`** — no citation given on purpose, only search terms, rather than risk fabricating one.
   - **`[非論文]`** — a blog post, model card, or specification. **Not peer-reviewed.** This matters more in this field than in most: several widely repeated terms and results were propagated by blog posts, and two of the architectures the 2026 literature benchmarks against have no technical report at all.
3. **No numbers.** The outline reports no benchmark scores. The few exceptions — the SAE random-baseline figures in W14, the chain-of-thought faithfulness rates in W10 — are there because the number *is* the teaching point, and each names its source. For a benchmark table on a slide, cite the original paper's table rather than a remembered value.
4. **Currency.** Written against arXiv through August–September 2026. Half-life varies enormously across the 14 weeks; the map below says which weeks need rewriting and which are settled.
5. **Closed models change under a fixed name.** Any slide carrying a specific score should record the date it was taken and say that it is the order of magnitude at the time of publication.

<!-- file: ledger.md -->
## The Compute Ledger: The Course's Shared Coordinate System

Hand this out in week 1, then fill in each layer's contribution as the course goes. It is the device that lets an abstract argument fall back onto a number at any point.

```
Training                                   Inference (per token)
─────────────────────────                  ─────────────────────────
C_train ≈ 6 · N · D  FLOPs                 prefill:  ≈ 2 · N · L_in  FLOPs   (compute bound)
  N = parameters                           decode:   ≈ 2 · N        FLOPs   (memory bound)
  D = training tokens                      KV cache: 2 · n_layer · n_kv_head · d_head
  6 = 2 forward + 4 backward                         · L · bytes_per_elem
                                           → what bounds decode is moving N and the KV
MoE: N_total sets memory,                    from HBM into SRAM, not the arithmetic
     N_active sets FLOPs
```

Three numbers students should work out for themselves in the first session, and revisit as the course goes:

1. The training FLOPs for a 7B dense model at Chinchilla-optimal ($D \approx 20N$), and how many years that is on a single T4.
2. The KV cache for that model at batch 1, 32k context, FP16 — in GB, against the T4's 16 GB. **That number is the shared motivation for W5, W6 and W11.**
3. Working backwards from the activation ratio: why a 2026 flagship is rational at roughly a trillion total parameters and tens of billions active, and where that design moves the cost *to*.

<!-- file: reading-questions.md -->
## Four Questions for Reading a Paper

Handed out in week 1 and used every week after. They are short because they have to be usable in the ten minutes before a seminar.

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
| **RAS-R** | Raschka, *Build a Reasoning Model (From Scratch)*, Manning `[驗]` (confirm the publication month) | Hand-built material for W9 and W10 — the DPO loss, verifiers, budget forcing |
| **XZ** | Xiao & Zhu, *Foundations of Large Language Models*, arXiv:2501.09223 (2025-01, last revised 2025-06) | **An arXiv monograph, not a published book.** Fine to assign as free lecture notes; do not list it as a textbook |
| **LLMBook** | 趙鑫、李軍毅、周昆、唐天一、文繼榮, *大語言模型*, 高等教育出版社, 2024 (13 chapters) | The Chinese companion. Its site carries lecture PDFs, code and the original slides on request — **the least-effort option given lectures in Mandarin and slides in English** |
| **BB** | Bishop & Bishop, *Deep Learning: Foundations and Concepts*, Springer, 2024 | Prerequisite mathematics and deep-learning background; free online edition |
| **UDL** | Prince, *Understanding Deep Learning*, MIT Press `[驗]`; GitHub **v5.0.3, 2026-02-09**, ch 1–21, CC BY-NC-ND | **The best catch-up text for a class with no prerequisites**: free PDF, notebooks, slides and an answer booklet, and still maintained |
| **HOLLM** | Alammar & Grootendorst, *Hands-On Large Language Models*, O'Reilly, 2024 (12 chapters) | The lab manual for Colab. The visualizations in ch 2 (tokens and embeddings) and ch 8 (RAG) are unusually good |
| **ZZM** | Zong, Zhao & Ma, *Natural Language Processing and Large Language Models: Theory, Hands-on Codes, and Case Studies*, Springer Nature Singapore / Tsinghua University Press, **2026** (ISBN 978-981-92-0681-0; **open access**, 400 pages) | **The English companion for W1–W3.** Ch 5 (n-grams, smoothing, perplexity, FNN- and LSTM-based LMs), ch 3 (word2vec, ELMo) and ch 4 (seq2seq through Transformer) line up almost section by section with W1 and W3; ch 2 is the least-effort catch-up chapter for a class with no prerequisites; ch 7 is the only treatment of Chinese word segmentation on this list. **Read it, do not run it** — see the note below |
| **LLMSurvey** | Zhao et al., *A Survey of Large Language Models*, arXiv:2303.18223 (**last revised 2026-03-18**, 144 pages, 1081 references, marked ongoing) | A reference, not a textbook |

### Three things to know about JM3

Checked directly against the Stanford page.

1. **It is still a draft with no print edition** (the site's own answer to "when will the book be finished?" is "don't ask"). **Freeze a PDF and cite the `2026-08-19` release in the syllabus**, because chapter numbers move between releases.
2. **The structure of the 2026-08-19 release** — the largest improvement since the 2024 version, since pretraining and post-training finally have separate chapters:
   - **Vol I — Large Language Models**: 1 Introduction (rewritten for this release) / 2 **Words and Tokens** / 3 N-gram LM / 4 Logistic Regression & Text Classification / 5 Embeddings / 6 Neural Networks / 7 **Transformers and Pretraining** / 8 **Post-training**
   - **Vol II — Advanced LLM Topics and Tools**: 9 Masked LM / 10 **Interpretability** (incomplete) / 11 **Information Retrieval and RAG** / 12 **Agents "[not written yet]"** / 13 MT / 14 RNNs & LSTMs / 15–17 speech
   - **Vol III — Annotating Linguistic Structure**: 18–26 (POS/NER, parsing, IE, SRL, coreference, discourse, conversation)
3. **W13 has no textbook** — ch 12 *Agents* is unwritten, so that week runs entirely on papers and lecture notes. Ch 10 *Interpretability* is also incomplete; use it with care in W14.

### Two limits on ZZM

1. **The code is on a different stack.** Chapters 7–15 are written entirely in PaddlePaddle / PaddleNLP against Baidu's ERNIE models and AI Studio: the book mentions `paddle` 474 times and `paddlenlp` 123 times, and `torch` and `pytorch` exactly zero times. This course runs on Hugging Face `transformers` with Qwen3, so **ZZM is reading, not lab material**.
2. **Coverage stops around 2023.** Ch 6 compresses instruction tuning and RLHF into a section each, and the substance of W5–W14 — RoPE and long context, MoE and state-space models, scaling laws, DPO and GRPO, reasoning and test-time compute, inference efficiency, RAG, agents, evaluation and interpretability — is simply not in the book.

The authors are candid about the application chapters themselves: ch 1.2 says they demonstrate "the implementation process of the task through a specific example, rather than getting a practical system with the best performance." That is worth passing on to students.

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
| W14 | Evaluation and interpretability | Korznikov et al., *Sanity Checks for Sparse Autoencoders*, arXiv:2602.14111 `[驗]` — with Miller, *Adding Error Bars to Evals* |

> **Why these.** The table leans deliberately towards critical work and negative results rather than the paper that first proposed each method. The originals are largely in the textbook — JM3 Vol I covers the spine of W1–W9 — while "here is where this method breaks" has no textbook and is worth far more to a graduate student. If you read one paper for a week, read the critical one.

<!-- file: derivations.md -->
## Derivations and Hand-Built Components

One board derivation and one component per week. The compute estimates assume free-tier Colab (T4, 16 GB).

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

**Students need no compute for this course.** It is lectures throughout, and every demo is run live in class on the lecturer's machine — a MacBook Pro with an M5 Max and 64 GB of unified memory. That machine has **no CUDA**, which decides most of this table.

| Purpose | First choice | Notes |
|:------------|:----------------|:--------------|
| Inference for in-class demos | **MLX (`mlx-lm`)**, or llama.cpp / Ollama (GGUF) | Native to Apple silicon. **64 GB of unified memory is this machine's real advantage**: models far larger than 16 GB load comfortably |
| Building components by hand | Plain `torch` (MPS backend, or CPU) | The hand-built labs in W3, W4, W6 and W9 are small enough that MPS — or even CPU — finishes them in minutes |
| Tokenizer experiments | `tokenizers`, `sentencepiece`, `tiktoken` | Pure CPU, hardware-independent. The W2 fertility table |
| Fine-tuning demos | **LoRA in `mlx-lm`** | W8. The `peft` + `bitsandbytes` QLoRA path does not run here — see below |
| Evaluation | `lm-evaluation-harness` | Pointed at an MLX or llama.cpp backend. The W14 workshop looks at per-item results, not just aggregate scores |
| RAG | `sentence-transformers` + `faiss-cpu`, or plain numpy | W12. Brute-force search on CPU is enough; no vector database needed |
| Interpretability | `transformer-lens` (MPS) | Activation patching in W14; fine at small model sizes |

### What cannot be demonstrated on this machine — and how it is taught instead

| Not available | Why | What replaces it |
|:-----------------|:--------|:---------------------|
| `bitsandbytes` 4/8-bit, QLoRA | CUDA only | MLX's own quantization and LoRA. The concept is identical; only the backend differs |
| `vLLM`'s PagedAttention and continuous batching | Not usable on Apple silicon | W11 teaches the argument and cites the SOSP 2023 numbers. PagedAttention's memory argument was always clearest on a whiteboard |
| **FP8 / NVFP4 measurements** | Apple silicon has no such tensor formats | W7 and W11 cite the figures from the Quartet and NVFP4-pretraining reports |
| FlashAttention, custom Triton kernels | CUDA only | The IO argument in W3 is a board derivation, not an implementation exercise |
| `auto-gptq`, `autoawq` | CUDA | MLX or GGUF quantization for the "what quantization changes besides perplexity" comparison |

> **The constraint is itself teaching material.** Saying in class "I cannot run this demo on this machine, because FP8 is a Blackwell tensor format" makes W11's central point better than any slide: **inference optimization is bound to specific hardware.** Whether the same `transformers` code runs at all, and how fast, is exactly what that week is about.

### Models for the demos

The demos were originally sized for 16 GB with students running them too, so they used 0.6B toys. With students watching rather than running, that ceiling is gone:

| Used for | Suggested size | Why it is worth scaling up |
|:-------------|:-------------------|:-------------------------------|
| Hand-built components (W3, W4, W6, W9) | Qwen3-0.6B / 1.7B | Small is already the right choice — and it finishes while the class is still watching |
| **Long-context behaviour (W5)** | A quantized mid-size model | The lost-in-the-middle curve is far more convincing on a real model than on 0.6B |
| **MoE routing (W6)** | A genuine open-weight MoE | No need to simulate with a tiny model plus four experts |
| **Reasoning (W10)** | A mid-size model with a thinking mode | Budget forcing and overthinking only show up on a real reasoning model |
| The 2026 flagships | Read the paper's table | DeepSeek-V4, Kimi K3 and the rest are still out of reach — that has not changed |

<!-- file: halflife.md -->
## Half-Life: What Needs Rewriting Next Year

The most useful table here for anyone teaching this material. Content half-life varies enormously across the 14 weeks, and being explicit about it saves a great deal of repeated preparation.

| Week | Half-life | What to rewrite | Notes |
|:--|:------|:------------------|:--------------------------|
| **W1** | **Very long (5 yr+)** | Almost nothing | N-grams, word2vec, RNNs and attention are settled, mechanism and failure modes alike. The compression framing is a stable spine |
| **W2** | **Long (3 yr)** | The tokenizer list and the fertility numbers | BPE and Unigram do not change. **What moves is the compute-optimal-tokenization line**, and the tokenizer versions in the demo |
| **W3** | **Long (3–5 yr)** | Almost nothing | Attention's mechanics, $1/\sqrt{d_k}$, the KV-cache formula, FlashAttention's IO argument and attention-is-not-explanation are all stable. **Only the MHA→MQA→GQA→MLA line grows a new entry** |
| **W4** | **Long (3 yr)** | The optimizer section | Parameter accounting, pre-LN, SwiGLU and the instability diagnostics are stable. **Muon-class optimizers and muP's standing under MoE are live** |
| **W5** | **Medium (2 yr)** | Extrapolation methods and long-context benchmarks | The RoPE derivation and lost-in-the-middle are stable. **What comes after YaRN, and how long context gets evaluated, are not** |
| **W6** | **Short (1 yr)** | **The model list and the activation ratios, every year** | The associativity identity and SSD duality are permanent mathematics. **Hybrid layer ratios, the flagship list and the activation-ratio trend all move**, and the DeepSeek line gains a paper a year |
| **W7** | **Medium (2–3 yr)** | The correction axes and the data pipelines | Chinchilla's derivation is stable, and the three-axis correction framework looks stable as of 2026. **Data pipelines gain a public result every year** |
| **W8** | **Medium (2 yr)** | Where the LoRA-variant question has settled | LoRA's mathematics and intruder dimensions are stable. **If the 2602.04998 negative result is overturned this week changes substantially**; model merging is comparatively settled |
| **W9** | **Short (6–12 mo)** | **The GRPO family and RL scaling, annually** | The DPO derivation and Bradley-Terry are permanent. **The good news is that the 2026 variant wars converged into "one normalization, several choices", so this week's half-life is longer than it looked in 2025.** RL compute scaling is new and active |
| **W10** | **Short (6 mo)** | **Recheck where each dispute stands** | CoT faithfulness has settled and can be fixed in place. **Whether RL extends the capability frontier, and where latent reasoning fails, are both moving**; the monitorability line only became citable in 2026 |
| **W11** | **Medium (2 yr)** | Quantization and speculative decoding | Roofline, the KV-cache formula and the losslessness proof are permanent. **Low-precision *training* and the failure modes of KV compression are live** |
| **W12** | **Medium (2 yr)** | Embedding models and agentic retrieval | BM25, the three retrieval paradigms, the three-term decomposition and the consequences of lost-in-the-middle are stable. **Embedding models turn over yearly and RAG evaluation is still fragmenting** |
| **W13** | **Very short (6 mo)** | **Effectively the whole week, annually** | Three things are stable: the model–harness split, the MAST taxonomy, and the structural argument about prompt injection. **The benchmarks turned over four times in eighteen months** (GAIA → Gaia2 → τ-bench → τ² → τ^τ), and the MCP ecosystem and its security literature move quarterly. If JM3 ch 12 lands, replan the week |
| **W14** | **Short (6–12 mo)** | **Interpretability may invert again** | Stable: superposition, attention-is-not-explanation, the contamination argument, the statistics of error bars. **Sparse autoencoders went from breakthrough to failing a sanity check across 2025–2026 and could turn again**; the scheming literature produces new results each quarter |

### Three maintenance disciplines

1. **Rescan `cs.CL` and `cs.LG` for the previous three months before each offering**, and reread the latest programs from ACL, EMNLP, NeurIPS, ICLR, ICML and COLM. Start with W6, W9, W10, W13 and W14.
2. **Every `[驗]` tag has to be checked.** These were confirmed only through a secondary mirror, so a date or an author order may be wrong.
3. **Keep the `[非論文]` tags.** Several widely circulated terms and results in this field originate in blog posts, model cards, or specifications — the propagation source for on-policy distillation, LoRA Without Regret, the lethal trifecta, vLLM V1, and the designs of Qwen3-Next and Llama 4. **Marking the nature of a source on the slide is itself good training**: it shows students that open weights are not open methods, and that influence is not evidence.
