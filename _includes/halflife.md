<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

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
