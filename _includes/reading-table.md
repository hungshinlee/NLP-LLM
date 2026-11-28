<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## One Paper per Week

| Week | Topic | If you read only one |
|:--|:---------------------|:---------------------------------------|
| [W1](weeks/w01.qmd) | From n-grams to seq2seq | Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015 |
| [W2](weeks/w02.qmd) | Tokenization | Tao et al., *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*, NeurIPS 2024 (pair with Limisiewicz et al., *Compute Optimal Tokenization*, arXiv:2605.01188) |
| [W3](weeks/w03.qmd) | Attention | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — with Jain & Wallace, *Attention is not Explanation* |
| [W4](weeks/w04.qmd) | The full block | Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, arXiv:2309.14322 |
| [W5](weeks/w05.qmd) | Position and long context | Liu et al., *Lost in the Middle*, TACL 12 (2024) |
| [W6](weeks/w06.qmd) | Architecture routes | Du et al., *Kimi Linear*, arXiv:2510.26692 — with Dao & Gu, *Mamba-2* |
| [W7](weeks/w07.qmd) | Pre-training and scaling | Hoffmann et al., *Chinchilla*, NeurIPS 2022 (arXiv:2203.15556), §3 — the three ways of estimating the optimum — with Besiroglu et al.'s replication, in full |
| [W8](weeks/w08.qmd) | SFT and PEFT | Lee et al., *Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning*, arXiv:2602.04998 (an ICML 2026 workshop paper), in full, read as a methodological review, with Def. 3.1, Alg. 1 and the section on reducing intruder dimensions in Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, NeurIPS 2025 |
| [W9](weeks/w09.qmd) | Alignment and RL | Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, ICML 2026 (arXiv:2506.10947), in full, with §4.1 and §4.3 of Wu et al., *Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination*, AAAI 2026 — two explanations of the same result |
| [W10](weeks/w10.qmd) | Reasoning and test-time compute | Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, NeurIPS 2025 (oral; arXiv:2504.13837), in full, with §4 of Wen et al., *Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs*, ICLR 2026 — two readings of the same crossing pass@k curves |
| [W11](weeks/w11.qmd) | Inference efficiency | Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 (oral; arXiv:2211.17192), in full, with §3–§4 of Chen et al., *The Pitfalls of KV Cache Compression*, ACL 2026 — one paper proves a speed-up that leaves the distribution unchanged, the other shows a speed-up after which the score holds and the behaviour does not |
| W12 | RAG | Weller et al., *On the Theoretical Limitations of Embedding-Based Retrieval*, ICLR 2026 (arXiv:2508.21038), in full, with §4–§5 of Wallat et al., *Correctness is not Faithfulness in RAG Attributions*, ICTIR 2025 — one paper computes a limit on *finding*, the other shows an illusion in *using*: a correct citation is not evidence that the source was used |
| W13 | Agentic systems | Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, arXiv:2503.13657 |
| W14 | Evaluation and interpretability | Korznikov et al., *Sanity Checks for Sparse Autoencoders*, arXiv:2602.14111 — with Miller, *Adding Error Bars to Evals* |

> **Why these.** The table leans deliberately towards critical work and negative results rather than the paper that first proposed each method. The originals are largely in the textbook — JM3 Vol I covers the spine of W1–W9 — while "here is where this method breaks" has no textbook and is worth far more to a graduate student. If you read one paper for a week, read the critical one.
