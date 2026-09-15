<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## One Paper per Week

| Week | Topic | If you read only one |
|:--|:---------------------|:---------------------------------------|
| [W1](weeks/w01.qmd) | From n-grams to seq2seq | Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015 |
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
