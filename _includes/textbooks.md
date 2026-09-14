<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

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
