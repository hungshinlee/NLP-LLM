<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

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
