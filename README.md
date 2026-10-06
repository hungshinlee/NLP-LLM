# NLP-LLM — Natural Language Processing and Large Language Models

**自然語言處理與大型語言模型** · 中文版說明：[README.zh-TW.md](README.zh-TW.md)

A graduate course (Master's / PhD): 14 weeks × 3 hours, lectures throughout. Lectures are given in Mandarin; slides and the course site are in English.

**Course site: <https://hungshinlee.github.io/NLP-LLM/>**
Lecturer: [Hung-Shin Lee (李鴻欣)](https://web.ntnu.edu.tw/~hslee/)

This repository holds the *public* side of the course — the website source, the slides, the classroom demo code and the handouts. The teaching materials it is generated from (the full course outline, speaker notes, exam questions and rubrics) live in a private repository and are not here; see [What is deliberately not here](#what-is-deliberately-not-here).

## The course in one sentence

For every layer of a modern LLM — from the tokenizer to the agent loop — explain **why it looks the way it does, and where it breaks.** Every component is introduced from the concrete failure it was built to fix, not from its definition. The course is pitched at the research level: derivation skeletons, the paper trail behind each idea, and open problems, aligned with the state of the field in 2025–2026.

Three threads run through all fourteen weeks and are asked of every layer: **Representation** (what is the unit here — byte, token, hidden state, KV entry, retrieved chunk — and who chose it, at what downstream cost), **Compute** (does this spend training or inference compute, and is it FLOPs-bound or memory-bandwidth-bound), and **Supervision** (where does this capability come from — the pre-training distribution, preference labels, a verifiable reward, retrieved knowledge, or extra compute at test time).

## Course map

| Part I — Architecture | Part II — Training | Part III — Systems & Scrutiny |
|---|---|---|
| W1 From n-grams to seq2seq: a compressed history | W7 Pre-training: scaling laws and data curation | W11 Inference efficiency: from FLOPs to memory bandwidth |
| W2 Tokenization: the world the model sees | W8 Post-training I: SFT, PEFT, and forgetting | W12 RAG and context engineering |
| W3 Transformer I: attention mechanics, complexity, and building it by hand | W9 Post-training II: preference alignment and RL | W13 Agentic systems: tools, planning, failure modes, security |
| W4 Transformer II: the full block, parameter accounting, training dynamics | W10 Reasoning and test-time compute | W14 Evaluation, interpretability, safety, open problems |
| W5 Positional encoding, long context, and the limits of context | | |
| W6 Architecture routes: MoE, SSMs, and linear attention | | |

**W3 and W4 are the hinge of the course.** Everything in Part III reads from two objects built there — the residual stream and the KV cache — so those two weeks are the ones not to miss.

## What is published so far

Weeks are opened on the site as they are taught. As of October 2026, **W1–W4** are open: week pages, slides, and (from W3 on) the demo code. W1 and W2 demos are not published except for two short text files that the W4 demos read. The remaining weeks appear on the course map as greyed-out placeholders; their pages are generated but not rendered, so there are no dead links.

| | W1 | W2 | W3 | W4 | W5–W14 |
|---|---|---|---|---|---|
| Week page (positioning, learning objectives, references) | ✓ | ✓ | ✓ | ✓ | generated, not yet published |
| Slides (`slides/wNN.qmd`, reveal.js) | ✓ | ✓ | ✓ | ✓ | — |
| Demo code (`demos/wNN_dK_*/`) | — | — | ✓ | ✓ | — |
| Handout | [Mathematical prerequisites (PDF)](handouts/w01-math-prerequisites.pdf) | | | | |

The **Syllabus** page carries the compute ledger (the shared coordinate system of the course), the four questions to ask of any paper, and a **half-life map** that says which weeks are stable for years and which need to be re-scanned against arXiv `cs.CL` / `cs.LG` and the latest ACL / EMNLP / NeurIPS / ICLR / ICML / COLM programmes before each offering. The **Resources** page lists the core textbooks, one paper per week, the derivations and hand-built components, and the toolchain the demos run on. **Supplements** holds four guides (in Chinese) for the midterm proof-of-concept report, the final paper, paper writing, and ten research topics sized for a single 16 GB GPU.

## For students

- **You do not need to run anything.** The course is lectures only. Every demo is run live by the lecturer, and every number on the slides comes from a rehearsal record checked into `demos/*/runs/rehearsal/`. The code is published so you can read it, re-run it, and change it.
- **The slides have no speaker notes.** Pressing `S` in a deck opens an empty notes panel. That is intentional, not a bug: the Chinese lecture script is part of the teaching materials and is not published.
- **Demos were tested on one machine** — a MacBook Pro M5 Max (64 GB unified memory, no CUDA). `demos/README.md` says which demos are portable (CPU-only PyTorch) and which need Apple silicon (`mlx`), and pins every package and model revision in `demos/versions.lock`.
- **References were checked one by one.** Every citation on the site has had its title and first author verified against the arXiv abstract page, the ACL Anthology, PMLR, or Crossref. Entries that could not be verified are dropped from the site rather than shown with a caveat. Where a source is a blog post, model card, or specification rather than a peer-reviewed paper, the sentence says so — that matters in this field, where several widely used terms and baselines have no technical report behind them.
- The site has a **presentation mode** (press `z`, or use the navbar button) that collapses the side panels for projection.

## Repository layout

Most of this repository is *generated*. Files marked ⚙︎ are overwritten on the next publish and must not be edited by hand.

| Path | What it is |
|---|---|
| `index.qmd`, `syllabus.qmd`, `resources.qmd`, `slides.qmd`, `supplements.qmd` | Hand-written site pages (English) |
| `docs/site-en.md` | Source of the English prose blocks pulled into Syllabus and Resources (compute ledger, four questions, textbooks, one-paper-per-week, derivations, toolchain) |
| `weeks/w01.qmd` … `w14.qmd` | ⚙︎ Week pages, filtered from the private course outline by `scripts/build_weeks.py` |
| `_includes/*.md` | ⚙︎ Generated fragments: course map, weekly schedule, reading table, derivations table, ledger, textbooks, toolchain |
| `slides/wNN.qmd`, `slides/theme.scss`, `slides/assets/` | ⚙︎ Slides with speaker notes **stripped**, published from the private originals |
| `demos/` | ⚙︎ Classroom demo code, rehearsal records, and student-facing READMEs, published by whitelist; `demos/README.md` is the entry point |
| `handouts/*.pdf` | ⚙︎ Handouts copied from the private repository |
| `supplements/*.md` | Hand-written supplementary guides (Chinese); the `.qmd` wrappers beside them are ⚙︎ generated |
| `scripts/build_weeks.py`, `scripts/visibility.py` | The filter: which sections of the outline are public (deny by default, whitelist), how citation markers are handled, which weeks are published |
| `styles.scss`, `_present-mode.html`, `_quarto.yml` | Site theme, presentation mode, Quarto configuration |
| `.github/workflows/publish.yml` | CI: a leak guard plus `quarto render`, deployed to GitHub Pages |
| `CLAUDE.md` | Maintainer notes: build procedure, rendering pitfalls, decisions |

## What is deliberately not here

The course outline (`course-outline.md`) is the single source of truth for the course and lives only in the private repository. Week pages on this site are produced from it by a **deny-by-default filter**: only three sections per week are public — *Positioning*, *Learning objectives*, and *References* — plus, from W3 on, a student-facing *Demos* section. Lesson timing, the misconception list, demo scripts for the lecturer, and the private design notes stay private. The same goes for the Chinese speaker notes in the slides, the lecturer's demo READMEs (rehearsal notes and fallbacks), and all assessment material (questions and rubrics).

If `docs/course-outline.md` ever appears in this repository, that is a regression, not a feature.

## How the site is built

The normal entry point is `bin/publish.py` in the private repository. In one run it re-extracts the speaker notes from the slide originals, publishes the stripped slides, copies the whitelisted demos and handouts, runs the filtered build of the week pages, and finally checks that nothing teacher-only leaked — a single surviving `::: {.notes}` block aborts the publish. It does **not** commit or push; that is done by hand so the diff is always seen.

```bash
cd ~/Course-Hub && python3 bin/publish.py nlp_llm
```

To rebuild only the week pages from a checkout of the outline (this path does not update slides or demos):

```bash
export COURSE_OUTLINE=~/Course-Hub/nlp_llm/course-outline.md
python3 scripts/build_weeks.py   # after every outline change
quarto preview                   # local preview (brew install --cask quarto)
quarto render                    # produces _site/
```

Pushing to `main` triggers GitHub Actions, which **does not** rebuild the week pages (CI has no access to the outline). It only runs a leak check and `quarto render`. The cost of that design is that the site drifts if the local build step is forgotten.

Opening a new week requires two coordinated edits: add the week to `PUBLISHED_WEEKS` in `scripts/build_weeks.py` (controls the links on the home page, course map and resources) and add its page to the `render` list and sidebar in `_quarto.yml` (controls whether the page exists).

## Language

The site skeleton and the body of every published week page are in English. The English for each week is written in the outline alongside the Chinese and extracted at build time; a week without an English block falls back to Chinese rather than breaking, and the build prints the list of weeks still to be translated. Two things stay Chinese on purpose: each week page's `subtitle` (the Chinese week title, kept for cross-reference) and the `supplements/` guides, which match the language of instruction.
