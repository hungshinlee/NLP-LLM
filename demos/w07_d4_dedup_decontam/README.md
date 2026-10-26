<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 4 — Near-duplicates across Common Crawl dumps, and MMLU against the corpus (slides p46–p47)

Two halves, both standard-library Python on the CPU. **(a) Live.** The instructor writes three functions from scratch in `dedup_live.py`, running the tests after each one, then runs them on demo 2's 200 MB sample of FineWeb-Edu (41,829 documents from ten Common Crawl dumps): MinHash signatures for every document, locality-sensitive hashing to find candidate near-duplicate pairs, the estimated and exact Jaccard of each pair, what the pairs are, and how the count moves when the LSH parameters $(b, r)$ change. **(b) Live.** The 14,042 questions of MMLU's test split are scanned for a 13-gram shared with the same corpus — GPT-3's decontamination rule — and the hits are printed; then two questions that MMLU-Redux marks as having a wrong answer key.

| Function | Signature | What it does |
|---|---|---|
| `shingles(text, n=5)` | → `set[str]` | lower-case, split on `\w+`, every run of 5 consecutive words joined by a space — the set $A$ on slide p44 (FineWeb also uses word 5-grams) |
| `minhash(shingles, k_h=128, seed=0)` | → `list[int]` | hash each shingle to a 64-bit integer $h$ (`blake2b`); the $i$-th signature is $\min_x (a_i h(x) + b_i) \bmod p$ with $p = 2^{61} - 1$ and $(a_i, b_i)$ drawn once from `random.Random(seed)`. A hash followed by a random linear map stands in for the random permutation $\pi$ of p44; the collision probability of one signature is the Jaccard similarity |
| `lsh_candidates(sigs, b=16, r=8)` | → `set[(i, j)]` | cut each signature into $b$ bands of $r$ rows; a band's key is `(band, tuple(sig[band*r:(band+1)*r]))`; any two documents sharing a key are a candidate pair. $\Pr[\text{candidate}] = 1 - (1 - J^{r})^{b}$, threshold about $(1/b)^{1/r}$ |

`dedup_final.py` is the finished reference implementation; `tests.py` holds the three tests and is shared by both files, so you can write your own version in `dedup_live.py` and test it step by step — functions you have not written yet are reported as `skip`, not as failures.

## The three tests

| | Checks | Passes when |
|---|---|---|
| (i) | the same 50-word paragraph shingled twice: exact Jaccard; once `minhash` exists, the 128 signatures too | $J = 1$ exactly; estimate $= 1$ (128/128 equal) |
| (ii) | two synthetic shingle sets of 1,000 elements built to have exactly $J = 0.75$, and two with $J = 0.30$ | the estimate is within $J \pm 2/\sqrt{k_h} = \pm 0.177$; the standard deviation $\sqrt{J(1-J)/k_h} \approx 0.04$ is printed too |
| (iii) | 60 synthetic pairs with $J$ from 0.10 to 0.99 run through `lsh_candidates` at $(16, 8)$ | every pair with $J > 0.9$ is a candidate. The fraction of candidates at each $J$ is printed next to the S-curve $1 - (1 - J^{r})^{b}$ — at $J = 0.6$ the curve says 0.24 and 3 of 8 pairs are candidates; at 0.7, 0.61 and 6 of 8; at 0.8, 0.95 and 8 of 8 |

## Requirements

| | |
|---|---|
| Packages | none beyond the standard library for `test`, `live`, `run`, `decontam`, `rehearse`, `replay`. `fetch` needs `pyarrow` and `huggingface_hub` (already in the shared `.venv` for demo 2's fetch). |
| Other folders | `../w07_corpus/fineweb_edu.jsonl` and `fineweb.jsonl` — demo 2's corpora; run `../w07_d2_scaling_fit/present.sh fetch --corpus both` once. |
| Network | `fetch` once: MMLU's test split (`cais/mmlu`, `all/test-*.parquet`, 7.5 MB as JSONL, MIT) and MMLU-Redux 2.0's Virology subset (`edinburgh-dawg/mmlu-redux-2.0`, 100 questions, CC BY 4.0), both saved under `data/` with the dataset revision in a `.meta.json`. Everything else is offline. |
| Time | signatures for the 41,829 FineWeb-Edu documents: 30 s on an Apple M5 Max (17 processes), 34 s for the 64,823 FineWeb documents; about 130 s on 3 processes of a small Linux VM. `run` caches the signatures in `runs/cache/`, so changing $(b, r)$ is instant (`rehearse` always recomputes, to record the time). The 13-gram scan takes 2–5 s per corpus. |
| Portability | nothing platform-specific. Built and run end to end on Linux/aarch64 with Python 3.10, then rehearsed on the course machine (Apple M5 Max, macOS 27, Python 3.12.14): every count below is identical on both — only `blake2b`, `random.Random(0)` and set arithmetic are involved. |

## Run it

```bash
cd demos                       # see ../README.md for creating .venv
cd w07_d2_scaling_fit && ./present.sh fetch --corpus both && cd ..   # once: the two corpora
cd w07_d4_dedup_decontam
./present.sh fetch             # once: MMLU test + MMLU-Redux Virology → data/
./present.sh test              # the three tests against dedup_final.py
./present.sh live              # the three tests against dedup_live.py (your version)
./present.sh run               # (a) on FineWeb-Edu: signatures → candidates at (16, 8) → Enter → (32, 4) → (8, 16) → (14, 8)
./present.sh run --impl live --corpus web --dumps CC-MAIN-2013-20,CC-MAIN-2022-21   # your functions, the other corpus, two dumps only
./present.sh decontam          # (b) 13-gram scan of MMLU against FineWeb-Edu → Enter → MMLU-Redux
./present.sh rehearse          # everything, both corpora, written to runs/rehearsal/
./present.sh replay            # the rehearsal record, one step per Enter (show: without pauses)
```

Without `present.sh`: `python demo.py run`, `python tests.py dedup_live`, and so on.

## What you should see

From `runs/rehearsal/` (rehearsed 2026-10-26 on the M5 Max; the counts are exact and machine-independent).

**(a) FineWeb-Edu, 41,829 documents, 10 dumps, $k_h = 128$.** At $(b, r) = (16, 8)$ — threshold $J \approx 0.71$ — there are **13 candidate pairs**: 12 across two dumps, 1 within a dump, 5 with the identical URL. Eight have exact $J \geq 0.9$ (six are $J = 1$). What they are: five pages re-crawled under the same URL in a later dump (a brain-teaser page, an H-Net review, a physics lecture note, a swissinfo article, a theology page); two pages whose URLs differ only in http/https, `www` or letter case (Wikipedia's `Black_skimmer` / `Black_Skimmer`, $J = 0.985$); two pairs on different sites — a company renaming (ihs.com → ihsmarkit.com, $J = 1$) and **a Mayo Clinic article reposted in full by a Nigerian news site** ($J = 0.713$); three same-site template pages (ancestry.com family-tree stubs, a leapfrog.com activity moved to a new path, two Merriam-Webster entries); one page that moved within a site. The single within-dump pair has $J = 0.525$: FineWeb's own per-dump MinHash (threshold 0.75) left nothing above 0.75 inside a dump, and our slightly lower threshold picks up a template page under it. Across dumps the picture is the opposite — eight of the twelve pairs are $J \geq 0.9$, the same page crawled in two different years, which per-dump deduplication does not see.

Changing $(b, r)$: **(32, 4)** (threshold 0.42) gives 74 candidates, of which 64 have $J < 0.75$ — template pages from ancestry.com, merriam-webster.com, catholic.org, nsf.gov; **(8, 16)** (threshold 0.88) gives 8, all of them $J \geq 0.9$; FineWeb's **(14, 8)** gives the same 13 as (16, 8). The eight high-similarity pairs are candidates under every setting; the parameters move only the left half of the S-curve.

**FineWeb (unfiltered), 64,823 documents, 8 dumps**: 6 pairs at (16, 8) — a blog post against its month-archive page ($J = 1$), a university department page that moved ($0.822$), and four same-site templates (a bookshop, a company directory, a journal site, a health forum) with $J$ from 0.43 to 0.74; 194 at (32, 4), 1 at (8, 16), 5 at (14, 8).

A 200 MB sample is about 2 % of `sample-10BT` and a $10^{-4}$ of FineWeb-Edu; a duplicate pair is seen only if both copies land in the sample, so 13 is a lower bound of a lower bound, not an estimate of the duplication rate. Read *what* the pairs are, not how many.

**(b) MMLU against the corpus.** 764,653 distinct 13-grams from 14,042 questions (question + choices); 231 questions have fewer than 13 words and cannot be caught by this rule. Against FineWeb-Edu, **37 questions share a 13-gram with the corpus** — but 34 of them because the question quotes a famous text as its stimulus ("This question refers to the following information …"): Roosevelt's Pearl Harbor address (3 questions), Kennedy's moon speech (4), King's *Letter from Birmingham Jail* (4), the Preamble to the U.S. Constitution (5), Patrick Henry (3), South Carolina's declaration of secession (4), the opening of *A Tale of Two Cities* (4), Churchill (2), Pericles' funeral oration (2) and the enacting clause of Australia's 1901 Immigration Restriction Act (3). Those texts are everywhere on the web; the overlap says nothing about whether the *question* leaked. Three hits are different: a clinical-knowledge item whose question plus correct answer — "a hypertonic solution is a solution that has a higher concentration of solutes than …" — appears as one sentence on a Biology 107 revision deck at studyblue.com; a government question that quotes "we hold these truths to be self-evident" in its own sentence (18 documents); and a professional-law choice that quotes the Constitution's copyright clause. Against unfiltered FineWeb: 11 questions (8 stimuli, 2 question sentences — one of them the American Marketing Association's definition of a brand — and 1 choice).

So the slide's question — zero, a handful, hundreds? — has a two-part answer: a few dozen hits, almost all of them public texts quoted by the benchmark, and one or two that look like the question itself was copied from a study site. An $n$-gram rule has false positives (quotation) and the detection has a floor (Oren et al.: a test set has to appear a few times before it can be detected at all); a clean scan of a sample proves little, and a dirty one has to be read.

**MMLU-Redux 2.0, Virology**: 57 of 100 questions are flagged (33 `wrong_groundtruth`, 14 `bad_question_clarity`, 4 `no_correct_answer`, 4 `multiple_correct_answers`, 2 `bad_options_clarity`) — Gema et al.'s 57 % for this subset. The two printed are epidemiology questions from a university sample exam, sitting in the Virology subset, with answer keys the re-annotators judge wrong (a cohort study keyed as a case series; "bias" keyed as "interaction").

## Files

| File | |
|---|---|
| `dedup_final.py` | `shingles`, `minhash`, `lsh_candidates`; helpers `jaccard`, `estimate`, `candidate_probability`, `permutations` |
| `dedup_live.py` | the version written in class (reset before the lecture) |
| `tests.py` | the three tests, shared by both implementations; `--replay` |
| `demo.py` | `fetch` / `run` / `decontam` / `rehearse` / `replay` / `show` |
| `demo_config.toml` | corpus paths, $n$, $k_h$, seed, $(b, r)$ and the settings tried afterwards, the 13-gram rule, the two datasets |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/` | `tests.json`; `dedup_edu.json`, `dedup_web.json` (every candidate pair at every setting, with $\widehat J$, exact $J$, dumps, URLs and a one-line classification); `decontam.json` (the hit questions, the shared 13-grams, the URLs); `summary.json` |
