<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 3 — Quality filtering: guess the source, score it, then train on each (slides p40–p41)

Two halves. **(a) Live.** Three pages sampled from FineWeb and three from FineWeb-Edu (fixed seed), shuffled: you guess which corpus each came from. Then the classifier that *built* FineWeb-Edu — `HuggingFaceFW/fineweb-edu-classifier`, a regression head on a 110M BERT, trained on 460k pages that Llama-3-70B-Instruct had scored 0–5 for "educational value" — scores the same six pages, and the sources are revealed. **(b) Recorded, replayed in class.** One of demo 2's byte-level GPTs (`s4`: $d = 192$, 7 layers, 3.15M parameters) is trained twice with the same seed and the same number of steps, once on the FineWeb sample and once on the FineWeb-Edu sample (equal bytes), and evaluated on **four held-outs, reported separately**: W1's English paragraph, a Wikipedia sample, and each corpus's own tail.

The point of splitting the report: "quality" has no definition, only a target. A model trained on filtered data should win on text that looks like the filter's taste; whether it also wins elsewhere is the measurement.

Two things to know before guessing. FineWeb-Edu is a *subset* of FineWeb (about 1.3T of 15T tokens), so a page drawn from FineWeb may well score 3 or more — it is unfiltered, not "non-educational". And the classifier reads at most the first 512 tokens of a page; the dataset was built the same way.

## What is being measured

| | |
|---|---|
| Pages | from each corpus, 2,000 candidate documents read at even spacing through the JSONL (the shards are sorted by Common Crawl dump), kept if 500–1,600 characters and `language_score ≥ 0.90`, three drawn with a fixed seed; the six are shuffled. `runs/rehearsal/pages.json` holds the text, URL and source of each. |
| Classifier score | the model's single logit (`score`); `int_score = round(clip(score, 0, 5))`; FineWeb-Edu keeps `int_score ≥ 3`. For the FineWeb-Edu pages the score stored in the dataset is printed next to ours — the same classifier, run by the dataset's authors. |
| Distribution | at rehearsal, 200 random pages from each corpus are scored and binned 0–5 (`distribution.json`). |
| The two runs | demo 2's `s4`, byte-level ($V = 256$, $L = 256$), 7,687 steps of $32 \times 256$ bytes = 63M bytes each (demo 2's $D = 20N$), AdamW $1.33 \times 10^{-3}$ with warmup and cosine, no dropout, seed 1 — everything read from demo 2's configuration. The two training sets are 200 MB each (equal to within 0.1 %). |
| Held-outs | `en_paragraph`: W1 demo 2's paragraph in English, 754 bytes (so a noisy single point); `wikipedia`: 290 whole articles, about 1 MiB, drawn with a seed from one row group of `wikimedia/wikipedia` 20231101.en (CC BY-SA 4.0); `edu_tail` and `web_tail`: the last ~1 MiB of documents of each sample, never trained on. Every 50 steps the loss on a fixed subset of each (up to 64 windows) is recorded; at the end W4's windowed measure runs over each whole held-out. bits/byte = nats ÷ ln 2 because a token is a byte. |

## Requirements

| | |
|---|---|
| Packages | `torch` (`2.14.0` pinned) and `transformers` (`5.17.0` pinned) for the classifier and the runs; `huggingface_hub` and `pyarrow` for `fetch` only. `replay`, `show` and `summarize` need none of them. |
| Other folders | `../w07_d2_scaling_fit/` (model, hyperparameters, corpus loader — and its `fetch --corpus both` must have been run: the two corpora live in `../w07_corpus/`), which in turn imports `../w04_d1_block_scratch/` and `../w03_d1_attention_scratch/`. Keep the folder layout. |
| Network | `fetch` once: the classifier (about 440 MB; its revision is pinned in `../versions.lock`) and one 23 MB row group of Wikipedia. Everything else runs offline. |
| Device | the classifier and the runs use `mps` on Apple silicon and fall back to the CPU (the classifier is fast either way, 0.1–0.4 s per page on a CPU; the two runs are about 10 min each on an M5 Max and far slower on a CPU). |
| Memory / time | the classifier is 110M parameters in fp32; the corpora are held in memory as byte tensors (200 MB each). The two runs took demo 2's `s4` 9.9 min each on the M5 Max. Weights go to `runs/weights/` (not committed). |
| Portability | nothing platform-specific; tested only on an Apple M5 Max (64 GB), macOS 27, Python 3.12. The pipeline was also exercised once on Linux/CPU with 12 steps per run and 20 MB corpora. |

## Run it

```bash
cd demos                       # see ../README.md for creating .venv
cd w07_d2_scaling_fit && ./present.sh fetch --corpus both && cd ..   # once: the two corpora (demo 2's fetch)
cd w07_d3_quality_filter
./present.sh fetch             # once: the classifier and the Wikipedia held-out
./present.sh sample            # draw the six pages (prints them with their sources; --seed N to redraw)
./present.sh pages             # in class: one page per Enter, sources hidden
./present.sh score             # in class: the classifier's score for each page, then the reveal, then the held-out texts' scores
./present.sh rehearse          # the recorded half: scores → distribution → the two runs → runs/rehearsal/ (JSON, summary, curves.svg)
./present.sh replay            # the rehearsal record, one held-out per Enter, then the slide's table
./present.sh show              # the same without pauses
```

Without `present.sh`: `python demo.py replay`, `python demo.py rehearse`, and so on, with `torch` and `transformers` importable.

## What you should see

Not rehearsed yet (2026-10-26): this section is filled in from `runs/rehearsal/summary.json` once the two runs have been made on the demo machine. From the container smoke test (CPU, 12 steps, 20 MB corpora — the training numbers there mean nothing) two things are already visible: our classifier scores match the scores stored in FineWeb-Edu to 0.01, and in a 20-page sample FineWeb pages averaged 1.45 (20 % at 3 or above) while FineWeb-Edu pages averaged 3.01 (all at 3 or above); the first 20 Wikipedia articles averaged 1.60.

A caveat the program prints every time: a three-million-parameter byte-level model, two 200 MB samples, one seed; six pages are six pages. What you see is the shape of the mechanism — quality is defined relative to a held-out — not the numbers of the FineWeb paper.

## Files

| File | |
|---|---|
| `sample.py` | `pick_pages`, `Classifier`, `score_distribution`, `fetch_classifier`, `fetch_wikipedia` / `load_wikipedia` |
| `train.py` | `Demo2` (loads demo 2, which loads W4 demo 1), `load_heldouts`, `train_run` |
| `demo.py` | `fetch` / `sample` / `pages` / `score` / `rehearse` / `train edu|web` / `summarize` / `replay` / `show` |
| `demo_config.toml` | sampling rule, classifier, which size, the four held-outs, the Wikipedia source |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/` | `pages.json`, `scores.json`, `distribution.json`, `edu.json`, `web.json` (settings, curves, finals), `summary.json` (the slide's table), `curves.svg` |
