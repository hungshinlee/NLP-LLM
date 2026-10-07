<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W2 · Demo 2 — BPE: one loop by hand, one line changed, and OOV is gone (slide p28)

A complete byte-pair-encoding trainer, encoder and decoder in about a hundred lines of plain Python — no packages, no virtual environment, no model. Two files:

- `bpe_live.py` is the **skeleton used in class**. `merge_pair`, `train`, `encode` and `decode` are written; `get_pairs` raises `NotImplementedError`. The instructor writes `get_pairs` live (two nested loops: over the words, over adjacent symbol pairs, weighted by the word's count), runs it, then changes **one line** in `to_symbols` — from "split into characters + end-of-word marker" to "split into UTF-8 bytes with a leading space" (the replacement line is already there as a comment) — and runs it again. Running the untouched skeleton stops at `NotImplementedError`; that is the clean state.
- `bpe_final.py` is the **reference implementation** with three tests and a `--show` mode that prints the merge table and the encode trace of each word.

The toy corpus is `{"new": 3, "newest": 5, "nest": 2, "net": 3, "widest": 2}`, the same one the slides use for the hand computation. It was chosen (by random search) so that the first six merges have no ties, which makes the paper-and-pencil answer unique; ties, when they occur, go to the lexicographically smallest pair.

## The three tests

| | Checks | Slide claim |
|---|---|---|
| (a) | the merge sequence matches the reference, and its first three steps match the hand computation on the slide: `n`+`e` (13), `t`+`_` (12), `s`+`t_` (9) | the algorithm you computed by hand is the one in the code |
| (b) | `decode(encode(w)) == w` for every word; in byte mode, for arbitrary UTF-8 strings | lossless (W1's premise (a)) |
| (c) | in byte mode 𠊎 (U+2028E) becomes four byte tokens `F0 A0 8A 8E` with no unknown; in character mode it has no id at all | the two ways of eliminating out-of-vocabulary tokens |

The character version prints `not in vocabulary` for 𠊎 — not the string `[UNK]`. A real `[UNK]` comes from `--bert 𠊎`, which sends the same character through `bert-base-chinese`; that needs `transformers` and the model's tokenizer files in the local Hugging Face cache (W2 demo 1's `fetch` provides them). **`--bert` has not been run by the instructor yet** (as of 2026-09-30); the expected output is `[UNK]`, but treat that as a prediction, not a record.

## Requirements

| | |
|---|---|
| Packages | none — standard library only (`re`, `collections`). Any Python 3. `--bert` additionally needs `transformers` and `bert-base-chinese` in the cache. |
| Device / memory / time | anything; milliseconds. |
| Portability | fully portable. Tested (tests (a)–(c), both modes) on macOS; the byte-mode round trip and the 𠊎 case were also verified on Linux (2026-09-30). |

## Run it

```bash
cd demos/w02_d2_bpe
python3 bpe_final.py            # three tests: a ok, b ok, c ok
python3 bpe_final.py --show     # merge table (with counts) and per-word encode traces, in both modes
python3 bpe_live.py             # stops at NotImplementedError until you write get_pairs
python3 bpe_final.py --bert 𠊎  # optional: the same character through bert-base-chinese (needs transformers + the cached tokenizer)
```

To do what the class does: open `bpe_live.py`, fill in `get_pairs` (hint: `Counter`, `zip(syms, syms[1:])`, multiply by the word count), run it and compare the merge table with the one you computed by hand; then swap the two lines in `to_symbols` and run it again — the merge table changes, the round trip still holds, and 𠊎 is now four tokens instead of an error. `git checkout bpe_live.py` restores the skeleton.

## What you should see

Character mode, the eight merges on the toy corpus in order: `n e → ne` (13), `t _ → t_` (12), `s t_ → st_` (9), `ne w → new`, `e st_ → est_`, `new est_ → newest_`, `ne t_ → net_`, `new _ → new_`. Encoding then applies them by rank: `nest → ne | st_`, `widest → w | i | d | est_`, and the unseen `newer → new | e | r | _`. Byte mode: the same algorithm over bytes; every word round-trips; `encode("𠊎")` gives the five symbols `20 f0 a0 8a 8e` (a leading space, then the four bytes of the character).

There is no `runs/rehearsal/` here: the demo has no model and nothing to record.

## Files

| File | |
|---|---|
| `bpe_final.py` | reference `train` / `encode` / `decode` in two modes (`--chars`, `--bytes`), three tests, `--show`, `--bert` |
| `bpe_live.py` | the classroom skeleton; only `get_pairs` is missing |
