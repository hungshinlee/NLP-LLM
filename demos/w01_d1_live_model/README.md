<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W1 · Demo 1 — same prompt, a 2026 model, live (slide p9)

The cold open of the course feeds eight words of Shakespeare to four n-gram models and watches them fail — locally (nonsense within a phrase) or globally (no thread across phrases). This demo hands the **same eight words** to a 2026 open-weight model, which fails in a third way: it is fluent at both scales, and then answers a concrete, checkable question **wrongly, in exactly the tone it uses when it is right**.

| Step | What happens | What to notice |
|---|---|---|
| 1 | The eight-word seed of the cold open (from `slides/assets/w01/w01-data.json`), wrapped in one English instruction ("Continue the following passage in the style of an Elizabethan play…"), is continued by the model | fluent, in register, and not a quotation |
| 2 | One Traditional Chinese question with a single verifiable answer (default: the length and opening year of the Hsuehshan Tunnel on National Freeway 5) | wrong on both numbers, with no hedging at all |

Keys during `show`: **Enter** runs the next step (the output streams token by token); **a** prints the verified answer and its source — decide for yourself first; **r** re-runs the last step (temperature 0 does not guarantee identical output); **q** quits. After each step the screen prints token counts, prompt and decode tokens/s, and peak memory — the gap between prompt and decode speed is the first sighting of W11's memory-bound decoding.

`demo_config.toml` holds five candidate questions, each with its verified answer, the source and the date it was checked, and regex patterns for a rough automatic pass/fail. The chosen one (`step2.chosen`) was picked by `probe` as the question the main model got wrong three times out of three with the same confident wording. All five are deliberately neutral facts — Taiwanese transport infrastructure and paper authorship — and all answers were checked against primary sources on 2026-09-16; if you add a question, do the same and write the source down.

## Requirements

| | |
|---|---|
| Platform | **Apple silicon only**: 4-bit MLX models run through `mlx-lm` (pinned in `../versions.lock`). |
| Models | main `mlx-community/gemma-4-26b-a4b-it-4bit` (≈ 15 GB on disk; MoE, about 4 B active parameters); backup `mlx-community/Qwen3.8-27B-4bit` (≈ 16 GB, dense). `../setup.sh --only w01_d1_live_model` downloads both (≈ 31 GB) and pins their revisions. Both are vision-language checkpoints; `mlx-lm` loads the text model only. |
| Memory | one model at a time; the Gemma 4 rehearsal peaked at 14.3 GB, so a 32 GB Mac is enough. |
| Time | Gemma 4 on the M5 Max: prompt 515 tokens/s, decode 138 tokens/s; each step is well under a second plus a few seconds to load. |
| Tested on | Apple M5 Max, 64 GB, macOS 26.6, Python 3.12.14, mlx 0.32.2, mlx-lm at commit `872ae88` (2026-09-16) |

## Run it

```bash
cd demos && ./setup.sh --no-fetch          # once: shared .venv with the pinned mlx / mlx-lm (see ../README.md)
cd w01_d1_live_model
(cd .. && ./setup.sh --only w01_d1_live_model)   # once, needs network, ≈ 31 GB for both models (no fetch subcommand in this demo's present.sh)
./present.sh check                         # versions, memory, one warm-up generation per model
./present.sh probe --repeat 3              # every candidate question, three times each, with the rough pass/fail
./present.sh                               # the demo on the main model: Enter = step 1, Enter = step 2, a = answer, r = rerun, q = quit
./present.sh backup                        # the same on Qwen3.8
./present.sh rehearse                      # both models, both steps → runs/rehearsal/step{1,2}-{primary,backup}.json
./present.sh replay                        # the instructor's recorded outputs (the screen says they are not live); replay-backup for Qwen3.8
./present.sh fake                          # the screens with placeholder text; no model needed
```

Step 1 uses chat mode on purpose: in raw-continuation mode Gemma 4 fell into a repetition loop under greedy decoding and Qwen3.8 started explaining the sentence instead of continuing it — both are instruction-tuned models and will not simply carry on without being asked. The instruction is shown on screen together with the seed.

## What you should see

From `runs/rehearsal/` (2026-09-16, Gemma 4, greedy):

- **Step 1** — seed `beside well in his person wrought To be`, continuation: *"the very image of a godhead's thought, / In virtue's bloom and wisdom's golden light, / To banish darkness from the eyes of sight. / No shadow falls where such a sun may rise…"* — iambic, rhymed, in register, and not a line from any play.
- **Step 2** — *國道 5 號雪山隧道全長幾公里？哪一年通車？* → the model answers **19.8 km** and **22 December 2006**, in bold, as a tidy bulleted list. The verified answer is about **12.9 km**, opened **16 June 2006** (Freeway Bureau, checked 2026-09-16). Three probe runs gave the identical wrong answer word for word.
- In `probe`, Gemma 4 got **all five** candidate questions wrong, each time identically across three runs. The backup model's records are in `step{1,2}-backup.json`.

If the model you run gets the question right, that is a fine outcome too: compare the wording of the right answer with the wording of the wrong one on the slide — the tone is the same, which is the point slide p36 (Kalai & Vempala) comes back to.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `check` / `probe` / `rehearse` / `show` / `replay`; `--model backup`, `--fake` |
| `demo_config.toml` | the two models, generation parameters, step 1 source and mode, the five verified questions |
| `present.sh` | entry point; `HF_HUB_OFFLINE=1` except for `fetch`; uses `../.venv/bin/python` |
| `runs/rehearsal/step1-primary.json`, `step2-primary.json`, `step1-backup.json`, `step2-backup.json` | the recorded outputs quoted above, with prompts, parameters, throughput and peak memory |
