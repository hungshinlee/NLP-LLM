<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W3 · Demo 3 — the same sentence, three rankings (slide p45)

One sentence goes into a small language model once. The demo asks the same question three ways — *for the model's prediction at the last position, which input tokens matter?* — and prints three rankings side by side:

| Column | How it is computed | What it is |
|---|---|---|
| **attention** | the last row of one head's attention matrix in one layer (weights sum to 1) | the thing people draw as a heat-map and read as "what the model looked at" |
| **gradient × input** | with gradients enabled on the input embeddings, one backward pass from the log-probability of the model's top-1 next token; score of token `s` = ⟨∂target/∂e_s, e_s⟩, ranked by absolute value and shown as a share | a first-order estimate of "how much would the target move if this token's embedding were removed" |
| **attention rollout** (Abnar & Zuidema, 2020) | per layer, average the heads, mix in the residual connection (`Ã = 0.5·A + 0.5·I`, rows renormalized), multiply the layers together, take the last row | the share of information flowing from each input to the output once the residual stream is accounted for |

Between the columns the demo prints Spearman rank correlations, and for the chosen layer it prints which token each of the 16 heads ranks first. Pressing Enter reveals the columns one at a time (attention → gradient × input → rollout → attention again for the next head), so you can commit to a prediction before each reveal.

The lesson is the week's first misconception: a clean heat-map and a measure of importance are different objects and routinely disagree. This is also the first appearance of the question W14 returns to with activation patching — what would count as *mechanistic* evidence.

## Requirements

| | |
|---|---|
| Packages | `torch 2.14.0`, `transformers 5.17.0` (pinned in `../versions.lock`). |
| Model | `Qwen/Qwen3-0.6B` (bf16 weights, run in float32; ≈ 1.2 GB of weights to download, revision pinned by `fetch`). Backup `Qwen/Qwen3-1.7B` with `--model backup`. |
| Device | **CPU by default** (`device = "cpu"` in `demo_config.toml`): one forward plus one backward over a 13-token sentence took 0.41 s. `mps` is accepted as an option but was not used for the rehearsal. |
| Memory | ≈ 3–4 GB free RAM (0.6B parameters in float32 ≈ 2.4 GB, plus activations and the embedding gradient). The 1.7B backup needs ≈ 7 GB. |
| Attention implementation | **must be `eager`.** `transformers`' default SDPA path does not return attention weights; on the pinned version `5.17.0`, loading with SDPA and then calling the model with `output_attentions=True` raises `TypeError` — it does not fall back to eager and does not return `None`. `inspect` reproduces this on your installation. |
| Portability | should run anywhere `torch` and `transformers` install and `Qwen/Qwen3-0.6B` can be downloaded; tested only on an Apple M5 Max, macOS 26.6, Python 3.12.14 (2026-09-29). |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv (torch + transformers are enough for this demo)
cd w03_d3_attn_vs_gradient
./present.sh fetch                    # once, needs network: downloads Qwen3-0.6B and Qwen3-1.7B into the Hugging Face cache, pins their revisions
./present.sh inspect                  # versions, layer/head counts, eager vs. sdpa behaviour with output_attentions=True, time per forward+backward
./present.sh                          # the demo: all sentences are computed first, then Enter reveals column by column; n = next sentence; q = quit
./present.sh rehearse                 # every sentence × every configured head → runs/rehearsal/rankings.json and *.svg
./present.sh replay                   # walks through the instructor's rehearsal record instead of a live run (the screen says so)
./present.sh fake                     # the screens with made-up numbers; no model needed
```

The sentences, the layer (`"mid"` = layer 14 of 28) and the heads shown (0, 5, 11) are in `demo_config.toml`. Add your own sentences there; sentences where the subject and the verb are separated by a clause make the three columns disagree most visibly. `rehearse` overwrites the files in `runs/rehearsal/` with your results; `git checkout` them to get the instructor's back.

## What you should see

From `runs/rehearsal/rankings.json` (2026-09-29, Qwen3-0.6B, eager, CPU, layer 14):

**Sentence 1** — *The keys that the old man who lives next door lost last week* → the model's top-1 next token is `were` (log-prob −1.43). The three columns disagree almost completely:

- attention (head 0) puts most weight on `lost` (0.29) and `The` (0.24); heads 5 and 11 put over 70 % of their weight on the first token `The`;
- gradient × input ranks `man` (22 %), `that` (18 %) and `week` (14 %) first; `keys`, the grammatical subject the verb has to agree with, gets 5 % and is in nobody's top five;
- rollout collapses onto the first token (share 1.000) — with the residual mixed in, every path leads back to position 0, which says more about the method than about the sentence;
- Spearman attention-vs-gradient×input: −0.10, −0.08, −0.35 for heads 0, 5, 11.

**Sentence 2** — the Traditional Chinese sentence (*那位住在隔壁、上星期把鑰匙弄丟的老先生今天終於* → `找到`) is the counter-example: attention and gradient × input largely agree (Spearman 0.84–0.89), both ranking `住在` and `那位` first. Sometimes the heat-map and the importance measure do line up — the point is that you cannot tell which case you are in by looking at the heat-map.

**Sentence 3** (*The article the senator wrote about the scandal was* → `published`) is the shorter backup.

Note the tokenization of sentence 2 in the record: `鑰` is split into two byte-level fragments (shown as `�`), which is the W2 story reappearing in a W3 demo.

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `inspect` / `rehearse` / `show` / `replay`, plus `--fake`; the SVG writer for the rehearsal figures needs no matplotlib |
| `demo_config.toml` | device, layer, heads, rollout residual, the sentences, the two models |
| `present.sh` | entry point; always `HF_HUB_OFFLINE=1` except for `fetch`, uses `../.venv/bin/python` |
| `runs/rehearsal/rankings.json`, `inspect.json` | the rehearsal record quoted above |
| `runs/rehearsal/s<i>_*.svg` | per-sentence figures: attention rows per head, the three rankings, the rollout matrix |
| `../common.py` | version lock and terminal helpers |
