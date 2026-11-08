<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 1 — SFT turns a base model into an assistant (slides p8–p9, p15, p42)

One model before and after supervised fine-tuning (SFT), on one small dataset, through one tool. `Qwen/Qwen3-1.7B-Base` is a *base* model: pre-trained only. LoRA adapters ($r = 16$, every linear layer of every block; one of them also on the embedding rows) are trained on 1,000 human-written instruction–answer pairs from `databricks/databricks-dolly-15k` with `mlx_lm.lora`, and the same three questions are put to the base model and the fine-tuned one. Four parts:

- **(a) Live.** Three questions — a fact, an instruction, one in Traditional Chinese. The base model gets the *raw text* (a base model sees a document, not a conversation); the fine-tuned model gets the same question through the tokenizer's chat template. Each output is printed in full and then judged on one thing you can see: did the model stop by itself, and on which token, or did it hit the 200-token limit?
- **(b) Live.** One training example, token by token, with the positions that carry the loss marked under mlx-lm's default (every token) and under `--mask-prompt` (the assistant's turn only). Then three probes: at three positions of the chat template, what probability does the *base* model give to the token the template says comes next (`<|im_start|>`, `assistant`, `<think>`)? That answers whether the control tokens were ever trained.
- **(c) Replayed.** Three adapters, same data, seed, steps and learning rate: A with `--mask-prompt`, B without, and C = A plus LoRA on the embedding rows (the LM head is tied to them). Their loss on 200 held-out examples (response tokens only, and all tokens), and their answers to the three questions side by side.
- **(d) Live.** The slide's one-line `mlx_lm.lora` command, run for 12 steps: train loss and peak memory every 4 steps. (Adapter C, used in (a), is the same command run for 500 steps.)

## What is being measured

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B-Base`, bf16, 1,720,574,976 parameters, 28 blocks, $d = 2048$, tied embeddings (Apache-2.0; revision pinned in `../versions.lock`). Its tokenizer ships Qwen3's chat template; its only declared stop token is `<\|endoftext\|>`. |
| Data | `databricks/databricks-dolly-15k` (CC BY-SA 3.0; 15,011 records written by Databricks employees, American English, eight categories). `prepare` keeps records with a 40–1,200-character response, a 10–500-character instruction and at most 1,500 characters of context, strips `[42]`-style citations, shuffles with a fixed seed and takes 1,000 / 100 / 200 for train / valid / test. Where a record has a context, it is appended to the instruction after a blank line. Format: mlx-lm's `{"messages": [{"role": "user", …}, {"role": "assistant", …}]}`. |
| Adapter | LoRA with `rank 16`, `scale 20`, `dropout 0`, on every `Linear` of every block (`--num-layers -1`: q, k, v, o, gate, up, down → 17,432,576 trainable parameters, 1.01 % of $N$); adapter C also on `model.embed_tokens` (mlx-lm's `LoRAEmbedding`, $\mathbf{A}$ is $V \times r$, and `as_linear` reaches the tied LM head: $(V + d)\,r = 2{,}463{,}744$ more, 19,896,320 in all). mlx-lm writes $y = Wx + s \cdot BAx$ with $s$ = `scale`, so `scale 20` is $\alpha / r = 20$, i.e. $\alpha = 320$ in Hu et al.'s notation. $\mathbf{A} \sim U(\pm 1/\sqrt{d_{\text{in}}})$, $\mathbf{B} = \mathbf{0}$; both stored in float32. |
| Training | Adam, constant learning rate $2 \times 10^{-5}$ (not tuned — demo 3 is the sweep), batch 4, 500 steps = 2 epochs, `max_seq_length 1024`, seed 20261108. mlx-lm sorts by length and pads to a multiple of 32; it does not pack. Validation on the 100 held-out examples every 50 steps. |
| The mask | mlx-lm's `ChatDataset`: tokens = `apply_chat_template(messages)`; with `--mask-prompt`, the loss starts at index `offset` = the length of `apply_chat_template(messages[:-1], add_generation_prompt=True)`. Qwen3's template renders the final assistant turn as `<\|im_start\|>assistant⏎<think>⏎⏎</think>⏎⏎{answer}<\|im_end\|>⏎`, so an empty think block is part of every target. |
| Stopping | Generation stops on `<\|endoftext\|>` *or* `<\|im_end\|>` for both models — the second is added by the program (`extra_eos` in `demo_config.toml`), because the base model's `generation_config.json` does not list it. Greedy decoding, at most 200 new tokens. |
| Held-out loss (c) | mlx-lm's own `evaluate()` on `data/test.jsonl`, in nats per token, twice: with `mask_prompt=True` (response tokens, including the think block and `<\|im_end\|>`) and `False` (every token). |
| Peak memory (d) | `mx.get_peak_memory()` as reported to mlx-lm's training callback at every report — the process-wide peak, so it includes the loaded weights. |

A caveat the program prints every time: one 1.7B model, one dataset of a thousand examples, one seed, one untuned learning rate. The shape of what SFT changes is the point; the numbers are this machine's.

## Requirements

| | |
|---|---|
| Packages | `mlx`, `mlx-lm` (git commit pinned in `../versions.lock`), `transformers` (tokenizer), `huggingface_hub`. Python 3.11+ for `tomllib` (3.10 works with `tomli`). `replay` and `replay-mask` need none of the MLX packages. |
| Models and data | `fetch` once (network): the model (about 3.4 GB) and the 13 MB Dolly file; both revisions are written to `../versions.lock`. Everything else runs offline. `prepare` is deterministic — re-running it reproduces `data/` byte for byte. |
| Device | **Apple silicon only** (MLX). `show` loads the base model twice (plain and with the adapter), about 7 GB; training peaks at the frozen 3.4 GB plus the adapter's 16 bytes per LoRA parameter plus activations — the program prints the measured value. |
| Time | one adapter: about 6.5–7 min for 500 steps (0.78 s/step, with validation every 50 steps); 1–4 s per generation (50–95 tok/s); 10–16 s per held-out evaluation. The whole rehearsal with three adapters: about 25 min. Run `rehearse --part live` in a fresh process, or its peak memory includes the models loaded before it. |
| Portability | nothing platform-specific apart from MLX; tested on an Apple M5 Max (64 GB), macOS 27.2, Python 3.12; the flow also in `--fake` mode on Linux. |

## Run it

```bash
cd demos                              # see ../README.md for creating .venv
cd w08_d1_sft_base_to_assistant
./present.sh fetch                    # once: the model and the dataset, revisions pinned
./present.sh prepare                  # data/{train,valid,test,example}.jsonl, data/meta.json, lora.yaml
./present.sh inspect                  # versions, config, template, stop tokens, N and N_LoRA against the slides' numbers, token lengths
./present.sh rehearse                 # train A and B → (a) → (b) → (c) → (d) → runs/rehearsal/
./present.sh rehearse --part train --adapter C      # or one part at a time: train | generate | mask | heldout | live
./present.sh                          # (a) in class: base vs. fine-tuned, three questions, Enter between outputs
./present.sh show --base-templated    # (a) with a third output per question: the base model through the chat template
./present.sh mask                     # (b) in class: the mask, then the three probes
./present.sh replay-mask              # (c) in class: the held-out table and the two adapters' answers (no model loaded)
./present.sh live-train               # (d) in class: the slide's command for 12 steps
./present.sh replay                   # the rehearsed (a), if the live run fails
./present.sh fake [show|mask|...]     # the same flow without mlx (fake outputs, marked as such)
```

Without `present.sh`: `python demo.py replay`, `python demo.py --part train rehearse`, and so on, with `mlx_lm` importable and `HF_HUB_OFFLINE=1` set. The equivalent `mlx_lm.lora` command line is printed before every training run; `lora.yaml` is what carries `rank` and `scale` (the CLI has no flag for them).

## What you should see

From `runs/rehearsal/` (2026-11-08, Apple M5 Max, macOS 27.2, Python 3.12, mlx 0.32.2, mlx-lm 0.32.0 at the pinned commit; greedy decoding, at most 200 new tokens).

**Three adapters, not two.** The first rehearsal trained A and B only. Both learned the *shape* of the format — two marker tokens each followed by a blank line, then the answer, then a marker — but in every place where `<think>`, `</think>` or `<|im_end|>` should be, they emitted some other token the base model had never been trained on (`𫟦`, `具有战士`, `Intialized`, …: ordinary rare entries of the vocabulary), and none of the three answers stopped. The reason is on the slides: LoRA on the Linears inside the blocks never touches the embedding table or the LM head, and for a token that pre-training never produced, that row is noise — indistinguishable from thousands of other untrained rows. Adapter C adds LoRA to the embedding (which Qwen3 ties to the LM head): 2,463,744 more parameters, and the three answers stop.

**(a) The three questions**

| | base, raw text | adapter C (chat template) |
|---|---|---|
| a fact (Austen, 1813) | does **not** stop (200 tokens): asks three more questions of its own ("Additionally, what is the significance…"), then answers in essay style; 1813 is right | stops on `<|im_end|>` after 29 tokens: one sentence; 1813 |
| an instruction (list vs. tuple) | **stops** on `<|endoftext|>` after 75 tokens: three numbered points | stops on `<|im_end|>` after 67 tokens: three points |
| in Traditional Chinese (Taipei 101) | **stops** on `<|endoftext|>` after 48 tokens: "310 m" (wrong) | stops on `<|im_end|>` after 37 tokens: "1,673 m" (wrong — 1,673 is in feet) |

So this base model already answers instruction-shaped text and stops — on the *document* end token. What SFT changed is the shape (one sentence, the conversation's end token) and not the facts: the Chinese answer is wrong before and after, differently. Through the chat template the base model repeats the question and then loops on a fragment (`umably`, `paque`): the control tokens mean nothing to it.

**(b) The mask and the probes.** The example (`data/example.jsonl`) is 28 tokens; `--mask-prompt` puts the loss on the last 14: `<think>`, a blank line, `</think>`, a blank line, eight answer tokens, `<|im_end|>`, a newline. Where the template says `<|im_start|>` should come next (after the user turn), the base model ranks it 51,940th and gives 0.56 to "What"; where it says `assistant`, 79,916th; where the training data has `<think>`, 18,461st. After training: A ranks `<think>` 10th ($p = 0.0004$; its top token is `𫟦` at 0.0006 — a flat distribution over untrained rows), B 12th, **C 1st** ($p = 0.99$). B — the only adapter with loss on the prompt — ranks `assistant` 1st after `<|im_start|>` ($p = 0.9999$); A and C, which never saw a loss there, keep the base's newline. And C, after the user turn, puts `<think>` / `</think>` / `<|im_end|>` at the top (0.39 / 0.27 / 0.13) with `<|im_start|>` still 123rd: only rows that were ever a target moved, and the three that moved share a direction (our reading, not a measured claim).

**(c) Held-out loss** (200 test examples, nats per token, mlx-lm's `evaluate()`):

| | response tokens only | all tokens |
|---|---|---|
| base | 2.717 | 2.995 |
| A · `--mask-prompt` | 2.012 | 2.857 |
| B · default | 1.973 | 2.125 |
| C · A + embedding rows | **1.721** | 2.587 |

A − B = +0.039: whether the prompt carries loss barely matters on this data (median response about 60 tokens). C − A = −0.291: the price of three rows that A can never predict. Training losses are not comparable across the mask (A 1.27, B 1.68, C 0.89 at step 500); validation loss bottoms out around step 250 and rises after — two epochs over 1,000 examples is already over-fitting.

**(d) Twelve steps** (a fresh process): train loss 2.500 / 2.219 / 1.809 at steps 4 / 8 / 12; peak memory 10.95 / 10.95 / 15.45 GB; 7.2 s. The slide's formula stops at 3.44 GB of frozen weights plus 0.32 GB for the adapter; the ten-odd GB above that are activations and the allocator, and they move with the longest sequence in the batch. The full 500-step runs took 392 s (A), 385 s (B) and 424 s (C).

## Files

| File | |
|---|---|
| `demo.py` | `fetch` / `prepare` / `inspect` / `rehearse` / `show` / `mask` / `replay-mask` / `live-train` / `replay`; training calls mlx-lm's `train_model` with the same arguments as the command line, plus a callback that records `peak_memory` |
| `demo_config.toml` | model, data source and filter, LoRA hyperparameters, the three questions, the mask example, the check against the slides' `w08-data.json` |
| `present.sh` | entry point; uses `../.venv/bin/python`, offline |
| `lora.yaml`, `lora_embed.yaml` | generated by `prepare`: `lora_parameters` for `mlx_lm.lora -c`; the second adds `keys` with `model.embed_tokens` (adapter C) |
| `data/` | generated by `prepare` from the pinned Dolly file (not published; `meta.json` records the filter, seed and source revision) |
| `runs/rehearsal/` | `inspect.json`, `adapters.json` (loss and peak-memory curves of A, B, C), `generate.json`, `mask.json` + `mask.txt`, `heldout.json`, `live_train.json` |
