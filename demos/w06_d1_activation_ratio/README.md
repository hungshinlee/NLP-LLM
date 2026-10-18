<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 1 — $N_{\text{total}}$, $N_{\text{active}}$ and $\rho$ from a `config.json` (slides p10–p11; the general formula is slide p13)

The compute ledger has said "memory $\propto N_{\text{total}}$, FLOPs $\propto N_{\text{active}}$" since week 1 without a number. The numbers live in each model's `config.json` — and reading an MoE config is the ticket to reading any 2026 open-weight model. This demo extends W4's `count_params.py` with an MoE branch and runs it on four models: Gemma 4 26B-A4B (the model on the instructor's machine, W1 demo 1), Qwen3-30B-A3B, Qwen3-Next-80B-A3B and Kimi Linear 48B-A3B.

Nothing is loaded. For the three remote models the script fetches only `config.json`, the safetensors index and the **header** of every shard (the first 8 bytes of a `.safetensors` file give the length of a JSON header listing every tensor's dtype and shape — two HTTP range requests per shard, no weights). For the local model it reads the same headers from the Hugging Face cache.

## What it computes

Two columns, in the spirit of W4's "exact vs. actual":

| Column | Source | What it adds up |
|---|---|---|
| **config (formula)** | `config.json` only | attention: the four projections of W4 p19 (GQA/MHA; a low-rank formula for MLA); linear/SSM layers where a formula is known (Qwen3-Next's Gated DeltaNet); the expert side $n_{\text{moe}}\,(n_E\,3\,d\,d_E + 3\,d\,d_S + n_E d)$; dense FFN layers $3\,d\,d_{\text{ff}}$; norms; embeddings |
| **file (safetensors)** | every shard's header | each tensor is classified by name and layer type (experts / shared expert / router / attention / linear / dense FFN / norms / embedding / head) and its shape multiplied out. Quantized MLX tensors are unpacked ($32/\text{bits}$ weights per `uint32`); scales and biases are listed but not counted. A multi-token-prediction module or a vision tower stored in the checkpoint is listed and **excluded** from $N$ |

Then, with $E$ = the routed-expert total (the only block that depends on $k$):

$$N_{\text{active}} = N_{\text{total}} - E + E\,\frac{k}{n_E}, \qquad \rho = \frac{N_{\text{active}}}{N_{\text{total}}}, \qquad \text{floor} = N_{\text{active}} - (\text{expert side}) = \text{attention} + \text{linear} + \text{norms} + \text{embeddings}.$$

$k$ and $n_E$ always come from the config — a checkpoint does not know how many experts a token uses. **And the key names are not standardised**: Qwen writes `num_experts_per_tok`, Kimi `num_experts_per_token`, Gemma 4 `top_k_experts` (nested under `text_config`). `./present.sh keys <model>` prints every key whose name contains `expert`, `moe`, `shared` or `router`, then says which key the script took for each quantity — if it recognised none, it says so rather than guessing.

Where the two columns disagree is the point of the exercise: the config did not say something. Gemma 4's config carries both `enable_moe_block` and a dense `intermediate_size` — whether each layer runs a dense FFN *and* a MoE block, or one *or* the other, only the tensor shapes can tell. Qwen3-Next's checkpoint stores an MTP module worth about 1.6 B parameters that is not part of the 80 B. Kimi Linear's KDA layers use low-rank projections whose rank is not in the config, so the formula column is left blank for them on purpose.

## Requirements

| | |
|---|---|
| Packages | **none** — standard library only (`tomllib` needs Python ≥ 3.11). `huggingface_hub`, if installed, is used only to locate the local cache. |
| Network | `./present.sh fetch` once, before class: three `config.json` files, three indices and a few dozen shard headers — tens of MB of transfer at most (Qwen3-Next lists some 70,000 tensors), stored compressed under `configs/` (under 1 MB). Everything else runs offline. |
| Local model | the Gemma 4 row needs `mlx-community/gemma-4-26B-A4B-it-4bit` in your Hugging Face cache (see `../w01_d1_live_model/`); without it the other three rows still run. |
| Memory / time | negligible — JSON parsing. |
| Portability | nothing platform-specific; the instructor runs it on an Apple M5 Max (macOS 27, Python 3.12). |

## Run it

```bash
cd demos                      # see ../README.md for the shared environment (optional here)
cd w06_d1_activation_ratio
./present.sh fetch            # once: configs + shard headers for the three remote models → configs/
./present.sh keys qwen30      # every expert/moe/shared/router key, and how the script reads them
./present.sh keys gemma       # same for the local Gemma 4 (nested text_config, top_k_experts)
./present.sh count            # four rows, two columns, then a summary table (the slide's table)
./present.sh count kimi       # one model: gemma, qwen30, qwen_next, kimi
./present.sh floor qwen30     # the floor: attention + norms + embeddings as a share of N_active
./present.sh replay           # the instructor's rehearsal record (runs/rehearsal/ratios.json), not a live run
```

Without `present.sh`: `python3 fetch_configs.py`, `python3 count_moe.py count`, `python3 count_moe.py keys qwen30`.

## What you should see

For Qwen3-30B-A3B the two columns agree to the parameter: $N_{\text{total}} = 30{,}532{,}122{,}624$, which is exactly the index's `total_size` of 61,064,245,248 bytes divided by 2 (bf16). With $n_E = 128$, $k = 8$: $N_{\text{active}} \approx 3.35$ B and $\rho \approx 11.0\%$ — well above $k/n_E = 6.25\%$, because the attention projections and the two embedding matrices (about 1.53 B, 46 % of $N_{\text{active}}$) are paid whatever $k$ is. That gap is the floor on slide p13.

From the rehearsal record `runs/rehearsal/ratios.json` (2026-10-18, Apple M5 Max, macOS 27.0.1, Python 3.12.14; every number from the safetensors headers, and for each fetched model the shapes × dtype bytes equal the index's `total_size`):

| model | $n_E$ / $k$ / $n_S$ | $N_{\text{total}}$ | $N_{\text{active}}$ | $\rho$ | $k/n_E$ | floor / $N_{\text{active}}$ |
|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B (text model; MLX 4-bit file) | 128 / 8 / 0 | 25.23 B | 3.82 B | 15.2 % | 6.25 % | 48 % |
| Qwen3-30B-A3B | 128 / 8 / 0 | 30.53 B | 3.35 B | 11.0 % | 6.25 % | 46 % |
| Qwen3-Next-80B-A3B | 512 / 10 / 1 | 79.67 B | 3.87 B | 4.9 % | 1.95 % | 56 % |
| Kimi Linear 48B-A3B | 256 / 8 / 1 | 49.12 B | 3.48 B | 7.1 % | 3.13 % | 50 % |

What the two columns disagreed on: Gemma 4 has dense FFN tensors (535 M in all) in **every one of its 30 layers next to the 128 experts** — so each layer runs a dense FFN *and* a MoE block, which is why its $\rho$ is the highest of the four; its checkpoint also carries a 0.57 B vision tower, listed and excluded. Qwen3-Next's checkpoint stores a 1.65 B MTP module (excluded), and its attention layers do have the output gate (the file agrees with the formula). Kimi Linear's 20 KDA layers come to 790 M, from the file only; its index `total_size / 2` is 82,560 above $N$ because a few small tensors are stored in fp32 — the bytes agree, the "÷ 2" does not.

## Files

| File | |
|---|---|
| `count_moe.py` | `keys` / `count` / `floor` / `rehearse` / `replay`; the key-name table is `moe_spec()`, the tensor classifier is `categorize()` |
| `fetch_configs.py` | configs, indices and shard headers via HTTP range requests; writes `configs/<org>--<name>/{config.json, model.safetensors.index.json, tensors.json.gz}` |
| `demo_config.toml` | the four models (in slide order) and the fetch/rehearse settings |
| `present.sh` | entry point |
| `runs/rehearsal/ratios.json` | the rehearsal record `replay` prints |
