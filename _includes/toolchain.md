<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## Toolchain

**Students need no compute for this course.** It is lectures throughout, and every demo is run live in class on the lecturer's machine — a MacBook Pro with an M5 Max and 64 GB of unified memory. That machine has **no CUDA**, which decides most of this table.

| Purpose | First choice | Notes |
|:------------|:----------------|:--------------|
| Inference for in-class demos | **MLX (`mlx-lm`)**, or llama.cpp / Ollama (GGUF) | Native to Apple silicon. **64 GB of unified memory is this machine's real advantage**: models far larger than 16 GB load comfortably |
| Building components by hand | Plain `torch` (MPS backend, or CPU) | The hand-built labs in W3, W4, W6 and W9 are small enough that MPS — or even CPU — finishes them in minutes |
| Tokenizer experiments | `tokenizers`, `sentencepiece`, `tiktoken` | Pure CPU, hardware-independent. The W2 fertility table |
| Fine-tuning demos | **LoRA in `mlx-lm`** | W8. The `peft` + `bitsandbytes` QLoRA path does not run here — see below |
| Evaluation | `lm-evaluation-harness` | Pointed at an MLX or llama.cpp backend. The W14 workshop looks at per-item results, not just aggregate scores |
| RAG | `sentence-transformers` + `faiss-cpu`, or plain numpy | W12. Brute-force search on CPU is enough; no vector database needed |
| Interpretability | `transformer-lens` (MPS) | Activation patching in W14; fine at small model sizes |

### What cannot be demonstrated on this machine — and how it is taught instead

| Not available | Why | What replaces it |
|:-----------------|:--------|:---------------------|
| `bitsandbytes` 4/8-bit, QLoRA | CUDA only | MLX's own quantization and LoRA. The concept is identical; only the backend differs |
| `vLLM`'s PagedAttention and continuous batching | Not usable on Apple silicon | W11 teaches the argument and cites the SOSP 2023 numbers. PagedAttention's memory argument was always clearest on a whiteboard |
| **FP8 / NVFP4 measurements** | Apple silicon has no such tensor formats | W7 and W11 cite the figures from the Quartet and NVFP4-pretraining reports |
| FlashAttention, custom Triton kernels | CUDA only | The IO argument in W3 is a board derivation, not an implementation exercise |
| `auto-gptq`, `autoawq` | CUDA | MLX or GGUF quantization for the "what quantization changes besides perplexity" comparison |

> **The constraint is itself teaching material.** Saying in class "I cannot run this demo on this machine, because FP8 is a Blackwell tensor format" makes W11's central point better than any slide: **inference optimization is bound to specific hardware.** Whether the same `transformers` code runs at all, and how fast, is exactly what that week is about.

### Models for the demos

The demos were originally sized for 16 GB with students running them too, so they used 0.6B toys. With students watching rather than running, that ceiling is gone:

| Used for | Suggested size | Why it is worth scaling up |
|:-------------|:-------------------|:-------------------------------|
| Hand-built components (W3, W4, W6, W9) | Qwen3-0.6B / 1.7B | Small is already the right choice — and it finishes while the class is still watching |
| **Long-context behaviour (W5)** | A quantized mid-size model | The lost-in-the-middle curve is far more convincing on a real model than on 0.6B |
| **MoE routing (W6)** | A genuine open-weight MoE | No need to simulate with a tiny model plus four experts |
| **Reasoning (W10)** | A mid-size model with a thinking mode | Budget forcing and overthinking only show up on a real reasoning model |
| The 2026 flagships | Read the paper's table | DeepSeek-V4, Kimi K3 and the rest are still out of reach — that has not changed |
