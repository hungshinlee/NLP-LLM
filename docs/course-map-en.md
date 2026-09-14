## Course Map

```
                    ┌──────────────────────────────────────────────┐
   W1  Compressed ─▶│  Goal: read every layer of a modern LLM      │
       history      │  tokenizer → transformer → training →        │
                    │  inference → agents                          │
                    └──────────────────────────────────────────────┘
                                        ▲
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
 Part I — Architecture (W1–W6)   Part II — Training (W7–W10)   Part III — Systems & Scrutiny (W11–W14)
        │                               │                               │
  W1 From n-grams to seq2seq      W7  Pre-training & scaling    W11 Inference efficiency
  W2 Tokenization                 W8  SFT, PEFT & forgetting    W12 RAG & context engineering
  W3 Attention: mechanics & code  W9  Preference alignment & RL  W13 Agentic systems
  W4 The full block & dynamics    W10 Reasoning & test-time      W14 Evaluation, interpretability,
  W5 Positional encoding &            compute                        safety
     long context
  W6 MoE, SSMs & linear attention
```

**Dependencies.** W3–W4 are the hinge of the course: W5, W6 and W11 all rest on understanding the residual stream and the KV cache. W2 is a prerequisite for W7 (the tokenizer shifts the compute-optimal configuration). W9 is a prerequisite for W10. **W3 and W4 must not be compressed**; if the course runs late, compress W6 (an architecture survey by design) and the engineering detail in W12.

**Three axes that run through every week.**

| Axis | The question to ask each week |
|:--|:--------------------------------|
| **Representation** | What is the unit at this layer — byte, token, hidden state, KV entry, retrieved chunk, latent thought? **Who chose that unit, and what does the choice cost downstream?** |
| **Compute** | Does this method spend *training* compute or *inference* compute? Is it FLOPs bound or memory-bandwidth bound? |
| **Supervision** | Where does this capability come from — the pre-training distribution, human preference labels, a verifiable reward, retrieved knowledge, or extra compute at test time? **The same behaviour can come from entirely different sources; that is the central dispute in W10 and W14.** |

**Four questions for reading a paper.** (1) Was the baseline actually tuned? (2) Is the comparison at equal compute — iso-FLOPs, iso-parameter, or iso-latency? (3) What is the metric rewarding? (4) Is the claim mechanistic or merely correlational?
