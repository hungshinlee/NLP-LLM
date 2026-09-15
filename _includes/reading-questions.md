<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## Four Questions for Reading a Paper

Handed out in week 1 and used every week after. They are short on purpose — they have to be usable in the ten minutes before a seminar.

1. **Was the baseline actually tuned?** The most common source of illusory progress in this field. W8 (LoRA variants) and W9 (GRPO variants) both have concrete cases where the improvement disappears once the baseline gets a learning-rate sweep.
2. **Is the comparison at equal compute?** Iso-FLOPs, iso-parameter and iso-latency are three different comparisons, and they frequently give opposite answers. W6 is built around this.
3. **What is the metric rewarding?** The gap between a metric and the capability it stands for is the subject of W14, but it comes up every week — most sharply in W11, where KV-cache compression leaves perplexity almost untouched and quietly breaks instruction following.
4. **Is the claim mechanistic or correlational?** "The attention weight is high", "the chain of thought wrote this step", "the reward went up" are all correlational. W3, W10 and W14 are the same error three times over, and the answer is the same each time: ask for an intervention, and ask what the strong baseline does.
