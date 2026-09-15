<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## Course Map

<div class="coursemap">
<p class="cm-goal"><strong>Endpoint — read a modern LLM layer by layer:</strong> for every layer from tokenizer to agent, say why it looks the way it does and where it breaks. W1 opens with the compression framing and the compute ledger that the remaining thirteen weeks keep filling in.</p>
<div class="cm-grid">
<div class="cm-col">
<div class="cm-part">Part I — Architecture</div>
<a class="cm-wk" href="weeks/w01.html"><span class="cm-n">W1</span><span class="cm-t">n-grams → seq2seq</span></a>
<div class="cm-wk cm-soon"><span class="cm-n">W2</span><span class="cm-t">Tokenization</span></div>
<div class="cm-wk cm-hinge cm-soon"><span class="cm-n">W3</span><span class="cm-t">Attention, by hand</span></div>
<div class="cm-wk cm-hinge cm-soon"><span class="cm-n">W4</span><span class="cm-t">The block &amp; dynamics</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W5</span><span class="cm-t">Position &amp; long context</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W6</span><span class="cm-t">MoE / SSM / linear</span></div>
</div>
<div class="cm-col">
<div class="cm-part">Part II — Training</div>
<div class="cm-wk cm-soon"><span class="cm-n">W7</span><span class="cm-t">Scaling laws &amp; data</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W8</span><span class="cm-t">SFT, PEFT, forgetting</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W9</span><span class="cm-t">Alignment &amp; RL</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W10</span><span class="cm-t">Reasoning &amp; test-time</span></div>
</div>
<div class="cm-col">
<div class="cm-part">Part III — Systems &amp; Scrutiny</div>
<div class="cm-wk cm-soon"><span class="cm-n">W11</span><span class="cm-t">Inference efficiency</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W12</span><span class="cm-t">RAG</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W13</span><span class="cm-t">Agentic systems</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W14</span><span class="cm-t">Evaluation &amp; interp.</span></div>
</div>
</div>
<p class="cm-note"><strong>W3 and W4 are the hinge.</strong> Everything in Part III reads from two objects built there — the <em>residual stream</em> and the <em>KV cache</em> — so those two weeks are the ones not to miss.</p>
</div>
