# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 1 的手寫部分：budget forcing（Muennighoff et al., s1, §3.1）的兩個動作，做成 mlx-lm 的 logits processor。

mlx-lm（../versions.lock 的 commit）的 `BatchGenerator.insert(..., logits_processors=[[...], ...])` 每一條序列可以掛自己的
processor：`processor(context, logits)`，context 是這一條到目前為止的 token（含 prompt），logits 是 (1, V)。
所以兩個動作都不必改取樣迴圈：

    逼它停      推理的 token 數到了上限 → 接下來幾步的 logits 換成 one-hot，依序送出「\\n</think>\\n\\n」
    不准它停    這一步抽到 </think> → 不送出，改送「Wait」，推理繼續；最多擋 waits 次

「抽到 </think>」是用同一個 sampler 先抽一次、看結果：抽到別的 token 就把那個 token 以 one-hot 送出（等於照原本的分布取樣），
抽到 </think> 才介入。這樣對任何 sampler（temperature、top-p、top-k）都與「先取樣、再把 </think> 換掉」同分布。

另外有三個不經過 processor 的「接好的 prompt」：把一段已經寫好的推理截到前 c 個 token 再接上結束記號（與「上限 c」同分布，
而且與原本那一段共用前綴）、把推理換成等長的「.」、在推理結束的地方接「Wait」繼續。demo 3 的 early answering 用第一個。

mlx 只在函數裡 import：切 token、接 prompt、數翻轉都是純標準函式庫。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ThinkIds:
    open: int                 # <think>
    close: int                # </think>
    dot: int                  # "."
    eos: set                  # 停止記號
    open_seq: list            # "<think>\n"
    close_seq: list           # "\n</think>\n\n"
    wait_seq: list            # "Wait"
    answer_prefix: list = field(default_factory=list)   # </think> 之後、回答之前接的一句（s1 的寫法是 "Final Answer:"）

    @property
    def tail(self) -> list:
        """close_seq 裡 </think> 之後的部分，加上 answer_prefix：不論推理怎麼結束，回答之前都是這幾個 token。"""
        return list(self.close_seq[self.close_seq.index(self.close) + 1:]) + list(self.answer_prefix)


def think_ids(tok, forcing_cfg: dict) -> ThinkIds:
    """從 tokenizer 查出記號的 id。<think>、</think>、"." 都必須是單一 token，不是就直接停。"""
    def enc(s: str) -> list[int]:
        return list(tok.encode(s, add_special_tokens=False))

    def single(s: str) -> int:
        ids = enc(s)
        if len(ids) != 1:
            raise SystemExit(f"{s!r} 在這個 tokenizer 裡不是單一 token（{ids}）：budget forcing 的遮罩要重寫")
        return ids[0]

    o, c = single(forcing_cfg["open"]), single(forcing_cfg["close"])
    close_seq = enc(forcing_cfg["close_text"])
    if close_seq.count(c) != 1:
        raise SystemExit(f"close_text {forcing_cfg['close_text']!r} 切出來沒有剛好一個 </think>：{close_seq}")
    ap = forcing_cfg.get("answer_prefix", "")
    return ThinkIds(open=o, close=c, dot=single(forcing_cfg["dot"]), eos=set(int(t) for t in tok.eos_token_ids),
                    open_seq=enc(forcing_cfg["open_text"]), close_seq=close_seq, wait_seq=enc(forcing_cfg["wait_text"]),
                    answer_prefix=enc(ap) if ap else [])


# ── 切開一段生成：推理／回答 ─────────────────────────────────────
def split_think(tokens: list[int], ids: ThinkIds) -> dict:
    """tokens：模型生成的 token（不含 prompt；可以含最後的停止記號）。
    think ＝ <think> 與 </think> 之間的全部 token；answer ＝ </think> 之後、停止記號之前。沒有 </think> 就是沒有回答。"""
    toks = list(tokens)
    while toks and toks[-1] in ids.eos:
        toks.pop()
    close_at = toks.index(ids.close) if ids.close in toks else None
    head = toks if close_at is None else toks[:close_at]
    open_at = head.index(ids.open) if ids.open in head else None
    think = head[(open_at + 1) if open_at is not None else 0:]
    return {"open_at": open_at, "close_at": close_at, "closed": close_at is not None, "think": think,
            "answer": [] if close_at is None else toks[close_at + 1:]}


# ── 接好的 prompt（不經過 processor）─────────────────────────────
def _head(natural: list[int], ids: ThinkIds) -> list[int]:
    """模型自己寫的開頭：到 <think> 為止（含）。模板已經替它開好的話是空的。"""
    s = split_think(natural, ids)
    return [] if s["open_at"] is None else list(natural[:s["open_at"] + 1])


def prefix_dots(prompt: list[int], ids: ThinkIds, n: int) -> list[int]:
    """推理換成 n 個「.」。"""
    return list(prompt) + ids.open_seq + [ids.dot] * int(n) + ids.close_seq + ids.answer_prefix


def prefix_capped(prompt: list[int], natural: list[int], ids: ThinkIds, cap: int) -> list[int]:
    """同一段推理只留前 cap 個 token，接上結束記號。cap 不小於原本的長度時就是原本那一段再接結束記號。"""
    s = split_think(natural, ids)
    return list(prompt) + _head(natural, ids) + s["think"][:int(cap)] + ids.close_seq + ids.answer_prefix


def prefix_wait(prompt: list[int], natural: list[int], ids: ThinkIds) -> list[int] | None:
    """在模型自己寫下 </think> 的地方不接 </think>、改接「Wait」。原本那一段沒有自己結束（撞到上限）就回傳 None。"""
    s = split_think(natural, ids)
    if not s["closed"]:
        return None
    return list(prompt) + _head(natural, ids) + s["think"] + ids.wait_seq


# ── processor：一條序列一個 ──────────────────────────────────────
class Forcer:
    """budget forcing 的狀態機。每一步被呼叫一次，決定「下一個 token」的 logits。

    think_cap    推理最多幾個 token（從 n_think0 算起；到了就送出 close_seq）
    answer_max   回答最多幾個 token（不含 </think> 後面固定接的那幾個；到了就送出停止記號）
    waits        還可以擋幾次 </think>
    n_think0     prompt 裡已經接好的推理 token 數（prefix_wait 的情形）
    events       ("wait", 當時的推理 token 數)、("cap", …)、("answer_cut", 回答的 token 數)
    """

    def __init__(self, ids: ThinkIds, sampler, think_cap: int, answer_max: int, waits: int = 0, n_think0: int = 0):
        self.ids, self.sampler = ids, sampler
        self.think_cap, self.answer_max, self.waits = int(think_cap), int(answer_max), int(waits)
        self.n_think, self.n_answer = int(n_think0), 0
        self.closed = self.done = False
        self.queue: list[int] = []
        self.events: list[tuple] = []
        self.calls = 0

    def _onehot(self, logits, token: int):
        import mlx.core as mx
        return mx.where(mx.arange(logits.shape[-1]) == int(token), 0.0, -float("inf")).astype(logits.dtype)[None]

    def _observe(self, token: int) -> None:
        if token in self.ids.eos:
            self.done = True
        elif self.closed:
            self.n_answer += 1
        elif token == self.ids.close:
            self.closed = True
            if not self.queue:                   # 模型自己結束的：補上固定接的那幾個 token
                self.queue = self.ids.tail
        elif token != self.ids.open:
            self.n_think += 1

    def __call__(self, context, logits):
        import mlx.core as mx
        if self.calls > 0:                       # 第 0 次看到的是 prompt 的最後一個 token
            self._observe(int(context[-1].item()))
        self.calls += 1
        if self.done:
            return logits
        if self.queue:
            return self._onehot(logits, self.queue.pop(0))
        if self.closed:
            if self.n_answer >= self.answer_max + len(self.ids.tail):
                self.events.append(("answer_cut", self.n_answer))
                return self._onehot(logits, min(self.ids.eos))
            return logits
        if self.n_think >= self.think_cap:       # 逼它停
            self.events.append(("cap", self.n_think))
            self.queue = list(self.ids.close_seq) + list(self.ids.answer_prefix)
            return self._onehot(logits, self.queue.pop(0))
        if self.waits > 0:                       # 不准它停：先用同一個 sampler 抽一次
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            drawn = int(self.sampler(logprobs).item())
            if drawn != self.ids.close:
                return self._onehot(logits, drawn)
            self.waits -= 1
            self.events.append(("wait", self.n_think))
            self.queue = list(self.ids.wait_seq)
            return self._onehot(logits, self.queue.pop(0))
        return logits

    def max_tokens(self) -> int:
        """給 BatchGenerator 的硬上限：狀態機自己會停，這只是保險。"""
        extra = len(self.ids.close_seq) + len(self.ids.answer_prefix) + (self.waits + 1) * max(1, len(self.ids.wait_seq))
        return max(8, self.think_cap - self.n_think + extra + self.answer_max + 8)


# ── 整批生成 ────────────────────────────────────────────────────
def generate(model, eos: set, items: list[dict], sampler, batch_size: int, progress=None) -> tuple[list[dict], dict]:
    """items[i] = {"prompt": [ids], "max_tokens": int, "forcer": Forcer 或沒有}。
    回傳每一條的 tokens（含停止記號）、finish（stop／length）、events；以及這一批的 token 數與秒數。
    progress(做完幾條, 共幾條)：每做完一條呼叫一次。"""
    from mlx_lm.generate import BatchGenerator
    if not items:
        return [], {"prompt_tokens": 0, "generation_tokens": 0, "seconds": 0.0}
    gen = BatchGenerator(model, stop_tokens=[[t] for t in sorted(eos)], sampler=sampler, max_tokens=max(it["max_tokens"] for it in items),
                         completion_batch_size=batch_size, prefill_batch_size=batch_size)
    procs = [[it["forcer"]] if it.get("forcer") is not None else [] for it in items]
    t0 = time.perf_counter()
    uids = gen.insert([list(it["prompt"]) for it in items], [int(it["max_tokens"]) for it in items], logits_processors=procs)
    out = {u: {"tokens": [], "finish": None} for u in uids}
    finished = 0
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                o = out[r.uid]
                o["tokens"].append(int(r.token))
                if r.finish_reason is not None:
                    o["finish"] = r.finish_reason
                    finished += 1
                    if progress is not None:
                        progress(finished, len(uids))
    gen.close()
    res = []
    for u, it in zip(uids, items):
        o = out[u]
        o["events"] = list(it["forcer"].events) if it.get("forcer") is not None else []
        res.append(o)
    info = {"prompt_tokens": int(stats.prompt_tokens), "generation_tokens": int(stats.generation_tokens), "seconds": time.perf_counter() - t0}
    return res, info


# ── 統計（純標準函式庫）─────────────────────────────────────────
def fit_q(counts: dict[int, tuple[int, int]]) -> dict:
    """counts[m] = (答對題數, 題數)。假設每一步獨立、每步做對的機率同為 q：Pr[全對] = q^m。
    一個參數的最大概似估計（黃金分割），另外回傳每個 m 各自的 acc^(1/m)。"""
    import math

    def nll(q: float) -> float:
        s = 0.0
        for m, (k, n) in counts.items():
            p = min(max(q ** m, 1e-12), 1 - 1e-12)
            s -= k * math.log(p) + (n - k) * math.log(1 - p)
        return s

    lo, hi, g = 0.5, 1 - 1e-9, (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(200):
        if nll(a) < nll(b):
            hi, b = b, a
            a = hi - g * (hi - lo)
        else:
            lo, a = a, b
            b = lo + g * (hi - lo)
    q = (lo + hi) / 2
    per_m = {m: ((k / n) ** (1 / m) if k > 0 else None) for m, (k, n) in counts.items()}
    half = math.log(2) / -math.log(q) if q < 1 else None
    return {"q": q, "half_life_steps": half, "per_m_q": per_m, "predicted": {m: q ** m for m in counts}}


def flips(left: list[int], here: list[int]) -> dict:
    """同一批題目、兩個條件的對錯（0／1）。right_to_wrong：左邊對、這裡錯。"""
    return {"right_to_wrong": sum(1 for a, b in zip(left, here) if a and not b),
            "wrong_to_right": sum(1 for a, b in zip(left, here) if not a and b)}
