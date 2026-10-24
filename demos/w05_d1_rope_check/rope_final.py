#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 1 的參考實作：RoPE 的三個函數——頻率、成對座標的旋轉、旋轉後的內積。純 `torch`，釘 CPU、float64。

課堂上授課者在 rope_live.py 從零寫同一份（寫的順序見投影片 p23 講稿的【操作】），每寫完一段就
`./present.sh live` 跑三個測試（測試本體在 tests.py，兩份實作共用）。這一份是 (1) 課前確認三個測試全過、
(2) 現場卡住兩分鐘後 `./present.sh show` 切過來、(3) 課後發給學生。

三個函數（tests.py 依名稱與簽名呼叫，live 版要一樣）：

  rope_angles(d_head, base=10000.0) -> (d_head/2,)
      θ_i = base^(−2i/d_head)，i = 0, …, d_head/2 − 1。float64。
  rotate(x, m, theta) -> 與 x 同形
      把 x 的第 (2i, 2i+1) 維當成一對平面座標，各自轉 m·θ_i。m 是位置（int 或 0 維張量），theta 是 rope_angles 的輸出。
      這裡用「相鄰配對」(0,1), (2,3), …；Hugging Face 的 Llama 實作用「前後兩半配對」(0, d/2), (1, d/2+1), …（rotate-half），
      只是座標的排列不同，數學上同一個 R_m。
  score(q, k, m, n, theta) -> 0 維張量
      (R_m q)ᵀ (R_n k)。測試 (a) 檢查的就是它只依賴 m − n。

為什麼釘 CPU、float64：測試 (a) 要把「只依賴 m − n」驗到機器精度（10⁻¹⁵ 量級），float32 只到 10⁻⁶ 量級，
而 Apple 的 Metal 後端沒有 double 型別——不是慢，是沒有（W3 demo 1 同一條理由）。

用法：
  python3 rope_final.py             # 三個測試（等同 ./present.sh）
  python3 rope_final.py --show      # d_head = 4 的小例子逐步印出 + 三個測試（現場退路）
  python3 rope_final.py --rehearse  # 三個測試 + 寫 runs/rehearsal/tests.json（彩排；講稿裡的實測值從這裡抄）
"""
import sys

import torch

torch.set_default_device("cpu")           # 釘 CPU：Metal 沒有 double
torch.set_default_dtype(torch.float64)    # 整個檔案預設 float64


def rope_angles(d_head, base=10_000.0):
    return base ** (-torch.arange(0, d_head, 2, dtype=torch.float64) / d_head)   # arange(0, d, 2) 就是 2i


def rotate(x, m, theta):
    x1, x2 = x[..., 0::2], x[..., 1::2]            # 第 2i 維與第 2i+1 維配成一對
    ang = m * theta                                # 這一對要轉的角度 m·θ_i
    c, s = torch.cos(ang), torch.sin(ang)
    y = torch.empty_like(x)
    y[..., 0::2] = x1 * c - x2 * s                 # 二維旋轉：四個乘法、兩個加法
    y[..., 1::2] = x1 * s + x2 * c
    return y


def score(q, k, m, n, theta):
    return rotate(q, m, theta) @ rotate(k, n, theta)


# ── 現場退路：d_head = 4 的小例子逐步印出 ──────────────────────────
def show():
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    d_head, base = 4, 100.0                          # 兩對座標、底取小一點讓第二對也看得出在轉
    th = rope_angles(d_head, base)
    print(f"d_head = {d_head}, base = {base}: theta = {th.tolist()}   (lambda = 2*pi/theta = {[round(v, 2) for v in (2 * torch.pi / th).tolist()]})\n")
    q = torch.randn(d_head)
    k = torch.randn(d_head)
    print("q =", q)
    for m in (0, 1, 2):
        print(f"rotate(q, m={m}) =", rotate(q, m, th), f"  |.| = {rotate(q, m, th).norm():.6f}  (the norm never changes)")
    print("\nscore(q, k, m, n) for pairs with the same m - n = 3:")
    for n in (0, 5, 10, 100):
        print(f"  m = {n + 3:>3}, n = {n:>3}: {score(q, k, n + 3, n, th).item():+.15f}")
    print("and with a different m - n:")
    for m, n in ((0, 0), (5, 0), (10, 3)):
        print(f"  m = {m:>3}, n = {n:>3}: {score(q, k, m, n, th).item():+.15f}")
    print()


if __name__ == "__main__":
    import tests
    if "--show" in sys.argv:
        show()
    tests.run(sys.modules[__name__], rehearse="--rehearse" in sys.argv, label="rope_final.py")
