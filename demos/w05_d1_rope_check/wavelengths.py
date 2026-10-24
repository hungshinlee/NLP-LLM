#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 1 的第二段（投影片 p24 的現場版）：d_head/2 對座標各自的波長 λ_i = 2π/θ_i，對訓練長度 L_train 看哪些對轉不滿一圈。

純算術、純標準函式庫（不需要 torch）：θ_i = base^(−2i/d_head)。
印兩個底（10⁴ 與 5×10⁵，Llama 3 的底）、兩個訓練長度（256：上週的 tiny GPT；8192：Llama 3 的 L_orig），
每一對一行，標出它在 L_train 內轉了幾圈（r_i = L_train/λ_i）、轉不滿一圈的對著色。
最後對一次投影片的數字：../../slides/assets/w05/w05-data.json 的 `rope`（make_demo_w05.py 第 1 節），兩邊應逐對相同。

    python3 wavelengths.py                   # 預設 d_head = 128
    python3 wavelengths.py --d-head 64       # 其他 head 維度
    python3 wavelengths.py --no-table        # 只印摘要與對照
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDE_JSON = HERE.parent.parent / "slides" / "assets" / "w05" / "w05-data.json"
BOLD, DIM, RESET, GREEN, RED, AMBER, BLUE, MAGENTA = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[35m"

BASES = (10_000, 500_000)
TRAIN = (256, 8192)


def thetas(d_head, base):
    return [base ** (-2.0 * i / d_head) for i in range(d_head // 2)]


def summary(d_head):
    out = {}
    for base in BASES:
        th = thetas(d_head, base)
        lam = [2 * math.pi / t for t in th]
        entry = {"lambda_0": lam[0], "lambda_max": lam[-1], "train": {}}
        for L in TRAIN:
            under = [i for i, l in enumerate(lam) if L / l < 1.0]
            entry["train"][L] = {"pairs_under_one_turn": len(under), "first_pair_under_one_turn": under[0] if under else None}
        out[base] = entry
    return out


def main():
    argv = sys.argv[1:]
    d_head = int(argv[argv.index("--d-head") + 1]) if "--d-head" in argv else 128
    n = d_head // 2
    print(f"{BOLD}d_head = {d_head}: {n} pairs, theta_i = base^(-2i/d_head), lambda_i = 2 pi / theta_i{RESET}\n")
    s = summary(d_head)
    for base in BASES:
        e = s[base]
        print(f"base = {base:>7,}: lambda_0 = {e['lambda_0']:.2f}  lambda_max = {e['lambda_max']:.3e}")
        for L in TRAIN:
            t = e["train"][L]
            print(f"    L_train = {L:>5}: {t['pairs_under_one_turn']:>2} of {n} pairs never complete one turn"
                  + (f" (pairs {t['first_pair_under_one_turn']}..{n - 1})" if t["first_pair_under_one_turn"] is not None else ""))
        print()

    if "--no-table" not in argv:
        print(f"{DIM}turns in L_train = 256 | 8192;  {RED}red{RESET}{DIM} = under one turn at 8192, {AMBER}amber{RESET}{DIM} = under one turn at 256 only{RESET}")
        hdr = f"{'i':>3} " + "".join(f"| {'base ' + format(b, ','):>14} {'r@256':>9} {'r@8192':>9} " for b in BASES)
        print(hdr)
        lams = {b: [2 * math.pi / t for t in thetas(d_head, b)] for b in BASES}
        for i in range(n):
            row = f"{i:>3} "
            for b in BASES:
                lam = lams[b][i]
                r256, r8k = TRAIN[0] / lam, TRAIN[1] / lam
                col = RED if r8k < 1 else (AMBER if r256 < 1 else "")
                row += f"| {col}{lam:>14.4g} {r256:>9.3g} {r8k:>9.3g}{RESET} "
            print(row)
        print()

    # 對照投影片的數字
    if SLIDE_JSON.exists():
        ref = json.loads(SLIDE_JSON.read_text(encoding="utf-8")).get("rope", {})
        if ref.get("d_head") == d_head:
            bad = []
            for base in BASES:
                r = ref["bases"].get(str(base), {})
                e = s[base]
                if r and abs(r["lambda_max"] - e["lambda_max"]) > 1e-9 * e["lambda_max"]:
                    bad.append(f"lambda_max @ base {base}")
                for L in TRAIN:
                    rt = r.get("train", {}).get(str(L), {})
                    if rt and rt["pairs_under_one_turn"] != e["train"][L]["pairs_under_one_turn"]:
                        bad.append(f"pairs under one turn @ base {base}, L {L}")
            if bad:
                print(f"{RED}differs from the slides' w05-data.json: {', '.join(bad)}{RESET}")
            else:
                print(f"{GREEN}matches the slides' w05-data.json (`rope`): same lambda_max and the same counts of pairs under one turn{RESET}")
        else:
            print(f"{DIM}w05-data.json has d_head = {ref.get('d_head')}, not compared{RESET}")
    else:
        print(f"{DIM}{SLIDE_JSON} not found; nothing to compare against{RESET}")


if __name__ == "__main__":
    main()
