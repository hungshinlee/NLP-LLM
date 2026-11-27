#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 demo 1（投影片 p15 前導頁、p16 示意圖、p17「Live: demo 1 (a)」、p25「Replay: demo 1 (b), (c)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live                 課堂 (a)：載 bf16 的模型，Enter 三次——讀一千個 token、寫 64 個、並排（之後印彩排的兩個模型）
#   ./present.sh replay               (a) 的彩排紀錄：兩個模型的 prefill 與 decode、量到的兩片屋頂（不載模型）
#   ./present.sh replay-batch         課堂 (b)(c)：彩排紀錄，Enter 三段——batch 掃描、長 context、KV 存少一點（不載模型）
#   ./present.sh live-batch 16        課堂（可選）：現場補 (b) 的一個點（預設 bf16；--model q4 換 4-bit）
#   ./present.sh test                 不載模型：算術對投影片的數；再用亂數小模型驗量法（幾秒）
#   ./present.sh inspect              課前：版本、機器、兩個模型的形狀與 bytes、KV 的 dtype、4-bit 檔的量化參數、我們的迴圈對 mlx-lm 的
#   ./present.sh rehearse             彩排全套：(a) → (b) → (c)，兩個模型；做過的點跳過
#                                     （--part a|b|c、--model bf16|q4、--force、--note "接電源、其他程式都關了"）
#   ./present.sh fetch                確認兩個模型在本機（都已在 ../versions.lock；缺的話 ./present.sh fetch --online）
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 選項（--model、--part、--force、--note）寫在子命令後面，例如 ./present.sh rehearse --part a --model q4
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  live|replay|replay-batch|test|inspect|rehearse|fetch)
    online=""
    rest=()
    for a in "$@"; do
      if [[ "$a" == "--online" ]]; then online="--online"; else rest+=("$a"); fi
    done
    if [[ -n "$online" ]]; then export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; fi
    exec "$PY" demo.py ${online} ${rest[@]+"${rest[@]}"} "$cmd" ;;
  live-batch)
    n="16"
    if [[ $# -gt 0 && "$1" != --* ]]; then n="$1"; shift; fi
    exec "$PY" demo.py "$@" live-batch "$n" ;;
  tiny)
    sub="rehearse"
    if [[ $# -gt 0 && "$1" != --* ]]; then sub="$1"; shift; fi
    exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,15p' "$0" >&2; exit 2 ;;
esac
