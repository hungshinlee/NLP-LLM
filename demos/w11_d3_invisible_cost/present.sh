#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 demo 3（投影片 p65 前導頁、p66 示意圖、p67「Replay: demo 3 (a)」、p70「Replay: demo 3 (b)」、p76「Live: demo 3 (c)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live-det             課堂 (c)：一個 prompt 單獨送兩次，再併進一個 batch 送，逐 token 對齊上色；最後印彩排的表
#                                     （--batch 16、--mode equal、--model q4、--prompt det-zh、--fixed）
#   ./present.sh replay-bits          課堂 (a)：彩排紀錄，Enter 四次——accuracy、flips 與雜訊底線、指令遵循、bits per byte（不載模型）
#   ./present.sh replay-evict         課堂 (b)：彩排紀錄，Enter 五次——system prompt、兩種政策、對照、一段實際的輸出（不載模型）
#   ./present.sh replay-det           現場 (c) 失敗：彩排的表與一個分岔的例子（不載模型）
#   ./present.sh replay               三段依序重播
#   ./present.sh test                 不載權重：規則與算術的測試，再用亂數小模型驗幾種生成方式（一分鐘內）
#   ./present.sh inspect              課前：版本、機器、兩百題從哪來、system prompt 的 token、mlx-lm 併批時補齊的那幾行、4-bit 兩種來源是否相同
#   ./present.sh probe                彩排前：每一段跑幾個，估全套要多久
#   ./present.sh rehearse             彩排全套：(c) → (a) → (b)；做過的格子跳過
#                                     （--part det|bits|evict、--only 8,4、--force、--note "接電源、其他程式都關了"）
#   ./present.sh rescore              不載模型：改了計分規則之後，從 runs/cache/ 的原始輸出重算三份紀錄的分數（--part det|bits|evict）
#   ./present.sh fetch                確認兩個模型與別的 demo 留下的檔案都在本機（缺模型：./present.sh fetch --online）
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 選項寫在子命令後面，例如 ./present.sh rehearse --part bits --only 4
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
  live-det|show|replay|replay-bits|replay-evict|replay-det|test|inspect|probe|rehearse|rescore|fetch)
    online=""
    rest=()
    for a in "$@"; do
      if [[ "$a" == "--online" ]]; then online="--online"; else rest+=("$a"); fi
    done
    if [[ -n "$online" ]]; then export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; fi
    exec "$PY" demo.py ${online} ${rest[@]+"${rest[@]}"} "$cmd" ;;
  tiny)
    sub="rehearse"
    if [[ $# -gt 0 && "$1" != --* ]]; then sub="$1"; shift; fi
    exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
