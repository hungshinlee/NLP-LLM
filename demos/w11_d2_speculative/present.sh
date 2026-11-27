#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 demo 2（投影片 p45 前導頁、p46 示意圖、p47「Live: demo 2 (a)」、p49「Replay: demo 2 (b)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live                 課堂 (a)：speculative_live.py 的函數本體寫好之後跑——五個測試，Enter 一個（不載模型）
#   ./present.sh live --final         寫錯了：用參考實作 speculative_final.py
#   ./present.sh replay-a             現場 (a) 失敗：印彩排的 tests.json（不載模型）
#   ./present.sh replay               課堂 (b)：彩排紀錄，Enter 四段——兩個 draft、加速比對 gamma、依文字種類、T = 0 對 greedy（不載模型）
#   ./present.sh show                 課堂（可選）：一個 prompt，先不用 draft、再用；草稿被留下的 token 是綠色
#                                     （--draft d17、--gamma 8、--prompt code-03、--target q4）
#   ./present.sh test                 不載權重：五個測試（參考實作；骨架寫了也一起測）、算術對投影片的數、亂數小模型上的量法（一分鐘內）
#   ./present.sh inspect              課前：版本、機器、四個模型、三個 tokenizer 是否相同、mlx-lm 的接受規則那幾行、draft 替自己草擬
#   ./present.sh probe                彩排前：每類一個 prompt 量一次，估全套要多久
#   ./present.sh rehearse             彩排全套：(a) → 每個 target 的基準、重疊、驗一趟的時間 → 掃 gamma；做過的格子跳過
#                                     （--part a|base|sweep、--target bf16|q4、--force、--note "接電源、其他程式都關了"）
#   ./present.sh fetch                確認四個模型在本機（都已在 ../versions.lock；缺的話 ./present.sh fetch --online）
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 選項寫在子命令後面，例如 ./present.sh rehearse --part sweep --target bf16
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
  live|replay|replay-a|show|test|inspect|probe|rehearse|fetch)
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
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,18p' "$0" >&2; exit 2 ;;
esac
