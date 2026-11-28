#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 1（投影片 p17 前導頁、p19「Live: demo 1 (a)」、p28「Replay: demo 1 (b)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live                 課堂 (a)：一題三種答法——不想、點點、想；Enter 一種（載模型；推理在載完之後先生成，第三步才印）
#   ./present.sh replay               課堂 (a)：讀 runs/rehearsal/chain.json，Enter 一欄，最後印推理長度與 q^m（不載模型）
#   ./present.sh replay-budget        課堂 (b)：讀 runs/rehearsal/budget.json，Enter 三段——上限、不准停、翻轉（不載模型）
#   ./present.sh replay-live          現場那一題失敗：印彩排的 live.json（不載模型）
#   ./present.sh test                 不載模型：狀態機、切 token、接 prompt、q 的擬合、題目（幾秒）
#   ./present.sh inspect              課前：版本、兩種模板的結尾、記號的 id、一段真的推理、兩個動作各做一次
#   ./present.sh probe                彩排前：tokens/s 對 batch size、每個 m 上三種答法的小樣本、估時（--part speed|levels 只做一段）
#   ./present.sh rehearse             彩排全套：(a) → (b) → 現場那一題；做過的跳過（--force 重跑；--part chain|budget|live 只做一段）
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 模型是 Qwen/Qwen3-8B（W8 demo 2 已經 fetch 過）；這裡沒有東西要 fetch。
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
  live|replay|replay-budget|replay-live|test|inspect|probe|rehearse) exec "$PY" demo.py "$@" "$cmd" ;;
  tiny)   sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
