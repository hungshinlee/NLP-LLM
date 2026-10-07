#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W1 demo 2（投影片 p28）課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用：載入兩個模型，按 Enter 逐步填表
#   ./present.sh replay        現場失敗時：顯示課前彩排的表格（畫面標明非現場）
#   ./present.sh inspect       課前挑模型：只比 tokenizer（很快）
#   ./present.sh rehearse      課前彩排：完整算一次並錄下退路
#   ./present.sh diag          打分結果可疑時：BOS 重複？帶／不帶 cache 一致？英文 sanity check
#   ./present.sh fake          沒有模型也能演練流程（數字是假的）
#
# macOS 內建的是 bash 3.2：空陣列要寫成 ${opts[@]+"${opts[@]}"}，否則 set -u 會報錯。
# 換模型組合或換文字：在任何子命令前加參數，例如
#   ./present.sh --pair qwen38 mistral7b rehearse
#   ./present.sh --text taigi rehearse
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

opts=()
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --pair) opts+=("$1" "$2" "$3"); shift 3 ;;
    --text) opts+=("$1" "$2"); shift 2 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|inspect|rehearse|diag) exec "$PY" demo.py ${opts[@]+"${opts[@]}"} "$cmd" "$@" ;;
  fake)      exec "$PY" demo.py --fake ${opts[@]+"${opts[@]}"} show "$@" ;;
  *) echo "不認得的子命令：$cmd" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
