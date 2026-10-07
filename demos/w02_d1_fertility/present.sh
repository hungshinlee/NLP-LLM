#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W2 demo 1（投影片 p9「Live: the full table」）課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用：載入九個 tokenizer，按 Enter 逐列填表；c = 帳本、l = 最長漢字 token、q = 離開
#   ./present.sh replay        現場失敗時：顯示課前彩排的表格（畫面標明非現場）
#   ./present.sh inspect       課前確認：模型類型、normalizer、數字規則、byte fallback、[UNK]、NFD、前導空白、控制 token
#   ./present.sh rehearse      課前彩排：整張表算一次並錄下退路（runs/rehearsal/table.json）
#   ./present.sh fetch         課前一週：下載九個 tokenizer（要網路；Llama、Mistral v0.1 要先接受授權）
#   ./present.sh fake          沒有 tokenizer 也能演練流程（數字是假的）
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TIKTOKEN_CACHE_DIR="$(pwd)/../.tiktoken-cache"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|inspect|rehearse) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  fake)   exec "$PY" demo.py --fake show "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
