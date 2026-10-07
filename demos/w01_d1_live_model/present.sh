#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W1 demo 1（投影片 p9）課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入主模型，按 Enter 逐步跑
#   ./present.sh backup        上課用：改用備用模型
#   ./present.sh replay        現場失敗時：重播課前彩排錄下的輸出（畫面標明非現場）
#   ./present.sh replay-backup 同上，重播備用模型的彩排
#   ./present.sh check         課前體檢（兩個模型各暖機一次）
#   ./present.sh probe [...]   課前挑題，例如 ./present.sh probe --repeat 5
#   ./present.sh rehearse      課前彩排並錄下退路
#   ./present.sh fake          沒有模型也能演練流程
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show)      exec "$PY" demo.py --model primary show "$@" ;;
  backup)    exec "$PY" demo.py --model backup show "$@" ;;
  replay)    exec "$PY" demo.py --model primary replay "$@" ;;
  replay-backup) exec "$PY" demo.py --model backup replay "$@" ;;
  check)     exec "$PY" demo.py check "$@" ;;
  probe)     exec "$PY" demo.py --model primary probe "$@" ;;
  rehearse)  "$PY" demo.py --model primary rehearse
             exec "$PY" demo.py --model backup rehearse ;;
  fake)      exec "$PY" demo.py --fake show "$@" ;;
  *) echo "不認得的子命令：$cmd" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
