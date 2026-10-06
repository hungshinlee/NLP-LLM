#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 3（投影片 p32「Live: skip one layer at a time, and measure the perplexity」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用：載模型 → 印 baseline → 停下來等學生預測 → Enter → 28 個點一個一個長出來 → 總表（存 runs/live/）
#   ./present.sh replay        現場失敗時：用課前彩排的紀錄走同一套畫面（畫面標明非現場）
#   ./present.sh inspect       課前確認：版本、device、decoder layer 回傳格式、hook 恆等化有沒有生效、一次前向幾秒 → runs/rehearsal/inspect.json
#   ./present.sh rehearse      課前彩排：baseline + window 1／2／4 → runs/rehearsal/curve.json 與 curve.svg
#   ./present.sh fetch         第一次：下載 Qwen3-0.6B 與 1.7B（W3 demo 3 已下載的話只確認 revision；要網路；先 unset HF_TOKEN）
#   ./present.sh fake          沒有模型也能演練流程（數字是假的）
#   加 --model backup 用 Qwen3-1.7B；--device cpu 覆蓋 device；--windows 1,2 覆蓋 window；--no-pause 不等 Enter
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 與 transformers 由 ../setup.sh 裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }
"$PY" -c "import torch, transformers" 2>/dev/null || { echo "共用環境裡缺 torch 或 transformers：重跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|inspect|rehearse) exec "$PY" demo.py "$@" "$cmd" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py --online "$@" fetch ;;
  fake)   exec "$PY" demo.py --fake "$@" show ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
