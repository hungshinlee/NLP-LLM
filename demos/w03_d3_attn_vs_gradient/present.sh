#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 3（投影片 p45「Live: the same sentence, three rankings」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用：先算好所有句子，然後 Enter 逐欄揭露（attention → gradient × input → rollout → 換 head …）；n = 下一句；q = 離開
#   ./present.sh replay        現場失敗時：用課前彩排的紀錄走同一套揭露（畫面標明非現場）
#   ./present.sh inspect       課前確認：版本、eager／sdpa 對 output_attentions 的行為、層數與 head 數、一次前向＋反向的耗時
#   ./present.sh rehearse      課前彩排：每句、每個 head 都算一遍 → runs/rehearsal/rankings.json 與 *.svg
#   ./present.sh fetch         第一次：下載 Qwen3-0.6B 與 1.7B（要網路；先 unset HF_TOKEN，見 ../README.md）
#   ./present.sh fake          沒有模型也能演練流程（數字是假的）
#   加 --model backup 用 Qwen3-1.7B，例如 ./present.sh rehearse --model backup
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
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
