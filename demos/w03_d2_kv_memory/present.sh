#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 2（投影片 p25「Live: predict the curve from config.json, then measure it」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用：Enter 一列一列量 KV cache（彩排時太慢的列顯示彩排值並標「非現場」）；q = 離開
#   ./present.sh replay        現場失敗時：只顯示課前彩排的表（畫面標明非現場）
#   ./present.sh inspect       課前確認：config.json 的三個數、KV 的 dtype、每 token 的 bytes、各 context 的預測值
#   ./present.sh rehearse      課前彩排：每個 context 各量一次並錄下（runs/rehearsal/curve.json）
#   ./present.sh fetch         換模型時才需要（要網路）；Llama 3.1 8B 與 Qwen3 8B 已在 W2 demo 3 下載並鎖定
#   ./present.sh fake          沒有 mlx 也能演練流程（數字是假的）
#   加 --model backup 用備用模型（Qwen3 8B），例如 ./present.sh rehearse --model backup
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|inspect|rehearse) exec "$PY" demo.py "$@" "$cmd" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py "$@" fetch ;;
  fake)   exec "$PY" demo.py --fake "$@" show ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
