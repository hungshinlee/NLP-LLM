#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 2（投影片 p26「Replay: demo 2 — NoPE vs. RoPE, then PI and NTK-aware without retraining」）課前與課堂的入口。
# 純 torch（訓練走 MPS，沒有就退 CPU）；不碰 Hugging Face。模型與語料都來自 ../w04_d1_block_scratch（從檔案路徑載入）。
#
#   ./present.sh rehearse            課前彩排（約 20 分鐘）：訓 nope、訓 rope → runs/weights/；量四組 → runs/rehearsal/*.json + curves.svg
#   ./present.sh train rope          只訓一個版本（nope|rope）→ runs/weights/<variant>.pt
#   ./present.sh measure             只量（讀 runs/weights/）：nope、rope、rope+PI、rope+NTK → runs/rehearsal/
#   ./present.sh replay              課堂（p26）：Enter 一條線——nope → rope → PI → NTK；最後印 p26 的表
#   ./present.sh show                只印表（替代 replay 的退路；不等 Enter）
#   ./present.sh estimate            不訓練：用 W4 demo 1 probe.json 的每步秒數估 rehearse 要多久
#   ./present.sh rehearse --steps 30 煙霧測試：30 步 + 全套量測，寫到 runs/smoke/
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  replay|show) exec "$PY" demo.py "$cmd" "$@" ;;       # 不需要 torch、不需要 GPU：只讀 runs/rehearsal/ 的 json
esac
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }

case "$cmd" in
  rehearse|train|measure|estimate) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
