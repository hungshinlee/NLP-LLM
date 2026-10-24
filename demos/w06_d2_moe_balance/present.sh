#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 2（投影片 p18「Replay: demo 2 — expert load and loss, three balancing rules」）課前與課堂的入口。
# 純 torch（訓練走 MPS，沒有就退 CPU）；不碰 Hugging Face。模型與語料都來自 ../w04_d1_block_scratch（從檔案路徑載入）。
#
#   ./present.sh estimate            不訓練：用 W4 demo 1 probe.json 的每步秒數估（MoE 會更慢，真正的估計是下一行）
#   ./present.sh rehearse --steps 30 煙霧測試：三組各 30 步 + 全套紀錄，寫到 runs/smoke/；印出的秒數乘 100 就是 rehearse 的時間
#   ./present.sh rehearse            課前彩排：三組各 3000 步 → runs/rehearsal/{none,switch,bias}.json + summary.json + curves.svg
#   ./present.sh train bias          只訓一組（none|switch|bias）→ runs/rehearsal/<setting>.json；之後 ./present.sh summarize
#   ./present.sh summarize           只從三個 json 重寫 summary.json 與 curves.svg
#   ./present.sh replay              課堂（p18）：Enter 一組——(i) → (ii) → (iii)；最後印 p18 的表
#   ./present.sh show                只印表（替代 replay 的退路；不等 Enter）
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  replay|show|summarize) exec "$PY" demo.py "$cmd" "$@" ;;       # 不需要 torch、不需要 GPU：只讀 runs/ 的 json
esac
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }
[[ -f ../w04_d1_block_scratch/block_final.py ]] || { echo "找不到 W4 demo 1 的 block_final.py（這個 demo 用它的 GPT）" >&2; exit 1; }

case "$cmd" in
  rehearse|train|estimate) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
