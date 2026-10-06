#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 2（投影片 p29「Replay: pre-LN vs. post-LN, no warmup」、p44「Replay: push the learning rate … then QK-norm」）課前與課堂的入口。
# 純 torch（MPS），用 demo 1 的 tiny GPT 與語料；不載任何預訓練模型、不碰網路（HF_HUB_OFFLINE=1 只是慣例）。
#
#   ./present.sh estimate          課前：不訓練，用 demo 1 probe.json 的每步秒數估三組要跑多久
#   ./present.sh rehearse          課前彩排：三組全部跑 → runs/rehearsal/{i,ii,iii}.json + .svg（幾十分鐘）
#   ./present.sh rehearse i        只跑設定 (i)：pre-LN vs. post-LN、不加 warmup；--depth 24、--lr 3e-3、--steps 1500 可覆蓋
#   ./present.sh rehearse ii       只跑設定 (ii)：LR 掃 {1,3,10}×；--mult 1,3,10,30 加一條
#   ./present.sh rehearse iii      只跑設定 (iii)：最高 LR 開 QK-norm；--z-loss 1e-4 多跑一條加 z-loss 的
#   ./present.sh replay i          課堂（p29）：表 + 兩條 loss／梯度範數曲線 + 第 0 步逐層梯度範數
#   ./present.sh replay ii         課堂（p44 前三列）：表 + 四條統計量曲線
#   ./present.sh replay iii        課堂（p44 第四列）：10× + QK-norm，與 (ii) 的 10× 並排
#   ./present.sh show i|ii|iii     只印表格（退路；投影機上曲線看不清時）
#   open runs/rehearsal/ii.svg     四個 panel 的靜態圖（replay 最後一行印路徑）
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝（2026-09-28 加）
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }
[[ -f ../w04_d1_block_scratch/block_final.py ]] || { echo "找不到 demo 1 的 block_final.py（這個 demo 用它的 GPT）" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  estimate)  exec "$PY" demo.py estimate ;;
  rehearse)  exec "$PY" demo.py rehearse "$@" ;;
  replay)    exec "$PY" demo.py replay "$@" ;;
  show)      exec "$PY" demo.py show "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,14p' "$0" >&2; exit 2 ;;
esac
