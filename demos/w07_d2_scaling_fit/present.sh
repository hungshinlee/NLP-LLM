#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 2（投影片 p33「Replay: demo 2 — five points, one ray, the extrapolation error」；p32 前導頁）課前與課堂的入口。
# 純 torch（訓練走 MPS，沒有就退 CPU）。模型來自 ../w04_d1_block_scratch（從檔案路徑載入）；語料是 FineWeb-Edu 的抽樣，fetch 一次，之後離線。
#
#   ./present.sh fetch [--corpus edu|web|both] [--mb 200]   只有這一步需要網路：串流 parquet shard → ../w07_corpus/*.jsonl（demo 3、4 共用）
#   ./present.sh estimate            不訓練：五個尺寸的 N、D、步數、lr，與粗估的時間
#   ./present.sh rehearse --steps 30 煙霧測試：五個尺寸各 30 步 + 擬合 + 總表 → runs/smoke/；印每個尺寸的 s/step 與換算的正式 run 分鐘數
#   ./present.sh rehearse            課前彩排：五個尺寸各訓到 D = 20N → runs/rehearsal/{s1..s5}.json + fit.json + summary.json + curves.svg；權重 → runs/weights/
#   ./present.sh train s3            只訓一個尺寸 → runs/rehearsal/s3.json；之後 ./present.sh fit
#   ./present.sh fit                 只從 json 重做擬合、envelope、summary.json 與 curves.svg
#   ./present.sh replay              課堂（p33）：Enter 一步——五個點 → 三點擬合與外插 → 五點擬合 → envelope → p33 的表
#   ./present.sh show                同 replay，不等 Enter
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝；pyarrow（只有 fetch 用）用 uv pip install 另裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  fetch)
    # 唯一要上網的一步。不設 HF_HUB_OFFLINE；這台機器的 GUI 環境曾注入 HF_TOKEN=PLACEHOLDER，會讓公開資料集也回 401（../README.md），遇到就拔掉。
    if [[ "${HF_TOKEN:-}" == "PLACEHOLDER" ]]; then unset HF_TOKEN; echo "（HF_TOKEN=PLACEHOLDER 已在這個 shell 裡拔掉）" >&2; fi
    exec "$PY" demo.py fetch "$@" ;;
  replay|show|fit) export HF_HUB_OFFLINE=1; exec "$PY" demo.py "$cmd" "$@" ;;     # 不需要 torch、不需要網路：只讀 runs/ 的 json
esac
export HF_HUB_OFFLINE=1
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }
[[ -f ../w04_d1_block_scratch/block_final.py ]] || { echo "找不到 W4 demo 1 的 block_final.py（這個 demo 用它的 GPT）" >&2; exit 1; }

case "$cmd" in
  estimate|rehearse|train) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
