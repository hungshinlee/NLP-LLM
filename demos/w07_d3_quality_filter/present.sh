#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 3（投影片 p41「Live + replay: demo 3 — guess the source, score it, two curves」；p40 前導頁）課前與課堂的入口。
# torch + transformers（分類器）；訓練走 MPS（沒有就退 CPU）。模型與語料來自 ../w07_d2_scaling_fit（先做完 demo 2 的 fetch --corpus both）。
#
#   ./present.sh fetch               一次、要網路：FineWeb-Edu 分類器（約 440 MB，revision → ../versions.lock）＋ Wikipedia held-out（約 1 MB → ../w07_corpus/）
#   ./present.sh sample [--seed 7] [--exclude id,id]   抽六頁、打亂、存 runs/rehearsal/pages.json，印出來（含來源；課前自己看，踩雷就換 seed）
#   ./present.sh pages               課堂：Enter 一頁，印六頁，不印來源
#   ./present.sh score               課堂：分類器給六頁打分（Enter 一頁）→ 揭曉來源 → held-out 文字的分數
#   ./present.sh rehearse [--steps N]  課前彩排：sample（若還沒有）→ score → 分數分布 → 兩個 run → runs/rehearsal/、runs/weights/；--steps 30 是煙霧測試 → runs/smoke/
#   ./present.sh train edu|web       只訓一邊；之後 ./present.sh summarize
#   ./present.sh summarize           只從兩個 json 重做 summary.json 與 curves.svg
#   ./present.sh replay              課堂：Enter 一步——設定 → 每個 held-out 一格的曲線 → p41 的表
#   ./present.sh show                同 replay，不等 Enter
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch、transformers 由 ../setup.sh 裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  fetch)
    # 唯一要上網的一步。不設 HF_HUB_OFFLINE；GUI 環境注入的 HF_TOKEN=PLACEHOLDER 會讓公開模型也回 401（../README.md），遇到就拔掉。
    if [[ "${HF_TOKEN:-}" == "PLACEHOLDER" ]]; then unset HF_TOKEN; echo "（HF_TOKEN=PLACEHOLDER 已在這個 shell 裡拔掉）" >&2; fi
    exec "$PY" demo.py fetch "$@" ;;
  replay|show|summarize) export HF_HUB_OFFLINE=1; exec "$PY" demo.py "$cmd" "$@" ;;     # 不需要 torch、不需要網路：只讀 runs/ 的 json
esac
export HF_HUB_OFFLINE=1
"$PY" -c "import torch, transformers" 2>/dev/null || { echo "共用環境裡沒有 torch 或 transformers：重跑 ../setup.sh --no-fetch" >&2; exit 1; }
[[ -f ../w07_d2_scaling_fit/train.py ]] || { echo "找不到 demo 2 的 train.py（這個 demo 用它的模型與語料）" >&2; exit 1; }
[[ -f ../w07_corpus/fineweb.jsonl && -f ../w07_corpus/fineweb_edu.jsonl ]] || { echo "缺語料：先 cd ../w07_d2_scaling_fit && ./present.sh fetch --corpus both" >&2; exit 1; }

case "$cmd" in
  sample|pages|score|rehearse|train) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
