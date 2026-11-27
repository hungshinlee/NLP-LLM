#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 3（投影片 p61 前導頁、p63「Replay: demo 3」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh replay               課堂：讀 runs/rehearsal/summary.json，Enter 逐段印——漏洞版的起點與終點、兩個分數對 √KL、
#                                     同一題訓練前後的回答、嚴格版的對照、硬幣獎勵的兩列（不載模型）
#   ./present.sh test                 不載權重：verifier、硬幣、clip 的目標函數、亂數小模型上的梯度與 KL（幾秒）
#   ./present.sh inspect              課前：沿用了 demo 2 的哪些設定；π_ref 在兩個 verifier 下各幾分；B = 0 時 KL 是 0
#   ./present.sh rehearse             彩排：leaky、strict、coin_1、coin_k 各一個子行程（做過的跳過，--force 重跑，--only count 只跑一個）→ 彙整
#   ./present.sh aggregate            只重新彙整 runs/rehearsal/runs/*.json → summary.json
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 模型、adapter、題目的級別與 learning rate 都讀 demo 2 的設定（../w09_d2_grpo_scratch/demo_config.toml）；這裡沒有東西要 fetch。
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-replay}"; shift || true
case "$cmd" in
  replay|test|inspect|rehearse|aggregate|run) exec "$PY" demo.py "$@" "$cmd" ;;
  tiny)   sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
