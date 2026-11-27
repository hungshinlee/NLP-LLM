#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 2（投影片 p52 前導頁、p54「Live: demo 2 (a)」、p55「Replay: demo 2 (b)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live                 課堂 (a)：advantage_live.py 的五行寫好之後跑——一題 G 段回答，六列逐列印，最後一列前停下來等 Enter
#   ./present.sh live --ref           寫錯了：用參考實作 advantage_final.py
#   ./present.sh replay               課堂 (b)：讀 runs/rehearsal/summary.json，Enter 印 G 的掃描、正確率對步數、開關（不載模型）
#   ./present.sh replay-a             現場 (a) 失敗：印彩排的 identity.json（不載模型）
#   ./present.sh test                 不載模型：advantage 的性質、恆等式（亂數向量）、verifier、題目產生器（幾秒）
#   ./present.sh inspect              課前：版本、模型與 adapter C、chat template、停止記號、旁路的參數量、B = 0 時輸出不變
#   ./present.sh probe                彩排前：每一級題目上 π_ref 的正確率、一步幾秒與整套估時、LR 小掃描（--part levels|timing|lr 只做一段）
#   ./present.sh rehearse             彩排全套：(a) 參考實作 → (b) 八個 run（各一個子行程；做過的跳過，--force 重跑，--only G8 只跑一個）→ 彙整
#   ./present.sh aggregate            只重新彙整 runs/rehearsal/runs/*.json → summary.json
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 模型沿用上週 demo 1（Qwen3-1.7B-Base 與 runs/weights/adapters_C）；這裡沒有東西要 fetch。
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
  live|replay|replay-a|test|inspect|probe|rehearse|aggregate|run) exec "$PY" demo.py "$@" "$cmd" ;;
  tiny)   sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
