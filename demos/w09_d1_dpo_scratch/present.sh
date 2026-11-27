#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 1（投影片 p33 前導頁、p35「Live: demo 1 (a)」、p36「Replay: demo 1 (b)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live                 課堂 (a)：dpo_live.py 的函數本體寫好之後跑——五個測試，Enter 一個（不載模型）
#   ./present.sh live --ref           寫錯了：用參考實作 dpo_final.py
#   ./present.sh replay               課堂 (b)：讀 runs/rehearsal/summary.json，Enter 逐段印——偏好對、三個 β 的表、曲線、其餘的尺（不載模型）
#   ./present.sh replay-a             現場 (a) 失敗：印彩排的 tests.json（不載模型）
#   ./present.sh test                 不載權重：五個測試（參考實作；骨架寫了也一起測）、規則與配對、亂數小模型上的梯度（幾秒）
#   ./present.sh inspect              課前：版本、模型與 adapter C、一個 Dolly prompt 過 chat template、B = 0 時輸出不變、記憶體裡有幾份權重
#   ./present.sh pairs                彩排第一步：π_ref 對每個 prompt 寫幾段回答 → 規則配對 → π_ref 的兩個 log 機率存檔 → π_ref 的 bits per byte
#   ./present.sh probe                彩排前：一步幾秒、LR 小掃描
#   ./present.sh rehearse             彩排全套：(a) 參考實作 → pairs（已有就跳過）→ 三個 β 各一個子行程（做過的跳過，--force 重跑，--only beta01 只跑一個）→ 彙整
#   ./present.sh aggregate            只重新彙整 runs/rehearsal/runs/*.json → summary.json
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載任何東西；寫到 runs/smoke/）；預設 rehearse
#
# 模型沿用上週 demo 1（Qwen3-1.7B-Base 與 runs/weights/adapters_C），設定讀 demo 2 的 demo_config.toml；這裡沒有東西要 fetch。
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
  live|replay|replay-a|test|inspect|pairs|probe|rehearse|aggregate|run) exec "$PY" demo.py "$@" "$cmd" ;;
  tiny)   sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,14p' "$0" >&2; exit 2 ;;
esac
