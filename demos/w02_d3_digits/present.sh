#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W2 demo 3（投影片 p43「Live: one model, one set of sums, three ways of writing the numbers」）課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用：載入主模型，Enter 逐列跑（4／7／10 位數，各 n_live 題）；q = 對照組、f = 彩排完整表、x = 離開
#   ./present.sh replay        現場失敗時：顯示課前彩排的表（畫面標明非現場）
#   ./present.sh inspect       課前：只載 tokenizer，看三種寫法各怎麼切、thinking 關不關得掉
#   ./present.sh scan          課前選模型：每個候選 × 三種位數 × 三種寫法各 n_scan 題（--only <key> 只跑一個）
#   ./present.sh rehearse      課前彩排：main 與 control 各一張完整的表（n_rehearse 題／格），錄進 runs/rehearsal/
#   ./present.sh fetch         課前一週：下載候選模型（要網路；--only <key> 只抓一個）
#   ./present.sh fake          沒有模型也能演練流程（正確率是假的）
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|inspect|scan|rehearse) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  fake)   exec "$PY" demo.py --fake show "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
