#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 3（投影片 p58 前導頁、p60「Live: demo 3 (a)」、p63「Replay: demo 3 (b)」、p70 的 (c)）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh live                 課堂 (a)：同一題問兩次，左右各印一段推理 → Enter 答案 → Enter 兩個 prompt（載模型）
#   ./present.sh replay               課堂 (a)：讀 runs/rehearsal/hint.json，Enter 一欄——switch、verbalise、alpha（不載模型）
#   ./present.sh replay-early         課堂 (b)：讀 early.json，推理截在 0–100% 時答案是不是已經定了（不載模型）
#   ./present.sh replay-lang          課堂 (c)：讀 lang.json，指令換語言時推理用什麼文字（不載模型）
#   ./present.sh replay-live          現場那一題失敗：印彩排存的那一對（不載模型；--pick K 看別的候選）
#   ./present.sh candidates           課前：列出現場那一題的候選（不載模型）
#   ./present.sh judge                課前：人工讀三十筆被帶走的推理，印字串規則與人的一致率（不載模型）
#   ./present.sh rescore              只改了 [mention] 的規則：不重新生成，重算 (a) 與候選（不載模型）
#   ./present.sh test                 不載模型：prompt、讀回答、規則、兩個率與 alpha、截點、文字種類，再用假資料走一遍畫面（幾秒）
#   ./present.sh fetch                確認 MMLU 的 test split 在本機（W7 demo 4 抓過就不必連網）
#   ./present.sh inspect              課前：模板、記號、模型自己怎麼寫回答、三種 prompt 的全文
#   ./present.sh probe                彩排前：每一科不加提示的正確率、switch 的小樣本、估時
#   ./present.sh rehearse             彩排全套：(a) → (b) → (c) → 候選；做過的跳過（--force 重跑；--part hint|early|lang|live 只做一段）
#   ./present.sh tiny [子命令]        亂數權重的小模型走真的流程（不下載模型；寫到 runs/smoke/）；預設 rehearse
#
# 模型是 Qwen/Qwen3-8B（W8 demo 2 已經 fetch 過）；題目是 W7 demo 4 抓過的 MMLU test split。
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
  live|replay|replay-early|replay-lang|replay-live|candidates|judge|rescore|test|inspect|probe|rehearse) exec "$PY" demo.py "$@" "$cmd" ;;
  fetch)  HF_HUB_OFFLINE=0 exec "$PY" demo.py --online "$@" fetch ;;
  tiny)   sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --tiny "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,19p' "$0" >&2; exit 2 ;;
esac
