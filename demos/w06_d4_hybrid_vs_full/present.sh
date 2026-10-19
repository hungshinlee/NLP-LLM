#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 4（投影片 p37「Replay + live: demo 4 — cache against L, and recall」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh                       上課用：載兩個模型；Enter 一題（兩個模型各答一次）、m 印 (a) 的表、r 印 (b) 的表、x 離開
#   ./present.sh replay                現場失敗時：只印課前彩排的 (a)(b)（不載模型）；replay memory / replay recall 只印一半
#   ./present.sh probe                 課前：每個模型答一個普通問題與 K = 5／20 的小題，看回覆成不成句
#   ./present.sh inspect               課前確認：兩個模型各載一次，印每層 cache 的類型、狀態張量的形狀與 dtype、KV 公式、固定狀態、
#                                      chat template 吃不吃 system、兩檔 prompt 在各自 tokenizer 下的 token 數
#   ./present.sh rehearse              課前彩排：每個模型先 (a) 1k → 32k 各一次 prefill，再 (b) 兩檔 × 五個位置 × Q 題 → runs/rehearsal/
#   ./present.sh rehearse --part memory              只做 (a)；--part recall 只做 (b)
#   ./present.sh rehearse --model falcon_h1          只做一個模型（qwen3_8b / falcon_h1 / mamba2）
#   ./present.sh rehearse --part recall --q 2        煙霧測試（每格 2 題，寫到 runs/smoke/）
#   ./present.sh fetch                 第一次：下載 Falcon-H1-7B-Instruct-4bit（要網路）；Qwen3-8B 已在 W3 demo 2 鎖定
#   ./present.sh fetch --only mamba2   若時間：純 Mamba-2 的第三條線
#   ./present.sh fake                  沒有 mlx 也能演練流程（數字與回答是假的）
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|inspect|probe|rehearse) exec "$PY" demo.py "$@" "$cmd" ;;
  replay)
    part="${1:-}"
    case "$part" in
      memory|recall) shift; exec "$PY" demo.py --part "$part" "$@" replay ;;
      *) exec "$PY" demo.py "$@" replay ;;
    esac ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py "$@" fetch ;;
  fake)   exec "$PY" demo.py --fake "$@" show ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,14p' "$0" >&2; exit 2 ;;
esac
