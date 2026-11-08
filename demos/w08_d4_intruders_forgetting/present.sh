#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 demo 4（投影片 p54 前導頁、p55「Live: demo 4 (a)」、p58「Replay: demo 4 (b)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh                      課堂 (a) = intruders：對 demo_config [intruders].matrices 的矩陣，數 full 與 LoRA 各有幾個 intruder dimension（p55 的表）。
#                                     函數是 intruders_live.py（現場寫）；寫錯了 ./present.sh intruders --ref 用參考實作；不載 mlx，純 numpy
#   ./present.sh replay               課堂 (b)：讀 runs/rehearsal/forgetting.json，Enter 一列印 p58 的表（不載模型）
#   ./present.sh replay --intruders   現場 (a) 失敗時：印彩排的 intruders.json
#   ./present.sh inspect              課前：版本、demo 3 的兩個權重資料夾、矩陣鍵名與形狀、四段 held-out 的 byte 數
#   ./present.sh scan                 彩排前：所有層 × q_proj／down_proj 的 intruder 數（挑 p55 要看的層；--layers 0,7,14 --names mlp.down_proj --float32 縮範圍）
#   ./present.sh forget               彩排 (b)：三個模型 × 四段文字的 bits-per-byte → runs/rehearsal/forgetting.json
#   ./present.sh rehearse             彩排全套：inspect → intruders --ref --rehearsal → forget（--intervene 加做介入）
#   ./present.sh intervene            可選：intruder 方向的奇異值乘 λ 再量 (b)（未在 Mac 實測）
#   ./present.sh fake [子命令]        沒有 mlx 也能演練流程（輸出是假的，畫面會標明）
#
# 權重沿用 demo 1（base）與 demo 3（runs/rehearsal/best.json 指向的兩個 run）；held-out 文字沿用 W1 demo 2、W2 demo 1、W7 demo 3（W7 的 fetch 要跑過）。
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-intruders}"; shift || true
case "$cmd" in
  intruders|replay|inspect|scan|forget|rehearse|intervene) exec "$PY" demo.py "$@" "$cmd" ;;
  fake)   sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --fake "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
