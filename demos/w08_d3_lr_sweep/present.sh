#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 demo 3（投影片 p48 前導頁、p49「Replay: demo 3」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh                      課堂 = replay：Enter 第一張圖（最終 loss 對 LR，兩條曲線）→ Enter 第二張圖（固定 LR 對各自最佳）→ p49 的表（讀彩排紀錄，不載模型）
#   ./present.sh inspect              課前：版本、資料與模型在不在、兩組 LR 網格（bf16 之後的值、PEFT 單位換算）、N_LoRA 對 w08-data.json
#   ./present.sh probe                彩排前：每個候選 LR 跑 30 步看發散點與一步幾秒（--steps 50 改步數；--group full 只做一組）→ 估整個掃描要多久
#   ./present.sh rehearse             彩排：base 的 held-out loss → 兩組各五個 run（每個 run 一個子行程）→ 交叉 run → curves.json、best.json；掛掉重跑會接著做
#   ./present.sh rehearse --group lora          只掃一組；--redo 重跑已有的；--no-cross 不做交叉
#   ./present.sh run --group full --lr 1.2e-5   手動補一個 run（之後 ./present.sh aggregate 重算 curves.json）
#   ./present.sh aggregate            只重算 curves.json／best.json
#   ./present.sh fake [子命令]        沒有 mlx 也能演練流程（輸出是假的，畫面會標明）
#
# 資料與模型沿用 demo 1：先在 ../w08_d1_sft_base_to_assistant 跑過 fetch 與 prepare。
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
  replay|inspect|probe|rehearse|run|aggregate) exec "$PY" demo.py "$@" "$cmd" ;;
  fake)   sub="${1:-replay}"; shift || true; exec "$PY" demo.py --fake "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
