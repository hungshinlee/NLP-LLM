#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 demo 1（投影片 p8 前導頁、p9「Live: demo 1 (a)」、p15「Live + replay: demo 1 (b) and (c)」、p42「Live: demo 1 (d)」）課前與課堂的入口。
# 一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh                      課堂 (a)：載 base 與 base+adapter C（demo_config 的 show_adapter；--adapter A 換）；三題，每題 Enter 印 base 的原文續寫、再 Enter 印 SFT 後的回答，標有沒有自己停
#   ./present.sh show --base-templated  同上，多印 base 套 chat template 的那一欄
#   ./present.sh mask                 課堂 (b)：一筆訓練資料逐 token 標 loss（default 對 --mask-prompt）；base 在 template 三個位置的探針
#   ./present.sh replay-mask          課堂 (c)：A／B／C 的 held-out response loss 與三題輸出並排（讀彩排紀錄，不載模型）
#   ./present.sh live-train           課堂 (d)：投影片那一行指令只跑 12 步，每 4 步印 train loss 與 peak memory
#   ./present.sh replay               現場失敗：印 (a) 的彩排紀錄（不載模型）
#   ./present.sh fetch                第一次：下載 Qwen3-1.7B-Base（約 3.4 GB）與 databricks-dolly-15k（13 MB），revision 寫進 ../versions.lock
#   ./present.sh prepare              抽 data/{train,valid,test}.jsonl 與 lora.yaml／lora_embed.yaml（純字元過濾 + seed；已進版控，重跑應該零 diff）
#   ./present.sh inspect              課前：版本、config、chat template、停止記號、N 與 N_LoRA 對 w08-data.json、資料的 token 長度
#   ./present.sh rehearse             課前彩排全套：訓 A／B／C → (a) → (b) → (c) → (d)；2026-11-08 實測一個 adapter 約 6.5 分鐘、生成 1–4 s 一題、held-out 10–16 s 一次
#   ./present.sh rehearse --part train --adapter C      只訓 C（--part 可重複：--part generate --part mask）
#   ./present.sh fake [子命令]        沒有 mlx 也能演練流程（輸出是假的，畫面會標明）
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
  show|mask|replay-mask|live-train|replay|inspect|rehearse|prepare) exec "$PY" demo.py "$@" "$cmd" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py --online "$@" fetch ;;
  fake)   sub="${1:-show}"; shift || true; exec "$PY" demo.py --fake "$@" "$sub" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac
