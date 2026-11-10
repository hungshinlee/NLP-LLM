#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 demo 2（投影片 p32 前導頁、p33「Live: demo 2 (a)」、p41「Replay: demo 2 (b)」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
# (a) 現場的算術——純標準函式庫，不載模型：
#   ./present.sh count                課前：config.json 從哪裡讀到、逐矩陣數 N、對 safetensors 的參數總數
#   ./present.sh table                p33：先印 N 與 N_LoRA，Enter 之後印四列（--all 不停；--modules q,v、--rank 8 --layers 16 換設定）
#   ./present.sh flops                p33：每個 token 的 FLOPs，6N 對 4N + 6 N_LoRA
#   ./present.sh mlx                  同一本帳照 mlx 實際存的 dtype 再算一次（(b) 的公式欄）
#   ./present.sh files                本機的權重檔依 dtype 加總（只讀 safetensors 的 header）
#   ./present.sh check                對投影片的數字來源 w08-data.json 逐個核對
# (b) 實測的峰值——MLX：
#   ./present.sh                      課堂 = replay：Enter 一列印 p41 的表，再印拆解（讀彩排紀錄，不載模型）
#   ./present.sh adam                 現場可跑（一秒）：一個 bf16 權重做一步 Adam——m、v 的 dtype、權重動了沒有
#   ./present.sh inspect              課前：版本、機器、模型在不在、每個設定的公式值、要跑哪幾個 run
#   ./present.sh fetch                下載缺的模型（Qwen/Qwen3-8B 的 bf16 約 16.4 GB；這一個會連網路）
#   ./present.sh rehearse             彩排：(a) 全部 → adam → 九個 run（各一個子行程）→ runs/rehearsal/；掛掉重跑會接著做
#   ./present.sh rehearse --setting qlora_8b    只做一個設定；--redo 重跑已有的
#   ./present.sh run --setting lora_8b --seq 1024 --grad-checkpoint     手動補一個 run（之後 ./present.sh aggregate）
#   ./present.sh tiny [子命令]        不下載任何東西：亂數權重的小模型走一遍真的 mlx 流程（檢查程式用）→ runs/tiny/
#   ./present.sh fake [子命令]        沒有 mlx 也能演練流程（數字是假的，畫面會標明）→ runs/fake/
#
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
  count|table|flops|mlx|files|check) exec "$PY" ledger.py "$cmd" "$@" ;;
  replay|adam|inspect|run|aggregate) exec "$PY" demo.py "$@" "$cmd" ;;
  fetch)    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; exec "$PY" demo.py --online "$@" fetch ;;
  rehearse) "$PY" ledger.py rehearse; "$PY" demo.py "$@" inspect; exec "$PY" demo.py "$@" rehearse ;;
  tiny)     sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --tiny "$@" "$sub" ;;
  fake)     sub="${1:-rehearse}"; shift || true; exec "$PY" demo.py --fake "$@" "$sub" ;;
  help|-h|--help) sed -n '2,24p' "$0"; exit 0 ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,24p' "$0" >&2; exit 2 ;;
esac
