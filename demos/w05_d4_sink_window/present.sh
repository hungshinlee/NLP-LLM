#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 4（投影片 p43「Replay (if time): demo 4 — a sliding window with and without the sink」）課前與課堂的入口。一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh               上課用 = replay：每條線 Enter 一次（分格的 perplexity 方塊圖），最後印 p43 的表；不載模型
#   ./present.sh replay        同上
#   ./present.sh inspect       課前確認：只載 tokenizer，數文字的 token；印 mlx-lm 在哪裡旋轉、RotatingKVCache(keep) 留了什麼；驗 RoPE 可合成／反轉
#   ./present.sh rehearse      課前彩排：同一段文字、四種 cache 的逐 token NLL → runs/rehearsal/{curve.json,curve.svg,tokens.json}
#   ./present.sh rehearse --chars 4000     煙霧測試（只取前 4000 字元，寫到 runs/smoke/）
#   ./present.sh rehearse --check          多跑一條：自己的 cache 關掉重編，對庫的 keep（應一致）
#   ./present.sh fetch         換模型時才需要（要網路）；Llama 3.1 8B 與 Qwen3 8B 已在 W2 demo 3 下載並鎖定
#   ./present.sh fake          沒有 mlx 也能演練 replay（曲線是假的）；fake-rehearse 產生假的 runs/rehearsal/curve.fake.json
#   加 --model backup 用備用模型（Qwen3 8B），例如 ./present.sh rehearse --model backup
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|inspect|rehearse) exec "$PY" demo.py "$@" "$cmd" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py "$@" fetch ;;
  fake)   exec "$PY" demo.py --fake "$@" replay ;;
  fake-rehearse) exec "$PY" demo.py --fake "$@" rehearse ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
