#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 1（投影片 p12「Live: demo 1 — N_total, N_active, ρ from the config」）課前與課堂的入口。純標準函式庫，不載任何模型。
#
#   ./present.sh fetch             課前（要網路）：抓三個遠端模型的 config.json、safetensors index 與每個 shard 的 header（不抓權重）→ configs/
#   ./present.sh keys qwen30       課堂第一步：印 config 裡所有含 expert／moe／shared／router 的鍵，再印這支程式怎麼解讀
#   ./present.sh keys gemma        同上，本機的 Gemma 4（鍵在 text_config 底下、top_k_experts、dense 的 intermediate_size 並存）
#   ./present.sh count             四列：逐項記帳（config 的式子 vs. safetensors 的形狀）→ N_total、N_active、ρ；最後一張總表照投影片的欄
#   ./present.sh count qwen30      只算一個（名字見 demo_config.toml：gemma、qwen30、qwen_next、kimi）
#   ./present.sh floor qwen30      地板：attention + norm + embedding 佔 N_active 的比例——接 p14
#   ./present.sh rehearse          課前彩排：count 全部 + 寫 runs/rehearsal/ratios.json（講稿裡的實測值從這裡抄）
#   ./present.sh replay            現場卡住：印彩排紀錄（畫面標明非現場）
#
# 除了 fetch 之外一律 HF_HUB_OFFLINE=1（本機 Gemma 4 只讀 HF cache 的 config.json 與 *.safetensors 的 header）。
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv（只為了問 huggingface_hub 本機快取的位置；沒有就退回系統 python3 + HF_HOME）
if [[ ! -x "$PY" ]]; then
  PY="python3"
  echo "（沒有共用環境 ../.venv，用系統 ${PY}；本機模型的位置改由 HF_HOME 決定）" >&2
fi

cmd="${1:-help}"; shift || true
case "$cmd" in
  fetch)    exec "$PY" fetch_configs.py "$@" ;;
  keys|count|floor|rehearse|replay)
            HF_HUB_OFFLINE=1 exec "$PY" count_moe.py "$cmd" "$@" ;;
  help|-h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
