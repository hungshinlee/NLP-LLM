#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 1（投影片 p9「Live: demo 1 (a)」、p22「Live: demo 1 (b)」）課前與課堂的入口。純標準函式庫，不載模型、不需要網路。
#
#   ./present.sh table           (a) C = 10^18 … 10^26 的 N*、D*、D*/N*、L*（閉式解 Eq. (4)；★ 是 p9 要填的五列）
#   ./present.sh points          (a) 三個真實點（W1 7B／140B、Chinchilla 70B／1.4T、Llama 3 8B／15T）：C、該 C 的最優、loss 差
#   ./present.sh inference       (b) 固定 loss、加推論項 2N·D_inf 重新最佳化：Enter 一列（--all 一次印完）
#   ./present.sh tiny            (c) W4 demo 1 的 tiny GPT 落在表上哪裡（讀 ../w04_d1_block_scratch/runs/rehearsal/）
#   ./present.sh check           對投影片的數字來源 ../../slides/assets/w07/w07-data.json 逐個核對
#   ./present.sh rehearse        課前彩排：四段全印 + check + 寫 runs/rehearsal/{table,points,inference,tiny,check}.json
#   ./present.sh replay [name…]  現場卡住：印彩排紀錄（畫面標明非現場；name 省略＝四段都印）
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv（3.12，有 tomllib）；沒有就退回系統 python3（optimal.py 自帶 toml 的簡易讀法）
if [[ ! -x "$PY" ]]; then
  PY="python3"
  echo "（沒有共用環境 ../.venv，用系統 ${PY}）" >&2
fi

cmd="${1:-help}"; shift || true
case "$cmd" in
  table|points|inference|tiny|check|rehearse|replay)
            HF_HUB_OFFLINE=1 exec "$PY" optimal.py "$cmd" "$@" ;;
  help|-h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
