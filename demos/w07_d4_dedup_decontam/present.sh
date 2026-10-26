#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 4（投影片 p47「Live: demo 4 — near-duplicates across dumps; MMLU vs. the corpus」；p46 前導頁）課前與課堂的入口。
# 純標準函式庫、CPU，不載模型；只有 fetch 要網路（MMLU 的 test split 與 MMLU-Redux 的 Virology 子集，幾 MB）。
# 語料共用 demo 2 的 ../w07_corpus/（先做完 ../w07_d2_scaling_fit/present.sh fetch --corpus both）。
#
#   ./present.sh fetch               一次、要網路：MMLU test → data/mmlu_test.jsonl；MMLU-Redux Virology → data/mmlu_redux_virology.jsonl
#   ./present.sh test                課前：三個測試對參考實作 dedup_final.py，全部 ok 才上台
#   ./present.sh live                課堂：三個測試對 dedup_live.py（寫到哪測到哪，沒寫的標 skip）
#   ./present.sh run [--impl live|final] [--corpus edu|web] [--dumps all|D1,D2] [--bands 16 --rows 8] [--also 32x4,8x16,14x8]
#                                    (a)：對語料算簽章（第一次約一分鐘、之後讀快取）→ 候選對、Ĵ 分布、兩對並排 → Enter 換 (b, r)
#   ./present.sh decontam [--corpus edu|web|both]
#                                    (b)：MMLU 的 14,042 題對語料掃 13-gram → 命中的題 → Enter → MMLU-Redux 標錯的兩題
#   ./present.sh rehearse            課前彩排：tests → run（edu、web，四組 (b, r)）→ decontam both → runs/rehearsal/*.json（講稿裡的實測值從這裡抄）
#   ./present.sh replay              現場失敗時：讀彩排紀錄，一步一個 Enter（畫面標明非現場）；show 同樣內容不停
#   ./present.sh reset               上課前把 dedup_live.py 清空到只剩開頭的註解、import 與 P = 2**61 - 1
#
# 課堂預設 --impl final；現場寫完三段之後用 ./present.sh run --impl live。
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv（這個 demo 只用標準函式庫；fetch 用 .venv 裡的 pyarrow 與 huggingface_hub）
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-test}"; shift || true
case "$cmd" in
  fetch)
    if [[ "${HF_TOKEN:-}" == "PLACEHOLDER" ]]; then unset HF_TOKEN; echo "（HF_TOKEN=PLACEHOLDER 已在這個 shell 裡拔掉）" >&2; fi
    exec "$PY" demo.py fetch "$@" ;;
  test)     exec "$PY" tests.py dedup_final "$@" ;;
  live)     exec "$PY" tests.py dedup_live "$@" ;;
  run|decontam|rehearse|replay|show)
    export HF_HUB_OFFLINE=1
    if [[ "$cmd" != "replay" && "$cmd" != "show" ]]; then
      [[ -f ../w07_corpus/fineweb_edu.jsonl ]] || { echo "缺語料：先 cd ../w07_d2_scaling_fit && ./present.sh fetch --corpus both" >&2; exit 1; }
    fi
    exec "$PY" demo.py "$cmd" "$@" ;;
  reset)    "$PY" - <<'PYEOF'
from pathlib import Path
p = Path("dedup_live.py")
head = []
for l in p.read_text(encoding="utf-8").splitlines(keepends=True):   # 只留開頭連續的註解行與 import
    if not (l.startswith("#") or l.startswith("import ") or l.startswith("P = ") or l.strip() == ""):
        break
    head.append(l)
p.write_text("".join(head), encoding="utf-8")
print("dedup_live.py 已清空到只剩 %d 行（註解 + import + P）" % len(head))
PYEOF
            ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,16p' "$0" >&2; exit 2 ;;
esac
