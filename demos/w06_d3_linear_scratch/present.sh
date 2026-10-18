#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 3（投影片 p24「Live: demo 3 — equal to float64, a crossover, a notebook filling up」）課前與課堂的入口。
# 純 torch、釘 CPU、float64，不碰 Hugging Face、不載模型。
#
#   ./present.sh               課前：對參考實作 linear_final.py 跑四個測試，全部 ok 才上台
#   ./present.sh live          課堂：對 linear_live.py 跑四個測試（寫到哪測到哪，沒寫的標 skip）
#   ./present.sh show          現場卡住兩分鐘：切到參考實作，L = 3、d_head = 2 的小例子逐步印出 + 四個測試
#   ./present.sh rehearse      課前彩排：四個測試 + 寫 runs/rehearsal/tests.json（講稿裡的實測值從這裡抄）
#   ./present.sh replay        現場連參考實作都跑不了時：顯示彩排紀錄（畫面標明非現場）
#   ./present.sh reset         上課前把 linear_live.py 清空到只剩開頭的註解與兩行 import
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法；$變數 後面不要緊接中文或全形標點，一律 ${var}。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-test}"; shift || true
case "$cmd" in
  replay) exec "$PY" tests.py --replay ;;             # 只讀 json，先於 torch 檢查
esac
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }

case "$cmd" in
  test)     exec "$PY" linear_final.py "$@" ;;
  live)     exec "$PY" tests.py linear_live "$@" ;;
  show)     exec "$PY" linear_final.py --show "$@" ;;
  rehearse) exec "$PY" linear_final.py --rehearse "$@" ;;
  reset)    "$PY" - <<'EOF'
from pathlib import Path
p = Path("linear_live.py")
head = []
for l in p.read_text(encoding="utf-8").splitlines(keepends=True):   # 只留開頭連續的註解行與兩行 import
    if not (l.startswith("#") or l.startswith("import torch")):
        break
    head.append(l)
p.write_text("".join(head), encoding="utf-8")
print("linear_live.py 已清空到只剩 %d 行（註解 + import）" % len(head))
EOF
            ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
