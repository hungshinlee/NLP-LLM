#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 1（投影片 p21「Live: attention and a KV cache, from scratch」）課前與課堂的入口。純 torch、釘 CPU，不碰 Hugging Face。
#
#   ./present.sh               課前：對參考實作 attention_final.py 跑五個測試，全部 ok 才上台
#   ./present.sh live          課堂：對 attention_live.py 跑五個測試（寫到哪測到哪，沒寫的標 skip）
#   ./present.sh show          現場卡住兩分鐘：切到參考實作，小例子逐步印出 + 五個測試
#   ./present.sh rehearse      課前彩排：五個測試 + 寫 runs/rehearsal/tests.json（講稿裡的實測值從這裡抄）
#   ./present.sh replay        現場連參考實作都跑不了時：顯示彩排紀錄（畫面標明非現場）
#   ./present.sh reset         上課前把 attention_live.py 清空到只剩開頭的註解
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝（2026-09-28 加）
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch（setup.sh 於 2026-09-28 加了 torch）" >&2; exit 1; }

cmd="${1:-test}"; shift || true
case "$cmd" in
  test)     exec "$PY" attention_final.py "$@" ;;
  live)     exec "$PY" tests.py attention_live "$@" ;;
  show)     exec "$PY" attention_final.py --show "$@" ;;
  rehearse) exec "$PY" attention_final.py --rehearse "$@" ;;
  replay)   exec "$PY" tests.py --replay ;;
  reset)    "$PY" - <<'EOF'
from pathlib import Path
p = Path("attention_live.py")
head = []
for l in p.read_text(encoding="utf-8").splitlines(keepends=True):   # 只留開頭連續的註解行
    if not l.startswith("#"):
        break
    head.append(l)
p.write_text("".join(head), encoding="utf-8")
print("attention_live.py 已清空到只剩 %d 行註解" % len(head))
EOF
            ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac
