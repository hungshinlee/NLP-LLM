#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 1（投影片 p19「Live: demo 1 — RoPE's relativity, checked to float64」）課前與課堂的入口。純 torch、釘 CPU、float64，不碰 Hugging Face。
#
#   ./present.sh               課前：對參考實作 rope_final.py 跑三個測試，全部 ok 才上台
#   ./present.sh live          課堂：對 rope_live.py 跑三個測試（寫到哪測到哪，沒寫的標 skip）
#   ./present.sh show          現場卡住兩分鐘：切到參考實作，d_head = 4 的小例子逐步印出 + 三個測試
#   ./present.sh rehearse      課前彩排：三個測試 + 寫 runs/rehearsal/tests.json（講稿裡的實測值從這裡抄）
#   ./present.sh replay        現場連參考實作都跑不了時：顯示彩排紀錄（畫面標明非現場）
#   ./present.sh wavelengths   p20 的現場版：64 對的波長對 L_train，轉不滿一圈的著色，最後對投影片的 w05-data.json（純標準函式庫）
#   ./present.sh reset         上課前把 rope_live.py 清空到只剩開頭的註解與 import torch
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-test}"; shift || true
case "$cmd" in
  wavelengths) exec "$PY" wavelengths.py "$@" ;;        # 不需要 torch，先於下面的檢查
esac
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }

case "$cmd" in
  test)     exec "$PY" rope_final.py "$@" ;;
  live)     exec "$PY" tests.py rope_live "$@" ;;
  show)     exec "$PY" rope_final.py --show "$@" ;;
  rehearse) exec "$PY" rope_final.py --rehearse "$@" ;;
  replay)   exec "$PY" tests.py --replay ;;
  reset)    "$PY" - <<'EOF'
from pathlib import Path
p = Path("rope_live.py")
head = []
for l in p.read_text(encoding="utf-8").splitlines(keepends=True):   # 只留開頭連續的註解行與 import torch
    if not (l.startswith("#") or l.startswith("import torch")):
        break
    head.append(l)
p.write_text("".join(head), encoding="utf-8")
print("rope_live.py 已清空到只剩 %d 行（註解 + import torch）" % len(head))
EOF
            ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
