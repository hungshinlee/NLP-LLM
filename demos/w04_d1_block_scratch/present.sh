#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 1（投影片 p16「Live: RMSNorm → SwiGLU → Block → GPT」、p20「Live: invert config.json」、p22 的 replay）課前與課堂的入口。
# 純 torch（MPS，測試走 CPU）；count 讀本機的 Hugging Face 快取，一律離線（HF_HUB_OFFLINE=1）。
#
#   ./present.sh test              課前：對參考實作 block_final.py 跑三個測試 (a)(b)(c)，全部 ok 才上台
#   ./present.sh live              課堂（p16）：對 block_live.py 跑三個測試（寫到哪測到哪；(a) 要等 GPT 寫完、(c) 逐段推進）
#   ./present.sh show              現場卡住兩分鐘：切到參考實作，逐段印 shape 與參數表 + 三個測試
#   ./present.sh count llama       課堂（p20）：讀 Llama 3.1 8B 的 config.json 印三欄；count qwen 第二個模型
#   ./present.sh count llama --config-only    只印七個鍵（學生先算第一欄）
#   ./present.sh peek              課堂（有時間才做）：現場跑最初幾百步，看 loss 開始掉；不存檔
#   ./present.sh probe             課前：MPS 可不可用、內建 SDPA 在 MPS 上 dropout_p > 0 的行為、一步要幾秒 → runs/rehearsal/probe.json
#   ./present.sh rehearse          課前彩排：三個測試 → runs/rehearsal/tests.json，再訓練 → runs/rehearsal/train.json + loss.svg
#   ./present.sh rehearse-train    只做訓練那一半
#   ./present.sh replay            現場連參考實作都跑不了時：顯示彩排的三個測試（含 count 的三欄）
#   ./present.sh replay train      課堂（p22）：印彩排的 loss 曲線與 BPB 對 W1 n-gram 的表（畫面標明非現場）
#   ./present.sh reset             上課前把 block_live.py 清到只剩「從這裡開始現場寫」那條線以上
#
# macOS 內建的是 bash 3.2：不要用 ${var,,} 這類 4.x 語法。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 nlp_llm/demos/.venv；torch 由 ../setup.sh 裝（2026-09-28 加）
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh --no-fetch" >&2; exit 1; }
"$PY" -c "import torch" 2>/dev/null || { echo "共用環境裡沒有 torch：重跑 ../setup.sh --no-fetch" >&2; exit 1; }

cmd="${1:-test}"; shift || true
case "$cmd" in
  test)      exec "$PY" block_final.py "$@" ;;
  live)      exec "$PY" tests.py block_live "$@" ;;
  show)      exec "$PY" block_final.py --show "$@" ;;
  count)     exec "$PY" count_params.py "$@" ;;
  peek)      exec "$PY" train.py peek "$@" ;;
  probe)     exec "$PY" train.py probe "$@" ;;
  rehearse)  "$PY" block_final.py --rehearse; exec "$PY" train.py rehearse "$@" ;;
  rehearse-train) exec "$PY" train.py rehearse "$@" ;;
  replay)
    if [[ "${1:-}" == "train" ]]; then exec "$PY" train.py replay; else exec "$PY" tests.py --replay; fi ;;
  reset)     "$PY" - <<'EOF'
from pathlib import Path
p = Path("block_live.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
keep = []
for l in lines:                                    # 留到「從這裡開始現場寫」那條線（含）
    keep.append(l)
    if l.startswith("# ── 從這裡開始現場寫"):
        break
else:
    raise SystemExit("block_live.py 裡找不到「從這裡開始現場寫」那條線，不動它")
p.write_text("".join(keep), encoding="utf-8")
print("block_live.py 已清到第 %d 行（線以上保留：註解、import、CausalSelfAttention）" % len(keep))
EOF
             ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac
