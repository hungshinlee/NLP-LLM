#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# nlp_llm/demos 所有 demo 共用的環境建置（只在 Apple silicon 的 macOS 上跑）。
#
#   ./setup.sh                  建（或更新）共用的 .venv，裝釘住的版本，下載每個 demo 要用的模型
#   ./setup.sh --no-fetch       只建環境，不下載模型
#   ./setup.sh --only w01_d2_bpb  只替某一個 demo 下載模型
#
# 版本政策（nlp_llm/course-outline.md 附錄 C）：
#   - mlx、transformers、torch 釘版本號；
#   - mlx-lm **釘 git commit，不用 PyPI wheel**（PyPI 版落後 repo 數月，gemma4 / qwen3_5 的支援在 main）。
#     第一次跑時取 main 的 HEAD 並寫進 versions.lock；之後都用鎖定的那個 commit。
#     要升級就刪掉 versions.lock 裡的 MLX_LM_COMMIT 那一行再跑。
#   - 模型 revision 也鎖在 versions.lock（以 repo 為鍵，多個 demo 共用同一個模型時共用同一個鎖）。
# 模型下載到 Hugging Face 的全域快取（~/.cache/huggingface/hub），不在這個資料夾。
set -euo pipefail
cd "$(dirname "$0")"

MLX_VERSION="0.32.2"          # 附錄 C，2026-09-16 查證
TRANSFORMERS_VERSION="5.17.0" # 同上；mlx-lm main 要求 >= 5.7.0
TORCH_VERSION="2.14.0"        # 附錄 C（2026-09-02 查證）；W3 demo 1 的手刻 attention 用，純 CPU。2026-09-28 加，在 Mac 上裝起來之前算未實測
PYTHON_VERSION="3.12"
MLX_LM_REPO="https://github.com/ml-explore/mlx-lm"

FETCH=1; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-fetch) FETCH=0 ;;
    --only) ONLY="$2"; shift ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "這台不是 Apple silicon 的 macOS。mlx 裝不起來；只能用各 demo 的 fake 模式演練流程。" >&2
fi
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
command -v git >/dev/null || { echo "需要 git" >&2; exit 1; }

MLX_LM_COMMIT=""
if [[ -f versions.lock ]]; then
  MLX_LM_COMMIT="$(grep -E '^MLX_LM_COMMIT=' versions.lock | cut -d= -f2 || true)"
fi
if [[ -z "$MLX_LM_COMMIT" ]]; then
  MLX_LM_COMMIT="$(git ls-remote "$MLX_LM_REPO" refs/heads/main | cut -f1)"
  echo "mlx-lm：鎖定 main 目前的 HEAD = $MLX_LM_COMMIT"
fi

[[ -x .venv/bin/python ]] || uv venv --python "$PYTHON_VERSION" .venv
uv pip install --python .venv/bin/python \
  "mlx==${MLX_VERSION}" \
  "transformers==${TRANSFORMERS_VERSION}" \
  "mlx-lm @ git+${MLX_LM_REPO}@${MLX_LM_COMMIT}" \
  "torch==${TORCH_VERSION}" \
  tiktoken opencc-python-reimplemented   # W2 demo 1：GPT-2／o200k 的 tokenizer、繁→簡；版本記在 requirements.lock.txt（2026-09-20 加，未釘）

# 寫回鎖定檔（保留各 demo fetch 寫進去的 MODEL_REVISION__*）。
# 格式與 common.py 的 write_lock 完全相同（一行註解 + 依鍵排序），內容沒變時重跑不會產生 git 變更。
touch versions.lock
{
  grep -E '^[A-Z0-9_]+=' versions.lock | grep -vE '^(MLX_LM_COMMIT|MLX_VERSION|TRANSFORMERS_VERSION|TORCH_VERSION|PYTHON_VERSION)=' || true
  echo "MLX_LM_COMMIT=${MLX_LM_COMMIT}"
  echo "MLX_VERSION=${MLX_VERSION}"
  echo "TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}"
  echo "TORCH_VERSION=${TORCH_VERSION}"
  echo "PYTHON_VERSION=${PYTHON_VERSION}"
} | LC_ALL=C sort > versions.lock.tmp
{
  echo "# 由 demos/setup.sh 與各 demo 的 fetch 產生。要換版本就刪掉對應那一行再重跑。"
  cat versions.lock.tmp
} > versions.lock
rm -f versions.lock.tmp
uv pip freeze --python .venv/bin/python > requirements.lock.txt
echo "完整的套件清單 → requirements.lock.txt"

if [[ "$FETCH" == 1 ]]; then
  for d in w[0-9][0-9]_d[0-9]*/; do
    d="${d%/}"
    [[ -n "$ONLY" && "$d" != "$ONLY" ]] && continue
    [[ -f "$d/demo.py" ]] || continue
    echo; echo "== ${d}：下載模型"
    .venv/bin/python "$d/demo.py" fetch
  done
fi
echo
echo "下一步：到各 demo 資料夾看 README，例如 cd w01_d1_live_model && ./present.sh check"
