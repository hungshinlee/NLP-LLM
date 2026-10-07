# NLP-LLM — 自然語言處理與大型語言模型

**Natural Language Processing and Large Language Models** · English version: [README.md](README.md)

碩士班／博士班課程：14 週 × 3 小時，全為 lectures。中文授課，投影片與課程網站為英文。

**課程網站：<https://hungshinlee.github.io/NLP-LLM/>**
授課教師：[李鴻欣 Hung-Shin Lee](https://web.ntnu.edu.tw/~hslee/)

這個 repo 放的是課程**對外公開**的部分——網站原始碼、投影片、課堂 demo 程式與講義。產生這些內容的教學材料（完整課程大綱、講稿、題目與 rubric）在另一個 private repo，不在這裡；見〈[刻意不放在這裡的東西](#刻意不放在這裡的東西)〉。

## 一句話說這門課

對現代 LLM 的每一層——從 tokenizer 到 agent loop——說清楚**它為什麼長成這樣、以及它在哪裡會壞掉**。每個元件都從它所解決的具體失效講起，而不是從定義講起。深度設定在研究所：含推導骨架、每個想法背後的論文脈絡與 open problems，對齊 2025–2026 的技術狀態。

三條主軸貫穿十四週，對每一層都問一次：**Representation**（這一層的單位是什麼——byte、token、hidden state、KV entry、retrieved chunk——誰選的、下游付出什麼代價）、**Compute**（這個方法花的是訓練算力還是推論算力、受限於 FLOPs 還是 memory bandwidth）、**Supervision**（這個能力從哪裡來——pre-training 分佈、偏好標注、可驗證的 reward、檢索到的知識，還是 test-time 的額外算力）。

## 課程地圖

| Part I — Architecture | Part II — Training | Part III — Systems & Scrutiny |
|---|---|---|
| W1 從 n-gram 到 seq2seq：一條被壓縮的技術史 | W7 Pre-training：scaling laws 與資料策展 | W11 推論效率：從 FLOPs 到 memory bandwidth |
| W2 Tokenization：模型看到的世界，以及它看不到的 | W8 Post-training I：SFT、PEFT 與遺忘 | W12 RAG 與 context engineering |
| W3 Transformer I：attention 的機制、複雜度與手刻 | W9 Post-training II：偏好對齊與強化學習 | W13 Agentic systems：工具、規劃、失效與安全 |
| W4 Transformer II：完整 block、參數記帳與訓練動力學 | W10 Reasoning 與 test-time compute | W14 評估、可解釋性、安全，與 open problems |
| W5 位置編碼、長文本與 context 的極限 | | |
| W6 架構路線：MoE、SSM 與 linear attention | | |

**W3 與 W4 是全課樞紐。** Part III 的每一週都建立在那兩週建出的兩個物件上——residual stream 與 KV cache——所以這兩週不能缺席。

## 目前已公開的內容

週次隨授課進度逐週開放。截至 2026 年 10 月，**W1–W4** 已開放：每週頁面、投影片，以及每一週的 demo 程式。其餘週次在課程地圖上以灰色占位顯示；頁面已產生但不 render，所以沒有死連結。

| | W1 | W2 | W3 | W4 | W5–W14 |
|---|---|---|---|---|---|
| 每週頁面（定位、learning objectives、參考資料） | ✓ | ✓ | ✓ | ✓ | 已產生，尚未公開 |
| 投影片（`slides/wNN.qmd`，reveal.js） | ✓ | ✓ | ✓ | ✓ | — |
| Demo 程式（`demos/wNN_dK_*/`） | ✓ | ✓ | ✓ | ✓ | — |
| 講義 | [數學先修（PDF）](handouts/w01-math-prerequisites.pdf) | | | | |

**Syllabus** 頁有算力帳本（全課共用的座標系）、讀任何一篇論文時要問的四個問題，以及**半衰期地圖**——標明哪些週次幾年不用動、哪些每次開課前要重掃 arXiv `cs.CL` / `cs.LG` 與 ACL / EMNLP / NeurIPS / ICLR / ICML / COLM 的最新議程。**Resources** 頁列核心教科書、每週一篇論文、推導與手刻元件清單，以及 demo 跑在上面的工具鏈。**Supplements** 有四份中文指南：期中 PoC 報告、期末論文、論文寫作，以及十個以單張 16 GB GPU 為預算的研究題目。

## 給修課同學

- **你不需要跑任何東西。** 本課全為 lectures，所有 demo 由授課者現場執行；投影片上的每個數字都來自 `demos/*/runs/rehearsal/` 裡進版控的彩排紀錄。程式公開是讓你能讀、能重跑、能改。
- **投影片沒有講稿。** 在 deck 裡按 `S` 打開的講稿欄是空的——那是刻意的，不是壞掉：中文講稿屬於教學材料，不公開。
- **Demo 只在一台機器上測過**——MacBook Pro M5 Max（64 GB 統一記憶體、無 CUDA）。`demos/README.md` 說明哪些 demo 可攜（純標準函式庫或純 CPU 的 PyTorch）、哪些需要 Apple silicon（`mlx`）；所有套件與模型版本釘在 `demos/versions.lock`。
- **引用逐篇查證過。** 站上每一筆引用都已對 arXiv 摘要頁、ACL Anthology、PMLR 或 Crossref 比對過 title 與第一作者。查不到的條目直接不上站，而不是附一句但書放上去。來源若是 blog、model card 或規格文件而非同儕審查論文，句子本身會寫明——這在本領域特別重要，有幾個廣為使用的術語與 baseline 背後根本沒有技術報告。
- 網站有**展示模式**（按 `z` 或點 navbar 的按鈕），會收起兩側欄位方便投影。

## Repo 結構

這個 repo 大部分是**產物**。標 ⚙︎ 的檔案在下次發佈時會被覆蓋，不要手改。

| 路徑 | 說明 |
|---|---|
| `index.qmd`、`syllabus.qmd`、`resources.qmd`、`slides.qmd`、`supplements.qmd` | 手寫的站台頁面（英文） |
| `docs/site-en.md` | Syllabus 與 Resources 引用的英文段落來源（算力帳本、四個問題、教科書、每週一篇論文、推導、工具鏈） |
| `weeks/w01.qmd` … `w14.qmd` | ⚙︎ 每週頁面，由 `scripts/build_weeks.py` 從 private 大綱過濾產生 |
| `_includes/*.md` | ⚙︎ 產生的片段：課程地圖、週次表、閱讀表、推導表、帳本、教科書、工具鏈 |
| `slides/wNN.qmd`、`slides/theme.scss`、`slides/assets/` | ⚙︎ **剝除講稿**後的投影片，從 private 正本發佈 |
| `demos/` | ⚙︎ 課堂 demo 程式、彩排紀錄與學生版 README，白名單制發佈；入口是 `demos/README.md` |
| `handouts/*.pdf` | ⚙︎ 從 private repo 複製過來的講義 |
| `supplements/*.md` | 手寫的補充教材（中文）；旁邊的 `.qmd` 包裝是 ⚙︎ 產物 |
| `scripts/build_weeks.py`、`scripts/visibility.py` | 過濾器：大綱哪些區塊公開（預設不公開、白名單制）、引用標記怎麼處理、哪些週次已開放 |
| `styles.scss`、`_present-mode.html`、`_quarto.yml` | 站台主題、展示模式、Quarto 設定 |
| `.github/workflows/publish.yml` | CI：洩漏檢查加 `quarto render`，部署到 GitHub Pages |
| `CLAUDE.md` | 維護筆記：建置流程、渲染陷阱、決策 |

## 刻意不放在這裡的東西

課程大綱（`course-outline.md`）是這門課唯一的真相來源，只存在於 private repo。站上的每週頁面是從它經**預設不公開的過濾器**產生的：每週只有三個區塊公開——*定位*、*Learning objectives*、*參考資料*——加上連到已公開程式的學生版 *Demos* 區塊。課堂時間分配、學生誤解清單、給授課者的 demo 腳本與教學設計備註都不公開。投影片裡的中文講稿、授課者版的 demo README（彩排筆記與退路）、所有評量材料（題目與 rubric）亦同。

若這個 repo 裡出現 `docs/course-outline.md`，那是回歸 bug，不是功能。

## 網站怎麼建

正常入口是 private repo 的 `bin/publish.py`。它一次做完：從投影片正本重抽講稿、發佈剝除講稿的投影片、複製白名單上的 demo 與講義、跑每週頁面的過濾建置，最後檢查沒有教師端內容外洩——產物裡只要殘留一個 `::: {.notes}` 區塊就中止發佈。它**不會**自動 commit 或 push，那兩步手動做，這樣一定看得到 diff。

```bash
cd ~/Course-Hub && python3 bin/publish.py nlp_llm
```

只要從大綱重建每週頁面時（這條路不更新投影片與 demo）：

```bash
export COURSE_OUTLINE=~/Course-Hub/nlp_llm/course-outline.md
python3 scripts/build_weeks.py   # 改完大綱後必跑
quarto preview                   # 本機預覽（brew install --cask quarto）
quarto render                    # 產出 _site/
```

Push 到 `main` 觸發 GitHub Actions，但 CI **不重建**每週頁面（它讀不到大綱），只做洩漏檢查與 `quarto render`。這個設計的代價是：忘記在本機重跑建置，網站就會漂移。

開放新的一週要同時改兩處：`scripts/build_weeks.py` 的 `PUBLISHED_WEEKS` 加上該週（控制首頁、課程地圖與 resources 是否給連結），以及 `_quarto.yml` 的 `render` 清單與 sidebar 加上該頁（控制頁面是否存在）。

## 語言

站台骨架與已公開每週頁的內文都是英文。每週的英文版寫在大綱裡與中文並列，建置時抽出；沒有英文區塊的週次退回中文而不會壞掉，建置結束會印出尚待翻譯的週次清單。兩處刻意保留中文：每週頁的 `subtitle`（中文週次標題，供對照）與 `supplements/` 的指南，後者對齊授課語言。
