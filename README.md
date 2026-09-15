# NLP-LLM

**自然語言處理與大型語言模型**（Natural Language Processing and Large Language Models）
碩士班／博士班課程；14 週 × 3 小時，全為 lectures。中文授課、英文投影片。

課程網站：**<https://hungshinlee.github.io/NLP-LLM/>**

課程以「理解一個現代 LLM 從 tokenizer 到 agent 的每一層為什麼長成這樣、以及它在哪裡會壞掉」為終點。每個元件都從它所解決的**具體失效**講起，而不是從定義講起。內容含推導骨架、論文脈絡與 open problems，切合 2025–2026 的技術狀態。

## 課程地圖

```
 Part I — 架構 (W1–W6)        Part II — 訓練 (W7–W10)      Part III — 系統與檢驗 (W11–W14)
 W1 從 n-gram 到 seq2seq      W7  Pre-training 與 scaling   W11 推論效率
 W2 Tokenization              W8  SFT、PEFT 與遺忘          W12 RAG 與 context engineering
 W3 Attention 機制與手刻      W9  偏好對齊與 RL             W13 Agentic systems
 W4 完整 block 與訓練動力學   W10 Reasoning 與 test-time     W14 評估、可解釋性、安全
 W5 位置編碼與長文本               compute
 W6 MoE、SSM 與 linear attention
```

**W3 與 W4 是全課樞紐**——W5、W6、W11 完全建立在 residual stream 與 KV cache 的理解上，不可壓縮。

## 檔案結構

| 路徑 | 說明 |
|---|---|
| `docs/site-en.md` | `syllabus` / `resources` 的英文片段來源（手寫，8 段） |
| `docs/course-map-en.md` | 課程地圖的英文 ASCII 版，保留備查，**已不上站**（首頁改用 HTML grid） |
| `weeks/`、`_includes/`、`slides.qmd`、`supplements/*.qmd` | ⚙︎ 由 `scripts/build_weeks.py` 自動產生，**勿手改** |
| `slides/` | ⚙︎ 由 `bin/publish_slides.py` 從 private repo 的正本**剝除中文講稿後**產生，**勿手改** |
| `supplements/*.md` | ★ 補充教材的來源（手寫，中文）。H1 下方需有 `en` / `order` / `summary` 三行註解 |
| `scripts/visibility.py` | 決定每週頁面哪些區塊會公開（預設不公開，白名單制） |
| `CLAUDE.md` | 站台建置、渲染陷阱與維護紀律 |

## 建置

課程大綱正本在 private repo（Course-Hub），**不在這個 repo 裡**。網站的每週頁面
由腳本從大綱**過濾後**產生——只公開定位、Learning objectives、參考資料三個區塊；
課堂時間分配、常見誤解排除、demo 腳本屬於授課用，不上站。

正常入口是 private repo 的 `bin/publish.py`——它一次做完重抽講稿、發佈剝除講稿的
投影片、過濾建置、洩漏檢查四件事，但**不會**自動 commit／push：

```bash
cd ~/Course-Hub && python3 bin/publish.py nlp_llm
```

只要重建每週頁面時也可以直接跑這裡的腳本（這條路不會更新投影片）：

```bash
export COURSE_OUTLINE=~/Course-Hub/nlp_llm/course-outline.md
python3 scripts/build_weeks.py   # 改完大綱後必跑
quarto preview                   # 本機預覽（需 brew install --cask quarto）
quarto render                    # 產出 _site/
```

Push 到 `main` 觸發 GitHub Actions，但 CI **不重建**每週頁面（它讀不到大綱），
只做 `quarto render` 加一道洩漏檢查。代價是忘記在本機重跑腳本時網站會漂移。

## 網站語言

站台骨架與每週頁的內文都是**英文**，只有每週頁的 `subtitle`（中文週次標題）與
`supplements/` 的內文維持中文——後者對齊「中文授課」。每週內文的英文版寫在大綱的
`<!-- en -->` 區塊裡，沒寫的週次維持中文而不會壞掉。**目前只有 W1 已轉英文。**

## 投影片

`slides/*.qmd` 是**剝除中文講稿後**的產物，正本在 private repo。所以站上的投影片按 `S`
打開的講稿欄是空的——**那是刻意的，不是壞掉**。

## 引用可信度標記

大綱的引用逐篇查證過，標記反映查證強度：

| 標記 | 意義 |
|---|---|
| 無標記 | 已在 `arxiv.org/abs/<id>` 比對 title 與 first author，可直接使用 |
| `[驗]` | 僅由二手來源確認，日期或作者順序可能有誤，**放投影片前請覆核** |
| `[題]` | 刻意只給檢索關鍵字，避免編造引用 |
| `[非論文]` | blog、model card 或規格文件，**不是同儕審查文獻** |

最後一級在本領域特別必要：有數個廣為流傳的術語與結論的傳播源是 blog，而 2026 年論文大量當作 baseline 的兩個架構根本沒有技術報告。

## 維護

**半衰期地圖**在 Syllabus 頁（來源是 `docs/site-en.md` 的 `halflife` 段），標明每週內容的
預期壽命：W1、W3 五年不用動；W6、W9、W10、W13、W14 的半衰期是 6–12 個月，
每次開課前需重掃 arXiv `cs.CL` / `cs.LG`，以及 ACL / EMNLP / NeurIPS / ICLR / ICML / COLM
的最新議程。
