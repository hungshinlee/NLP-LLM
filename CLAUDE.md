# CLAUDE.md — NLP-LLM 課程網站

這個 repo 只放**學生看得到的東西**：Quarto 站台、投影片，以及由課程大綱
**過濾後產生**的每週頁面。

教師端的正本——課程大綱、題目與 rubric、教學設計決定、demo 約束——在 private repo
`hungshinlee/Course-Hub` 的 `nlp_llm/`，那裡的 `CLAUDE.md` 是完整的工作手冊。
**先讀那一份再動這裡的東西。**

## 鐵則

1. **大綱不在這個 repo，也不得搬回來。** 從 Course-Hub 複製任何內容進來，
   唯一的合法通道是 `scripts/build_weeks.py` + `scripts/visibility.py` 的過濾輸出。
2. **`weeks/*.qmd` 是產物且已過濾。** 手改會在下次 build 被無聲覆蓋，
   更糟的是可能把教師內容帶上站。要改內容請改 Course-Hub 的大綱。
3. **`slides/*.qmd` 也是產物。** 投影片正本（含 `::: {.notes}` 中文講稿）在
   Course-Hub 的 `nlp_llm/slides/`；這裡的版本是 `bin/publish_slides.py` **剝除講稿後**
   的產物。**不要手動 `cp` 一份 `.qmd` 過來**，那會把講稿一起帶出去。
4. **不確定某段內容該不該公開 → 不公開。**

## 0. 誰跑什麼

`device_bash` 跑的是使用者機器上一個**獨立的 Linux VM**，只掛載連進來的資料夾，
**不是** macOS 環境本身——brew 裝的工具、macOS 的 python env、GUI 應用都看不到。

網路（2026-09-14 實測，修正舊版「對外連線全被擋」的敘述）：

| 項目 | 狀況 |
|---|---|
| HTTPS 到允許清單內的主機 | **可以**。`git ls-remote https://github.com/…` 成功 |
| SSH 傳輸層 | **可以**，proxy 有 CONNECT 隧道。host key 從 `api.github.com/meta` 取得寫進 `~/.ssh/known_hosts`（不要用 TOFU） |
| SSH 認證 | **不行**，VM 裡沒有私鑰 → `git push` 由使用者在 Mac 上執行 |
| 裸 TCP（`/dev/tcp`）、DNS 直查 | 不行 |

刪檔預設被擋，要先 `device_request_delete_permission`（**每個 session 重新授權一次**）。
沒授權就 commit 會留下 `.git/HEAD.lock`，**下一次 commit 會卡住**——授權後
`find .git -name '*.lock' -delete` 清掉即可。

`quarto` 裝在 macOS 上，VM 看不到；要驗證 render 就在雲端容器裡另裝一份。

## 1. 建置

**正常入口是 Course-Hub 的 `bin/publish.py`**，它會依序重抽講稿、發佈剝除講稿的投影片、
跑過濾建置、再做洩漏檢查：

```bash
cd ~/Course-Hub && python3 bin/publish.py nlp_llm
```

只要重建每週頁面時，也可以直接跑這裡的腳本（但這條路**不會**更新投影片）：

```bash
export COURSE_OUTLINE=~/Course-Hub/nlp_llm/course-outline.md
python3 scripts/build_weeks.py     # 產生過濾後的 weeks/*.qmd 與 _includes/
```

兩條路都**不會**自動 commit／push，那兩步由使用者在 Mac 上做。

CI **不重建**每週頁面（它讀不到大綱），只跑 `quarto render`，外加一個不需要大綱的
洩漏檢查。過濾規則在 `scripts/visibility.py` 的設定區：本課程
`DROP_UNVERIFIED = False`、`STRIP_MARKERS = ()`——四級引用標記是逐條查證過的、
Syllabus 有向學生說明，所以原樣上站。（Speech-AI 相反，那是刻意的。）

`build_weeks.py` 跑完會印出**尚未有英文內文的週次**當 TODO，見下面「網站語言」。

---

### 已定案的網站決策

| 決策 | 內容 | 理由 |
|---|---|---|
| 工具鏈 | **Quarto** | 原生 KaTeX（大綱有大量行內／display math）；同一套 `.qmd` 之後可 render revealjs 投影片 |
| 大綱切頁 | 網站只放 14 週各一頁 | 1670 行單頁在手機上不可用。**不要產生把全部串起來的 all.qmd**（Speech-AI 已移除那一頁） |
| KaTeX 版本 | **釘在 0.18.7**（`_quarto.yml` 的 `html-math-method.url`） | Quarto 預設載 `katex@latest`，上游改版會無聲壞掉 |
| 附錄 E | **不上網站**，只存在 `docs/course-outline.md` | 那是授課者私用的課程設計備註 |
| 附錄 D（半衰期地圖） | **上網站**，放 `syllabus.qmd` | 「哪些內容穩定、哪些六個月就過時」對學生與其他教師都有價值，是本課最誠實的一頁 |
| 網站語言 | **一律英文**（2026-09-15 起，與 Speech-AI 對齊）。英文：`index.qmd`、`syllabus.qmd`、`resources.qmd`、`slides.qmd`、`supplements.qmd`、navbar／sidebar／footer、每週索引、每週頁與補充教材頁的 `title`、**每週頁的內文**、頁內 TOC 標題、搜尋與 repo-actions（後三項靠 `_quarto.yml` 的 `language:` 覆寫）。**`lang:` 必須寫 `zh-TW` 不能寫 `zh-Hant`**：Quarto 只有 `_language-zh-TW.yml`，給 `zh-Hant` 會退回簡體的 `_language-zh.yml`。中文：每週頁的 `subtitle`（對照用）、`supplements/*.md` 的全部內文 | 修課學生與外部訪客都要照顧，對外門面一律英文。站名 `自然語言處理與大型語言模型` 不翻 |
| 每週內文的英文 | 寫在大綱公開區塊的 `<!-- en --> … <!-- /en -->` 裡，`visibility.py` 上站時只取英文並把 `###` 標題換成 `SECTION_EN`。**沒寫英文的週次維持中文，不會壞掉**；`build_weeks.py` 會列出清單。**目前只有 W1 已轉英文** | 逐週遷移，不必一次翻完 14 週 |
| 週次索引 | 英文標題為連結、中文標題為次行（`[…]{.wk-zh}`） | 連結文字與點進去的頁面標題要對得起來 |
| 首頁課程地圖 | **HTML/CSS grid**，由 `build_weeks.py` 從週次資料產生（樣式在 `styles.scss` 的 `.coursemap`）。最上方 `.cm-goal` 是終點框 | 方塊可點、文字進得了站內搜尋、窄螢幕自動疊成一欄——這三件事 ASCII 與 SVG 都做不到。短標籤在腳本的 `SHORT`，缺一週會 `sys.exit`；`docs/course-map-en.md` 保留備查但**不再上站** |
| 表格欄寬 | 用 pipe table 分隔列的**破折號長度**指定比例，不要留 `|---|---|` | Pandoc 依破折號長度分配欄寬；全部等長就是均分，雙語標題那一欄會被擠到不能看 |
| 投影片 | **由授課者自行製作**，AI 不代勞（沿用 Speech-AI 的指示）。正本在 private repo，這裡是剝除講稿的產物。`build_weeks.py` 偵測 `slides/wNN.qmd` 存在才注入連結 | 不存在就不給連結，避免死連結 |
| 展示模式 | 按 `z` 或點 navbar 的按鈕切換，`localStorage` 持久化、換頁保持（`_present-mode.html` + `styles.scss` 的 `html.present-mode`） | 投影時 zoom in 會落在「兩側導覽還沒收起、內容卻被擠掉」的區間。**不能用 Quarto 內建的 reader-mode**：那顆按鈕的 inline `onclick` 呼叫 `window.quartoToggleReader`，由 `quarto.js` 在 `include-in-header` 之後才載入，會吃掉同名覆寫。**關鍵**：Quarto 是 named-line grid，只把側欄 `display:none` 不會收回欄寬，要讓 `main.content` 跨到 `page-start / page-end` |

### 本機預覽

容器與此 VM 都沒裝 Quarto，需要在 Mac 上安裝一次：

```bash
brew install --cask quarto
quarto preview          # 本機即時預覽
quarto render           # 產出 _site/
```

### 手機閱讀：實際會壞的三個地方

`styles.scss` 針對這三項各有處理，改樣式前先了解為什麼這樣寫：

1. **寬表格**（文獻速查表、推導表、教科書表、半衰期表）→ `≤768px` 時 `table` 改
   `display:block; overflow-x:auto`，並給 cell `min-width`，保留換行可讀性而非壓成窄柱。
2. **ASCII 圖**（`pre`）→ `white-space:pre` + `overflow-x:auto`，在容器內橫向捲動，
   不要讓它撐開頁面。
3. **行內 `code` 的長識別字** → 見下面「已踩過的渲染陷阱」第 6 項。

### 首次上線的一次性設定（**已完成，2026-09**）

GitHub repo → **Settings → Pages → Build and deployment → Source 設為 `GitHub Actions`**。workflow 用的是 `upload-pages-artifact` + `deploy-pages`，不走 `gh-pages` 分支，所以 Source 選錯會 deploy 失敗。

⚠️ **force push 不會觸發 GitHub Actions。** 新舊 HEAD 沒有共同祖先時，`on.push.paths`
的路徑過濾算不出變更清單，workflow 被跳過。用 Actions 頁面的 **Run workflow**
（`workflow_dispatch` 已在 workflow 裡），或推一個正常的 commit 帶動。

---

## 5.5 投影片（W1 起，`slides/w01.qmd` 是模板）

**正本不在這個 repo**（2026-09-15 起）：`nlp_llm/slides/w01.qmd` 才是可編輯的來源，
含 33 段 `::: {.notes}` 中文講稿；這裡的 `slides/w01.qmd` 是 `bin/publish_slides.py`
剝除講稿後的產物，**手改下次發佈會被覆蓋**。`theme.scss` 與 `assets/` 同樣是複製品。

格式決策整套沿用 Speech-AI（見該 repo 的 CLAUDE.md 第 4.7 節），這裡只記本 repo 特有的部分。

| 決策 | 內容 |
|---|---|
| 格式 | **Quarto revealjs**，`theme.scss` 從 Speech-AI 移植，只加了一個 `.gen` class |
| 尺寸 | `width: 1280`、`height: 800`、`margin: 0.04`（實測 `Reveal.getScale()` = 0.960） |
| 字級 | 正文 34px；表格**絕對 26px**（不要改回 em，會與 `.smaller` 相乘）；圖內文字 22–26px |
| `navigation-mode` | **`linear`**（必須，否則章節堆疊會讓左右鍵跳章） |
| footer | 只用每張的 `::: {.footer}`，不要設全域 `footer:` |
| 標題 | **assertion-evidence**：標題寫主張不寫主題 |
| **不從大綱自動生成** | 大綱是閱讀密度，投影片是講述密度。自動轉換必然產生 bullet 洪流 |
| 講稿 | 中文，寫在正本的 `::: {.notes}`。**公開版沒有講稿**（reveal 的 notes 只靠 CSS 隱藏，`?showNotes=true` 就看得到，而且 `.qmd` 在 GitHub 上直接可讀——兩條路都要堵）。可列印版由 `bin/extract_notes.py` 抽成 `nlp_llm/teaching-notes/w01.md` |
| 講稿裡不要寫 LaTeX | presenter view 是另一個視窗、沒載 KaTeX 的 CSS，MathML 與 HTML 兩份會同時顯示（畫面上出現 `ℓℓ`）；平板上讀的 `.md` 也不 render 數學。一律用 Unicode 純文字 |

### `.gen` class（本 repo 新增）

W1 要並列展示四段模型生成的文字。`.gen` 用襯線體 + 淺底把「機器產生的資料」和投影片的論述文字在視覺上分開；`.gen.hit` 是紅底，標記逐字複製訓練語料的那一段；`.gen .n` 是左上角的 `n = ?` 標記。

### W1 的素材全部是現算的

**投影片上沒有任何印象值。** 兩支腳本負責：

兩支都已搬到 private repo 的 `nlp_llm/slides/scripts/`（`ROOT` 是 `nlp_llm/slides/`）：

- `make_demo_w01.py` — 下載 tinyshakespeare（公版，存到 gitignore 的 `_corpus/`），算出 n=1/2/4/8 的生成、sparsity、bits-per-byte、以及 PPMI+SVD 的 cosine 與最近鄰，寫進 `assets/w01/w01-data.json`（**進版控**，所以不重跑也能 render）。
- `make_figs_w01.py` — 讀那份 JSON 產生三張 SVG。

用莎士比亞有兩個理由：公版可自由使用，而且 **JM3 第 3 章就有這張經典的 n-gram 生成圖**，等於把課本那張圖用自己算的數字重跑一次。

**W1 的三個關鍵數字**（換語料就要重算，不要沿用）：held-out 的 4-gram 有 **95.0%** 沒見過、5-gram **99.0%**；bits-per-byte 在 **n=2** 觸底（1.861）而非 n 越大越好；`love` 的最近鄰是 **`hate`（0.412）**，而兩個同義詞對（good/excellent 0.212、sword/blade 0.163）都低於所有反義詞對。最後這一條是「cosine ≠ 語意相似」最有力的證據。

### 五個誤解的編號要與大綱對齊

deck 依出現順序編號：1 perplexity 不可跨 tokenizer 比 → 2 LLM 只是更大的 n-gram → 3 cosine ≠ 語意相似 → 4 attention 不是 Transformer 發明的 → 5 RNN 輸在平行化不是品質。**改動任何一條時，大綱 W1 的誤解表要一起改。**

### 又踩到的兩個坑（版面驗證用）

1. **`Reveal.slide(i)` 在有垂直堆疊時只走水平軸。** 章節分隔頁（`#`）會產生 stack，所以 `Reveal.slide(i)` 逐一遞增只會走過 7 個 stack 而不是 41 張，量出來的「無溢出」是假的。**正確做法是用 `Reveal.next()` 逐張前進。**
2. **`document.querySelector('section.present')` 會先命中外層 stack**，不是當前那張。量到的是整疊的高度與所有圖片。**要取 `document.querySelectorAll('section.present')` 的最後一個。**

修正後的實測：**41 張、`Reveal.getScale()` = 0.960、1280×800 下無垂直溢出、三張 SVG 的 `naturalWidth` 皆非 0。**

### SVG 的 C2PA 標記

透過 Cowork 的檔案傳輸寫進 repo 的 SVG 會被加上 C2PA 來源標記（每張多約 8 KB，仍是合法 SVG，能正常 render）。圖是由 `nlp_llm/slides/scripts/make_figs_w01.py` 決定性產生的，所以**在 Mac 上重跑一次那支腳本**就會得到乾淨的版本；否則下次重跑時 diff 會很大。

---

## 6. 已踩過的渲染陷阱（新增內容時檢查）

前四項是從 Speech-AI 帶過來的，第五、六項是本 repo 新發現的。

1. **markdown 表格的 cell 裡不能出現裸的 `|`**——行內數學請用 `\lvert … \rvert` 或 `\mid`。
2. **KaTeX 不支援 `\mathbb{1}`**，用 `\mathbf{1}`。（W10 的 self-consistency 式子踩過）
3. **`\|` 在數學裡改用 `\lVert … \rVert`**。pandoc 會把段落中的 `\|` 當成轉義豎線；雖然 `$…$` 內通常安全，但統一寫法省事。（W9 的 KL、W11 的範數踩過）
4. **`$$` display math 在清單裡要縮排到該項的內容欄**，否則會變成 code block。
5. **縮排在清單項目裡的 markdown 表格，前後都必須有空行**，否則 pandoc 會當成段落續行，整張表變成一坨文字。W14 的「貫串主題」表與附錄 E 的 demo 表都踩過。**修法是插空行，不是把表格拉到頂層**（那會脫離清單）。
6. **行內 `code` 的長識別字會撐破手機版面**。`torch.nn.functional.scaled_dot_product_attention` 在 390px 下把頁面撐寬 89px。上游把行內 code 的 `white-space` 設成 `pre`，**光給 `overflow-wrap: anywhere` 不會生效——必須先把 `white-space` 收回 `normal`**。`styles.scss` 的 `code:not(pre code)` 已處理，`pre` 區塊仍維持 `white-space:pre` + 橫向捲動。（Speech-AI 的 `styles.scss` 已於 2026-09-14 同步同一條修正。）

### 版面驗證方法

改完樣式或新增大量內容後，用 Playwright 量 18 個頁面在 390 / 768 / 1280px 下的 `document.documentElement.scrollWidth === clientWidth`。已驗證結果：**三個寬度下 18 頁全部無頁面級橫向溢出**；寬表格與 ASCII 圖各自在容器內捲動。

⚠️ **一個量測陷阱**：雲端沙箱與本機 VM 都連不到 jsdelivr（KaTeX 的 CDN，403），所以沙箱裡看到的是**未渲染的 LaTeX 原始碼**的寬度。溢出的結論**若指向 `.math` 元素就不可信**，必須在使用者機器的瀏覽器上確認。（Speech-AI 曾為此加了一條錯的 CSS 又撤回。）但**若指向非數學元素（如上面第 6 項的 `code`）就是真的**——先查出溢出元素是誰，再決定要不要相信。

---

## 維護紀律

- 動了 Course-Hub 的大綱之後：(a) 重跑 `build_weeks.py`；(b) 同步 Claude Project
  的副本；(c) 改了週次結構或標題要回頭檢查 README；(d) 改週次標題時**中英文一起改**
  （英文寫在標題下一行的 `<!-- en: -->`）；(e) 改到網站也在用的共用區段要同步
  `docs/site-en.md`（手寫，不會自動更新）；改週次分組或主題則同步
  `scripts/build_weeks.py` 的短標籤與分組設定。
- **W6、W9、W10、W13、W14 的文獻半衰期是 6–12 個月**（完整的半衰期地圖在
  `_includes/halflife.md`，上站在 Syllabus 頁）。每次開課前重掃 arXiv `cs.CL` / `cs.LG`
  近三個月，以及 ACL / EMNLP / NeurIPS / ICLR / ICML / COLM 的最新議程。
- 動了投影片或講稿之後跑 `bin/publish.py`：它會重抽 `teaching-notes/`、把剝除講稿的
  版本發佈到這裡，並在產物殘留 `::: {.notes}` 時中止。
- Commit 訊息要說明**改了哪一週、改了什麼層級**（結構／內容／引用）。
