# CLAUDE.md — NLP-LLM 課程網站

這個 repo 是課程網站的**發佈產物**：Quarto 站台、過濾後的每週頁面、剝除講稿的投影片、demo 程式、講義與補充教材。
課程的正本與工作手冊在授課者的 private repo，不在這裡。**這份檔案在 GitHub 上公開，只寫這個 repo 怎麼維護，不寫教學內容。**

## 規則

1. **不要手改產物。** `weeks/`、`_includes/`、`slides/`、`demos/`、`handouts/`、`supplements/*.qmd` 與 `supplements/` 裡的報告範本
   由 private repo 的發佈腳本產生（README 的檔案表標 ⚙︎ 的那些），手改會在下次發佈被覆蓋。要改內容，改正本、重新發佈。
2. **不要從 private repo 手動複製任何東西進來**，只走發佈腳本。
3. **不確定某段內容該不該公開時，不公開。**

## 可以直接改的

站台本身：`_quarto.yml`、`index.qmd`、`syllabus.qmd`、`resources.qmd`、`slides.qmd`、`supplements.qmd`、
`styles.scss`、`_present-mode.html`、`docs/site-en.md`、`scripts/build_weeks.py`、`scripts/visibility.py`、`.github/workflows/publish.yml`。
這些檔案的設計理由與踩過的坑（手機版面、KaTeX 版本、展示模式、渲染陷阱、量測方法）記在 private repo，改之前先讀那邊。

## 建置與部署

push 到 `main` 觸發 `.github/workflows/publish.yml`：先跑洩漏檢查，再 `quarto render`，部署到 GitHub Pages（Settings → Pages 的 Source 是 **GitHub Actions**）。
force push 不會觸發 workflow，用 Actions 頁面的 **Run workflow**。本機預覽：`quarto preview`。
