# 期中研究計畫與概念驗證（PoC）教戰守策：從問題定義到初期驗證
<!-- en: Midterm: From Problem Statement to Proof of Concept -->
<!-- order: 2 -->
<!-- summary: How to frame a falsifiable hypothesis and position it against prior art, the fatal flaws reviewers look for first, and where AI tools belong in the process. -->

這份期中報告本質上是一篇預備投稿至 ACL、EMNLP、NAACL、COLM 或 ICLR 的 **Research Proposal 兼初期概念驗證（Proof of Concept, PoC）報告**。頂級會議（Top Conferences）的 Area Chair (AC) 與 Reviewer 在審閱 Proposal 或初期稿件時，核心關注點在於：**問題是否具備真實價值（Non-trivial & Non-artificial）**、**底層假說是否合理且可證偽（Falsifiable Hypothesis）**，以及**研究在學術坐標系中的定位是否明確（Novelty & Positioning）**。

本指南為修課同學擬定各章節撰寫規範、評量標準與 AI 協同邊界。

---

## 一、 核心章節架構與撰寫標準

### 1. 題目選定與痛點分析 (Problem Formulation & Motivation)
* **拒絕偽問題與平庸評估（Trivial Evaluation）**：嚴禁選定「將現成開源 LLM 套用在某公開資料集上看分數」之類的題目。若無涉及深入的 Failure Mode 分析或機理探討，單純的 API 呼叫測試無法構成頂會研究。
* **鎖定具體的結構性失效（Structural Failure Mode）**：
  * 現有 SOTA（State of the Art）或主流範式在哪裡遭遇瓶頸？
  * 是 Context Length 擴展後的 Attention Dispersion？推論鏈（Chain-of-Thought）在特定複合邏輯下的 Step-level Hallucination？還是繁體中文 / 低資源語言（Low-resource Languages）在 Tokenizer 壓縮率不對稱與語義空間對齊上的劣勢？
* **明確提出核心研究問題（Research Questions, RQs）**：
  * 必須定義 1–2 個具體、精準且可量化驗證的 RQ。
  * 例如：*RQ1: 在多跳推理（Multi-hop Reasoning）場景中，動態剪枝檢索上下文能否在降低 40% KV-cache 佔用的同時維持推論準確率？*

### 2. 文獻探討與學術定位 (Related Work & Positioning)
* **分類架構（Taxonomy）勝過流水帳清單**：
  * 嚴禁寫成「A 做了什麼、B 做了什麼」的清單式整理。
  * 必須依據方法論或底層機制建立多維度分類（例如：*Parametric Memory Adaptation* vs. *Non-parametric Retrieval Augmentation*，或 *Pre-generation Filtering* vs. *Post-generation Self-Correction*）。
* **確立「相對座標」與 Delta**：
  * 必須有一段明確論述本研究與現有 Benchmark 或 SOTA 的邊界差異（Conceptual / Methodological Delta）。Reviewer 依此判定研究的邊際貢獻（Novelty）。
* **文獻真實性嚴格查驗（Hallucination-free Citations）**：
  * 嚴格禁止直接採信由 LLM 生成的參考文獻。所有引用必須於 **ACL Anthology**、**Semantic Scholar** 或 **arXiv** 核實作者、年份、發表會議與核心貢獻。

### 3. 解決方案 (Proposed Methodology)
* **假說驅動（Hypothesis-Driven Architecture）**：
  * 清楚說明「為何該架構/機制能解決目標痛點」。若採用了特定 PEFT（如 LoRA / DoRA）、設計了動態 Routing 機制、或是導入了 Process-based 驗證，必須提出機理解釋。
* **符號與數學規範**：
  * 集中定義輸入、轉換與輸出符號（如輸入序列 $X = \{x_1, \dots, x_n\}$、潛在空間表示 $\mathbf{h} \in \mathbb{R}^d$、狀態轉移條件機率 $P_	heta(y_t \mid y_{<t}, X)$）。
  * 繪製高自明性（Self-contained）的系統架構圖（Architecture Diagram），清晰標注資料流向與張量維度變換。

### 4. 資料集選定與處理 (Data Curation & Benchmark Setup)
* **防範資料污染（Data Contamination）**：
  * 主流開源或閉源 LLM 的預訓練語料可能已包含常見公開 Benchmark（如 GSM8K、MMLU）。若使用現成資料集，需說明如何防範或檢驗資料洩漏。
* **領域語料治理（Data Cleaning & Pipeline）**：
  * 若建構特定領域語料（如法規判決、醫療紀錄、台灣本土語料），需詳述清洗規則、過濾邏輯（Heuristic filtering / Quality filtering）、Token 長度分佈、多樣性指標（Diversity metrics）與標註準則（Annotation Guidelines）。

### 5. 實驗規劃與初期驗證 (Experimental Design & Initial PoC)
* **基線模型（Baselines）層次配置**：
  * *Naive Baseline*：Zero-shot / Few-shot Direct Prompting。
  * *Competitive Baseline*：同等運算預算下的公認開源 SOTA（如 Llama-3-8B / Qwen-2.5-7B 的常規微調或成熟框架）。
* **評估指標（Evaluation Metrics）**：
  * 客觀任務：Exact Match (EM)、Macro-F1、Pass@k。
  * 生成任務：若引入 LLM-as-a-judge，必須附帶評分量表（Rubric）、對齊人工評估的一致性檢驗規劃，以及防範 Position Bias 的雙向評估（Swap Evaluation）機制。
* **PoC 階段驗證成果**：
  * 期中報告必須證明 Pipeline 已經跑通。不需要繳交全量資料結果，但必須展示 **Small-scale（50–100 樣本）的初期定量結果**與 **初步質化案例分析（Initial Qualitative Error Cases）**，證明核心假說具備可行性。

---

## 二、 常見致命傷與 Reviewer 視角對照

| 學生常見盲點 | Reviewer 典型質疑 (Reasons to Reject) | 修正與防禦標準 |
| :--- | :--- | :--- |
| **動機空泛**：「LLM 在 X 任務表現不佳，因此我提出 Y」 | 表現不佳的具體機理為何？是參數量限制、語料分佈偏差，還是推理長度受限？ | 進行精準的 Failure Mode 分析，指認出結構性或表徵層面的具體缺陷。 |
| **文獻堆砌**：羅列 10 篇論文但無結構分類 | 看不出本研究在現有學術光譜中的相對位置在哪裡。 | 建立清晰的分類矩陣（Taxonomy Table），凸顯既有方法的盲區與本研究的 Delta。 |
| **評估主觀**：「輸出結果經肉眼檢驗感覺品質優異」 | 缺乏客觀標準與可復現性，屬於無效實驗結果。 | 設計標準量化指標、自動化測試集，或嚴格定義 Rubric 進行盲測（Blind Test）。 |
| **掩蓋失敗**：初期實驗只挑選成功案例呈現 | 提出的方案邊界條件為何？是否存在 Cherry-picking？ | 主動披露初步 PoC 的失敗樣例（Failure Cases），並轉化為期末消融實驗的方向。 |

---

## 三、 善用 AI 工具（以 Gemini 為例）之邊界規範

AI 工具在學術研究中是思考共振與工程加速器，嚴禁作為認知外包工具。

### 1. 鼓勵使用的情境
* **對抗性審稿模擬（Adversarial Reviewer Simulation）**：
  * 將 Motivation 與 Proposed Method 提供給模型，指令其以 ACL Senior Reviewer 角度挑刺，專注尋找「非必然推論跳躍（Non-sequitur）」與「潛在未控制變量」。
* **工程腳本除錯與加速**：
  * 輔助編寫 Hugging Face Dataset 加載腳本、vLLM 推論平行化配置、DeepSpeed ZeRO-Stage 參數調校。
* **學術表達修飾**：
  * 將已完成草稿轉換為被動語態較少、動詞精準、句構嚴謹的頂會寫作風格。

### 2. 嚴格禁止的情境
* **直接委由 AI 產出文獻探討（Literature Review Generation）**：
  * 語言模型具備嚴重的虛構文獻傾向（Hallucinated Citations），嚴禁由模型自行搜集並撰寫文獻引用。
* **虛構實驗數值或 Baseline 評分**：
  * 所有基準數據必須來自本地執行 Log 或已發表的論文原文。
* **概念黑盒子化**：
  * 學生必須徹底理解提議架構中每一個數學運算與張量維度轉換，不得在不清楚其數學意義的情況下採納 AI 給予的演算法架構。
