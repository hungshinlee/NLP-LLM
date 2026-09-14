# 期末完整論文（MVP）教戰守策：從 PoC 跨越至 Top Conference 錄取門檻
<!-- en: Final Paper: From PoC to a Conference-Ready Manuscript -->
<!-- order: 3 -->
<!-- summary: The section-by-section structure, the experiments that establish soundness, the qualitative error analysis reviewers actually read, and a rubric aligned with ARR's scoring dimensions. -->

期末報告是將期中經過概念驗證（PoC）的研究方案，全面擴展為具備完整實證、消融實驗與嚴密論證的**完整論文投稿稿件（Camera-ready Candidate / MVP）**。

在 ACL Rolling Review (ARR) 以及 ICLR、NeurIPS、EMNLP 等頂尖會議的審稿機制中，高分錄取（Accept）的關鍵在於**完整性（Soundness）**與**嚴謹度（Rigor）**。本指南提供完整論文寫作與實驗收斂標準。

---

## 一、 論文結構與章節撰寫規範

稿件統一採用標準 LaTeX 模板（如 ACL 或 ICLR 雙欄格式），正文篇幅建議為 Short Paper（4 頁）或 Long Paper（8 頁），References 與 Appendix 頁數不限。

### 1. Title & Abstract
* **Title**：精準揭露核心創新架構與任務範疇，禁止泛泛之論（如 *A Novel Framework for LLM Reasoning*）。建議直接包含機制與目標（例如：*Context-Aware Dynamic Routing for Efficient Long-Document Summarization*）。
* **Abstract（200 字內，五段式微縮模型）**：
  1. *Context (1句)*：確立任務領域與核心重要性。
  2. *Defect / Tension (1–2句)*：現有 SOTA 的核心結構性失效（Failure Mode）。
  3. *Proposed Method (1–2句)*：提出之方法及其底層核心機理（Core Mechanism）。
  4. *Key Results (1–2句)*：主指標數據提昇幅度（明確給出 Delta，如 *outperforms Llama-3-70B by 4.2% in Macro-F1*）。
  5. *Significance (1句)*：本研究之方法論或實證貢獻意涵。

### 2. Introduction
* **第一頁視覺錨點（Figure 1）**：
  * 必須置於第 1 頁或第 2 頁上方。嚴禁放通用流程方塊圖。
  * 必須設計為「對比圖」（Baseline vs. Proposed）或「直觀現象圖」，直接呈現 Baseline 為何出錯、本研究機制如何修復該錯誤。圖說（Caption）必須自明（Self-contained）。
* **貢獻條列（Explicit Contributions）**：
  * 於導論結尾明確列出 3 點 Contribution：
    1. *Conceptual / Theoretical*：揭露何種現象、機制失效或提出何種新範式。
    2. *Methodological*：提出具備數學自洽性與泛化性的具體架構/演算法。
    3. *Empirical / Resource*：在標準 Benchmark 上顯著提升效能、開源高品質領域資料集或診斷性評估套件。

### 3. Problem Formulation & Methodology
* **數學符號嚴格一致**：
  * 在章節開頭集中宣告符號體系（純量 $x$、向量 $\mathbf{x}$、矩陣 $\mathbf{X}$、分佈 $\mathcal{P}$、參數集 $\Theta$）。
* **拒絕積木式拼貼（Mechanistic Justification）**：
  * 提出的每一個模組（Module）、損失函數項（Loss Term）或約束條件（Constraint），都必須在文中闡明其理論必要性。每個設計都必須直接對應一個後續的消融實驗變數。

### 4. Experimental Setup
* **Datasets**：詳述來源、規模、資料切分比例（Train/Dev/Test）、授權（License），以及針對資料洩漏（Data Contamination）所做的具體檢驗措施。
* **Baselines 覆蓋標準**：
  * *Naive Baselines*：Zero-shot / Few-shot Direct Prompting。
  * *Competitive SOTA Baselines*：公認具備競爭力之開源模型（如 Llama-3、Qwen-2.5 家族）或代表性頂會專利方法。
  * *Oracle / Upper Bound（若適用）*：完美檢索或人工黃金標註輔助時的理論極限。
* **Evaluation Metrics & Protocol**：
  * 客觀指標明列計算方式；若涉及 LLM-as-a-judge，必須在 Appendix 揭露完整 Prompt Template、模型具體版本編號、Temperature 參數，並提供與人類評估的一致性指標（Cohen's $\kappa$ 或 Spearman's $
ho$）。

---

## 二、 核心實驗驗證維度（Soundness 之基石）

一篇頂會論文在實驗設計上必須滿足三層檢驗：

| 實驗維度 | 核心檢驗目的 | Reviewer 審查標準 |
| :--- | :--- | :--- |
| **Main Results** | 證明提議方法在整體性能上超越 SOTA 與競爭基線。 | 必須包含多隨機種子（Multiple Seeds）測試並標註標準差（Standard Deviation）。關鍵指標需檢驗統計顯著性（Statistical Significance, $p < 0.05$）。 |
| **Ablation Study** | 證明系統中「每一個元件」都是不可或缺的。 | 逐一移除或替換核心組件（如 Module A、Loss Term B、Prompt Strategy C）。若移除某組件後性能未顯著衰退，代表該設計屬於無效增量。 |
| **In-depth Analysis** | 探討方法的行為邊界與機理魯棒性。 | 包括：序列長度變化測試、雜訊耐受度、推論延遲（Latency）與吞吐量（Throughput）之 Pareto Frontier 分析。 |

---

## 三、 質化錯誤分析（Qualitative Error Analysis）

頂會 Reviewer 普遍高度重視 Error Analysis。若缺少質化分析，論文將顯得浮於表面。

1. **建構錯誤分類體系（Error Taxonomy）**：
   * 從測試集中隨機抽樣至少 50–100 個預測失敗（Prediction Errors）樣本。
   * 將錯誤分類，例如：*Context Misalignment*（檢索上下文未包含關鍵事實）、*Reasoning Drift*（推論過程中斷）、*Over-refusal*（安全邊界過激拒絕）。
2. **量化統計分佈**：
   * 繪製錯誤類別分佈圖，量化對比 Proposed Method 與 Baseline 在各類錯誤上的比例變化。
3. **典型 Case Study 展示**：
   * 使用 LaTeX Table 呈現 1–2 個典型失敗樣例（包含 Input、Gold Reference、Baseline Output、Proposed Output）。
   * 逐句深入剖析提議方法在此案例中為何失效，並具體指出未來改進空間。

---

## 四、 局限性與倫理聲明（Limitations & Ethics）

ARR 與主流會議目前將 Limitations 設為強制審查項目。缺少此節將直接面臨退稿風險。

* **Computational Overhead**：誠實揭露訓練所需的 GPU-hours、峰值顯存（Peak VRAM）、推論延遲或 API Token 成本。
* **Generalization Bounds**：明確界定本方法的適用範圍。若僅在繁體中文或特定法律語料上驗證，需誠實說明尚未在多語言、跨領域環境中獲得泛化驗證。
* **Risks & Biases**：評估模型是否存在潛在毒性（Toxicity）、幻覺誤導或領域特有偏見（Domain-specific Bias）。

---

## 五、 審稿評分量表（對齊 ARR 評審維度）

論文將依據頂會標準進行綜合評定：

* **Soundness (1–5 分)**：實驗設計是否公正？基線是否充分且未被弱化？消融實驗是否完備？數據結論是否與推論吻合？
* **Excitement / Novelty (1–5 分)**：提出的觀點是真正推進了領域認知，還是僅為工程性質的堆砌？
* **Reproducibility (1–5 分)**：超參數（Learning Rate, Batch Size, Scheduler）、資料切分、Prompt Templates 與開源代碼庫是否完整透明？
* **Clarity & Presentation (1–5 分)**：論文敘事邏輯是否順暢？圖表是否具備高自明性？LaTeX 排版是否有 Overfull hbox 等格式缺陷？
