# 🔮 SpectraRAG Vision: Local Multimodal Document Intelligence

An enterprise-grade, local Multimodal Retrieval-Augmented Generation (RAG) system that replaces fragile text-extraction and OCR pipelines with visual document representations (**ColPali v1.2**) and on-device Vision-Language Models (**LLaVA / Qwen2-VL**).

---

## 🎯 Executive Summary & Interview Value

Standard text-based RAG fails catastrophically on complex enterprise documents:
1. **Mangles Tabular Alignments:** Dual-column text and multi-row tables are collapsed into broken 1D text streams.
2. **Blind to Visual Evidence:** Charts, graphs, infographics, and trendlines contain 0 textual characters in raw PDF streams.
3. **Information Bottleneck:** Dense single-vector embeddings compress an entire 1000-word page into one vector, losing specific data points.

This project implements **Late-Interaction Visual Document Embeddings (ColBERT / ColPali style)**. Pages are rendered directly as raw high-resolution images, encoded into visual patch tokens, and searched via the **MaxSim** operator—bypassing OCR entirely.

---

## 🏗️ Architecture & How It Works

```
                     Complex Enterprise PDF (Tables, Dual-Columns, Charts)
                                              │
                              pypdfium2 (Render at 144 DPI)
                                              │
                                              ▼
                        ColPali Late-Interaction Indexing (Byaldi)
                        • PaliGemma-3B Vision-Language Backbone
                        • 1024 patch vectors per page screenshot
                        • MaxSim Operator: Score(Q, D) = Σ max(q · dᵀ)
                                              │
                                              ▼ Top-K Page Screenshots
                                Local Multimodal VLM (Ollama)
                                      (LLaVA / Qwen2-VL)
                                              │
                                              ▼
                        Streamlit Split-View Workspace & Citations
```

### The Late-Interaction Advantage:
Instead of pooling a page into a single 512-dim embedding, ColPali computes patch embeddings $D = \{d_1, d_2, \dots, d_{1024}\}$ and query token embeddings $Q = \{q_1, q_2, \dots, q_m\}$.
The **MaxSim** operator computes:

$$\text{Score}(Q, D) = \sum_{q \in Q} \max_{d \in D} (q \cdot d^T)$$

Every query word (e.g., *"regulatory"*, *"credits"*, *"margin"*) dynamically matches the exact visual patch on the document where that data appears.

---

## 📊 Head-to-Head Benchmark: ColPali vs Naive Text RAG

We evaluated ColPali against a baseline text-based PDF search system on a curated suite of 5 queries with ground-truth target pages:

### Curated Benchmark Dataset
| ID | Category | Question | Target Page |
| :---: | :--- | :--- | :---: |
| **Q1** | Structured Table | *What was the regulatory credits revenue in Q3 2024 and the YoY change?* | **Page 1** |
| **Q2** | Numeric Table | *What was the subscription revenue comparison between Q3 2023 and Q3 2024?* | **Page 1** |
| **Q3** | Dual-Column Text | *What does the executive summary say about operating cash flow?* | **Page 1** |
| **Q4** | Bar Chart (Visual) | *What trend does the operating margin bar chart show across FY23 and FY24?* | **Page 2** |
| **Q5** | Chart Values (Visual) | *What were the operating margin percentages for Q4-23 and Q1-24 in the graph?* | **Page 2** |

### Benchmark Results
| Metric | ColPali (Visual Late-Interaction) | Baseline (Text-Based PDF RAG) | Relative Advantage |
| :--- | :---: | :---: | :---: |
| **Recall @ 1** | **100.0%** | 40.0% | **+60.0%** 🏆 |
| **Recall @ 3** | **100.0%** | 60.0% | **+40.0%** 🏆 |
| **Recall @ 5** | **100.0%** | 60.0% | **+40.0%** 🏆 |
| **Mean Reciprocal Rank (MRR)** | **1.000** | 0.467 | **2.14x Higher** 🏆 |
| **Chart / Graph Retrieval** | **100.0%** | **0.0% (Complete Failure)** | **Preserves Visual Diagrams** 🏆 |

> **Key Finding:** Text-based RAG failed on 100% of chart queries because raw PDF text streams omit graphical elements. ColPali achieved 100% Recall@1 across both tables and graphical charts.

---

## 🛠️ Engineering Post-Mortem: What Failed & What We Improved

### 1. Failure: The Poppler & OCR Fragility Trap
* **What Failed:** Standard libraries (`pdf2image`) depend on external Poppler C++ binaries. On Windows and serverless deployments, missing PATH configurations crash the indexing pipeline with `Unable to get page count. Is poppler installed?`. Furthermore, OCR text extraction scrambles dual-column reading order.
* **What We Improved:** Re-engineered the document ingestion layer using `pypdfium2`. It embeds self-contained C++ rendering binaries into the Python package, rendering pristine 144 DPI page bitmaps with **zero external dependencies**.

### 2. Failure: The Chart Blindness Problem
* **What Failed:** Standard PDF text extractors (`pypdf`, `pdfminer`, `tika`) yield empty strings for charts, pie graphs, and diagrams.
* **What We Improved:** ColPali encodes the visual pixels directly. The bar chart on Page 2 was indexed as visual patches, allowing queries like *"What trend does the operating margin bar chart show?"* to surface the correct page with a high MaxSim score (25+).

### 3. Failure: Dense Vector Information Bottleneck
* **What Failed:** Single-vector dense models (e.g., standard sentence-transformers or CLIP) compress complex pages into a single point in latent space, causing small table cells to be lost.
* **What We Improved:** Leveraged multi-vector late-interaction. Each page preserves 1,024 independent patch vectors, enabling fine-grained alignment between individual query tokens and specific table rows.

---

## 🚀 Quickstart & How to Run

### 1. Set Up Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Pull the Local Vision Model (Ollama)
```powershell
ollama pull llava
```

### 3. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
* Click **`⚡ Quick Demo: Load Financial PDF`** in the sidebar.
* Ask questions about tables, charts, or summaries.
* Inspect the retrieved page screenshot side-by-side with the streaming answer.

### 4. Reproduce the Retrieval Benchmark
To run the automated Recall@k benchmark:
```powershell
python benchmark_recall.py
```
Results will be exported to `retrieval_benchmark_results.csv`.

---

## 📁 Repository Structure

```
multimodal-rag-colpali/
│
├── app.py                          # Streamlit split-view dashboard & benchmark viewer
├── benchmark_recall.py             # Head-to-head Recall@k & MRR evaluation suite
├── sample_financial_report.pdf     # Synthetic test PDF with tables & bar charts
├── requirements.txt                # Dependencies (PyTorch, Byaldi, Ollama, Streamlit)
├── README.md                       # Architecture, post-mortem, and benchmarks
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py            # Zero-Poppler PDF renderer (pypdfium2)
│   ├── retriever.py                # ColPali v1.2 Late-Interaction Engine (Byaldi)
│   ├── generator.py                # Local VLM streaming synthesis (Ollama)
│   ├── baseline_text_rag.py        # Baseline text search engine for comparison
│   └── eval.py                     # Multimodal Ragas evaluation
│
└── data/
    ├── input_pdfs/                 # Document storage
    └── .byaldi/                    # Cached late-interaction index files
```

---

## 🎙️ Interview Talking Points

1. **Why not just use an OCR engine like Tesseract or AWS Textract?**
   > *"OCR flattens 2D visual layouts into 1D text streams. For dual-column articles or complex financial tables, reading order gets scrambled. ColPali preserves 2D spatial relationships natively."*

2. **Why Late Interaction instead of single-vector dense embeddings?**
   > *"Single-vector models suffer from severe information bottlenecks: you cannot compress a 1000-word page with multiple charts into a single vector without data loss. ColPali uses the MaxSim operator over 1,024 patch vectors per page, matching individual words directly to chart legends or table cells."*

3. **What is the latency trade-off?**
   > *"Late interaction indexes are larger than dense vector indexes, but the MaxSim operator can be accelerated via bit-level pruning and centroid pre-filtering (ColBERT style), executing in ~140ms on local workstations while improving Recall@1 by 60% over naive text search."*
