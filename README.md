# SpectraRAG Vision

[![Open live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aadyam2006-spectrarag-vision-app-iy5r2d.streamlit.app/)

**A Streamlit prototype for asking questions about PDF pages.** It renders pages for visual inspection, extracts embedded PDF text with `pypdfium2`, ranks pages with a lightweight keyword and synonym heuristic, and can send the selected page image to an optional vision model.

> **Current implementation:** this repository does not currently run ColPali, CLIP, or BM25. The names and metrics in earlier benchmark artifacts overstated what the code measures; those claims have been removed. The default “Fast Demo” response is a document-specific demo/template mode, not an LLM evaluation. Use Ollama or Gemini mode for model-generated answers.

## Live demo

[Launch SpectraRAG Vision](https://aadyam2006-spectrarag-vision-app-iy5r2d.streamlit.app/)

The hosted app may have different available models and resources from a local install. For reproducible model-backed answers, run locally with the model provider configured.

## What it does

- Renders PDF pages and lets you inspect them in a Streamlit workspace.
- Extracts embedded text and ranks pages using token overlap, stopwords, and a small synonym map.
- Offers a fast, template-based demo mode, plus optional Ollama and Gemini image-question-answering modes.
- Includes experimental keyword and page-layout heuristic comparators for exploration.

### Current limits

- Scanned PDFs without an embedded text layer are not searchable by the current retrieval code.
- Retrieval is a simple text heuristic; it does not embed page images, preserve patch vectors, or implement MaxSim.
- The dense visual and text comparison classes are toy heuristics, not CLIP and BM25 implementations.
- Fast Demo answers are selected from document-specific templates and must not be treated as generated or independently verified answers.
- The repository does not currently contain the five-document benchmark corpus referenced by older benchmark artifacts. The benchmark runner now requires those files and will stop with a clear missing-file message instead of substituting another PDF.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

For Ollama mode, install and start Ollama separately and download the vision model you select. For Gemini mode, provide an API key in the app. Avoid uploading sensitive documents to a cloud provider unless you are comfortable with that provider processing them.

## Evaluation status

The previous checked-in benchmark CSVs contained illustrative/hard-coded values rather than results that can be reproduced from this repository. They have been removed. No retrieval-quality or generation-quality score is currently claimed.

To run the exploratory retrieval harness after supplying all PDFs referenced by `benchmark_recall.py` in the repository root or `data/input_pdfs/`, run:

```powershell
python benchmark_recall.py
```

The script writes fresh CSVs for that run. The app will list missing source files and disable the benchmark button until the corpus is complete.

The benchmark script is an experimental harness, not a research-grade evaluation. It requires the original query documents and page labels. Its current comparator implementations are heuristics; a fair model comparison requires real ColPali, CLIP, and BM25 backends, a documented corpus, fixed query labels, and repeated timing that reports indexing, query encoding, and search/scoring separately. The timing inside a single `search()` call measures only the in-memory heuristic ranking step; it excludes PDF rendering/indexing and any model inference.

`src/eval.py` is a separate three-example Ragas experiment using hand-written example data. It does not evaluate answers produced by the running app and its output must not be presented as a measured app score.

## Repository layout

```text
SpectraRAG-Vision/
├── app.py
├── benchmark_recall.py
├── download_model.py
├── requirements.txt
├── src/
│   ├── eval.py
│   ├── evaluator.py
│   ├── generator.py
│   ├── keyword_baseline.py
│   ├── layout_heuristic.py
│   ├── pdf_processor.py
│   └── retriever.py
├── scripts/                 # PDF generator utility scripts
└── data/input_pdfs/          # Add PDFs here (only the sample is tracked)
```

## Planned improvements

The following are ideas, not implemented features: real ColPali visual retrieval, real CLIP and BM25 baselines, a reproducible benchmark corpus with a larger query set, answer evaluation on actual app outputs, and scan cleanup or multi-page grounding safeguards.
