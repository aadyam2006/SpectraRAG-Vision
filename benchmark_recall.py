"""Exploratory comparison of three hand-built PDF page-ranking heuristics.

This script does not run ColPali, CLIP, BM25, or evaluate generated answers.
The source PDFs referenced by BENCHMARK_DATASET must be supplied locally.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.retriever import TextOverlapRetriever
from src.keyword_baseline import KeywordOverlapBaseline
from src.layout_heuristic import LayoutHeuristicBaseline
from src.evaluator import SystemEvaluator


# ==============================================================================
# EXPLORATORY QUERY SET (35 queries, small and manually labeled)
# ==============================================================================
BENCHMARK_DATASET: List[Dict[str, Any]] = [
    # Category 1: Structured Tables
    {
        "id": 1,
        "category": "Structured Table",
        "question": "What was the regulatory credits revenue in Q3 2024 and the YoY change?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Table Cell"
    },
    {
        "id": 2,
        "category": "Structured Table",
        "question": "What is the total invoice amount and GST tax rate applied?",
        "doc": "MS STAR COMPUTERS INVOICE .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Invoice Summary Table"
    },
    {
        "id": 3,
        "category": "Structured Table",
        "question": "What was the subscription revenue comparison between Q3 2023 and Q3 2024?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Revenue Table"
    },
    {
        "id": 4,
        "category": "Structured Table",
        "question": "What is the HSN code and quantity listed for the cartridge product?",
        "doc": "MS STAR COMPUTERS INVOICE .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Line Item Table"
    },

    # Category 2: Financial Statements
    {
        "id": 5,
        "category": "Financial Statement",
        "question": "What were the Energy Storage segment revenues and gross profit for Tesla in Q3?",
        "doc": "tesla_q3_operational_highlights.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Segment Financials"
    },
    {
        "id": 6,
        "category": "Financial Statement",
        "question": "What was the total cash, cash equivalents, and investments reported at quarter-end?",
        "doc": "tesla_q3_operational_highlights.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Liquidity Statement"
    },
    {
        "id": 7,
        "category": "Financial Statement",
        "question": "What was the YoY percentage increase in operating cash flow reported for ACME Tech?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Cash Flow Highlights"
    },

    # Category 3: Bar & Line Charts
    {
        "id": 8,
        "category": "Bar & Line Chart",
        "question": "What trend does the operating margin bar chart show across FY23 and FY24?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Bar Chart Layout"
    },
    {
        "id": 9,
        "category": "Bar & Line Chart",
        "question": "What was the operating margin percentage reported for Q4-23 in the bar chart?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Bar Chart Data Label"
    },
    {
        "id": 10,
        "category": "Bar & Line Chart",
        "question": "What was the Megapack deployment volume milestone shown in the energy storage chart?",
        "doc": "tesla_q3_operational_highlights.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Deployment Metric Chart"
    },

    # Category 4: Graphical Diagrams & Visual Plots
    {
        "id": 11,
        "category": "Graphical Diagram",
        "question": "What visual coordinate geometry or triangle diagram is illustrated in Section C?",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Geometry Figure"
    },
    {
        "id": 12,
        "category": "Graphical Diagram",
        "question": "Where is the system architecture diagram illustrating the AI Code Mentor pipeline?",
        "doc": "Aadya_Mendiratta_Resume .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Project Visual Flow"
    },
    {
        "id": 13,
        "category": "Graphical Diagram",
        "question": "What graphical flowchart represents fraud detection data processing in the NIMS report?",
        "doc": "Abhinav_Goel_NIMS_Practical_Training_Report_Final.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Flowchart Diagram"
    },

    # Category 5: Dual-Column Documents
    {
        "id": 14,
        "category": "Dual-Column Document",
        "question": "What does the executive summary state in the left column regarding operating margin recovery?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Dual-Column Left"
    },
    {
        "id": 15,
        "category": "Dual-Column Document",
        "question": "What commentary is provided in the right column regarding freight costs and supply chain?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Dual-Column Right"
    },
    {
        "id": 16,
        "category": "Dual-Column Document",
        "question": "What two parallel columns contrast the Technical Skills against the Projects list?",
        "doc": "Aadya_Mendiratta_Resume .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Dual Column CV Layout"
    },

    # Category 6: Text-Heavy Pages
    {
        "id": 17,
        "category": "Text-Heavy Page",
        "question": "What does the candidate state in their Professional Summary about machine learning?",
        "doc": "Aadya_Mendiratta_Resume .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Paragraph Text"
    },
    {
        "id": 18,
        "category": "Text-Heavy Page",
        "question": "What are the general instructions and time allowed listed for the Mathematics exam?",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Exam Instructions"
    },
    {
        "id": 19,
        "category": "Text-Heavy Page",
        "question": "What is the detailed student declaration and acknowledgement statement?",
        "doc": "Abhinav_Goel_NIMS_Practical_Training_Report_Final.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Formal Declaration"
    },

    # Category 7: Mixed Text + Visuals
    {
        "id": 20,
        "category": "Mixed Text + Visual",
        "question": "Which page pairs the operating margin bar chart with the forward-looking guidance text?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Chart with Annotation"
    },
    {
        "id": 21,
        "category": "Mixed Text + Visual",
        "question": "What question presents a polynomial equation alongside an explanatory coordinate graph?",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 3,
        "evidence_type": "Formula with Graph"
    },
    {
        "id": 22,
        "category": "Mixed Text + Visual",
        "question": "Which page pairs the company logo with the authorized signatory and tax breakdown stamp?",
        "doc": "MS STAR COMPUTERS INVOICE .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Stamp & Signature"
    },

    # Category 8: Numerical Precision
    {
        "id": 23,
        "category": "Numerical Precision",
        "question": "What is the exact regulatory credit dollar figure for Q3-24 ($715M) and Q3-23 ($280M)?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Exact Numerical Value"
    },
    {
        "id": 24,
        "category": "Numerical Precision",
        "question": "What was the exact energy storage gross margin percentage reported by Tesla (30.5%)?",
        "doc": "tesla_q3_operational_highlights.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Percentage Metric"
    },
    {
        "id": 25,
        "category": "Numerical Precision",
        "question": "What is the specific marks allotment (80 marks, 3 hours) printed in the header?",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Exam Numerical Header"
    },
    {
        "id": 26,
        "category": "Numerical Precision",
        "question": "What is the exact CGST and SGST percentage rate printed on the invoice (9% each)?",
        "doc": "MS STAR COMPUTERS INVOICE .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Tax Rate Number"
    },

    # Category 9: Layout-Dependent Questions
    {
        "id": 27,
        "category": "Layout-Dependent",
        "question": "Where is the buyer billing address located relative to the seller details on the invoice?",
        "doc": "MS STAR COMPUTERS INVOICE .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Spatial Top-Left vs Right"
    },
    {
        "id": 28,
        "category": "Layout-Dependent",
        "question": "Which section is positioned directly beneath Technical Skills on the candidate resume?",
        "doc": "Aadya_Mendiratta_Resume .pdf",
        "ground_truth_page": 1,
        "evidence_type": "Vertical Stack Order"
    },
    {
        "id": 29,
        "category": "Layout-Dependent",
        "question": "Where are the Table 1 footnote citations positioned on the financial report page?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Page Footer Coordinate"
    },
    {
        "id": 30,
        "category": "Layout-Dependent",
        "question": "What section header appears immediately below the general CBSE examination instructions?",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Section A Boundary"
    },

    # Category 10: Multi-Page Questions
    {
        "id": 31,
        "category": "Multi-Page Reasoning",
        "question": "Compare the operating margin reported in the Table on Page 1 with the Graph on Page 2.",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Multi-Page Visual Synthesis"
    },
    {
        "id": 32,
        "category": "Multi-Page Reasoning",
        "question": "How do the questions in Section A (Page 1) transition into Section B short answers (Page 2)?",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Cross-Section Transition"
    },
    {
        "id": 33,
        "category": "Multi-Page Reasoning",
        "question": "Synthesize the project objectives on Page 1 with the technical outcomes on Page 2.",
        "doc": "Abhinav_Goel_NIMS_Practical_Training_Report_Final.pdf",
        "ground_truth_page": 2,
        "evidence_type": "Multi-Section Evaluation"
    },
    {
        "id": 34,
        "category": "Multi-Page Reasoning",
        "question": "How does the executive revenue commentary on Page 1 align with the margin visual on Page 2?",
        "doc": "sample_financial_report.pdf",
        "ground_truth_page": 1,
        "evidence_type": "Dual-Page Synthesis"
    },
    {
        "id": 35,
        "category": "Multi-Page Reasoning",
        "question": "Contrast the foundational algebra MCQs on Page 1 with the applied case study on Page 3.",
        "doc": "maths-class-x-question-bank.pdf",
        "ground_truth_page": 3,
        "evidence_type": "Longitudinal Progression"
    }
]


# ==============================================================================
# BENCHMARK EXECUTION HARNESS
# ==============================================================================
def run_research_benchmark(dataset: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the experimental heuristic comparison on the supplied PDF corpus."""
    if dataset is None:
        dataset = BENCHMARK_DATASET

    print("=" * 78)
    print("      EXPERIMENTAL PDF PAGE-RANKING HEURISTIC COMPARISON")
    print("      PDF text overlap vs page-layout heuristic vs keyword overlap")
    print(f"      Total queries: {len(dataset)} (small exploratory dataset)")
    print("=" * 78)

    # Initialize all three engines
    text_overlap_retriever = TextOverlapRetriever()
    layout_retriever = LayoutHeuristicBaseline()
    keyword_retriever = KeywordOverlapBaseline()

    # Document index cache
    indexed_docs = set()

    text_overlap_records = []
    dense_records = []
    text_records = []
    detailed_table = []

    print("\n[Phase 1/2] Executing Tri-Way Retrieval Across 35 Benchmark Queries...")

    for idx, item in enumerate(dataset):
        doc_name = item["doc"]
        target_page = item["ground_truth_page"]
        query = item["question"]
        category = item["category"]

        # Resolve exact document path. Never substitute a different PDF.
        candidates = [Path(doc_name), Path("data/input_pdfs") / doc_name]
        doc_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if doc_path is None:
            missing = Path("data/input_pdfs") / doc_name
            raise FileNotFoundError(
                f"Required benchmark PDF is missing: {doc_name}. Add it at "
                f"'{missing}' (or in the repository root); no substitute is used."
            )

        # Ensure document is indexed across all 3 engines
        doc_key = doc_path.name
        if doc_key not in indexed_docs:
            text_overlap_retriever.index_documents(str(doc_path.resolve()), max_pages=0, overwrite=False)
            layout_retriever.index_pdf(doc_path, max_pages=0)
            keyword_retriever.index_pdf(doc_path, max_pages=0)
            indexed_docs.add(doc_key)

        # 1. PDF text-overlap search
        t0 = time.perf_counter()
        cp_res = text_overlap_retriever.search(query, top_k=5)
        cp_lat = (time.perf_counter() - t0) * 1000.0
        cp_pages = [getattr(r, "page_num", 1) for r in cp_res]
        c_r1 = 1 if cp_pages and cp_pages[0] == target_page else 0
        c_r3 = 1 if target_page in cp_pages[:3] else 0
        c_r5 = 1 if target_page in cp_pages[:5] else 0
        c_rank = cp_pages.index(target_page) + 1 if target_page in cp_pages else 0
        c_mrr = 1.0 / c_rank if c_rank > 0 else 0.0

        text_overlap_records.append({
            "id": item["id"], "r@1": c_r1, "r@3": c_r3, "r@5": c_r5, "mrr": c_mrr, "latency_ms": cp_lat
        })

        # 2. Page-layout heuristic
        t0 = time.perf_counter()
        dense_res = layout_retriever.search(query, top_k=5)
        dense_lat = (time.perf_counter() - t0) * 1000.0
        dense_pages = [r["page_num"] for r in dense_res]
        d_r1 = 1 if dense_pages and dense_pages[0] == target_page else 0
        d_r3 = 1 if target_page in dense_pages[:3] else 0
        d_r5 = 1 if target_page in dense_pages[:5] else 0
        d_rank = dense_pages.index(target_page) + 1 if target_page in dense_pages else 0
        d_mrr = 1.0 / d_rank if d_rank > 0 else 0.0

        dense_records.append({
            "id": item["id"], "r@1": d_r1, "r@3": d_r3, "r@5": d_r5, "mrr": d_mrr, "latency_ms": dense_lat
        })

        # 3. Keyword-overlap baseline
        t0 = time.perf_counter()
        text_res = keyword_retriever.search(query, top_k=5)
        text_lat = (time.perf_counter() - t0) * 1000.0
        text_pages = [r["page_num"] for r in text_res]
        t_r1 = 1 if text_pages and text_pages[0] == target_page else 0
        t_r3 = 1 if target_page in text_pages[:3] else 0
        t_r5 = 1 if target_page in text_pages[:5] else 0
        t_rank = text_pages.index(target_page) + 1 if target_page in text_pages else 0
        t_mrr = 1.0 / t_rank if t_rank > 0 else 0.0

        text_records.append({
            "id": item["id"], "r@1": t_r1, "r@3": t_r3, "r@5": t_r5, "mrr": t_mrr, "latency_ms": text_lat
        })

        detailed_table.append({
            "Query ID": f"Q{item['id']:02d}",
            "Category": category,
            "Target Page": f"P{target_page}",
            "PDF text overlap top": f"P{cp_pages[0]}" if cp_pages else "None",
            "Page-layout heuristic top": f"P{dense_pages[0]}" if dense_pages else "None",
            "Keyword overlap top": f"P{text_pages[0]}" if text_pages else "None",
            "PDF text overlap hit": "Pass" if c_r1 else "Miss",
            "Page-layout heuristic hit": "Pass" if d_r1 else "Miss",
            "Keyword overlap hit": "Pass" if t_r1 else "Miss"
        })

    # Calculate System-Level Retrieval Metrics
    cp_metrics = SystemEvaluator.calculate_retrieval_metrics(text_overlap_records)
    dense_metrics = SystemEvaluator.calculate_retrieval_metrics(dense_records)
    text_metrics = SystemEvaluator.calculate_retrieval_metrics(text_records)

    # Format Comparative Summary
    summary_matrix = [
        {"Metric": "Recall @ 1", "PDF text overlap": f"{cp_metrics['recall@1']}%", "Page-layout heuristic": f"{dense_metrics['recall@1']}%", "Keyword overlap": f"{text_metrics['recall@1']}%"},
        {"Metric": "Recall @ 3", "PDF text overlap": f"{cp_metrics['recall@3']}%", "Page-layout heuristic": f"{dense_metrics['recall@3']}%", "Keyword overlap": f"{text_metrics['recall@3']}%"},
        {"Metric": "Recall @ 5", "PDF text overlap": f"{cp_metrics['recall@5']}%", "Page-layout heuristic": f"{dense_metrics['recall@5']}%", "Keyword overlap": f"{text_metrics['recall@5']}%"},
        {"Metric": "Mean Reciprocal Rank", "PDF text overlap": f"{cp_metrics['mrr']:.3f}", "Page-layout heuristic": f"{dense_metrics['mrr']:.3f}", "Keyword overlap": f"{text_metrics['mrr']:.3f}"},
        {"Metric": "Mean search call (ms)", "PDF text overlap": f"{cp_metrics['latency_avg_ms']}", "Page-layout heuristic": f"{dense_metrics['latency_avg_ms']}", "Keyword overlap": f"{text_metrics['latency_avg_ms']}"},
        {"Metric": "P95 search call (ms)", "PDF text overlap": f"{cp_metrics['latency_p95_ms']}", "Page-layout heuristic": f"{dense_metrics['latency_p95_ms']}", "Keyword overlap": f"{text_metrics['latency_p95_ms']}"}
    ]

    summary_df = pd.DataFrame(summary_matrix)
    print("\n" + "=" * 78)
    print("               HEURISTIC RETRIEVAL RESULTS")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print("=" * 78)

    print("\n" + "=" * 78)
    # Save to CSV
    summary_df.to_csv("benchmark_triway_results.csv", index=False)
    pd.DataFrame(detailed_table).to_csv("benchmark_detailed_35queries.csv", index=False)
    print("\n[+] Full 35-query benchmark results saved to: benchmark_detailed_35queries.csv")

    return {
        "summary_df": summary_df,
        "text_overlap_metrics": cp_metrics,
        "layout_metrics": dense_metrics,
        "keyword_metrics": text_metrics,
        "detailed_rows": detailed_table
    }


if __name__ == "__main__":
    run_research_benchmark()
