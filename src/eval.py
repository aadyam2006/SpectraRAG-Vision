"""Standalone Ragas smoke experiment over three hand-written examples.

This script does not evaluate outputs from the running SpectraRAG app.
"""

import pandas as pd
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas import evaluate


def run_multimodal_evaluation():
    """
    Evaluates visual document retrieval and answer generation faithfulness.
        Runs Ragas metrics on a tiny illustrative dataset, not app-generated answers.
    """
    sample_dataset = {
        "question": [
            "What was the total operating profit margin in Q3 according to the breakdown chart?",
            "What is the YoY percentage growth for automotive regulatory credits?",
            "Compare Megapack deployment numbers between the two reported quarters."
        ],
        "contexts": [
            ["[Visual Evidence: Page 4 bar chart showing GAAP Operating Margin rebounding to 10.8%]"],
            ["[Visual Evidence: Page 4 table row 'Regulatory credits' displaying $739M vs $286M (+158% YoY)]"],
            ["[Visual Evidence: Page 7 Energy Storage table showing 6.9 GWh deployed]"]
        ],
        "answer": [
            "Based on the bar chart on page 4, the operating margin rebounded to 10.8% in Q3.",
            "According to the financial summary table on page 4, regulatory credits surged by +158% YoY to $739M.",
            "Megapack storage deployments reached 6.9 GWh on page 7, showing a significant margin expansion."
        ],
        "ground_truth": [
            "10.8%",
            "+158% ($739 million vs $286 million)",
            "6.9 GWh"
        ]
    }

    print("[Evaluation] Preparing Ragas evaluation dataset...")
    eval_dataset = EvaluationDataset.from_dict(sample_dataset)

    print("[Evaluation] Running Ragas benchmark (Faithfulness, Relevancy, Context Recall)...")
    try:
        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall]
        )
        df = result.to_pandas()
        print("\n=== Benchmark Summary ===")
        print(df[["faithfulness", "answer_relevancy", "context_recall"]])
        df.to_csv("benchmark_results.csv", index=False)
        print("Results exported to benchmark_results.csv")
    except Exception as e:
        print(f"[Evaluation Note] To run full online LLM judge evaluation with Ragas, configure OPENAI_API_KEY. Error: {e}")


if __name__ == "__main__":
    run_multimodal_evaluation()
