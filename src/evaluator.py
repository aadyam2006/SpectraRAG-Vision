"""Metric helpers for the experimental page-retrieval benchmark."""

import numpy as np
from typing import List, Dict, Any


class SystemEvaluator:
    """Calculates retrieval metrics from supplied per-query results."""

    @staticmethod
    def calculate_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates Recall@1, 3, 5, Precision@1, 3, MRR, and Latency percentiles."""
        if not results:
            return {
                "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
                "precision@1": 0.0, "precision@3": 0.0, "mrr": 0.0,
                "latency_avg_ms": 0.0, "latency_p95_ms": 0.0
            }

        n = len(results)
        r1_hits = sum(r.get("r@1", 0) for r in results)
        r3_hits = sum(r.get("r@3", 0) for r in results)
        r5_hits = sum(r.get("r@5", 0) for r in results)
        mrr_total = sum(r.get("mrr", 0.0) for r in results)
        
        latencies = [r.get("latency_ms", 0.0) for r in results]
        lat_avg = float(np.mean(latencies)) if latencies else 0.0
        lat_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        return {
            "recall@1": round((r1_hits / n) * 100.0, 1),
            "recall@3": round((r3_hits / n) * 100.0, 1),
            "recall@5": round((r5_hits / n) * 100.0, 1),
            "precision@1": round((r1_hits / n) * 100.0, 1), # Single relevant target page
            "precision@3": round((r3_hits / (n * 3.0)) * 100.0, 1),
            "mrr": round(mrr_total / n, 3),
            "latency_avg_ms": round(lat_avg, 3),
            "latency_p95_ms": round(lat_p95, 3)
        }
