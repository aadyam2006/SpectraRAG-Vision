"""Toy page-layout heuristic baseline (not a CLIP implementation)."""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import pypdfium2 as pdfium


class LayoutHeuristicBaseline:
    """
    Uses simple grayscale thumbnail statistics as a layout proxy.
    It does not create learned embeddings and is not comparable to CLIP.
    """
    def __init__(self):
        self.page_descriptors = []

    def index_pdf(self, pdf_path: str | Path, max_pages: int = 15):
        """Indexes pages using grayscale thumbnail statistics."""
        self.page_descriptors = []
        try:
            with pdfium.PdfDocument(pdf_path) as pdf:
                limit = min(len(pdf), max_pages) if max_pages > 0 else len(pdf)
                for i in range(limit):
                    page = pdf[i]
                    # Render a thumbnail and retain simple grayscale statistics.
                    bitmap = page.render(scale=0.5)
                    pil_img = bitmap.to_pil().convert("L") # Grayscale
                    arr = np.array(pil_img.resize((32, 16)), dtype=np.float32)
                    
                    # These values are a toy heuristic, not a learned representation.
                    self.page_descriptors.append({
                        "page_num": i + 1,
                        "mean_luminance": float(np.mean(arr)),
                        "variance": float(np.var(arr))
                    })
            print(f"[Layout heuristic] Indexed {len(self.page_descriptors)} page thumbnails.")
        except Exception as e:
            print(f"[Layout heuristic] Indexing error: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rank pages with keyword intent and thumbnail variance heuristics.
        """
        if not self.page_descriptors:
            return []

        q_lower = query.lower()
        # Visual query intent weights
        is_visual_query = any(k in q_lower for k in ["chart", "graph", "diagram", "trend", "visual", "bar", "plot", "drawing"])
        is_table_query = any(k in q_lower for k in ["table", "revenue", "credits", "tax", "gst", "invoice", "total"])

        scored = []
        for p in self.page_descriptors:
            # Thumbnail variance is used as a rough proxy for page visual complexity.
            visual_weight = p["variance"] / 1000.0
            
            if is_visual_query:
                # This hand-built heuristic gives visually varied pages a small boost.
                score = 0.65 + min(visual_weight * 0.15, 0.25)
            elif is_table_query:
                # Small heuristic boost for table-like query terms.
                score = 0.50 + min(visual_weight * 0.08, 0.20)
            else:
                score = 0.40 + (0.05 * (p["page_num"] == 1))

            scored.append({
                "page_num": p["page_num"],
                "score": round(float(score), 3)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
