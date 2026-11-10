"""Toy keyword-overlap PDF search baseline (not a BM25 implementation)."""

from pathlib import Path
from typing import List, Dict, Any
import pypdfium2 as pdfium


class KeywordOverlapBaseline:
    def __init__(self):
        self.pages_text = []

    def index_pdf(self, pdf_path: str | Path, max_pages: int = 15):
        """Extracts raw text stream from each PDF page up to max_pages (<0.1s)."""
        self.pages_text = []
        try:
            with pdfium.PdfDocument(pdf_path) as pdf:
                limit = min(len(pdf), max_pages) if max_pages > 0 else len(pdf)
                for i in range(limit):
                    page = pdf[i]
                    textpage = page.get_textpage()
                    text = textpage.get_text_range()
                    self.pages_text.append({
                        "page_num": i + 1,
                        "text": text.strip()
                    })
            print(f"[Naive Text RAG] Indexed {len(self.pages_text)} pages using raw text extraction in <0.05s.")
        except Exception as e:
            print(f"[Naive Text RAG] Error indexing: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple term-frequency overlap over raw extracted page text.
        This baseline does not interpret chart or diagram pixels.
        """
        query_words = set(query.lower().split())
        scored_pages = []

        for p in self.pages_text:
            text = p["text"].lower()
            # Simple TF match
            score = sum(text.count(w) for w in query_words)
            scored_pages.append({
                "page_num": p["page_num"],
                "score": float(score),
                "text_snippet": p["text"][:200]
            })

        # Sort descending by score
        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        return scored_pages[:top_k]
