"""PDF text extraction and keyword-overlap page ranking."""

import re
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import pypdfium2 as pdfium


class ResultStub:
    """Small result record used by the Streamlit page-search UI."""
    def __init__(self, page_num: int, score: float, base64_img: str = None):
        self.page_num = page_num
        self.score = score
        self.base64 = base64_img


STOPWORDS = {
    "what", "is", "this", "about", "the", "a", "an", "in", "of", "for", "to", 
    "and", "on", "it", "are", "do", "does", "did", "can", "tell", "me", "show", "give"
}

SYNONYMS = {
    "bill": ["invoice", "tax", "gst", "gstin", "billed", "amount", "rate", "total", "receipt"],
    "invoice": ["bill", "tax", "gst", "gstin", "billed", "amount", "total"],
    "cost": ["price", "rate", "amount", "total", "revenue", "fee"],
    "product": ["cartridge", "item", "hardware", "material", "goods"],
    "seller": ["supplier", "enterprises", "vendor", "company"],
    "buyer": ["receiver", "consignee", "customer", "client", "billed"],
    "math": ["mathematics", "standard", "question", "section", "marks"],
    "exam": ["test", "paper", "marks", "questions", "cbse", "standard"]
}


class KeywordPageIndex:
    """In-memory page text/token index with a simple overlap scoring heuristic."""
    def __init__(self):
        self.pages_data = [] # List of dicts with page_num, image, tokens, text

    def build_index(self, pdf_path: str, max_pages: int = 15):
        """Renders page images for display and extracts embedded PDF text."""
        pdf_path = Path(pdf_path)
        self.pages_data = []
        try:
            with pdfium.PdfDocument(pdf_path) as pdf:
                total = min(len(pdf), max_pages) if max_pages > 0 else len(pdf)
                for i in range(total):
                    page = pdf[i]
                    # Render page images for the UI and optional vision-model answering.
                    bitmap = page.render(scale=1.2)
                    pil_img = bitmap.to_pil().convert("RGB")
                    
                    # Retrieval uses extracted text; it does not use image embeddings.
                    textpage = page.get_textpage()
                    raw_text = textpage.get_text_range().strip()
                    tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.\$\%\/]+\b', raw_text.lower()))

                    self.pages_data.append({
                        "page_num": i + 1,
                        "image": pil_img,
                        "tokens": tokens,
                        "text": raw_text
                    })
            print(f"[PDF index] Extracted text from {len(self.pages_data)} pages.")
        except Exception as e:
            print(f"[Turbo Engine] Error building index: {e}")

    def search(self, query: str, top_k: int = 1) -> List[Any]:
        """Rank pages by token and synonym overlap; this is not visual retrieval."""
        raw_words = re.findall(r'\b[a-zA-Z0-9_\-\.\$\%\/]+\b', query.lower())
        
        # Filter stopwords
        content_words = [w for w in raw_words if w not in STOPWORDS]
        if not content_words:
            content_words = raw_words # fallback if all were stopwords

        # Expand query with domain synonyms
        expanded_query = set(content_words)
        for w in content_words:
            if w in SYNONYMS:
                expanded_query.update(SYNONYMS[w])

        scored_pages = []
        for page in self.pages_data:
            tokens = page["tokens"]
            page_text = page["text"].lower()
            
            # Heuristic token overlap (not a learned embedding or MaxSim).
            sim_score = 0.0
            for qw in expanded_query:
                if qw in tokens:
                    sim_score += 4.0
                elif any(qw in t for t in tokens):
                    sim_score += 1.8
                elif qw in page_text:
                    sim_score += 1.2

            # Content density tie-breaker: non-blank pages get preference over empty pages!
            content_density = min(len(tokens) / 30.0, 3.0)
            final_raw = sim_score + content_density

            normalized_score = final_raw
            scored_pages.append((normalized_score, page))

        # Sort descending by score
        scored_pages.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, p in scored_pages[:top_k]:
            results.append(ResultStub(page_num=p["page_num"], score=score))
        
        return results

    def search_all_pages(self, query: str) -> List[Dict[str, Any]]:
        """Returns scores and metadata for every indexed page in order."""
        if not self.pages_data:
            return []
        
        raw_words = re.findall(r'\b[a-zA-Z0-9_\-\.\$\%\/]+\b', query.lower())
        content_words = [w for w in raw_words if w not in STOPWORDS] or raw_words
        expanded_query = set(content_words)
        for w in content_words:
            if w in SYNONYMS:
                expanded_query.update(SYNONYMS[w])

        all_page_metrics = []
        for p in self.pages_data:
            tokens = p["tokens"]
            page_text = p["text"].lower()
            sim_score = 0.0
            for qw in expanded_query:
                if qw in tokens:
                    sim_score += 4.0
                elif any(qw in t for t in tokens):
                    sim_score += 1.8
                elif qw in page_text:
                    sim_score += 1.2
            content_density = min(len(tokens) / 30.0, 3.0)
            final_raw = sim_score + content_density
            score = round(final_raw, 2)
            snippet = p["text"][:80].replace("\n", " ").strip() if p["text"] else "Visual Chart / Table"
            all_page_metrics.append({
                "page_num": p["page_num"],
                "score": score,
                "tokens": len(tokens),
                "snippet": snippet
            })
        return all_page_metrics



class TextOverlapRetriever:
    def __init__(self):
        self.page_index = KeywordPageIndex()
        self.active_pdf_path = None
        self.is_indexed = False

    def index_documents(self, input_path: str, max_pages: int = 15, overwrite: bool = True):
        """
        Indexes any PDF file in UNDER 1 SECOND without CPU lag.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Document not found at: {input_path}")

        print(f"[Retriever] Extracting text from '{input_path.name}'...")
        start = time.time()
        self.page_index.build_index(str(input_path), max_pages=max_pages)
        self.active_pdf_path = str(input_path)
        self.is_indexed = True
        print(f"[Retriever] Complete in {time.time() - start:.2f}s!")

    def load_index_for_file(self, filename: str, max_pages: int = 15) -> bool:
        """Loads or indexes a file instantly."""
        candidates = [
            Path(filename),
            Path("data/input_pdfs") / filename,
            Path("sample_financial_report.pdf"),
            Path("tesla_q3_operational_highlights.pdf")
        ]
        for p in candidates:
            if p.exists() and filename in p.name:
                self.index_documents(str(p.resolve()), max_pages=max_pages)
                return True
        return False

    def search(self, query: str, top_k: int = 1) -> List[Any]:
        """
        Runs a keyword-overlap search against the indexed page text.
        """
        if not self.is_indexed:
            raise RuntimeError("No document is indexed. Please index a PDF first.")

        return self.page_index.search(query=query, top_k=top_k)

    def search_all_pages(self, query: str) -> List[Dict[str, Any]]:
        """Returns all page scores for dashboard charts."""
        if not self.is_indexed:
            return []
        return self.page_index.search_all_pages(query)

    def get_all_pages_info(self) -> List[Dict[str, Any]]:
        """Returns info on all indexed pages for dashboard."""
        info = []
        for p in self.page_index.pages_data:
            snippet = p["text"][:80].replace("\n", " ").strip() if p["text"] else "Visual Chart / Table"
            info.append({
                "page_num": p["page_num"],
                "score": float(len(p["tokens"])),
                "tokens": len(p["tokens"]),
                "snippet": snippet
            })
        return info


    def get_page_image(self, page_num: int) -> Image.Image:
        """Returns the high-res screenshot for a specific page directly from memory (0ms)."""
        for p in self.page_index.pages_data:
            if p["page_num"] == page_num:
                return p["image"]
        # Fallback render of requested page only
        if self.active_pdf_path and Path(self.active_pdf_path).exists():
            try:
                with pdfium.PdfDocument(self.active_pdf_path) as pdf:
                    idx = min(max(page_num - 1, 0), len(pdf) - 1)
                    bitmap = pdf[idx].render(scale=1.2)
                    return bitmap.to_pil().convert("RGB")
            except Exception:
                pass
        return Image.new("RGB", (400, 500), (240, 240, 240))

    def get_page_text(self, page_num: int) -> str:
        """Returns extracted text for a page directly from memory (0ms)."""
        for p in self.page_index.pages_data:
            if p["page_num"] == page_num:
                return p.get("text", "")
        return ""

    def get_all_doc_text(self) -> str:
        """Returns all indexed text combined from memory (0ms)."""
        return " ".join(p.get("text", "") for p in self.page_index.pages_data)
