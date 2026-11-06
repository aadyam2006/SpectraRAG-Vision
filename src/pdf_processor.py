"""
src/pdf_processor.py - PDF to high-fidelity PIL image renderer.
Uses pypdfium2 (self-contained binaries, no external Poppler required on Windows).
"""

from pathlib import Path
from typing import List
from PIL import Image
import pypdfium2 as pdfium


class PDFProcessor:
    @staticmethod
    def convert_pdf_to_images(pdf_path: str | Path, scale: float = 2.0) -> List[Image.Image]:
        """
        Renders each PDF page to a PIL Image using pypdfium2.
        scale=2.0 renders at ~144 DPI for optimal visual clarity with ColPali & Vision-LLMs.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        pdf = pdfium.PdfDocument(pdf_path)
        images = []
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            images.append(pil_image)

        return images
