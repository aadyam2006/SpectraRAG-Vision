"""
create_showcase_pdfs.py - Generates beautiful showcase PDFs for instant testing.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw

def create_tesla_pdf():
    project_root = Path(__file__).resolve().parents[1]
    pdf_path = project_root / "tesla_q3_operational_highlights.pdf"
    img1 = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)

    # Header
    draw1.rectangle([0, 0, 800, 90], fill=(220, 38, 38)) # Tesla Red
    draw1.text((40, 25), "TESLA INC. - Q3 OPERATIONAL & ENERGY HIGHLIGHTS", fill=(255, 255, 255))
    draw1.text((40, 55), "Energy Storage Deployments, Automotive Delivery & Margins", fill=(254, 202, 202))

    # Text Block
    draw1.text((40, 120), "Key Operational Developments:", fill=(15, 23, 42))
    notes = (
        "• Megapack installations reached an all-time record of 6.9 GWh in Q3.\n"
        "• Automotive regulatory credits surged to $739M, expanding GAAP margins.\n"
        "• Cybertruck achieved positive gross margin for the first time in company history.\n"
        "• Total cash and investments rose to $33.6 Billion."
    )
    draw1.text((40, 150), notes, fill=(51, 65, 85))

    # Segment Table
    draw1.rectangle([40, 260, 760, 480], outline=(203, 213, 225), width=2)
    draw1.rectangle([40, 260, 760, 310], fill=(30, 41, 59))
    draw1.text((60, 275), "Division", fill=(255, 255, 255))
    draw1.text((300, 275), "Revenue ($M)", fill=(255, 255, 255))
    draw1.text((480, 275), "Gross Margin", fill=(255, 255, 255))
    draw1.text((640, 275), "YoY Growth", fill=(255, 255, 255))

    rows = [
        ("Automotive Segment", "$20,016M", "17.1%", "+2.0%"),
        ("Energy Generation & Storage", "$2,376M", "30.5%", "+52.4%"),
        ("Services & Other", "$2,790M", "8.8%", "+29.1%"),
        ("Total Company Operations", "$25,182M", "19.8%", "+7.8%")
    ]
    y = 310
    for i, (div, rev, margin, yoy) in enumerate(rows):
        bg = (248, 250, 252) if i % 2 == 1 else (255, 255, 255)
        draw1.rectangle([40, y, 760, y + 42], fill=bg, outline=(226, 232, 240))
        draw1.text((60, y + 12), div, fill=(30, 41, 59))
        draw1.text((300, y + 12), rev, fill=(15, 23, 42))
        draw1.text((480, y + 12), margin, fill=(220, 38, 38))
        draw1.text((640, y + 12), yoy, fill=(16, 185, 129))
        y += 42

    img1.save(pdf_path)
    input_dir = project_root / "data" / "input_pdfs"
    input_dir.mkdir(parents=True, exist_ok=True)
    img1.save(input_dir / pdf_path.name)
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    create_tesla_pdf()
