"""
Generate sample PDF using PIL (already installed in .venv)
"""
import os
from PIL import Image, ImageDraw, ImageFont

paths = [
    r"C:\Users\Admin\.gemini\antigravity\scratch\multimodal-rag-colpali\sample_financial_report.pdf",
    r"C:\Users\Admin\.gemini\antigravity\scratch\multimodal-rag-colpali\data\input_pdfs\sample_financial_report.pdf"
]

# Page 1: Financial Table
img1 = Image.new("RGB", (800, 1000), color=(255, 255, 255))
draw1 = ImageDraw.Draw(img1)

# Header
draw1.rectangle([0, 0, 800, 100], fill=(30, 41, 59))
draw1.text((40, 30), "ACME TECH CORP - Q3 FINANCIAL REPORT", fill=(255, 255, 255))
draw1.text((40, 60), "Unaudited Operational & Revenue Breakdown (FY 2024)", fill=(148, 163, 184))

# Paragraph text
draw1.text((40, 130), "Executive Summary:", fill=(15, 23, 42))
exec_summary = (
    "Q3 marked strong margin expansion driven by enterprise cloud contracts and higher\n"
    "regulatory credit recognition. Operating cash flow surged by 34% year-over-year.\n"
    "Supply chain efficiencies and regional manufacturing offset freight pressures."
)
draw1.text((40, 160), exec_summary, fill=(51, 65, 85))

# Table Box
draw1.rectangle([40, 260, 760, 520], outline=(203, 213, 225), width=2)
# Table Header
draw1.rectangle([40, 260, 760, 310], fill=(79, 70, 229))
draw1.text((60, 275), "Financial Metric ($M)", fill=(255, 255, 255))
draw1.text((320, 275), "Q3 2023", fill=(255, 255, 255))
draw1.text((480, 275), "Q3 2024", fill=(255, 255, 255))
draw1.text((640, 275), "YoY Change", fill=(255, 255, 255))

# Rows
rows = [
    ("Subscription Revenue", "$3,120", "$4,410", "+41.3%"),
    ("Hardware & Systems", "$2,850", "$2,990", "+4.9%"),
    ("Regulatory Credits", "$280", "$715", "+155.4%"),
    ("Operating Income", "$1,150", "$1,890", "+64.3%"),
    ("Operating Margin", "18.5%", "23.2%", "+470 bps")
]

y = 310
for i, (m, q3, q4, yoy) in enumerate(rows):
    bg_color = (248, 250, 252) if i % 2 == 1 else (255, 255, 255)
    draw1.rectangle([40, y, 760, y + 42], fill=bg_color, outline=(226, 232, 240))
    draw1.text((60, y + 12), m, fill=(30, 41, 59))
    draw1.text((320, y + 12), q3, fill=(71, 85, 105))
    draw1.text((480, y + 12), q4, fill=(15, 23, 42))
    draw1.text((640, y + 12), yoy, fill=(16, 185, 129))
    y += 42

draw1.text((40, 550), "* Note: Regulatory credits surged by +155.4% to $715M due to automotive milestones.", fill=(100, 116, 139))


# Page 2: Bar Chart
img2 = Image.new("RGB", (800, 1000), color=(255, 255, 255))
draw2 = ImageDraw.Draw(img2)

draw2.rectangle([0, 0, 800, 80], fill=(30, 41, 59))
draw2.text((40, 30), "Operating Margin Trend FY23-FY24 (%)", fill=(255, 255, 255))

# Draw Bar Chart
bars = [
    ("Q4-23", 16.8, (148, 163, 184)),
    ("Q1-24", 14.5, (148, 163, 184)),
    ("Q2-24", 17.2, (148, 163, 184)),
    ("Q3-24", 23.2, (79, 70, 229))  # Highlighted
]

chart_bottom = 500
chart_scale = 14  # pixels per %

draw2.line([100, chart_bottom, 700, chart_bottom], fill=(148, 163, 184), width=2)

for i, (label, val, col) in enumerate(bars):
    x = 160 + i * 140
    bar_height = int(val * chart_scale)
    y_top = chart_bottom - bar_height
    draw2.rectangle([x, y_top, x + 70, chart_bottom], fill=col)
    draw2.text((x + 15, y_top - 25), f"{val}%", fill=(15, 23, 42))
    draw2.text((x + 15, chart_bottom + 15), label, fill=(71, 85, 105))

draw2.text((100, 560), "Key Takeaway: Operating margin reached 23.2% in Q3-24, reversing previous declines.", fill=(51, 65, 85))

# Save both
for p in paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    img1.save(p, save_all=True, append_images=[img2])
    print("Saved sample PDF to:", p)
