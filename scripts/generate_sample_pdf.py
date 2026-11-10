"""
Script to generate a sample multi-column financial PDF with a table and chart
for trying the PDF preview and text-search prototype.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pathlib import Path

output_pdf = Path(__file__).resolve().parents[1] / "data" / "input_pdfs" / "sample_financial_report.pdf"
os.makedirs("data/input_pdfs", exist_ok=True)

with PdfPages(output_pdf) as pdf:
    # PAGE 1: Title, dual column text, and financial summary table
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Title
    fig.text(0.1, 0.93, "ACME TECH CORP - Q3 FINANCIAL REPORT", fontsize=18, fontweight='bold', color='#1e293b')
    fig.text(0.1, 0.90, "Unaudited Operational & Revenue Breakdown | Fiscal Year 2024", fontsize=11, color='#64748b')
    
    # Dual column text
    col1_text = (
        "Executive Summary:\n"
        "Q3 marked strong margin expansion driven\n"
        "by enterprise cloud contracts and higher\n"
        "regulatory credit recognition. Operating\n"
        "cash flow surged by 34% year-over-year.\n"
        "Supply chain efficiencies offset freight costs."
    )
    col2_text = (
        "Segment Outlook:\n"
        "The North American enterprise division\n"
        "outperformed expectations, reaching $4.8B.\n"
        "Hardware shipments grew moderately while\n"
        "Subscription services achieved record gross\n"
        "margins of 71.4%."
    )
    fig.text(0.1, 0.77, col1_text, fontsize=10, color='#334155', linespacing=1.4)
    fig.text(0.55, 0.77, col2_text, fontsize=10, color='#334155', linespacing=1.4)

    # Financial Table
    table_data = [
        ["Financial Metric ($M)", "Q3 2023", "Q3 2024", "YoY Change"],
        ["Subscription Revenue", "$3,120", "$4,410", "+41.3%"],
        ["Hardware & Systems", "$2,850", "$2,990", "+4.9%"],
        ["Regulatory Credits", "$280", "$715", "+155.4%"],
        ["Operating Income", "$1,150", "$1,890", "+64.3%"],
        ["Operating Margin", "18.5%", "23.2%", "+470 bps"]
    ]
    table = plt.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.45, 0.8, 0.22]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4f46e5')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f8fafc')
        cell.set_edgecolor('#cbd5e1')

    # Notes
    fig.text(0.1, 0.38, "Note: Regulatory credits surged by +155.4% due to milestone deliveries recognized in September.", fontsize=9, fontstyle='italic', color='#64748b')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # PAGE 2: Chart Page (Operating Margin & Revenue Mix)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
    
    quarters = ['Q4-23', 'Q1-24', 'Q2-24', 'Q3-24']
    margins = [16.8, 14.5, 17.2, 23.2]
    colors = ['#94a3b8', '#94a3b8', '#94a3b8', '#4f46e5']
    
    ax1.bar(quarters, margins, color=colors, width=0.45)
    ax1.set_title("Operating Margin Trend FY23-FY24 (%)", fontsize=13, fontweight='bold', color='#1e293b', pad=12)
    ax1.set_ylabel("Margin %", fontsize=10)
    ax1.set_ylim(0, 30)
    for i, v in enumerate(margins):
        ax1.text(i, v + 0.8, f"{v}%", ha='center', fontweight='bold' if i == 3 else 'normal', color='#1e293b' if i == 3 else '#64748b')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Pie / Distribution Chart
    labels = ['Enterprise Subscriptions', 'Hardware Systems', 'Regulatory Credits', 'Professional Services']
    shares = [54, 31, 9, 6]
    chart_colors = ['#4f46e5', '#38bdf8', '#34d399', '#f59e0b']
    ax2.pie(shares, labels=labels, autopct='%1.0f%%', startangle=140, colors=chart_colors)
    ax2.set_title("Q3 2024 Revenue Breakdown by Segment", fontsize=13, fontweight='bold', color='#1e293b', pad=12)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("Generated sample financial PDF at:", output_pdf)
