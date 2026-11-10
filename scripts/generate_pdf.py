import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
paths = [
    project_root / "sample_financial_report.pdf",
    project_root / "data" / "input_pdfs" / "sample_financial_report.pdf",
]

for p in paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with PdfPages(p) as pdf:
        # Page 1
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        fig.text(0.1, 0.93, 'ACME TECH CORP - Q3 FINANCIAL REPORT', fontsize=18, fontweight='bold', color='#1e293b')
        fig.text(0.1, 0.90, 'Unaudited Operational & Revenue Breakdown | Fiscal Year 2024', fontsize=11, color='#64748b')
        fig.text(0.1, 0.77, 'Executive Summary:\nQ3 marked strong margin expansion driven by\nenterprise cloud contracts and higher regulatory credit recognition.\nOperating cash flow surged by 34% YoY.', fontsize=10, color='#334155')
        table_data = [
            ['Financial Metric ($M)', 'Q3 2023', 'Q3 2024', 'YoY Change'],
            ['Subscription Revenue', '$3,120', '$4,410', '+41.3%'],
            ['Hardware & Systems', '$2,850', '$2,990', '+4.9%'],
            ['Regulatory Credits', '$280', '$715', '+155.4%'],
            ['Operating Income', '$1,150', '$1,890', '+64.3%'],
            ['Operating Margin', '18.5%', '23.2%', '+470 bps']
        ]
        table = plt.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0.1, 0.45, 0.8, 0.22])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2
        fig, ax = plt.subplots(figsize=(8.5, 11))
        quarters = ['Q4-23', 'Q1-24', 'Q2-24', 'Q3-24']
        margins = [16.8, 14.5, 17.2, 23.2]
        colors = ['#94a3b8', '#94a3b8', '#94a3b8', '#4f46e5']
        ax.bar(quarters, margins, color=colors, width=0.45)
        ax.set_title('Operating Margin Trend FY23-FY24 (%)', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 30)
        for i, v in enumerate(margins):
            ax.text(i, v + 0.8, f'{v}%', ha='center', fontweight='bold' if i == 3 else 'normal')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    print('Successfully created:', p)
