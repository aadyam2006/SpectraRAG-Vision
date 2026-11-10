"""
app.py - SpectraRAG Vision: PDF Q&A prototype.
Comprehensive implementation featuring:
1. Modern Outstaff Neo-Dashboard UI
2. Multi-Page Document Preview & Filmstrip Gallery
3. Embedded-text page ranking with a PDF page preview
4. Page references and click-to-jump navigation
5. Live Retrieval Evaluation (Recall@K, MRR, Precision@K) with Altair charts
6. Side-by-side comparison of prototype ranking heuristics
7. Multi-Turn Chat Experience with Persistent Conversation History
8. Robust Error Handling (file validation, empty query protection, graceful fallbacks)
"""

import streamlit as st
from pathlib import Path
import tempfile
import io
import time
import base64
import math
import re
import datetime
import pandas as pd
import altair as alt
from PIL import Image, ImageDraw
import pypdfium2 as pdfium

# Core modules
from src.retriever import TextOverlapRetriever
from src.generator import MultimodalGenerator
from src.keyword_baseline import KeywordOverlapBaseline
from src.layout_heuristic import LayoutHeuristicBaseline
from benchmark_recall import BENCHMARK_DATASET, run_research_benchmark

# 1. Page Configuration
st.set_page_config(
    page_title="SpectraRAG Vision • Multimodal Document Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Outstaff Neo-Dashboard CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    .stApp {
        background-color: #F4F5FA !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: #1E293B !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E8ECF4 !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #334155 !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Outstaff Cards */
    .outstaff-card {
        background: #FFFFFF;
        border-radius: 22px;
        border: 1px solid #E8ECF4;
        box-shadow: 0 6px 24px rgba(100, 116, 139, 0.04);
        padding: 22px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .outstaff-card:hover {
        box-shadow: 0 10px 32px rgba(100, 116, 139, 0.07);
    }
    
    .card-title {
        font-size: 17px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .card-subtitle {
        font-size: 12px;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 14px;
    }

    /* Outstaff Capsule Bars */
    .capsule-group {
        display: flex;
        align-items: flex-end;
        justify-content: space-around;
        height: 150px;
        padding: 8px 0;
    }
    .capsule-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        flex: 1;
    }
    .capsule-val {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
    }
    .capsule-track {
        width: 32px;
        height: 110px;
        background: #E0F2FE;
        border-radius: 16px;
        display: flex;
        align-items: flex-end;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }
    .capsule-fill {
        width: 100%;
        background: #0284C7;
        border-radius: 16px;
        transition: height 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .capsule-fill.highlight {
        background: linear-gradient(180deg, #38BDF8 0%, #0284C7 100%);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
    }
    .capsule-lbl {
        font-size: 12px;
        font-weight: 600;
        color: #94A3B8;
    }

    /* Status Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .pill-done { background: #D1FAE5; color: #047857; }
    .pill-progress { background: #DBEAFE; color: #1E40AF; }
    .pill-todo { background: #EDE9FE; color: #6D28D9; }
    .pill-cyan { background: #E0F2FE; color: #0369A1; }
    .pill-dark { background: #18181B; color: #FFFFFF; }
    .pill-gray { background: #F1F5F9; color: #475569; }
    .pill-red { background: #FEE2E2; color: #B91C1C; }

    /* Modern Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        padding-bottom: 8px;
        flex-wrap: wrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 9999px !important;
        padding: 0 16px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E8ECF4 !important;
        color: #64748B !important;
        font-weight: 700 !important;
        font-size: 12.5px !important;
        white-space: nowrap !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #18181B !important;
        border-color: #18181B !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(24, 24, 27, 0.2) !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: 1px solid #E8ECF4 !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.2s ease !important;
        padding: 8px 16px !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        border-color: #8B5CF6 !important;
        color: #8B5CF6 !important;
        box-shadow: 0 6px 16px rgba(139, 92, 246, 0.15) !important;
    }

    .dark-metric-card {
        background: #18181B;
        border-radius: 20px;
        padding: 22px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.14);
    }

    .ai-assistant-card {
        background: linear-gradient(145deg, #EEF2FF 0%, #E0E7FF 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 20px;
        padding: 18px;
        margin-top: 24px;
    }

    .task-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .task-table th {
        text-align: left;
        color: #94A3B8;
        font-weight: 600;
        padding: 10px 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .task-table td {
        padding: 12px;
        border-bottom: 1px solid #F8FAFC;
        color: #1E293B;
    }
    .task-table tr:hover td {
        background-color: #F8FAFC;
    }

    /* Citation Box */
    .citation-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 10px 16px;
        margin-top: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Filmstrip Card */
    .page-thumb-card {
        background: #FFFFFF;
        border: 1px solid #E8ECF4;
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .page-thumb-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        border-color: #8B5CF6;
    }
</style>
""", unsafe_allow_html=True)


# 3. Helper Functions: Page Highlighting & Thumbnail Rendering
def render_circular_ring(pct: int, label: str, sublabel: str, color: str) -> str:
    r = 38
    c = 2 * math.pi * r
    offset = c * (1 - (min(pct, 100) / 100.0))
    return f'<div style="text-align: center; display: inline-block; margin: 0 8px;"><svg width="86" height="86" viewBox="0 0 100 100"><circle cx="50" cy="50" r="{r}" fill="none" stroke="#F1F5F9" stroke-width="8" /><circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="8" stroke-dasharray="{c:.2f}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round" transform="rotate(-90 50 50)" /><text x="50" y="56" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="800" fill="#0F172A">{label}</text></svg><div style="font-size: 11px; font-weight: 600; color: #64748B; margin-top: 4px;">{sublabel}</div></div>'



@st.cache_data(show_spinner=False)
def get_cached_disk_page_image(pdf_path: str, page_num: int) -> Image.Image:
    """Fallback: Renders base page from disk once and caches in RAM (<0.2s)."""
    if not pdf_path or not Path(pdf_path).exists():
        return Image.new("RGB", (400, 500), (245, 245, 245))
    try:
        with pdfium.PdfDocument(pdf_path) as pdf:
            idx = min(max(page_num - 1, 0), len(pdf) - 1)
            bitmap = pdf[idx].render(scale=1.2)
            return bitmap.to_pil().convert("RGB")
    except Exception:
        return Image.new("RGB", (400, 500), (245, 245, 245))


def render_page_with_highlight(pdf_path: str, page_num: int, query: str = "") -> Image.Image:
    """Render a page preview and optionally highlight query terms in extracted text."""
    if not pdf_path:
        return Image.new("RGB", (400, 500), (245, 245, 245))

    # Reuse the page image rendered while extracting PDF text.
    base_img = None
    if "retriever" in st.session_state and hasattr(st.session_state.retriever, "get_page_image"):
        base_img = st.session_state.retriever.get_page_image(page_num)

    if base_img is None:
        base_img = get_cached_disk_page_image(pdf_path, page_num)

    if not query:
        return base_img

    # The overlay is an approximate visual guide, not a text-coordinate match.
    pil_img = base_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    iw, ih = pil_img.size
    q_lower = query.lower()

    # Choose a rough page region based on query keywords.
    if any(k in q_lower for k in ["skill", "tech", "programming", "language", "stack", "tool", "framework"]):
        y0, y1 = int(ih * 0.25), int(ih * 0.44)
    elif any(k in q_lower for k in ["project", "work", "app", "mentor"]):
        y0, y1 = int(ih * 0.44), int(ih * 0.70)
    elif any(k in q_lower for k in ["certif", "nptel", "course"]):
        y0, y1 = int(ih * 0.75), int(ih * 0.88)
    elif any(k in q_lower for k in ["education", "college", "degree", "btech"]):
        y0, y1 = int(ih * 0.11), int(ih * 0.24)
    elif any(k in q_lower for k in ["megapack", "storage", "energy", "deployment"]):
        y0, y1 = int(ih * 0.28), int(ih * 0.52)
    elif any(k in q_lower for k in ["regulatory", "credits", "subscription", "revenue"]):
        y0, y1 = int(ih * 0.20), int(ih * 0.45)
    elif any(k in q_lower for k in ["margin", "chart", "trend", "graph"]):
        y0, y1 = int(ih * 0.32), int(ih * 0.65)
    elif any(k in q_lower for k in ["math", "question", "section", "formula", "mark", "solve"]):
        y0, y1 = int(ih * 0.20), int(ih * 0.55)
    else:
        y0, y1 = int(ih * 0.22), int(ih * 0.52)

    draw.rectangle([25, y0, iw - 25, y1], fill=(139, 92, 246, 35), outline=(139, 92, 246, 230), width=3)
    return Image.alpha_composite(pil_img, overlay).convert("RGB")


@st.cache_data(show_spinner=False)
def get_cached_thumbnails(pdf_path_str: str, max_thumbs: int = 6):
    """Renders small thumbnails for gallery once and caches them in memory."""
    if not pdf_path_str or not Path(pdf_path_str).exists():
        return []
    try:
        with pdfium.PdfDocument(pdf_path_str) as pdf:
            count = min(len(pdf), max_thumbs)
            thumbs = []
            for i in range(count):
                bm = pdf[i].render(scale=0.35)
                thumbs.append(bm.to_pil().convert("RGB"))
            return thumbs
    except Exception:
        return []



# 4. State Initialization (One-time, zero reload overhead)
if "retriever" not in st.session_state or not hasattr(st.session_state.retriever, "search_all_pages"):
    from src.retriever import TextOverlapRetriever
    st.session_state.retriever = TextOverlapRetriever()

if "text_rag" not in st.session_state:
    st.session_state.text_rag = KeywordOverlapBaseline()

if "dense_rag" not in st.session_state:
    st.session_state.dense_rag = LayoutHeuristicBaseline()

if "generator" not in st.session_state:
    from src.generator import MultimodalGenerator
    st.session_state.generator = MultimodalGenerator(mode="fast_demo")

if "active_pdf_path" not in st.session_state:
    sample_path = Path("sample_financial_report.pdf")
    if sample_path.exists():
        st.session_state.active_pdf_path = str(sample_path.resolve())
        st.session_state.active_pdf_name = sample_path.name
        st.session_state.retriever.load_index_for_file(sample_path.name, max_pages=5)
        st.session_state.text_rag.index_pdf(sample_path, max_pages=5)
        st.session_state.dense_rag.index_pdf(sample_path, max_pages=5)
    else:
        st.session_state.active_pdf_path = None
        st.session_state.active_pdf_name = None

# Chat History state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = "What is this document about?"
if "last_top_page" not in st.session_state:
    st.session_state.last_top_page = 1
if "last_top_score" not in st.session_state:
    st.session_state.last_top_score = 0.0
if "last_retrieval_ms" not in st.session_state:
    st.session_state.last_retrieval_ms = 0
if "inspect_page" not in st.session_state:
    st.session_state.inspect_page = 1
if "highlight_region" not in st.session_state:
    st.session_state.highlight_region = True
if "comparator_mode" not in st.session_state:
    st.session_state.comparator_mode = False


# 5. Sidebar Navigation & Vault (Error-Handled)
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding-left: 4px;">
        <div style="width: 36px; height: 36px; border-radius: 12px; background: linear-gradient(135deg, #8B5CF6 0%, #38BDF8 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);">
            <span style="font-size: 18px; color: white;">✦</span>
        </div>
        <div>
            <div style="font-size: 18px; font-weight: 800; color: #0F172A; letter-spacing: -0.5px;">SpectraRAG</div>
            <div style="font-size: 11px; font-weight: 600; color: #94A3B8;">Vision Intelligence v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Engine Settings")
    gen_mode = st.selectbox(
        "Synthesis Engine",
        [
            "⚡ Fast Demo Templates (no LLM)",
            "💻 Local Ollama (LLaVA on CPU)",
            "☁️ Gemini Vision API (Cloud)"
        ],
        index=0
    )
    if "Fast Demo" in gen_mode:
        st.session_state.generator.mode = "fast_demo"
    elif "Gemini" in gen_mode:
        st.session_state.generator.mode = "gemini"
        gemini_key = st.text_input("Gemini API Key", type="password")
        st.session_state.generator.api_key = gemini_key
    else:
        st.session_state.generator.mode = "ollama"
        st.session_state.generator.model_name = "llava"

    top_k = st.slider("Top-K Retrieved Pages", min_value=1, max_value=4, value=1)
    
    st.divider()
    st.markdown("### 📂 Document Vault")
    
    st.markdown("**⚡ Instant Showcase PDFs:**")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📊 Acme Tech Q3", use_container_width=True):
            p = Path("sample_financial_report.pdf")
            if p.exists():
                st.session_state.retriever.load_index_for_file(p.name, max_pages=max_pages)
                st.session_state.text_rag.index_pdf(p, max_pages=max_pages)
                st.session_state.dense_rag.index_pdf(p, max_pages=max_pages)
                st.session_state.active_pdf_path = str(p.resolve())
                st.session_state.active_pdf_name = p.name
                st.session_state.inspect_page = 1
                st.toast(f"Loaded: {p.name}")
                st.rerun()
    with col_btn2:
        if st.button("🚗 Tesla Q3", use_container_width=True):
            p = Path("tesla_q3_operational_highlights.pdf")
            if p.exists():
                st.session_state.retriever.index_documents(str(p.resolve()), max_pages=max_pages, overwrite=False)
                st.session_state.text_rag.index_pdf(p, max_pages=max_pages)
                st.session_state.dense_rag.index_pdf(p, max_pages=max_pages)
                st.session_state.active_pdf_path = str(p.resolve())
                st.session_state.active_pdf_name = p.name
                st.session_state.inspect_page = 1
                st.toast(f"Loaded: {p.name}")
                st.rerun()

    st.markdown("**— Upload Any PDF from PC —**")
    max_pages = st.slider("Pages to Index (Fast CPU Mode)", min_value=1, max_value=15, value=5)
    
    # Error Handling for File Upload
    uploaded_file = st.file_uploader("Upload PDF (Financials, Papers, Invoices, CVs)", type=["pdf"], key="pdf_uploader")
    
    if uploaded_file is not None and uploaded_file.name != st.session_state.get("active_pdf_name"):
        # Validate PDF extension
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.error("⚠️ Unsupported file format. Please upload a valid PDF.")
        else:
            save_dir = Path("data/input_pdfs")
            save_dir.mkdir(parents=True, exist_ok=True)
            dest_path = save_dir / uploaded_file.name
            
            try:
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"⚡ Indexing '{uploaded_file.name}' ({max_pages} pages in <1s)..."):
                    st.session_state.retriever.index_documents(str(dest_path.resolve()), max_pages=max_pages, overwrite=True)
                    st.session_state.text_rag.index_pdf(dest_path, max_pages=max_pages)
                    st.session_state.dense_rag.index_pdf(dest_path, max_pages=max_pages)
                    st.session_state.active_pdf_path = str(dest_path.resolve())
                    st.session_state.active_pdf_name = uploaded_file.name
                    st.session_state.inspect_page = 1
                    st.session_state.messages.clear()
                    st.toast(f"Indexed successfully: {uploaded_file.name}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    if st.session_state.active_pdf_name:
        st.markdown(f"""
        <div style="background: #F1F5F9; border-radius: 12px; padding: 10px 14px; margin-top: 10px; border: 1px solid #E2E8F0;">
            <div style="font-size: 11px; font-weight: 600; color: #64748B;">ACTIVE DOCUMENT</div>
            <div style="font-size: 13px; font-weight: 700; color: #0F172A; word-break: break-all;">{st.session_state.active_pdf_name}</div>
        </div>
        """, unsafe_allow_html=True)

    # Clear Chat Button
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages.clear()
            st.toast("Conversation cleared")
            st.rerun()

    st.markdown("""
    <div class="ai-assistant-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="font-size: 14px; font-weight: 800; color: #4338CA;">AI Assistant</div>
            <span class="pill pill-todo">PDF text search</span>
        </div>
        <p style="font-size: 12px; color: #4B5563; margin: 8px 0 12px 0; line-height: 1.4;">
            Searches text embedded in the PDF and previews the matching page.
        </p>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span class="pill pill-cyan">Turbo Engine</span>
            <span style="font-size: 18px;">⚡</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# 6. Main Dashboard View
active_file_name = st.session_state.get("active_pdf_name", "sample_financial_report.pdf")
pages_info = st.session_state.retriever.get_all_pages_info() if hasattr(st.session_state.retriever, "get_all_pages_info") else []
num_pages = len(pages_info) if pages_info else 1

# Top Header Bar
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; margin-bottom: 20px;">
    <div>
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #0F172A; letter-spacing: -0.5px;">Dashboard</h1>
            <span class="pill pill-cyan">Text search active</span>
            <span class="pill pill-todo">Prototype</span>
        </div>
        <div style="margin-top: 4px; font-size: 13px; font-weight: 500; color: #64748B;">
            Active Document: <strong style="color: #0F172A;">{active_file_name}</strong> ({num_pages} pages) • PDF text-overlap ranking
        </div>
    </div>
    <div style="background: #EDE9FE; border-radius: 18px; padding: 10px 22px; display: flex; align-items: center; gap: 16px; border: 1px solid rgba(139, 92, 246, 0.2); box-shadow: 0 4px 14px rgba(139, 92, 246, 0.08);">
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #6D28D9; text-transform: uppercase;">Text-overlap score</div>
            <div style="font-size: 22px; font-weight: 800; color: #4C1D95;">{st.session_state.last_top_score:.1f}</div>
        </div>
        <div style="font-size: 22px; background: #FFFFFF; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
            ⚡
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# 7. Navigation Tabs (All 8 Features Covered)
tab_studio, tab_preview, tab_compare, tab_analytics, tab_benchmark = st.tabs([
    "⚡ Studio",
    "📑 Page Gallery",
    "⚖️ Compare Heuristics",
    "📊 Analytics",
    "🔬 Benchmark"
])



# ==============================================================================
# TAB 1: PDF Q&A (Text Retrieval + Page References + Chat History)
# ==============================================================================
with tab_studio:
    if not st.session_state.active_pdf_path:
        st.info("👈 Select a showcase document or upload any PDF in the sidebar to begin.")
    else:
        # Split Screen Layout
        col_evidence, col_synthesis = st.columns([1.05, 1], gap="large")

        # LEFT: Retrieved Page Preview
        with col_evidence:
            st.markdown("""
            <div class="card-title">
                <span>🔍 Retrieved Page Preview</span>
                <span class="pill pill-cyan">Page preview</span>
            </div>
            <div class="card-subtitle">Inspect the selected PDF page; retrieval uses embedded text, not image patches</div>
            """, unsafe_allow_html=True)

            # Controls: Region Highlight Toggle & Page Selector
            ctrl_col1, ctrl_col2 = st.columns([1.2, 1])
            with ctrl_col1:
                st.session_state.highlight_region = st.toggle("Show approximate highlight", value=st.session_state.highlight_region)
            with ctrl_col2:
                curr_inspect = min(max(int(st.session_state.get("inspect_page", 1)), 1), max(num_pages, 1))
                selected_page = st.number_input("Inspect Page", min_value=1, max_value=max(num_pages, 1), value=curr_inspect, step=1)
                st.session_state.inspect_page = selected_page

            # Render Page Screenshot (with or without region bounding box)
            if st.session_state.highlight_region:
                annotated_img = render_page_with_highlight(
                    st.session_state.active_pdf_path,
                    st.session_state.inspect_page,
                    st.session_state.last_query
                )
            else:
                annotated_img = render_page_with_highlight(
                    st.session_state.active_pdf_path,
                    st.session_state.inspect_page,
                    ""
                )

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E8ECF4; border-radius: 14px; padding: 10px 16px; margin: 10px 0; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="pill pill-todo">Page {st.session_state.inspect_page}</span>
                    <span style="font-size: 13px; font-weight: 700; color: #0F172A;">Selected PDF page</span>
                </div>
                <span class="pill pill-cyan">Text score: {st.session_state.last_top_score:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

            st.caption("The highlight is approximate and does not identify exact text coordinates.")
            st.image(annotated_img, caption=f"Document page preview • Page {st.session_state.inspect_page}", use_container_width=True)

        # RIGHT: Document Q&A with Conversation History
        with col_synthesis:
            st.markdown("""
            <div class="card-title">
                <span>💡 Document Q&amp;A</span>
                <span class="pill pill-todo">Executive Synthesis</span>
            </div>
            <div class="card-subtitle">Ask questions and see which page the text search retrieved</div>
            """, unsafe_allow_html=True)

            # Quick Prompt Chips based on document type
            st.markdown("<div style='font-size: 11px; font-weight: 700; color: #64748B; margin-bottom: 6px;'>QUICK PROMPTS:</div>", unsafe_allow_html=True)
            chip_col1, chip_col2 = st.columns(2)
            sample_query = None

            if "tesla" in active_file_name.lower():
                with chip_col1:
                    if st.button("🔋 Megapack deployments & margin?", use_container_width=True):
                        sample_query = "What were the Megapack storage deployments and gross margin in Q3?"
                with chip_col2:
                    if st.button("🚗 Cybertruck gross margin milestone?", use_container_width=True):
                        sample_query = "What milestone did Cybertruck achieve for gross margin?"
            elif "invoice" in active_file_name.lower() or "bill" in active_file_name.lower():
                with chip_col1:
                    if st.button("🧾 What is this bill about?", use_container_width=True):
                        sample_query = "What is this bill about and who is the customer?"
                with chip_col2:
                    if st.button("💰 Total amount & GST rate?", use_container_width=True):
                        sample_query = "What is the total invoice amount and GST tax rate applied?"
            elif "resume" in active_file_name.lower() or "cv" in active_file_name.lower():
                with chip_col1:
                    if st.button("💻 What are the technical skills?", use_container_width=True):
                        sample_query = "what are the tech skills?"
                with chip_col2:
                    if st.button("🚀 What projects are listed?", use_container_width=True):
                        sample_query = "what projects did she do?"
            elif "math" in active_file_name.lower() or "question" in active_file_name.lower() or "cbse" in active_file_name.lower():
                with chip_col1:
                    if st.button("📐 Questions breakdown & topics?", use_container_width=True):
                        sample_query = "What topics, question types, and structure are covered in this question bank?"
                with chip_col2:
                    if st.button("📝 High-weightage units on this page?", use_container_width=True):
                        sample_query = "What are the core mathematics topics and formulas shown on this page?"
            else:
                with chip_col1:
                    if st.button("📈 Regulatory credits revenue & YoY?", use_container_width=True):
                        sample_query = "What was the regulatory credits revenue in Q3 2024 and YoY change?"
                with chip_col2:
                    if st.button("📊 Operating margin bar chart trend?", use_container_width=True):
                        sample_query = "What trend does the operating margin bar chart show across FY23-FY24?"

            # Render Past Chat History
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="🧑‍💻"):
                            st.markdown(f"**{msg['content']}**")
                    else:
                        with st.chat_message("assistant", avatar="🔮"):
                            st.markdown(msg["content"])
                            st.markdown(f'<div class="citation-card"><div style="display: flex; align-items: center; gap: 8px;"><span class="pill pill-done">📌 Retrieved Page {msg.get("target_page", 1)}</span><span style="font-size: 12px; font-weight: 600; color: #64748B;">{active_file_name}</span></div><span class="pill pill-cyan">Search {msg.get("latency_ms", 0)}ms</span></div>', unsafe_allow_html=True)

            # Chat Input Box
            query = st.chat_input("Ask about charts, tables, diagrams, or numbers in this document...")
            if sample_query:
                query = sample_query

            # Error Handling for Empty Queries
            if query is not None:
                query_clean = query.strip()
                if not query_clean:
                    st.warning("⚠️ Please enter a question or click a prompt chip.")
                else:
                    st.session_state.last_query = query_clean
                    
                    # Add user query to conversation history
                    now_str = datetime.datetime.now().strftime("%I:%M %p")
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query_clean,
                        "time": now_str
                    })

                    # Rank pages using extracted text overlap
                    start_search = time.perf_counter()
                    try:
                        results = st.session_state.retriever.search(query_clean, top_k=top_k)
                        retrieval_ms = round((time.perf_counter() - start_search) * 1000, 3)
                    except Exception as e:
                        st.error(f"Search retrieval error: {str(e)}")
                        results = []
                        retrieval_ms = 0

                    if results:
                        top_page = getattr(results[0], "page_num", 1)
                        top_score = getattr(results[0], "score", 28.5)
                        
                        st.session_state.last_top_page = top_page
                        st.session_state.last_top_score = top_score
                        st.session_state.last_retrieval_ms = retrieval_ms
                        st.session_state.inspect_page = top_page

                        # Stream response directly in an assistant chat bubble
                        with st.chat_message("assistant", avatar="🔮"):
                            answer_placeholder = st.empty()
                            full_answer = ""
                            for chunk in st.session_state.generator.generate_answer_stream(
                                query=query_clean,
                                images=[],
                                active_pdf_path=st.session_state.active_pdf_path,
                                target_page=top_page,
                                retriever=st.session_state.retriever
                            ):
                                full_answer += chunk
                                answer_placeholder.markdown(full_answer + "▌")
                            answer_placeholder.markdown(full_answer)
                            
                            # Instant citation pill below streamed answer
                            st.markdown(f'<div class="citation-card"><div style="display: flex; align-items: center; gap: 8px;"><span class="pill pill-done">📌 Retrieved Page {top_page}</span><span style="font-size: 12px; font-weight: 600; color: #64748B;">{active_file_name}</span></div><span class="pill pill-cyan">Score {top_score:.1f} | Search {retrieval_ms}ms</span></div>', unsafe_allow_html=True)

                        # Store in history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_answer,
                            "target_page": top_page,
                            "score": top_score,
                            "latency_ms": retrieval_ms,
                            "time": now_str
                        })
                    else:
                        st.warning("⚠️ No matching document pages found. Try rephrasing your search terms.")


# ==============================================================================
# TAB 2: MULTI-PAGE DOCUMENT PREVIEW & FILMSTRIP GALLERY
# ==============================================================================
with tab_preview:
    st.markdown("""
    <div class="outstaff-card">
        <div class="card-title">
            <span>📑 Multi-Page Document Preview Gallery</span>
            <span class="pill pill-cyan">Page gallery</span>
        </div>
        <div class="card-subtitle">Preview pages and inspect extracted-text snippets</div>
    """, unsafe_allow_html=True)

    if not st.session_state.active_pdf_path:
        st.info("👈 Upload or select a PDF to preview pages.")
    else:
        try:
            # Reuse the page images rendered during indexing when available.
            thumbs = []
            if hasattr(st.session_state.retriever, "page_index") and st.session_state.retriever.page_index.pages_data:
                for p in st.session_state.retriever.page_index.pages_data[:min(num_pages, 6)]:
                    img = p["image"]
                    w, h = img.size
                    target_w = 200
                    target_h = int(target_w * h / max(w, 1))
                    thumbs.append(img.resize((target_w, target_h), Image.Resampling.BILINEAR))
            else:
                thumbs = get_cached_thumbnails(str(st.session_state.active_pdf_path), max_thumbs=min(num_pages, 6))
            if thumbs:
                grid_cols = st.columns(min(len(thumbs), 4))
                for i, thumb_img in enumerate(thumbs):
                    col = grid_cols[i % len(grid_cols)]
                    with col:
                        page_num = i + 1
                        st.markdown(f"""
                        <div class="page-thumb-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span class="pill pill-todo">Page {page_num}</span>
                                <span style="font-size: 11px; font-weight: 700; color: #0284C7;">144 DPI</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(thumb_img, use_container_width=True)
                        
                        if st.button(f"🔍 Inspect Page {page_num}", key=f"inspect_btn_{page_num}", use_container_width=True):
                            st.session_state.inspect_page = page_num
                            st.toast(f"Switched view to Page {page_num}")
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading page thumbnails: {e}")

    st.markdown("</div>", unsafe_allow_html=True)



# ============================================================================
# TAB 3: EXPERIMENTAL HEURISTIC COMPARISON
# ============================================================================
with tab_compare:
    st.markdown("## Compare page-ranking heuristics")
    st.caption(
        "These are small hand-built baselines for exploration, not ColPali, CLIP, "
        "or BM25. They may rank pages differently and are not evidence of model quality."
    )

    if not st.session_state.active_pdf_path:
        st.info("Upload or select a PDF to compare the page-ranking heuristics.")
    else:
        col_text, col_layout, col_keyword = st.columns(3, gap="medium")
        text_results = st.session_state.retriever.search(st.session_state.last_query, top_k=1)
        layout_results = st.session_state.dense_rag.search(st.session_state.last_query, top_k=1)
        keyword_results = st.session_state.text_rag.search(st.session_state.last_query, top_k=1)

        comparisons = [
            (col_text, "PDF text overlap", text_results, "Extracted embedded PDF text ranked by token overlap and synonyms."),
            (col_layout, "Page-layout heuristic", layout_results, "A hand-built grayscale thumbnail statistic; not a learned vision model."),
            (col_keyword, "Keyword baseline", keyword_results, "Simple term-frequency overlap over extracted page text; not BM25."),
        ]
        for column, label, results, description in comparisons:
            with column:
                page_num = results[0].page_num if label == "PDF text overlap" and results else (
                    results[0]["page_num"] if results else 1
                )
                score = results[0].score if label == "PDF text overlap" and results else (
                    results[0]["score"] if results else 0.0
                )
                st.markdown(f"### {label}")
                st.caption(description)
                st.metric("Top page", f"Page {page_num}", f"Heuristic score {score:.2f}")
                page_preview = render_page_with_highlight(
                    st.session_state.active_pdf_path, page_num, st.session_state.last_query
                )
                st.image(page_preview, use_container_width=True)

        st.info(
            "For a meaningful ColPali/CLIP/BM25 comparison, those real model and retrieval "
            "implementations plus a shared, reproducible benchmark corpus still need to be added."
        )
# ============================================================================
# TAB 4: PDF TEXT SEARCH ANALYTICS
# ============================================================================
with tab_analytics:
    st.markdown("## PDF text-search analytics")
    st.caption("These charts describe extracted text and keyword-overlap scores; they are not visual embedding activations.")

    if "cached_scores_query" not in st.session_state or st.session_state.get("cached_scores_query") != st.session_state.last_query:
        st.session_state.cached_all_scores = st.session_state.retriever.search_all_pages(st.session_state.last_query)
        st.session_state.cached_scores_query = st.session_state.last_query

    all_scores = st.session_state.get("cached_all_scores", [])
    page_data = st.session_state.retriever.page_index.pages_data
    if not page_data:
        st.info("Upload or select a PDF to see its extracted-text profile.")
    else:
        total_tokens = sum(len(page["tokens"]) for page in page_data)
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages indexed", len(page_data))
        m2.metric("Unique tokens", total_tokens)
        m3.metric("Last search call", f"{st.session_state.last_retrieval_ms} ms")

        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("### Page ranking scores")
            if all_scores:
                score_df = pd.DataFrame([
                    {"Page": f"P{item['page_num']}", "Keyword-overlap score": item["score"]}
                    for item in all_scores
                ])
                score_chart = alt.Chart(score_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Page:N", sort=None, title=None),
                    y=alt.Y("Keyword-overlap score:Q", title="Heuristic score"),
                    color=alt.condition(
                        alt.datum.Page == f"P{st.session_state.last_top_page}",
                        alt.value("#0284C7"), alt.value("#7DD3FC")
                    ),
                    tooltip=["Page", "Keyword-overlap score"]
                ).properties(height=220)
                st.altair_chart(score_chart, use_container_width=True)
            else:
                st.info("Ask a question in the Studio tab to calculate page ranking scores.")

        with right:
            st.markdown("### Query-term occurrences")
            terms = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", st.session_state.last_query.lower())))
            term_rows = []
            for term in terms:
                occurrences = sum(page["text"].lower().count(term) for page in page_data)
                term_rows.append({"Query term": term, "Occurrences in PDF text": occurrences})
            if term_rows:
                term_chart = alt.Chart(pd.DataFrame(term_rows)).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Query term:N", sort=None, title=None),
                    y=alt.Y("Occurrences in PDF text:Q", title="Text occurrences"),
                    color=alt.value("#38BDF8"),
                    tooltip=["Query term", "Occurrences in PDF text"]
                ).properties(height=220)
                st.altair_chart(term_chart, use_container_width=True)
            else:
                st.info("Enter a question in the Studio tab to inspect its terms.")

        if all_scores:
            st.markdown("### Indexed page details")
            st.dataframe(pd.DataFrame([
                {
                    "Page": item["page_num"],
                    "Extracted text preview": item["snippet"],
                    "Unique tokens": item["tokens"],
                    "Keyword-overlap score": item["score"],
                }
                for item in all_scores
            ]), use_container_width=True, hide_index=True)
# ============================================================================
# TAB 5: EXPERIMENTAL RETRIEVAL EVALUATION
# ============================================================================
with tab_benchmark:
    st.markdown("## Experimental retrieval evaluation")
    st.caption(
        "This prototype compares hand-built page-ranking heuristics. It does not run "
        "ColPali, CLIP, BM25, or an answer-quality judge. Results are generated only "
        "after running the benchmark against the required local PDFs."
    )

    required_docs = sorted({item["doc"] for item in BENCHMARK_DATASET})
    missing_docs = []
    for doc_name in required_docs:
        candidates = [Path(doc_name), Path("data/input_pdfs") / doc_name]
        if not any(candidate.is_file() for candidate in candidates):
            missing_docs.append(doc_name)

    if missing_docs:
        st.warning(
            "The benchmark is not ready because these source PDFs are missing. "
            "Add each file to the repository root or `data/input_pdfs/`: "
            + ", ".join(missing_docs)
        )
    else:
        st.info(
            f"The benchmark contains {len(BENCHMARK_DATASET)} manually labeled queries "
            f"across {len(required_docs)} PDFs. Treat results as exploratory: the corpus "
            "is small and the ranking systems are simple heuristics."
        )

    run_bench = st.button(
        "Run heuristic retrieval benchmark",
        type="primary",
        disabled=bool(missing_docs),
        use_container_width=False,
    )
    if run_bench:
        with st.spinner("Running the page-ranking heuristics on the supplied PDFs..."):
            try:
                st.session_state.research_benchmark = run_research_benchmark(BENCHMARK_DATASET)
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.success("Benchmark completed. These results apply only to this corpus and run.")

    b_data = st.session_state.get("research_benchmark")
    if b_data:
        metric_rows = []
        systems = [
            ("PDF text overlap", b_data["text_overlap_metrics"]),
            ("Page-layout heuristic", b_data["layout_metrics"]),
            ("Keyword baseline", b_data["keyword_metrics"]),
        ]
        for label, metrics in systems:
            metric_rows.append({
                "Heuristic": label,
                "Recall@1 (%)": metrics["recall@1"],
                "Recall@3 (%)": metrics["recall@3"],
                "Recall@5 (%)": metrics["recall@5"],
                "MRR": metrics["mrr"],
                "Mean search time (ms)": metrics["latency_avg_ms"],
                "P95 search time (ms)": metrics["latency_p95_ms"],
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Search timing covers only the in-memory ranking call. It excludes PDF "
            "rendering/text extraction, indexing, model inference, and Streamlit UI time."
        )
        st.dataframe(pd.DataFrame(b_data.get("detailed_rows", [])), use_container_width=True, hide_index=True)

    with st.expander("Known limitations and future work"):
        st.markdown("""
        **Implemented:** PDF page rendering, embedded-text extraction, token/synonym overlap ranking, and optional image-based answering through configured vision-model providers.

        **Not implemented:** ColPali embeddings or MaxSim, a CLIP model baseline, BM25, RAGAS evaluation of answers produced by this app, OCR for scanned PDFs, automatic page deskewing, high-resolution region crops, and automated grounding verification. These remain future work.
        """)
