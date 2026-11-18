from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st

# Find the root 'fall2026-gasp' folder (one level above 'app/') and add it to Python's import path.
# This guarantees that 'from app.components import ...' works inside every page in app/pages/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add src/ so 'from gasp import ...' works where the package isn't installed (Streamlit Cloud)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Configure the browser tab title and use the full-width layout across all pages
st.set_page_config(
    page_title="GASP · Natural Gas Storage Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components import inject_term_tip_css

# Load the tooltip styles once, so every page's terms get the pop-up definitions
inject_term_tip_css()

# Make the "open navigation" button obvious: a gray tile containing the arrow and a "Pages" label
st.markdown(
    """
    <style>
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background: #E5E2DC;
        border-radius: 14px;
        height: 3rem;
        width: auto;
        padding: 0 1rem 0 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] button svg,
    [data-testid="collapsedControl"] button svg {
        width: 1.6rem;
        height: 1.6rem;
        color: #18222C;
    }
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="stSidebarCollapsedControl"] button::after,
    [data-testid="collapsedControl"] button::after {
        content: "Pages";
        position: static;
        font-size: 0.95rem;
        font-weight: 600;
        color: #18222C;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Register all 8 required pages in the exact order listed in the Piece 1 instructions.
# Setting default=True on Home makes the app open on the Home page automatically.
pages = [
    st.Page("pages/home.py", title="Home", default=True),
    st.Page("pages/track_record.py", title="Track Record"),
    st.Page("pages/explain.py", title="Explain"),
    st.Page("pages/what_if.py", title="What-If"),
    st.Page("pages/winter_outlook.py", title="Winter Outlook"),
    st.Page("pages/models.py", title="Models"),
    st.Page("pages/data.py", title="Data"),
    st.Page("pages/about.py", title="About"),
]

# Build the sidebar navigation menu and run whichever page the user clicked
nav = st.navigation(pages, position="top")
nav.run()
