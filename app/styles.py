# app/styles.py
# Global CSS polish for Supply Command AI
# Apply via st.markdown(get_styles(), unsafe_allow_html=True)

def get_styles():
    return """
    <style>
    /* ── FONTS ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    * { font-family: 'Inter', sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace; }

    /* ── HIDE STREAMLIT BRANDING ────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }

    /* ── MAIN BACKGROUND ────────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background: #060912;
    }
    [data-testid="stMain"] {
        background: #060912;
    }

    /* ── SIDEBAR ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0a0f1e !important;
        border-right: 1px solid #1a2340;
    }
    [data-testid="stSidebar"] * {
        color: #e8edf8;
    }

    /* ── TOP NAV BAR ────────────────────────────────────────── */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: #0a0f1e;
        border-bottom: 1px solid #1a2340;
        margin: -1rem -1rem 2rem -1rem;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .top-nav-brand {
        font-size: 16px;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 0.5px;
    }
    .top-nav-brand span {
        color: #5a6a8a;
        font-weight: 400;
        font-size: 13px;
        margin-left: 8px;
    }
    .top-nav-pills {
        display: flex;
        gap: 4px;
        background: #0f1628;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #1a2340;
    }
    .top-nav-pill {
        padding: 6px 20px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        color: #5a6a8a;
        text-decoration: none;
    }
    .top-nav-pill.active {
        background: #00d4ff;
        color: #060912;
        font-weight: 600;
    }
    .top-nav-status {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #5a6a8a;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #00ff9d;
        box-shadow: 0 0 6px #00ff9d;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── METRIC CARDS ───────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #0a0f1e;
        border: 1px solid #1a2340;
        border-radius: 10px;
        padding: 20px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #00d4ff44;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #5a6a8a !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #e8edf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    /* ── CHAT MESSAGES ──────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: #0a0f1e !important;
        border: 1px solid #1a2340 !important;
        border-radius: 10px !important;
        margin-bottom: 12px !important;
        padding: 16px !important;
    }
    [data-testid="stChatMessageContent"] {
        color: #e8edf8 !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
    }

    /* ── CHAT INPUT ─────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        background: #0a0f1e !important;
        border: 1px solid #1a2340 !important;
        border-radius: 10px !important;
        color: #e8edf8 !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
    }

    /* ── BUTTONS ────────────────────────────────────────────── */
    [data-testid="baseButton-primary"],
    .stButton > button {
        background: rgba(0,212,255,0.1) !important;
        border: 1px solid #00d4ff !important;
        color: #00d4ff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.2s !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stButton > button:hover {
        background: rgba(0,212,255,0.2) !important;
        box-shadow: 0 0 12px rgba(0,212,255,0.2) !important;
    }

    /* ── SELECTBOX / ROLE SELECTOR ──────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background: #0a0f1e !important;
        border: 1px solid #1a2340 !important;
        border-radius: 6px !important;
        color: #e8edf8 !important;
    }

    /* ── EXPANDERS (alert details) ──────────────────────────── */
    [data-testid="stExpander"] {
        background: #0a0f1e !important;
        border: 1px solid #1a2340 !important;
        border-radius: 8px !important;
        margin-top: 4px !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #00d4ff44 !important;
    }

    /* ── DIVIDERS ───────────────────────────────────────────── */
    hr {
        border-color: #1a2340 !important;
        margin: 24px 0 !important;
    }

    /* ── HEADINGS ───────────────────────────────────────────── */
    h1, h2, h3 {
        color: #e8edf8 !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }
    h1 { font-size: 24px !important; }
    h2 { font-size: 18px !important; }
    h3 { font-size: 15px !important; }

    /* ── DATAFRAMES / TABLES ────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1a2340 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* ── SPINNER ────────────────────────────────────────────── */
    [data-testid="stSpinner"] {
        color: #00d4ff !important;
    }

    /* ── SUCCESS / WARNING / ERROR BOXES ────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border-left-width: 4px !important;
        font-size: 13px !important;
    }

    /* ── SCROLLBAR ──────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #060912; }
    ::-webkit-scrollbar-thumb {
        background: #1a2340;
        border-radius: 2px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #00d4ff44; }

    /* ── CAPTION / SMALL TEXT ───────────────────────────────── */
    [data-testid="stCaptionContainer"] {
        color: #5a6a8a !important;
        font-size: 11px !important;
    }

    /* ── RADIO BUTTONS (navigation) ─────────────────────────── */
    [data-testid="stRadio"] label {
        font-size: 13px !important;
        color: #5a6a8a !important;
        font-weight: 500 !important;
    }
    [data-testid="stRadio"] label:hover {
        color: #e8edf8 !important;
    }

    /* ── PAGE PADDING ───────────────────────────────────────── */
    .main .block-container {
        padding: 1rem 2rem 2rem !important;
        max-width: 1200px !important;
    }
    </style>
    """
