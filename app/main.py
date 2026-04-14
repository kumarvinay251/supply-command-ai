"""
Supply Command AI — Streamlit Entry Point
app/main.py

Run with:
    cd "Control Tower"
    streamlit run app/main.py

This file:
    1. Sets page config (must be first Streamlit call)
    2. Initialises session state
    3. Renders the sidebar (logo, role selector, navigation)
    4. Routes to the correct page component
"""

import sys
import os

# ── Ensure project root is on the path so agents/services/database import ──────
# WHY sys.path manipulation here?
#   Streamlit runs app/main.py from its own working directory.
#   Adding the project root explicitly means every `from agents.X import Y`
#   works exactly as it does in the CLI tests.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Page config — MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="Supply Command AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import page renderers after page config ────────────────────────────────────
from app.pages.dashboard import render as render_dashboard
from app.pages.chat      import render as render_chat
from app.styles          import get_styles

# ── Apply global CSS polish ────────────────────────────────────────────────────
st.markdown(get_styles(), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
#  Set once on first load; never overwrite existing values.
# ═══════════════════════════════════════════════════════════════════════════════

_ROLE_ACCESS = {
    "Operations Manager": {
        "level":       "Full operational + limited financial",
        "tables":      "shipments, suppliers_master",
        "color":       "#378ADD",
        "icon":        "⚙️",
    },
    "CFO": {
        "level":       "Full access — all tables and financials",
        "tables":      "shipments, suppliers_master, financial_impact",
        "color":       "#2ECC71",
        "icon":        "💼",
    },
    "Demand Planner": {
        "level":       "Operational only — no financial data",
        "tables":      "shipments, suppliers_master (row-capped)",
        "color":       "#E67E22",
        "icon":        "📦",
    },
}

def _init_session_state() -> None:
    defaults = {
        "role":          "Operations Manager",
        "page":          "Dashboard",
        "chat_history":  [],    # list of {role, content, metadata}
        "last_result":   None,  # most recent run_pipeline() return dict
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        # ── Brand header ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding: 20px 0 16px;">
            <div style="font-size:20px; font-weight:800;
                        color:#00d4ff; letter-spacing:-0.5px;">
                Supply Command AI
            </div>
            <div style="font-size:11px; color:#5a6a8a;
                        margin-top:4px; letter-spacing:1px;">
                GLOBALMEDTECH INC.
            </div>
        </div>
        <hr style="border-color:#1a2340; margin: 0 0 20px;">
        """, unsafe_allow_html=True)

        # ── Role selector ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:10px; color:#5a6a8a;
                    letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:8px;">
            Active Role
        </div>""", unsafe_allow_html=True)

        role = st.selectbox(
            "",
            ["Operations Manager", "CFO", "Demand Planner"],
            index=["Operations Manager", "CFO", "Demand Planner"].index(
                st.session_state["role"]
            ),
            label_visibility="collapsed",
        )

        # Update role in session state; clear chat if role changes
        if role != st.session_state["role"]:
            st.session_state["role"]         = role
            st.session_state["chat_history"] = []
            st.session_state["last_result"]  = None

        # ── Role description badge ────────────────────────────────────────────
        role_desc = {
            "Operations Manager": ("🟢", "Full operational + limited financial"),
            "CFO":                ("🟡", "Full financial + projections access"),
            "Demand Planner":     ("🔵", "Operational only — no financial data"),
        }
        icon, desc = role_desc[role]
        st.markdown(f"""
        <div style="background:#0f1628; border:1px solid #1a2340;
                    border-radius:6px; padding:10px 12px; margin-top:8px;">
            <div style="font-size:12px; font-weight:600;
                        color:#e8edf8;">{icon} {role}</div>
            <div style="font-size:11px; color:#5a6a8a;
                        margin-top:3px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:10px; color:#5a6a8a;
                    letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:8px;">
            Navigation
        </div>""", unsafe_allow_html=True)

        pages = ["📊 Dashboard", "💬 Chat"]
        nav_selection = st.radio(
            "",
            pages,
            index=0 if st.session_state["page"] == "Dashboard" else 1,
            label_visibility="collapsed",
        )
        st.session_state["page"] = nav_selection.split(" ", 1)[1]

        # ── Architecture info ─────────────────────────────────────────────────
        st.markdown("<br><hr style='border-color:#1a2340'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:10px; color:#5a6a8a; line-height:1.8;">
            <div style="letter-spacing:2px; text-transform:uppercase;
                        margin-bottom:8px;">Architecture</div>
            <div>1 LLM call per query</div>
            <div>SQL templates · FAISS RAG</div>
            <div>LangGraph orchestration</div>
            <div>Human-in-loop @ action keywords</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _render_sidebar()

    page = st.session_state["page"]
    role = st.session_state["role"]

    if page == "Dashboard":
        render_dashboard(role)
    elif page == "Chat":
        render_chat(role)
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
else:
    # Streamlit calls the script as a module — render on import too
    main()
