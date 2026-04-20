"""
Supply Command AI — Chat Page
app/pages/chat.py

The primary interface for asking supply chain questions.

Layout:
    Left col (2/3 width)  — chat history + input + answer display
    Right col (1/3 width) — agent trace panel (what ran, cost, confidence)

Data flow:
    user types question
        → run_pipeline(query, role)        ← one call to graph.py
        → result dict returned             ← structured answer + metadata
        → answer rendered with sources     ← expander with SQL + citations
        → agent trace rendered             ← right panel updated
        → result stored in chat_history    ← persists across reruns
"""

import sys
import os
import time
import re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from services.graph import run_pipeline
from app.styles    import get_styles


def clean_trace_text(text) -> str:
    """Strip any HTML tags from agent trace field values before interpolation."""
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', str(text)).strip()


# ── Agent metadata for trace panel display ────────────────────────────────────
_AGENT_META = {
    "db_agent":        {"icon": "🔍", "label": "DB Agent",       "desc": "SQL template → SQLite"},
    "rag_agent":       {"icon": "📄", "label": "RAG Agent",       "desc": "FAISS vector search"},
    "roi_agent":       {"icon": "💰", "label": "ROI Agent",       "desc": "Financial calculation"},
    "executive_agent": {"icon": "🧠", "label": "Executive Agent", "desc": "GPT-4o-mini formatter"},
    "planning_agent":  {"icon": "🗺️", "label": "Planning Agent",  "desc": "Intent classification"},
}

# ── Confidence colour thresholds ──────────────────────────────────────────────
def _confidence_color(score: float) -> str:
    if score >= 0.85:
        return "#2ECC71"   # green
    if score >= 0.70:
        return "#F1C40F"   # yellow
    return "#E67E22"       # orange


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    if score >= 0.70:
        return "Moderate"
    return "Low"


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT TRACE PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def _render_agent_trace(result: dict) -> None:
    """Render the right-column agent trace panel for a completed pipeline run."""

    st.markdown(
        "<p style='font-size:0.8rem; color:#AAAAAA; "
        "font-weight:600; letter-spacing:1px;'>AGENT TRACE</p>",
        unsafe_allow_html=True,
    )

    agents_used  = result.get("agents_used", [])
    tokens_used  = result.get("tokens_used", 0)
    cost_usd     = result.get("cost_usd", 0.0)
    exec_ms      = result.get("execution_time_ms", 0)
    groundedness = result.get("groundedness_score", 1.0)
    plan_text    = result.get("plan_explanation", "")

    # ── Planning Agent is always first ───────────────────────────────────────
    all_agents = ["planning_agent"] + [a for a in agents_used if a != "planning_agent"]

    for agent_key in all_agents:
        meta   = _AGENT_META.get(agent_key, {"icon": "🤖", "label": agent_key, "desc": ""})
        is_llm = agent_key == "executive_agent"
        toks   = tokens_used if is_llm else 0
        _label = clean_trace_text(meta['label'])
        _desc  = clean_trace_text(meta['desc'])

        st.markdown(
            f"""
            <div style="background:#1E2329; border-radius:6px;
                        padding:10px 12px; margin-bottom:8px;
                        border-left:3px solid #378ADD;">
                <div style="display:flex; justify-content:space-between;
                            align-items:center;">
                    <span style="font-size:0.9rem;">
                        {meta['icon']} <b>{_label}</b>
                    </span>
                    <span style="font-size:0.72rem; color:#2ECC71;">✅ Complete</span>
                </div>
                <div style="font-size:0.72rem; color:#888888; margin-top:4px;">
                    {_desc}
                    {"&nbsp;·&nbsp;" + str(toks) + " tokens" if is_llm and toks else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    conf_score = result.get("confidence", 0.0)
    conf_color = _confidence_color(conf_score)
    conf_label = _confidence_label(conf_score)

    st.markdown(
        f"""
        <div style="font-size:0.78rem; line-height:2.0;">
            <span style="color:#AAAAAA;">Confidence</span>&nbsp;&nbsp;
            <span style="color:{conf_color}; font-weight:600;">
                {conf_score:.0%} — {conf_label}
            </span><br>
            <span style="color:#AAAAAA;">Groundedness</span>&nbsp;&nbsp;
            <span style="color:#FFFFFF; font-weight:600;">
                {groundedness:.0%}
            </span><br>
            <span style="color:#AAAAAA;">LLM Tokens</span>&nbsp;&nbsp;
            <span style="color:#FFFFFF; font-weight:600;">{tokens_used}</span><br>
            <span style="color:#AAAAAA;">API Cost</span>&nbsp;&nbsp;
            <span style="color:#FFFFFF; font-weight:600;">
                ${cost_usd:.4f}
            </span><br>
            <span style="color:#AAAAAA;">Pipeline Time</span>&nbsp;&nbsp;
            <span style="color:#FFFFFF; font-weight:600;">{exec_ms}ms</span><br>
            <span style="color:#AAAAAA;">Agents Used</span>&nbsp;&nbsp;
            <span style="color:#FFFFFF; font-weight:600;">
                {len(agents_used)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Plan explanation (collapsed) ──────────────────────────────────────────
    # Only show investigation plan if steps exist AND it is not a simple metric
    plan_steps        = result.get("plan_steps", [])
    metric_definition = result.get("metric_definition", "")
    query_type        = result.get("query_type", "")

    if (
        plan_steps
        and len(plan_steps) > 0
        and metric_definition not in ("SIMPLE_METRIC", "SIMPLE_COUNT")
        and query_type != "METRIC_QUERY"
    ):
        with st.expander("📋 View Investigation Plan", expanded=False):
            st.markdown(
                f"<pre style='font-size:0.72rem; color:#CCCCCC; "
                f"white-space:pre-wrap;'>{plan_text}</pre>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  ANSWER DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def _render_answer(result: dict, msg_index: int = 0) -> None:
    """Render one complete answer with human approval handling and source expander."""

    answer   = result.get("answer", "")
    blocked  = result.get("blocked", False)
    approval = result.get("human_approval_required", False)
    sources  = result.get("sources", [])
    sql_list = result.get("sql_shown", [])
    conf     = result.get("confidence", 0.0)
    warnings = result.get("warnings", [])

    # ── RBAC block ────────────────────────────────────────────────────────────
    if blocked:
        reason = result.get("blocked_reason", "Access restricted.")
        st.error(f"🔒 **Access Restricted**\n\n{reason}")
        st.info(
            "💡 **Suggestion:** Switch to a role with higher access level "
            "(Operations Manager or CFO) using the sidebar role selector.",
            icon="💡",
        )
        return

    # ── Human approval required ───────────────────────────────────────────────
    if approval:
        approval_reason  = result.get("approval_reason",  "")
        impact_summary   = result.get("impact_summary",   "")

        st.warning(
            f"**⚠️ Human Approval Required**\n\n"
            f"This decision requires review before action is taken.\n\n"
            f"**Reason:** {approval_reason}\n\n"
            f"**Impact:** {impact_summary}",
        )

        col_approve, col_escalate, _ = st.columns([1, 1, 3])
        with col_approve:
            if st.button("✅ Approve", key=f"approve_{msg_index}", type="primary"):
                st.success("Approval recorded. Proceeding with recommendation.")
        with col_escalate:
            if st.button("🔼 Escalate", key=f"escalate_{msg_index}"):
                st.info("Escalated to Senior Management. Notification sent.")

        st.divider()

    # ── Main answer text ──────────────────────────────────────────────────────
    # Strip warning headers that were prepended in executive_agent
    # (they are surfaced separately above via st.warning)
    display_answer = answer
    if display_answer.startswith("⚠️"):
        lines = display_answer.split("\n\n", 1)
        if len(lines) > 1:
            display_answer = lines[1]

    st.markdown(display_answer)

    # ── Sources + SQL expander ────────────────────────────────────────────────
    with st.expander("🔎 View Sources, SQL & Evidence", expanded=False):

        # Confidence bar
        conf_color = _confidence_color(conf)
        conf_label = _confidence_label(conf)
        st.markdown(
            f"""
            <div style="margin-bottom:12px;">
                <span style="font-size:0.8rem; color:#AAAAAA;">
                    Confidence Score
                </span><br>
                <div style="background:#2A2F38; border-radius:4px;
                            height:8px; margin-top:4px; width:100%;">
                    <div style="background:{conf_color}; border-radius:4px;
                                height:8px; width:{conf:.0%};"></div>
                </div>
                <span style="font-size:0.75rem; color:{conf_color};">
                    {conf:.0%} — {conf_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sources
        if sources:
            st.markdown("**Sources cited:**")
            for src in sources:
                st.markdown(f"- {src}")

        # SQL queries
        if sql_list:
            st.markdown("**SQL executed:**")
            for i, sql in enumerate(sql_list, 1):
                st.code(sql, language="sql")
        else:
            st.caption("No SQL queries — answer from RAG or financial calculation.")

        # Warnings
        if warnings:
            for w in warnings:
                st.caption(f"⚠️ {w}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE QUESTIONS PER ROLE
# ═══════════════════════════════════════════════════════════════════════════════

_SAMPLE_QUESTIONS = {
    "Operations Manager": [
        "Which supplier has the highest delay rate?",
        "What are the top delay reasons this year?",
        "Which supplier is causing the most financial damage and what should we do?",
        "Show me the monthly delay trend for 2024",
    ],
    "CFO": [
        "What is our total supply chain cost and ROI?",
        "What was the impact of the Red Sea crisis on our costs?",
        "Show the AI savings progression since go-live",
        "What is our financial exposure from underperforming suppliers?",
    ],
    "Demand Planner": [
        "Which shipments are currently delayed?",
        "What is SupplierB's on-time delivery rate?",
        "Show me high-risk shipments this month",
        "Which supplier has the best SLA performance?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def render(role: str) -> None:
    """Main entry point called by app/main.py."""

    st.markdown(get_styles(), unsafe_allow_html=True)

    # ── Top nav bar ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="top-nav">
        <div class="top-nav-brand">
            🏭 Supply Command AI
            <span>GlobalMedTech Inc.</span>
        </div>
        <div class="top-nav-status">
            <div class="status-dot"></div>
            Live · supply_chain.db · 1 LLM call/query
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Initialise session state ──────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Back button ───────────────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Dashboard"):
            st.session_state["page"] = "Dashboard"
            st.rerun()

    # ── Page header ───────────────────────────────────────────────────────────
    col_title, col_role = st.columns([3, 1])
    with col_title:
        st.markdown(
            "<h2 style='margin-bottom:0;'>💬 Ask Supply Command AI</h2>",
            unsafe_allow_html=True,
        )
    with col_role:
        role_colors = {
            "Operations Manager": "#378ADD",
            "CFO":                "#2ECC71",
            "Demand Planner":     "#E67E22",
        }
        color = role_colors.get(role, "#378ADD")
        st.markdown(
            f"""
            <div style="background:{color}20; border:1px solid {color};
                        border-radius:6px; padding:8px 12px;
                        text-align:center; margin-top:8px;">
                <span style="font-size:0.8rem; color:{color};
                             font-weight:600;">{role}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Two-column layout: chat left, trace right ──────────────────────────────
    col_chat, col_trace = st.columns([2, 1])

    with col_chat:
        # ── Sample questions (shown when chat is empty) ────────────────────
        if not st.session_state.get("chat_history"):
            st.markdown(
                "<p style='color:#888888; font-size:0.85rem;'>"
                "Try one of these questions for your role:</p>",
                unsafe_allow_html=True,
            )
            samples = _SAMPLE_QUESTIONS.get(role, [])
            btn_cols = st.columns(2)
            for i, q in enumerate(samples):
                with btn_cols[i % 2]:
                    if st.button(q, key=f"sample_{i}", use_container_width=True):
                        st.session_state["_pending_query"] = q

        # ── Chat history ───────────────────────────────────────────────────
        for i, entry in enumerate(st.session_state.get("chat_history", [])):
            with st.chat_message(entry["role"]):
                if entry["role"] == "user":
                    st.markdown(entry["content"])
                else:
                    _render_answer(entry["metadata"], msg_index=i)

        # ── Chat input ─────────────────────────────────────────────────────
        pending = st.session_state.pop("_pending_query", None)
        user_input = st.chat_input(
            placeholder=f"Ask a supply chain question as {role}...",
        ) or pending

        if user_input:
            # Add user message to history
            st.session_state["chat_history"].append({
                "role":     "user",
                "content":  user_input,
                "metadata": {},
            })

            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)

            # ── Run pipeline with progress feedback ────────────────────────
            with st.chat_message("assistant"):
                with st.spinner("Planning investigation..."):
                    time.sleep(0.3)   # brief pause so spinner shows

                status_placeholder = st.empty()

                status_placeholder.markdown(
                    "<span style='color:#888888; font-size:0.85rem;'>"
                    "⚙️ Running agents...</span>",
                    unsafe_allow_html=True,
                )

                result = run_pipeline(
                    query         = user_input,
                    role          = role,
                    forced_metric = None,
                    forced_entity = None,
                )

                status_placeholder.empty()

                # Render the answer — use len(chat_history) as index since
                # this message hasn't been appended to history yet
                _render_answer(result, msg_index=len(st.session_state.get("chat_history", [])))

            # Store assistant result in history
            st.session_state["chat_history"].append({
                "role":     "assistant",
                "content":  result.get("answer", ""),
                "metadata": result,
            })

            # Store as last result for the trace panel
            st.session_state["last_result"] = result

            st.rerun()

    # ── Agent trace panel ─────────────────────────────────────────────────────
    with col_trace:
        last_result = st.session_state.get("last_result")

        if last_result and st.session_state.get("chat_history"):
            _render_agent_trace(last_result)
        else:
            # Empty state
            st.markdown(
                """
                <div style="background:#1E2329; border-radius:8px;
                            padding:20px; text-align:center; margin-top:40px;">
                    <p style="font-size:1.5rem;">🗺️</p>
                    <p style="color:#888888; font-size:0.85rem;">
                        Agent trace will appear here after your first question.
                    </p>
                    <p style="color:#555555; font-size:0.75rem; margin-top:8px;">
                        DB · RAG · ROI · Executive
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Architecture hint
            st.markdown(
                """
                <div style="margin-top:20px; font-size:0.75rem;
                            color:#555555; line-height:1.8;">
                    <b style="color:#777777;">How it works</b><br>
                    1️⃣ Input guardrails<br>
                    2️⃣ Planning Agent classifies intent<br>
                    3️⃣ DB Agent queries SQLite<br>
                    4️⃣ RAG Agent searches Annual Report<br>
                    5️⃣ ROI Agent calculates impact<br>
                    6️⃣ Executive Agent formats answer<br>
                    <br>
                    <b style="color:#777777;">1 LLM call per question</b><br>
                    All routing is coded Python.
                </div>
                """,
                unsafe_allow_html=True,
            )
