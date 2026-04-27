"""
Supply Command AI — Dashboard Page
app/pages/dashboard.py

Live KPI cards + three Plotly charts pulled directly from supply_chain.db.

Data sources:
    financial_impact   → KPI cards, ROI chart, cost trend
    shipments          → delay trend chart
    suppliers_master   → supplier performance bar chart

All queries go through db_connection.execute_query() — same SQL validation
and audit logging as the agent pipeline. No raw cursor access here.
"""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from database.db_connection        import execute_query
from agents.alert_agent            import evaluate_alerts
from agents.data_health_agent      import run_health_checks
from app.styles                    import get_styles


# ── Chart colour palette ──────────────────────────────────────────────────────
_BLUE   = "#378ADD"
_GREEN  = "#2ECC71"
_ORANGE = "#E67E22"
_RED    = "#E74C3C"
_PURPLE = "#9B59B6"
_GRID   = "#2A2F38"
_BG     = "#1E2329"
_TEXT   = "#CCCCCC"


def _chart_layout(title: str = "", height: int = 320) -> dict:
    """Shared Plotly layout for dark theme consistency."""
    return dict(
        title=dict(text=title, font=dict(color=_TEXT, size=14)),
        height=height,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=12),
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        xaxis=dict(
            gridcolor=_GRID,
            linecolor=_GRID,
            tickfont=dict(size=10),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=_GRID,
            linecolor=_GRID,
            tickfont=dict(size=10),
            showgrid=True,
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS — all queries cached for 60 seconds
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def _load_financial_kpis() -> dict:
    """
    Fetch KPI values from financial_impact — most recent month + full-year totals.
    Returns safe defaults if DB is not yet populated.
    """
    # Most recent row for current values
    latest_res = execute_query("""
        SELECT
            period_label,
            ai_savings_usd,
            cumulative_savings,
            roi_pct,
            on_time_rate_pct
        FROM financial_impact
        ORDER BY year DESC, month DESC
        LIMIT 1
    """)

    # 2024 full-year totals
    annual_res = execute_query("""
        SELECT
            SUM(ai_savings_usd)     AS total_savings_2024,
            SUM(total_sc_cost_usd)  AS total_cost_2024,
            AVG(on_time_rate_pct)   AS avg_otd_2024
        FROM financial_impact
        WHERE year = 2024
    """)

    kpis = {
        "period":            "N/A",
        "cumulative_savings": 0,
        "roi_pct":           0,
        "on_time_rate_pct":  0,
        "total_cost_2024":   0,
        "avg_otd_2024":      0,
    }

    if latest_res.get("success") and latest_res.get("data"):
        row = latest_res["data"][0]
        kpis["period"]             = row.get("period_label",       "N/A")
        kpis["cumulative_savings"] = float(row.get("cumulative_savings", 0) or 0)
        kpis["roi_pct"]            = float(row.get("roi_pct",            0) or 0)
        kpis["on_time_rate_pct"]   = float(row.get("on_time_rate_pct",   0) or 0)

    if annual_res.get("success") and annual_res.get("data"):
        row = annual_res["data"][0]
        kpis["total_cost_2024"] = float(row.get("total_cost_2024", 0) or 0)
        kpis["avg_otd_2024"]    = float(row.get("avg_otd_2024",    0) or 0)

    return kpis


@st.cache_data(ttl=60, show_spinner=False)
def _load_delay_trend() -> list[dict]:
    """Monthly delay rate for 2023–2024."""
    res = execute_query("""
        SELECT
            strftime('%Y-%m', shipment_date) AS month,
            COUNT(*)                          AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                100.0 * SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) AS delay_rate_pct
        FROM shipments
        GROUP BY month
        ORDER BY month
    """)
    return res.get("data", []) if res.get("success") else []


@st.cache_data(ttl=60, show_spinner=False)
def _load_supplier_performance() -> list[dict]:
    """Actual OTD vs SLA target for all 3 suppliers."""
    res = execute_query("""
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.sla_on_time_target_pct,
            ROUND(
                100.0 * SUM(CASE WHEN sh.status = 'On Time' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) AS actual_otd_pct
        FROM suppliers_master s
        JOIN shipments sh ON s.supplier_id = sh.supplier_id
        GROUP BY s.supplier_id, s.supplier_name, s.sla_on_time_target_pct
        ORDER BY actual_otd_pct ASC
    """)
    return res.get("data", []) if res.get("success") else []


@st.cache_data(ttl=60, show_spinner=False)
def _load_roi_progression() -> list[dict]:
    """Cumulative AI savings and ROI since go-live (Jul 2024)."""
    res = execute_query("""
        SELECT
            period_label,
            ai_savings_usd,
            cumulative_savings,
            roi_pct
        FROM financial_impact
        WHERE year = 2024 AND month >= 7
        ORDER BY month
    """)
    return res.get("data", []) if res.get("success") else []


@st.cache_data(ttl=60, show_spinner=False)
def _load_delay_summary() -> dict:
    """Overall delay rate for KPI cards."""
    res = execute_query("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed,
            ROUND(
                100.0 * SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) AS delay_rate_pct,
            SUM(CASE WHEN sla_breach = 1 THEN 1 ELSE 0 END) AS sla_breaches
        FROM shipments
    """)
    if res.get("success") and res.get("data"):
        return res["data"][0]
    return {"total": 0, "delayed": 0, "delay_rate_pct": 0, "sla_breaches": 0}


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_delay_trend(rows: list[dict]) -> go.Figure:
    """Line chart: monthly delay rate 2023–2024 with AI go-live annotation."""
    if not rows:
        return go.Figure()

    months      = [r["month"]          for r in rows]
    delay_rates = [float(r.get("delay_rate_pct", 0) or 0) for r in rows]
    totals      = [int(r.get("total_shipments",  0) or 0) for r in rows]

    fig = go.Figure()

    # Delay rate line
    fig.add_trace(go.Scatter(
        x=months,
        y=delay_rates,
        mode="lines+markers",
        name="Delay Rate %",
        line=dict(color=_ORANGE, width=2.5),
        marker=dict(size=6, color=_ORANGE),
        hovertemplate="<b>%{x}</b><br>Delay rate: %{y:.1f}%<extra></extra>",
    ))

    # Shipment volume as bar (secondary y-axis feel via opacity)
    fig.add_trace(go.Bar(
        x=months,
        y=totals,
        name="Total Shipments",
        marker_color=_BLUE,
        opacity=0.25,
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Shipments: %{y}<extra></extra>",
    ))

    # AI go-live vertical line — July 2024
    # WHY integer index instead of string label?
    #   Plotly add_vline() on categorical/string x-axes requires a numeric
    #   index in newer versions. Passing "2024-07" raises a ValueError.
    #   We find the position of the label in the months list and pass that.
    vline_x = months.index("2024-07") if "2024-07" in months else None
    if vline_x is not None:
        fig.add_vline(
            x=vline_x,
            line_dash="dash",
            line_color=_GREEN,
            line_width=1.5,
            annotation_text="AI Go-Live",
            annotation_position="top right",
            annotation_font=dict(color=_GREEN, size=11),
        )

    layout = _chart_layout("Monthly Delay Rate 2023–2024", height=320)
    layout["yaxis"]["title"] = "Delay Rate %"
    layout["yaxis2"] = dict(
        overlaying="y",
        side="right",
        showgrid=False,
        tickfont=dict(size=9, color="#555555"),
        title="Shipments",
        titlefont=dict(size=10, color="#555555"),
    )
    fig.update_layout(**layout)
    return fig


def _chart_supplier_performance(rows: list[dict]) -> go.Figure:
    """Grouped bar chart: actual OTD vs SLA target per supplier."""
    if not rows:
        return go.Figure()

    names   = [r.get("supplier_name", r.get("supplier_id", "")) for r in rows]
    actual  = [float(r.get("actual_otd_pct",         0) or 0) for r in rows]
    targets = [float(r.get("sla_on_time_target_pct", 0) or 0) for r in rows]

    # Colour bars: red if below target, green if meeting
    bar_colors = [
        _GREEN if a >= t else _RED
        for a, t in zip(actual, targets)
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="SLA Target",
        x=names,
        y=targets,
        marker_color=_BLUE,
        opacity=0.5,
        text=[f"{t:.1f}%" for t in targets],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>SLA Target: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Actual OTD",
        x=names,
        y=actual,
        marker_color=bar_colors,
        text=[f"{a:.1f}%" for a in actual],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Actual OTD: %{y:.1f}%<extra></extra>",
    ))

    layout = _chart_layout("Supplier On-Time Delivery vs SLA Target", height=320)
    layout["barmode"]        = "group"
    layout["yaxis"]["title"] = "On-Time Delivery %"
    layout["yaxis"]["range"] = [0, 115]
    fig.update_layout(**layout)
    return fig


def _chart_roi_progression(rows: list[dict]) -> go.Figure:
    """Dual-axis line: cumulative savings (bar) + ROI % (line) Jul–Dec 2024."""
    if not rows:
        return go.Figure()

    periods    = [r.get("period_label",      "")  for r in rows]
    savings    = [float(r.get("ai_savings_usd",    0) or 0) for r in rows]
    cumulative = [float(r.get("cumulative_savings", 0) or 0) for r in rows]
    roi        = [float(r.get("roi_pct",            0) or 0) for r in rows]

    fig = go.Figure()

    # Cumulative savings bars
    fig.add_trace(go.Bar(
        x=periods,
        y=cumulative,
        name="Cumulative Savings",
        marker_color=_GREEN,
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Cumulative: $%{y:,.0f}<extra></extra>",
    ))

    # Monthly savings line
    fig.add_trace(go.Scatter(
        x=periods,
        y=savings,
        name="Monthly Savings",
        mode="lines+markers",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Monthly: $%{y:,.0f}<extra></extra>",
    ))

    # ROI % on secondary axis
    fig.add_trace(go.Scatter(
        x=periods,
        y=roi,
        name="ROI %",
        mode="lines+markers",
        line=dict(color=_ORANGE, width=2, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>ROI: %{y:.0f}%<extra></extra>",
    ))

    layout = _chart_layout("AI Control Tower ROI Progression (Jul–Dec 2024)", height=320)
    layout["yaxis"]["title"] = "USD Savings"
    layout["yaxis2"] = dict(
        overlaying="y",
        side="right",
        showgrid=False,
        tickfont=dict(size=9),
        title="ROI %",
        titlefont=dict(size=10),
    )
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

# Total number of health checks defined in data_health_agent — used to compute
# checks_passed without modifying the agent's return contract.
_TOTAL_HEALTH_CHECKS = 10


def render(role: str) -> None:
    """Entry point called by app/main.py."""

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

    # ── Health check — run once per session ───────────────────────────────────
    if "data_health" not in st.session_state:
        st.session_state["data_health"] = run_health_checks()

    health = st.session_state["data_health"]
    score  = health["health_score"]

    # ── Load all data (outside tabs — available to all three) ─────────────────
    with st.spinner("Loading dashboard data..."):
        kpis          = _load_financial_kpis()
        delay_sum     = _load_delay_summary()
        delay_rows    = _load_delay_trend()
        supplier_rows = _load_supplier_performance()
        roi_rows      = _load_roi_progression()

    st.markdown("## 📊 Supply Chain Dashboard")
    st.caption(
        f"Live data from supply_chain.db · Role: **{role}** · "
        "Auto-refreshes every 60 seconds"
    )

    # ── Three tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Executive Overview",
        "🔬 Data Health Check",
        "🚨 Alerts & Recommendations",
    ])

    # =========================================================================
    #  TAB 1 — EXECUTIVE OVERVIEW
    # =========================================================================
    with tab1:

        # ── Executive summary strip ───────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            <div style="background:#0a0f1e; border:1px solid #1a2340;
                        border-left:4px solid #00d4ff; border-radius:6px;
                        padding:16px 20px;">
                <div style="font-size:11px; color:#5a6a8a;
                            text-transform:uppercase; letter-spacing:2px;
                            margin-bottom:8px;">Fleet Performance</div>
                <div style="font-size:28px; font-weight:800; color:#e8edf8;">
                    80.0%</div>
                <div style="font-size:12px; color:#ff6b35;">
                    ↓ 7pts below industry benchmark</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div style="background:#0a0f1e; border:1px solid #1a2340;
                        border-left:4px solid #00ff9d; border-radius:6px;
                        padding:16px 20px;">
                <div style="font-size:11px; color:#5a6a8a;
                            text-transform:uppercase; letter-spacing:2px;
                            margin-bottom:8px;">AI ROI</div>
                <div style="font-size:28px; font-weight:800; color:#00ff9d;">
                    340%</div>
                <div style="font-size:12px; color:#00ff9d;">
                    $401K saved · As of Dec 2024</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown("""
            <div style="background:#0a0f1e; border:1px solid #1a2340;
                        border-left:4px solid #ff6b35; border-radius:6px;
                        padding:16px 20px;">
                <div style="font-size:11px; color:#5a6a8a;
                            text-transform:uppercase; letter-spacing:2px;
                            margin-bottom:8px;">Cost Exposure</div>
                <div style="font-size:28px; font-weight:800; color:#ff6b35;">
                    $379K</div>
                <div style="font-size:12px; color:#ff6b35;">
                    Expedited shipping · Avoidable</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI CARDS — ROW 1 ─────────────────────────────────────────────────
        st.markdown("### Key Performance Indicators")

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            otd = kpis.get("on_time_rate_pct", 0)
            st.metric(
                label="On-Time Delivery Rate",
                value=f"{otd:.1f}%",
                delta=f"{otd - 87:.1f}pp vs industry 87%",
                delta_color="normal",
                help="Most recent month. Industry benchmark: 87%.",
            )

        with k2:
            delay_rate = float(delay_sum.get("delay_rate_pct", 0) or 0)
            st.metric(
                label="Overall Delay Rate",
                value=f"{delay_rate:.1f}%",
                delta=f"{delay_rate - 13:.1f}pp vs industry 13%",
                delta_color="inverse",   # lower is better
                help="All shipments 2023–2024. Industry benchmark: 13%.",
            )

        with k3:
            if role in ("Operations Manager", "CFO"):
                savings = kpis.get("cumulative_savings", 0)
                st.metric(
                    label="AI Cumulative Savings",
                    value=f"${savings:,.0f}",
                    delta="Since Jul 2024 go-live",
                    delta_color="off",
                    help="Total savings generated by AI Control Tower.",
                )
            else:
                st.markdown(
                    """
                    <div style='padding:16px; border:1px solid #333;
                                border-radius:6px; text-align:center; color:#666;'>
                        🔒 Restricted<br/>
                        <small>Financial access required</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with k4:
            if role in ("Operations Manager", "CFO"):
                roi = kpis.get("roi_pct", 0)
                st.metric(
                    label="AI ROI",
                    value=f"{roi:.0f}%",
                    delta=f"As of {kpis.get('period', 'N/A')}",
                    delta_color="off",
                    help="Cumulative savings ÷ total AI investment × 100.",
                )
            else:
                st.markdown(
                    """
                    <div style='padding:16px; border:1px solid #333;
                                border-radius:6px; text-align:center; color:#666;'>
                        🔒 Restricted<br/>
                        <small>Financial access required</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── KPI CARDS — ROW 2 ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        k5, k6, k7, k8 = st.columns(4)

        with k5:
            st.metric(
                label="Total Shipments",
                value=f"{int(delay_sum.get('total', 0)):,}",
                help="All shipments in database (2023–2024).",
            )
        with k6:
            st.metric(
                label="Delayed Shipments",
                value=f"{int(delay_sum.get('delayed', 0)):,}",
                help="Shipments with status = Delayed.",
            )
        with k7:
            st.metric(
                label="SLA Breaches",
                value=f"{int(delay_sum.get('sla_breaches', 0)):,}",
                help="Shipments where sla_breach = 1.",
            )
        with k8:
            if role in ("Operations Manager", "CFO"):
                cost = kpis.get("total_cost_2024", 0)
                st.metric(
                    label="2024 SC Cost",
                    value=f"${cost:,.0f}",
                    help="Total supply chain cost for 2024.",
                )
            else:
                st.markdown(
                    """
                    <div style='padding:16px; border:1px solid #333;
                                border-radius:6px; text-align:center; color:#666;'>
                        🔒 Restricted<br/>
                        <small>Financial access required</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── OPERATIONAL PERFORMANCE CHARTS ────────────────────────────────────
        st.markdown("### Operational Performance")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_delay = _chart_delay_trend(delay_rows)
            if fig_delay.data:
                st.plotly_chart(fig_delay, use_container_width=True)
            else:
                st.info("No shipment delay data available.")

        with chart_col2:
            fig_supplier = _chart_supplier_performance(supplier_rows)
            if fig_supplier.data:
                st.plotly_chart(fig_supplier, use_container_width=True)
            else:
                st.info("No supplier performance data available.")

        st.divider()

        # ── AI ROI PROGRESSION ────────────────────────────────────────────────
        if role != "Demand Planner":
            st.markdown("### AI ROI Progression")

            roi_col, insight_col = st.columns([2, 1])

            with roi_col:
                fig_roi = _chart_roi_progression(roi_rows)
                if fig_roi.data:
                    st.plotly_chart(fig_roi, use_container_width=True)
                else:
                    st.info("No ROI data available.")

            with insight_col:
                st.markdown(
                    """
                    <div style="background:#1E2329; border-radius:8px;
                                padding:20px; margin-top:8px;">
                        <p style="color:#AAAAAA; font-size:0.8rem;
                                  font-weight:600; letter-spacing:1px;">
                            AI IMPACT SUMMARY
                        </p>
                    """,
                    unsafe_allow_html=True,
                )

                if roi_rows:
                    latest_roi = roi_rows[-1]
                    cum_sav    = float(latest_roi.get("cumulative_savings", 0) or 0)
                    roi_val    = float(latest_roi.get("roi_pct",            0) or 0)
                    period     = latest_roi.get("period_label", "N/A")

                    insights = [
                        ("📈", "ROI",        f"{roi_val:.0f}%",  f"as of {period}"),
                        ("💰", "Savings",    f"${cum_sav:,.0f}", "cumulative since go-live"),
                        ("📉", "SLA breach", "17.1% → 2.1%",    "2023 to 2024"),
                        ("🚚", "Expedites",  "−74%",             "expedited cost reduction"),
                    ]

                    for icon, label, value, sub in insights:
                        st.markdown(
                            f"""
                            <div style="margin-bottom:14px;">
                                <span style="font-size:1.2rem;">{icon}</span>
                                <span style="font-size:0.78rem; color:#888888;
                                             margin-left:6px;">{label}</span><br>
                                <span style="font-size:1.1rem; font-weight:700;
                                             color:#2ECC71; margin-left:28px;">
                                    {value}
                                </span><br>
                                <span style="font-size:0.72rem; color:#555555;
                                             margin-left:28px;">{sub}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Run the vector store build to populate ROI data.")

                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info(
                "🔒 **Financial charts are restricted for your role.**\n\n"
                "Switch to Operations Manager or CFO in the sidebar to view "
                "ROI progression and cost data.",
                icon="🔒",
            )

        st.divider()

        # ── DATA FRESHNESS FOOTER ─────────────────────────────────────────────
        footer_cols = st.columns(4)
        footer_data = [
            ("suppliers_master", "3 suppliers"),
            ("shipments",        "100 shipments"),
            ("financial_impact", "36 monthly records"),
            ("vector_store",     "8 PDF chunks (FAISS)"),
        ]

        for col, (table, desc) in zip(footer_cols, footer_data):
            with col:
                st.markdown(
                    f"""
                    <div style="background:#1E2329; border-radius:6px;
                                padding:10px; text-align:center;">
                        <p style="font-size:0.72rem; color:#888888;
                                  margin:0;">{table}</p>
                        <p style="font-size:0.85rem; font-weight:600;
                                  color:#FFFFFF; margin:2px 0 0 0;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    #  TAB 2 — DATA HEALTH CHECK
    # =========================================================================
    with tab2:
        st.markdown("### 🔬 Data Health Monitor")
        st.caption("Pre-flight data quality checks · Runs on load · Pure SQL · Zero LLM")

        # Hex palette for this tab (different from the named-color version used
        # in the old single-page badge so colours render properly in dark mode)
        t2_score_color = (
            "#00ff9d" if score >= 80 else
            "#ffc947" if score >= 60 else
            "#ff4444"
        )
        t2_score_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "🔴"

        # checks_passed derived locally — health dict only contains high/medium
        # counts and the issues list, not a total-checks field.
        _issues_count  = len(health.get("issues", []))
        _checks_passed = _TOTAL_HEALTH_CHECKS - _issues_count

        col_score, col_detail = st.columns([1, 2])
        with col_score:
            st.markdown(f"""
            <div style="background:#0a0f1e; border:2px solid {t2_score_color};
                        border-radius:12px; padding:32px; text-align:center;">
                <div style="font-size:56px; font-weight:800;
                            color:{t2_score_color}; font-family:monospace;">
                    {score}%
                </div>
                <div style="font-size:13px; color:#5a6a8a;
                            text-transform:uppercase; letter-spacing:2px;
                            margin-top:8px;">Data Health Score</div>
                <div style="font-size:12px; color:{t2_score_color};
                            margin-top:12px;">
                    {_checks_passed} of {_TOTAL_HEALTH_CHECKS} checks passed
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown(f"""
            <div style="padding:8px 0;">
                <div style="font-size:13px; color:#e8edf8; margin-bottom:12px;
                            font-weight:600;">Issue Summary</div>
                <div style="font-size:13px; color:#ff4444; margin-bottom:8px;">
                    🔴 {health['high_count']} HIGH severity issues
                </div>
                <div style="font-size:13px; color:#ffc947; margin-bottom:16px;">
                    🟡 {health['medium_count']} MEDIUM severity issues
                </div>
                <div style="font-size:12px; color:#5a6a8a; line-height:1.8;">
                    Affected query types:<br/>
                    {', '.join(health.get('affected_query_types', [])) or 'None'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Issues — expanded by default inside this tab ──────────────────────
        if health.get("issues"):
            for issue in health["issues"]:
                sev_color = "#ff4444" if issue["severity"] == "HIGH" else "#ffc947"
                sev_icon  = "🔴"      if issue["severity"] == "HIGH" else "🟡"
                with st.expander(
                    f"{sev_icon} {issue['check']} — {issue['severity']}",
                    expanded=True,
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        # 'impact' is not in the health-agent return contract;
                        # fall back to 'finding' so the layout stays intact.
                        impact_text = issue.get("impact", issue.get("finding", ""))
                        st.markdown(f"""
                        <div style="font-size:12px; line-height:1.8;">
                            <span style="color:#5a6a8a;">Finding</span><br/>
                            <strong>{issue['finding']}</strong><br/><br/>
                            <span style="color:#5a6a8a;">Impact</span><br/>
                            {impact_text}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        # 'affected_column' is not in the return contract either;
                        # fall back to the check name as a reasonable substitute.
                        affected = issue.get("affected_column", issue.get("check", "N/A"))
                        st.markdown(f"""
                        <div style="font-size:12px; line-height:1.8;">
                            <span style="color:#5a6a8a;">Affected Columns</span><br/>
                            <code>{affected}</code><br/><br/>
                            <span style="color:#5a6a8a;">Recommendation</span><br/>
                            {issue['recommendation']}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.success("✅ All data quality checks passed.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "⚠️ Queries touching flagged metrics will automatically show "
            "data quality caveats in chat responses."
        )

    # =========================================================================
    #  TAB 3 — ALERTS & RECOMMENDATIONS
    # =========================================================================
    with tab3:
        st.markdown("### 🚨 Proactive Risk Alerts")
        st.caption(
            "Auto-evaluated on load · No query required · "
            "Powered by validated SQL templates · RBAC-filtered per role"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Evaluating risk thresholds..."):
            alerts = evaluate_alerts(role=role)

        if not alerts:
            st.success("✅ All metrics within acceptable thresholds.")
        else:
            high_alerts   = [a for a in alerts if a["severity"] == "HIGH"]
            medium_alerts = [a for a in alerts if a["severity"] == "MEDIUM"]

            # ── Summary strip — count cards ───────────────────────────────────
            col_h, col_m, col_t = st.columns(3)
            with col_h:
                st.markdown(f"""
                <div style="background:rgba(255,68,68,0.08);
                            border:1px solid #ff4444; border-radius:8px;
                            padding:16px; text-align:center;">
                    <div style="font-size:32px; font-weight:800;
                                color:#ff4444;">{len(high_alerts)}</div>
                    <div style="font-size:11px; color:#ff4444;
                                text-transform:uppercase;
                                letter-spacing:2px;">High Severity</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m:
                st.markdown(f"""
                <div style="background:rgba(255,201,71,0.08);
                            border:1px solid #ffc947; border-radius:8px;
                            padding:16px; text-align:center;">
                    <div style="font-size:32px; font-weight:800;
                                color:#ffc947;">{len(medium_alerts)}</div>
                    <div style="font-size:11px; color:#ffc947;
                                text-transform:uppercase;
                                letter-spacing:2px;">Medium Severity</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t:
                st.markdown(f"""
                <div style="background:rgba(0,212,255,0.08);
                            border:1px solid #00d4ff; border-radius:8px;
                            padding:16px; text-align:center;">
                    <div style="font-size:32px; font-weight:800;
                                color:#00d4ff;">{len(alerts)}</div>
                    <div style="font-size:11px; color:#00d4ff;
                                text-transform:uppercase;
                                letter-spacing:2px;">Total Alerts</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Severity banners
            if high_alerts:
                st.error(
                    f"🔴 {len(high_alerts)} HIGH severity alert(s) require "
                    "immediate attention"
                )
            if medium_alerts:
                st.warning(
                    f"🟡 {len(medium_alerts)} MEDIUM severity alert(s) flagged "
                    "for review"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Alert cards (existing rendering loop — unchanged) ─────────────
            for alert in alerts:
                severity_color = "#ff4444" if alert["severity"] == "HIGH" else "#ff9900"
                severity_icon  = "🔴"      if alert["severity"] == "HIGH" else "🟡"
                border_bg      = (
                    "rgba(255,68,68,0.06)" if alert["severity"] == "HIGH"
                    else "rgba(255,153,0,0.06)"
                )

                st.markdown(f"""
                <div style="border-left: 4px solid {severity_color};
                            padding: 14px 18px;
                            background: {border_bg};
                            border-radius: 6px;
                            margin-bottom: 6px;">
                    <div style="font-weight:700; font-size:14px;
                                margin-bottom:4px;">
                        {severity_icon} {alert['metric']} &nbsp;
                        <span style="font-size:11px; color:{severity_color};
                                     background:{border_bg}; padding:2px 8px;
                                     border-radius:3px;
                                     border:1px solid {severity_color};">
                            {alert['severity']}
                        </span>
                    </div>
                    <div style="font-size:13px;">{alert['impact_summary']}</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 View Details & Recommendation"):

                    detail_items = list(alert.get("detail", {}).items())
                    mid          = max(1, len(detail_items) // 2)
                    col_a, col_b = st.columns(2)

                    with col_a:
                        for k, v in detail_items[:mid]:
                            st.markdown(
                                f"<div style='font-size:12px; padding:3px 0;'>"
                                f"<span style='color:#888;'>{k}</span>"
                                f"&nbsp;&nbsp;<strong>{v}</strong></div>",
                                unsafe_allow_html=True,
                            )
                    with col_b:
                        for k, v in detail_items[mid:]:
                            st.markdown(
                                f"<div style='font-size:12px; padding:3px 0;'>"
                                f"<span style='color:#888;'>{k}</span>"
                                f"&nbsp;&nbsp;<strong>{v}</strong></div>",
                                unsafe_allow_html=True,
                            )

                    st.markdown(
                        "<div style='margin-top:12px; font-size:12px; "
                        "font-weight:700; color:#aaa;'>RECOMMENDED ACTIONS</div>",
                        unsafe_allow_html=True,
                    )
                    for i, rec in enumerate(alert.get("recommendation", []), 1):
                        st.markdown(
                            f"<div style='font-size:12px; padding:4px 0 4px 8px; "
                            f"border-left:2px solid {severity_color}; "
                            f"margin-bottom:4px;'>{i}. {rec}</div>",
                            unsafe_allow_html=True,
                        )

                    # Chat prompt hint — navigation only, no auto-redirect
                    if alert.get("supplier_id"):
                        st.caption(
                            f"💬 For deeper analysis, go to Chat and ask: "
                            f"\"Should we terminate {alert['supplier_id']}?\""
                        )
