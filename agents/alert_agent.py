"""
Supply Command AI — Proactive Risk Alert Agent
agents/alert_agent.py

Option 2: Data-driven recommendations with detail + recommendation structure.
All SQL uses confirmed column names from schema inspection.
No LLM. No RAG. No session-state passing. DB only.

SQL safety: All queries pass through execute_query() → validate_sql() as per
CLAUDE.md principle 8 ("SQL validated before execution — always").

Returns a list of structured alert dicts with:
    alert_id, severity, metric, supplier_id,
    impact_summary, detail (dict), recommendation (list)
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from database.db_connection import execute_query
from services.logger import get_logger

log = get_logger("alert_agent")

# ── Alert thresholds — locked, do not change without stakeholder sign-off ──────
THRESHOLDS: dict[str, float] = {
    "delay_rate_pct":       15.0,    # flag if any supplier exceeds 15 %
    "otd_gap_vs_benchmark":  5.0,    # flag if fleet OTD gap > 5 pp vs benchmark
    "expedited_cost_usd": 150_000.0, # flag if total expedited_ship_usd > $150 K
    "sla_breaches":         10.0,    # flag if total SLA breaches > 10
}

INDUSTRY_OTD_BENCHMARK: float = 87.0

# Per-role alert visibility — Demand Planner cannot see financial alerts
ROLE_BLOCKED_ALERTS: dict[str, list[str]] = {
    "Demand Planner":     ["EXPEDITED_COST"],
    "Operations Manager": [],
    "CFO":                [],
}


def evaluate_alerts(role: str = "Operations Manager") -> list[dict]:
    """
    Run threshold checks against live DB and return structured alert list.

    Each alert dict contains:
        alert_id      — unique string identifier
        severity      — "HIGH" | "MEDIUM"
        metric        — human-readable metric name
        supplier_id   — supplier this alert concerns, or None for fleet-wide
        impact_summary — one-sentence headline (shown in the card)
        detail        — dict of {label: value} pairs for the expander table
        recommendation — list of strings (ordered action steps)

    Returns empty list if all metrics are within thresholds.
    """
    alerts: list[dict] = []
    can_see_financials = role in ("Operations Manager", "CFO")

    # ── CHECK 1 — Per-supplier delay rate ────────────────────────────────────
    # One big JOIN so we have SLA, OTD, breach counts, and spend in one pass.
    # WHY JOIN instead of two queries?
    #   Alert detail expanders need SLA target, OTD, and spend in one dict.
    #   A single JOIN is faster than 3 sequential queries and avoids
    #   data-sync issues between separate result sets.
    try:
        res = execute_query("""
            SELECT
                s.supplier_id,
                s.supplier_name,
                s.sla_on_time_target_pct,
                s.annual_spend_usd_2024,
                s.risk_tier,
                COUNT(sh.shipment_id)                                    AS total_shipments,
                SUM(CASE WHEN sh.status = 'Delayed' THEN 1 ELSE 0 END)  AS delayed_shipments,
                ROUND(
                    100.0 * SUM(CASE WHEN sh.status = 'Delayed' THEN 1 ELSE 0 END)
                    / COUNT(sh.shipment_id), 2
                )                                                         AS delay_rate_pct,
                ROUND(
                    100.0 * SUM(CASE WHEN sh.status = 'OnTime' THEN 1 ELSE 0 END)
                    / COUNT(sh.shipment_id), 2
                )                                                         AS otd_pct,
                SUM(CASE WHEN sh.sla_breach = 'Yes' THEN 1 ELSE 0 END)  AS sla_breaches
            FROM suppliers_master s
            JOIN shipments sh ON s.supplier_id = sh.supplier_id
            GROUP BY s.supplier_id
            ORDER BY delay_rate_pct DESC
        """)
        supplier_rows = res.get("data", [])

        for row in supplier_rows:
            rate = float(row.get("delay_rate_pct") or 0)
            if rate <= THRESHOLDS["delay_rate_pct"]:
                continue

            severity     = "HIGH" if rate >= 20.0 else "MEDIUM"
            delayed      = int(row.get("delayed_shipments") or 0)
            total        = int(row.get("total_shipments") or 0)
            otd_pct      = float(row.get("otd_pct") or 0)
            sla_target   = float(row.get("sla_on_time_target_pct") or 92.0)
            sla_gap      = round(sla_target - otd_pct, 1)
            sup          = row["supplier_id"]
            spend        = row.get("annual_spend_usd_2024") or 0
            risk_tier    = row.get("risk_tier") or "N/A"

            # Best alternative = other supplier with lowest delay rate
            alternatives = [r for r in supplier_rows if r["supplier_id"] != sup]
            best_alt = (
                min(alternatives, key=lambda x: float(x.get("delay_rate_pct") or 0))
                if alternatives else None
            )

            rec_lines = []
            if best_alt:
                rec_lines.append(
                    f"Redirect 30–40% of volume to "
                    f"{best_alt['supplier_id']} "
                    f"(OTD {best_alt['otd_pct']}% — best available alternative)."
                )
            rec_lines.append(
                f"Issue formal performance improvement notice to "
                f"{sup} — SLA gap is {sla_gap} points."
            )
            rec_lines.append(
                f"Review in 30 days. Escalate to termination if delay rate "
                f"remains above {THRESHOLDS['delay_rate_pct']}%."
            )

            alerts.append({
                "alert_id":       f"DELAY_{sup}",
                "severity":       severity,
                "metric":         "Delay Rate",
                "supplier_id":    sup,
                "impact_summary": (
                    f"{sup} delay rate at {rate}% — "
                    f"{delayed} of {total} shipments delayed."
                ),
                "detail": {
                    "Delay Rate":       f"{rate}%",
                    "OTD":              f"{otd_pct}%",
                    "SLA Target":       f"{sla_target}%",
                    "SLA Gap":          f"{sla_gap} points",
                    "SLA Breaches":     str(row.get("sla_breaches", 0)),
                    "Annual Spend":     f"${spend:,.0f}",
                    "Risk Tier":        risk_tier,
                    "Best Alternative": (
                        f"{best_alt['supplier_id']} "
                        f"(OTD {best_alt['otd_pct']}%)"
                        if best_alt else "None identified"
                    ),
                },
                "recommendation": rec_lines,
            })

        log.debug(f"evaluate_alerts | CHECK 1 (delay rate) | alerts so far: {len(alerts)}")
    except Exception as exc:
        log.error(f"evaluate_alerts | CHECK 1 FAILED | {exc}")
        supplier_rows = []

    # ── CHECK 2 — Fleet OTD gap vs industry benchmark ─────────────────────────
    try:
        res = execute_query("""
            SELECT
                ROUND(
                    100.0 * SUM(CASE WHEN status = 'OnTime' THEN 1 ELSE 0 END)
                    / COUNT(*), 2
                )                                                         AS fleet_otd,
                COUNT(*)                                                  AS total_shipments,
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)     AS total_delayed
            FROM shipments
        """)
        data = res.get("data", [])
        if data:
            fleet_otd = float(data[0].get("fleet_otd") or 0)
            gap       = round(INDUSTRY_OTD_BENCHMARK - fleet_otd, 1)
            if gap > THRESHOLDS["otd_gap_vs_benchmark"]:
                worst = supplier_rows[0] if supplier_rows else None
                alerts.append({
                    "alert_id":       "OTD_GAP",
                    "severity":       "MEDIUM",
                    "metric":         "Fleet OTD vs Benchmark",
                    "supplier_id":    None,
                    "impact_summary": (
                        f"Fleet OTD at {fleet_otd}% vs industry benchmark "
                        f"{INDUSTRY_OTD_BENCHMARK}% — gap of {gap} points."
                    ),
                    "detail": {
                        "Fleet OTD":          f"{fleet_otd}%",
                        "Industry Benchmark": f"{INDUSTRY_OTD_BENCHMARK}%",
                        "Gap":                f"{gap} percentage points",
                        "Total Shipments":    str(data[0]["total_shipments"]),
                        "Total Delayed":      str(data[0]["total_delayed"]),
                        "Biggest Drag":       (
                            f"{worst['supplier_id']} at "
                            f"{worst['delay_rate_pct']}% delay rate"
                            if worst else "N/A"
                        ),
                    },
                    "recommendation": [
                        (
                            f"Closing the gap requires addressing "
                            f"{worst['supplier_id'] if worst else 'top delayed supplier'}"
                            f" — highest contributor to fleet delay."
                        ),
                        f"Target: bring fleet OTD above "
                        f"{INDUSTRY_OTD_BENCHMARK}% within 90 days.",
                        "Implement weekly OTD review cadence with all suppliers.",
                    ],
                })
        log.debug(f"evaluate_alerts | CHECK 2 (OTD gap) | alerts so far: {len(alerts)}")
    except Exception as exc:
        log.error(f"evaluate_alerts | CHECK 2 FAILED | {exc}")

    # ── CHECK 3 — Total expedited cost (financial roles only) ─────────────────
    # WHY skip Demand Planner?
    #   financial_impact table is restricted to Operations Manager and CFO
    #   per RBAC config in agents/guardrails.py. Running this query for a
    #   Demand Planner would violate the access control contract.
    if can_see_financials:
        try:
            res = execute_query("""
                SELECT
                    SUM(expedited_ship_usd)  AS total_expedited,
                    SUM(delay_penalty_usd)   AS total_penalty,
                    SUM(total_avoidable_usd) AS total_avoidable,
                    SUM(ai_savings_usd)      AS total_ai_savings
                FROM financial_impact
            """)
            data = res.get("data", [])
            if data:
                total_exp = float(data[0].get("total_expedited") or 0)
                if total_exp > THRESHOLDS["expedited_cost_usd"]:
                    overage     = total_exp - THRESHOLDS["expedited_cost_usd"]
                    penalty     = float(data[0].get("total_penalty")   or 0)
                    avoidable   = float(data[0].get("total_avoidable") or 0)
                    ai_savings  = float(data[0].get("total_ai_savings") or 0)
                    alerts.append({
                        "alert_id":       "EXPEDITED_COST",
                        "severity":       "HIGH",
                        "metric":         "Total Expedited Shipping Cost",
                        "supplier_id":    None,
                        "impact_summary": (
                            f"Total expedited shipping cost at ${total_exp:,.0f} "
                            f"— exceeds threshold by ${overage:,.0f}."
                        ),
                        "detail": {
                            "Expedited Cost":     f"${total_exp:,.0f}",
                            "Threshold":          f"${THRESHOLDS['expedited_cost_usd']:,.0f}",
                            "Overage":            f"${overage:,.0f}",
                            "Delay Penalties":    f"${penalty:,.0f}",
                            "Total Avoidable":    f"${avoidable:,.0f}",
                            "AI Savings to Date": f"${ai_savings:,.0f}",
                        },
                        "recommendation": [
                            "Expedited cost is driven by delayed suppliers — "
                            "address top delay offenders first.",
                            f"Avoidable cost pool is ${avoidable:,.0f} — "
                            "highest ROI reduction opportunity.",
                            "Expand AI intervention coverage to reduce "
                            "reactive expediting.",
                        ],
                    })
            log.debug(f"evaluate_alerts | CHECK 3 (expedited cost) | alerts so far: {len(alerts)}")
        except Exception as exc:
            log.error(f"evaluate_alerts | CHECK 3 FAILED | {exc}")

    # ── CHECK 4 — SLA breaches ────────────────────────────────────────────────
    try:
        res = execute_query("""
            SELECT
                SUM(CASE WHEN sla_breach = 'Yes' THEN 1 ELSE 0 END) AS total_breaches,
                COUNT(*)                                              AS total_shipments
            FROM shipments
        """)
        data = res.get("data", [])
        if data:
            breaches = int(data[0].get("total_breaches") or 0)
            total_s  = int(data[0].get("total_shipments") or 0)
            if breaches > THRESHOLDS["sla_breaches"]:
                # Top offender — separate quick query
                top_res = execute_query("""
                    SELECT
                        supplier_id,
                        SUM(CASE WHEN sla_breach = 'Yes' THEN 1 ELSE 0 END) AS breaches
                    FROM shipments
                    GROUP BY supplier_id
                    ORDER BY breaches DESC
                    LIMIT 1
                """)
                top_breach = (top_res.get("data") or [{}])[0]
                breach_rate = round(100.0 * breaches / total_s, 1) if total_s else 0
                alerts.append({
                    "alert_id":       "SLA_BREACH",
                    "severity":       "MEDIUM",
                    "metric":         "SLA Breaches",
                    "supplier_id":    None,
                    "impact_summary": (
                        f"{breaches} SLA breaches recorded — "
                        f"exceeds threshold of {int(THRESHOLDS['sla_breaches'])}."
                    ),
                    "detail": {
                        "Total Breaches":  str(breaches),
                        "Threshold":       str(int(THRESHOLDS["sla_breaches"])),
                        "Total Shipments": str(total_s),
                        "Breach Rate":     f"{breach_rate}%",
                        "Top Offender":    (
                            f"{top_breach.get('supplier_id', 'N/A')} "
                            f"({top_breach.get('breaches', 0)} breaches)"
                            if top_breach.get("supplier_id") else "N/A"
                        ),
                    },
                    "recommendation": [
                        (
                            f"Initiate corrective action plan with "
                            f"{top_breach.get('supplier_id', 'top supplier')}"
                            f" — highest SLA breach count."
                        ),
                        "Require weekly breach reporting from all "
                        "suppliers above threshold.",
                        "Review SLA terms at next contract renewal — "
                        "tighten penalty clauses.",
                    ],
                })
        log.debug(f"evaluate_alerts | CHECK 4 (SLA breaches) | alerts so far: {len(alerts)}")
    except Exception as exc:
        log.error(f"evaluate_alerts | CHECK 4 FAILED | {exc}")

    # ── Role filter ───────────────────────────────────────────────────────────
    blocked = ROLE_BLOCKED_ALERTS.get(role, [])
    if blocked:
        alerts = [a for a in alerts if a["alert_id"] not in blocked]

    # Sort HIGH first, then MEDIUM
    alerts.sort(key=lambda a: 0 if a["severity"] == "HIGH" else 1)

    log.info(
        f"evaluate_alerts | role='{role}' | total={len(alerts)} | "
        f"high={sum(1 for a in alerts if a['severity'] == 'HIGH')} | "
        f"medium={sum(1 for a in alerts if a['severity'] == 'MEDIUM')}"
    )
    return alerts
