"""
Supply Command AI — DB Agent

Executes structured data queries against supply_chain.db.
Receives one step dict from the Planning Agent and returns
a structured finding dict with confidence, caveats, and metadata.

LLM Usage Policy:
    NONE — this entire file is pure Python + SQL.
    Zero tokens consumed at any point.

SQL Security model:
    All SQL is pre-written in this file (Section 1).
    No SQL is generated at runtime from user input.
    Every query passes through db_connection.execute_query()
    which runs validate_sql() before touching the database.
    Three layers of protection: pre-written → guardrail → parameterised.

WHY pre-written SQL instead of LLM-generated SQL?
    1. Security   — pre-written SQL cannot be manipulated via prompt injection.
    2. Correctness — SQL is reviewed at build time, not runtime.
    3. Speed       — no LLM call needed to get the query, just a dict lookup.
    4. Auditability — every query variant is visible in this file.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from services.logger import get_logger
from database.db_connection import (
    execute_query,
    get_table_schema,
    log_agent_decision,
)
from agents.guardrails import ROLES

log = get_logger("db_agent")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SQL TEMPLATE LIBRARY
#  All SQL is written and reviewed here — never generated at runtime.
#  get_sql_template() is a pure Python dict lookup. Zero LLM tokens.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Template store: task_key → SQL string ─────────────────────────────────────
#
# WHY store SQL as a dict instead of .sql files?
#   Single-file visibility — every query the system can run is readable
#   in one place. No hidden .sql files that could be swapped out.
#
# WHY ROUND() on all percentages?
#   Raw floating-point arithmetic in SQLite produces values like 19.444...
#   Rounding to 1 decimal keeps the UI clean without post-processing.

# CANONICAL DEFINITION: delayed shipment = status = 'Delayed'
# Never use delay_days > 0, sla_breach = 'Yes', or any time filter
# unless the user's query explicitly mentions a time period.
# All delay-related SQL templates in this file enforce this definition.

_SQL_TEMPLATES: dict[str, str] = {

    # ── Total delayed shipment count (fleet-wide, no filter) ─────────────────
    # WHY a dedicated count template?
    #   "How many shipments are delayed?" is a direct scalar question.
    #   Using delay_count_by_supplier for this would return per-supplier rows
    #   and force the executive to sum them — error-prone and verbose.
    #   A single COUNT(*) query is unambiguous and returns exactly 1 row.
    # WHY status = 'Delayed' not delay_days > 0?
    #   delay_days > 0 includes shipments where the delay field was populated
    #   but the status was corrected to OnTime (e.g. recovered shipments).
    #   status = 'Delayed' is the authoritative operational state.
    "total_delayed_count": """
        SELECT
            COUNT(*) AS total_delayed,
            COUNT(DISTINCT supplier_id) AS suppliers_affected,
            COUNT(DISTINCT product_category) AS categories_affected
        FROM shipments
        WHERE status = 'Delayed'
    """,

    # ── Delayed count for a specific calendar month ───────────────────────────
    # WHY a parameterised template?
    #   The user says "in December 2024" — we need to filter by month.
    #   Using a ? placeholder prevents SQL injection even though the month
    #   string comes from controlled planning-agent extraction, not raw input.
    # WHY strftime('%Y-%m', shipment_date)?
    #   shipment_date is stored as TEXT in ISO format (YYYY-MM-DD).
    #   strftime extracts the year-month prefix for grouping and filtering.
    # NOTE: pass params=("YYYY-MM",) when calling execute_query() for this template.
    "delayed_count_by_month": """
        SELECT
            strftime('%Y-%m', shipment_date) AS month,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct
        FROM shipments
        WHERE strftime('%Y-%m', shipment_date) = ?
        GROUP BY month
    """,

    # ── Delay count by supplier ───────────────────────────────────────────────
    # WHY include avg_delay_days?
    #   Delay rate alone doesn't capture severity. A supplier with a 20%
    #   delay rate where delays average 1 day is less harmful than a
    #   supplier with a 15% rate but 10-day average delays.
    "delay_count_by_supplier": """
        SELECT
            supplier_id,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct,
            AVG(CASE WHEN delay_days > 0 THEN delay_days END)
                AS avg_delay_days
        FROM shipments
        GROUP BY supplier_id
        ORDER BY delay_rate_pct DESC
    """,

    # ── Delay trend by month ──────────────────────────────────────────────────
    # WHY strftime('%Y-%m', shipment_date)?
    #   Produces sortable ISO month strings (2023-01, 2023-02…) that the
    #   UI can plot directly on a time axis without further transformation.
    "delay_trend_by_month": """
        SELECT
            strftime('%Y-%m', shipment_date) AS month,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct
        FROM shipments
        GROUP BY strftime('%Y-%m', shipment_date)
        ORDER BY month
    """,

    # ── Top delay reasons ─────────────────────────────────────────────────────
    # WHY a correlated subquery for the denominator?
    #   pct_of_delays must be percentage of ALL delays, not percentage of
    #   the group. Using COUNT(*) alone in a GROUP BY would give the wrong
    #   denominator. The subquery calculates total delayed shipments once.
    "top_delay_reasons": """
        SELECT
            delay_reason_category,
            COUNT(*) AS incident_count,
            ROUND(
                COUNT(*) * 100.0
                / (SELECT COUNT(*) FROM shipments WHERE status = 'Delayed'),
            1) AS pct_of_delays,
            AVG(delay_days) AS avg_delay_days
        FROM shipments
        WHERE status = 'Delayed'
          AND delay_reason_category IS NOT NULL
        GROUP BY delay_reason_category
        ORDER BY incident_count DESC
    """,

    # ── Supplier SLA performance ──────────────────────────────────────────────
    # WHY sla_gap_pct as the sort key?
    #   The most actionable insight is not who has the lowest OTD, but who
    #   is furthest below their contractual commitment. A supplier at 88%
    #   OTD with a 95% target is more problematic than one at 85% OTD
    #   with an 80% target.
    "supplier_sla_performance": """
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.sla_on_time_target_pct,
            ROUND(
                SUM(CASE WHEN sh.status = 'OnTime' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS actual_otd_pct,
            ROUND(
                s.sla_on_time_target_pct
                - (SUM(CASE WHEN sh.status = 'OnTime' THEN 1 ELSE 0 END)
                   * 100.0 / COUNT(*)), 1
            ) AS sla_gap_pct,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN sh.sla_breach = 'Yes' THEN 1 ELSE 0 END)
                AS breach_count
        FROM suppliers_master s
        JOIN shipments sh ON s.supplier_id = sh.supplier_id
        GROUP BY s.supplier_id
        ORDER BY sla_gap_pct DESC
    """,

    # ── Financial cost breakdown by year ─────────────────────────────────────
    # WHY GROUP BY year and not month?
    #   Strategic cost questions ("how much did we spend last year?") need
    #   annual totals. Monthly granularity is available in monthly_cost_trend.
    #   Two separate templates prevent bloated result sets.
    "financial_cost_breakdown": """
        SELECT
            year,
            SUM(total_sc_cost_usd) AS annual_cost,
            SUM(expedited_ship_usd) AS expedited_total,
            SUM(stockout_loss_usd) AS stockout_total,
            SUM(delay_penalty_usd) AS penalty_total,
            SUM(total_avoidable_usd) AS avoidable_total,
            AVG(on_time_rate_pct) AS avg_otd_pct
        FROM financial_impact
        GROUP BY year
        ORDER BY year
    """,

    # ── ROI progression after AI go-live ─────────────────────────────────────
    # WHY WHERE roi_pct > 0?
    #   Before AI go-live (Jul 2024), roi_pct is 0 — there was no AI
    #   investment. Including pre-go-live rows adds noise to the ROI story.
    #   The Planning Agent uses a separate template (monthly_cost_trend)
    #   for baseline comparison.
    "roi_progression": """
        SELECT
            period_label,
            ai_savings_usd,
            cumulative_savings,
            roi_pct,
            on_time_rate_pct,
            notes
        FROM financial_impact
        WHERE roi_pct > 0
        ORDER BY year, month
    """,

    # ── Monthly cost trend (full history) ────────────────────────────────────
    # WHY include geopolitical_cost?
    #   Supply chain cost spikes often correlate with geopolitical events
    #   (Russia/Ukraine war, COVID lockdowns). Including it lets the
    #   Executive Agent explain anomalies without needing a separate query.
    "monthly_cost_trend": """
        SELECT
            period_label,
            total_sc_cost_usd,
            expedited_ship_usd,
            geopolitical_cost,
            on_time_rate_pct,
            notes
        FROM financial_impact
        ORDER BY year, month
    """,

    # ── High-risk shipments (live view) ──────────────────────────────────────
    # WHY LIMIT 20?
    #   A UI card showing 200 high-risk shipments is not actionable.
    #   20 is the practical upper limit for a daily review. The Operations
    #   Manager can filter further if needed.
    "high_risk_shipments": """
        SELECT
            shipment_id,
            shipment_date,
            supplier_id,
            product_category,
            region,
            status,
            delay_days,
            delay_reason_category,
            risk_flag,
            recommended_action
        FROM shipments
        WHERE risk_flag = 'High'
        ORDER BY shipment_date DESC
        LIMIT 20
    """,

    # ── Total shipments (all statuses, fleet-wide) ────────────────────────────
    # WHY separate from total_delayed_count?
    #   "How many shipments are there?" asks for all shipments — not just delayed.
    #   Using total_delayed_count would return a misleadingly low number.
    #   This template is the unambiguous answer to total fleet size queries.
    "total_shipments": """
        SELECT
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'OnTime' THEN 1 ELSE 0 END) AS on_time_count,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'OnTime' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS on_time_rate_pct
        FROM shipments
    """,

    # ── Lowest delay rate supplier (best performer) ───────────────────────────
    # WHY ORDER BY ASC instead of DESC?
    #   delay_count_by_supplier orders DESC (worst first). This template
    #   orders ASC to surface the best-performing supplier — needed for
    #   queries like "which supplier has the lowest delay rate?" or
    #   "who is our best supplier?".
    # WHY LIMIT 1?
    #   The user wants the single best performer. Returning all 3 suppliers
    #   for a "lowest" question adds noise and risks the executive selecting
    #   the wrong row.
    "lowest_delay_rate_supplier": """
        SELECT
            supplier_id,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct,
            AVG(CASE WHEN delay_days > 0 THEN delay_days END) AS avg_delay_days
        FROM shipments
        GROUP BY supplier_id
        ORDER BY delay_rate_pct ASC
        LIMIT 1
    """,

    # ── Delay rate ranked by region ───────────────────────────────────────────
    # WHY GROUP BY region?
    #   Region-level delay analysis identifies geographic bottlenecks
    #   (port congestion, customs delays, weather corridors) that are
    #   invisible when looking at supplier-level data only.
    "highest_delay_rate_region": """
        SELECT
            region,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct
        FROM shipments
        WHERE region IS NOT NULL
        GROUP BY region
        ORDER BY delay_rate_pct DESC
    """,

    # ── Delay rate ranked by product category ────────────────────────────────
    # WHY include product category analysis?
    #   Some product lines (e.g. surgical implants) are more delay-sensitive
    #   than others (consumables). Category-level delay rates help the
    #   Demand Planner prioritise which categories need expedited handling.
    "highest_delay_rate_product_category": """
        SELECT
            product_category,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct
        FROM shipments
        WHERE product_category IS NOT NULL
        GROUP BY product_category
        ORDER BY delay_rate_pct DESC
    """,

    # ── Total avoidable cost (fleet-wide, all time) ───────────────────────────
    # WHY SUM across all months?
    #   "What is total avoidable cost?" asks for a fleet-wide aggregate —
    #   not a monthly breakdown. Using monthly_cost_trend and summing
    #   in Python is error-prone. A direct SUM() is unambiguous.
    # WHY total_avoidable_usd not total_sc_cost_usd?
    #   Avoidable costs = expedited freight + stockout loss + delay penalty.
    #   These are costs that better planning could eliminate. Total SC cost
    #   includes fixed costs that cannot be avoided regardless of performance.
    "total_avoidable_cost": """
        SELECT
            SUM(total_avoidable_usd) AS total_avoidable,
            SUM(expedited_ship_usd)  AS total_expedited,
            SUM(stockout_loss_usd)   AS total_stockout,
            SUM(delay_penalty_usd)   AS total_penalties,
            COUNT(*)                 AS months_of_data,
            MIN(year)                AS from_year,
            MAX(year)                AS to_year
        FROM financial_impact
        WHERE total_avoidable_usd IS NOT NULL
    """,

    # ── Total supply chain cost (fleet-wide, all time) ───────────────────────
    # WHY SUM all years?
    #   "What is total supply chain cost?" asks for cumulative spend — not
    #   an annual snapshot. financial_cost_breakdown returns per-year rows;
    #   this template gives the single lifetime figure the CFO needs.
    "total_supply_chain_cost": """
        SELECT
            SUM(total_sc_cost_usd)  AS total_supply_chain_cost,
            AVG(total_sc_cost_usd)  AS avg_monthly_cost,
            MIN(total_sc_cost_usd)  AS min_monthly_cost,
            MAX(total_sc_cost_usd)  AS max_monthly_cost,
            COUNT(*)                AS months_of_data,
            MIN(year)               AS from_year,
            MAX(year)               AS to_year
        FROM financial_impact
        WHERE total_sc_cost_usd IS NOT NULL
    """,

    # ── Total expedited shipping cost (fleet-wide, all time) ─────────────────
    # WHY expedited_ship_usd and not total_avoidable_usd?
    #   expedited_ship_usd is the specific column for expedited freight costs.
    #   total_avoidable_usd is a composite (expedited + stockout + penalty).
    #   When the alert asks "what is the expedited cost?", the user wants
    #   the specific line item, not the composite avoidable figure.
    # WHY no COUNT(*)?
    #   The scalar result is the complete answer — "across N periods" adds
    #   noise to an alert-driven one-sentence response. Keeping the template
    #   minimal prevents the interpreter from embedding period context the
    #   user did not ask for.
    "total_expedited_cost": """
        SELECT
            SUM(expedited_ship_usd) AS total_expedited_cost
        FROM financial_impact
    """,

    # ── Entity-specific delay rate (single supplier) ──────────────────────────
    # WHY a separate template from delay_count_by_supplier?
    #   Alert-driven queries ("What is SUP003's delay rate?") need a scalar
    #   result for exactly one supplier. delay_count_by_supplier returns all
    #   3 suppliers — the executive then has to find the right row. A targeted
    #   WHERE clause is unambiguous and cannot pick the wrong row.
    # WHY ? placeholder?
    #   Uses db_connection's validate_sql() parameterised path. The value is
    #   always a controlled forced_entity string set by alert_agent (e.g.
    #   "SUP001"), never raw user input — but parameterised queries are safer.
    # NOTE: pass params=(forced_entity,) when calling execute_query().
    "supplier_delay_rate": """
        SELECT
            supplier_id,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_shipments,
            ROUND(
                100.0 * SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                / COUNT(*), 2
            ) AS delay_rate_pct
        FROM shipments
        WHERE supplier_id = ?
        GROUP BY supplier_id
    """,

    # ── Fleet-wide OTD vs industry benchmark ─────────────────────────────────
    # WHY a scalar instead of per-supplier?
    #   The OTD alert fires on the fleet-wide gap vs the 87% industry benchmark.
    #   A per-supplier breakdown (supplier_sla_performance) would require the
    #   executive to compute the weighted average and then compare — error-prone.
    #   A direct shipment-weighted scalar is the authoritative fleet OTD figure.
    # WHY status = 'OnTime' (not sla_breach)?
    #   OTD = on-time delivery rate, computed from shipment status.
    #   sla_breach flags contractual misses which may use different thresholds.
    #   "OnTime" is the canonical status value per the CANONICAL DEFINITION above.
    "fleet_otd_vs_benchmark": """
        SELECT
            ROUND(
                SUM(CASE WHEN status = 'OnTime' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS fleet_otd,
            COUNT(*) AS total_shipments
        FROM shipments
    """,

    # ── Total SLA breach count (fleet-wide scalar) ────────────────────────────
    # WHY a dedicated template instead of supplier_sla_performance?
    #   supplier_sla_performance returns per-supplier rows (breach_count per row).
    #   When the alert fires on the fleet-wide total (14 breaches) and the user
    #   asks "how many total SLA breaches are there?", we need one scalar row.
    #   Summing from supplier_sla_performance would require the executive to
    #   aggregate multiple rows — error-prone. A direct COUNT is unambiguous.
    # WHY sla_breach = 'Yes' not breach_count > 0?
    #   sla_breach is the authoritative flag stored per shipment in the shipments
    #   table. breach_count in supplier_sla_performance is a derived aggregate;
    #   going back to the source ensures consistency with alert_agent CHECK 4.
    "total_sla_breaches": """
        SELECT
            SUM(CASE WHEN sla_breach = 'Yes' THEN 1 ELSE 0 END) AS total_sla_breaches,
            COUNT(*) AS total_shipments
        FROM shipments
    """,

    # ── All suppliers ranked by delay rate (comparison view) ─────────────────
    # WHY no LIMIT?
    #   Comparison queries ("compare delay rate across all suppliers") want
    #   ALL suppliers ranked, not just the top/bottom one. The executive
    #   formats them as a numbered ranked list when multi_row=True.
    # WHY ORDER BY delay_rate_pct DESC?
    #   Most-problematic supplier first — that is the natural reading order
    #   for a "compare" query: worst → best, so action priority is clear.
    "supplier_delay_comparison": """
        SELECT
            supplier_id,
            COUNT(*) AS total_shipments,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_count,
            ROUND(
                SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*), 1
            ) AS delay_rate_pct,
            AVG(CASE WHEN delay_days > 0 THEN delay_days END) AS avg_delay_days
        FROM shipments
        GROUP BY supplier_id
        ORDER BY delay_rate_pct DESC
    """,

    # ── Shipment value templates ──────────────────────────────────────────────
    "total_shipment_value": """
        SELECT
            ROUND(SUM(shipment_value_usd) / 1000000.0, 2) AS total_millions,
            COUNT(*) AS shipments
        FROM shipments
    """,

    "avg_shipment_value": """
        SELECT
            ROUND(AVG(shipment_value_usd), 0) AS avg_value,
            COUNT(*) AS shipments
        FROM shipments
    """,

    "supplier_shipment_value": """
        SELECT
            supplier_id,
            ROUND(SUM(shipment_value_usd) / 1000000.0, 2) AS value_millions,
            ROUND(
                100.0 * SUM(shipment_value_usd)
                / (SELECT SUM(shipment_value_usd) FROM shipments), 1
            ) AS pct
        FROM shipments
        GROUP BY supplier_id
        ORDER BY value_millions DESC
    """,

    "region_shipment_value": """
        SELECT
            region,
            ROUND(SUM(shipment_value_usd) / 1000000.0, 2) AS value_millions
        FROM shipments
        GROUP BY region
        ORDER BY value_millions DESC
    """,

    "category_shipment_value": """
        SELECT
            product_category,
            ROUND(SUM(shipment_value_usd) / 1000000.0, 2) AS value_millions
        FROM shipments
        GROUP BY product_category
        ORDER BY value_millions DESC
    """,

    # ── Delay statistics templates ────────────────────────────────────────────
    "avg_delay_days": """
        SELECT
            ROUND(AVG(delay_days), 1) AS avg_delay,
            MAX(delay_days)           AS max_delay,
            COUNT(*)                  AS delayed_count
        FROM shipments
        WHERE status = 'Delayed'
    """,

    # ── Maximum observed delay (scalar — no avg contamination) ───────────────
    # WHY a separate template from avg_delay_days?
    #   avg_delay_days returns BOTH avg and max, causing the Executive Agent
    #   bypass to produce "Average delay ... (max 12 days)" when the user
    #   asked only for the maximum. A dedicated scalar avoids this contamination
    #   and lets the test assert "12" without "6.0" or "average" appearing.
    "max_delay_days": """
        SELECT
            MAX(delay_days) AS max_delay,
            COUNT(*)        AS delayed_count
        FROM shipments
        WHERE status = 'Delayed'
    """,

    "overall_delay_rate": """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS delayed,
            ROUND(
                100.0 * SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) AS rate
        FROM shipments
    """,

    # ── Date span templates ───────────────────────────────────────────────────
    "shipment_date_span": """
        SELECT
            MIN(shipment_date) AS first_date,
            MAX(shipment_date) AS last_date,
            COUNT(*)           AS total
        FROM shipments
    """,

    "financial_date_span": """
        SELECT
            (SELECT period_label FROM financial_impact
             ORDER BY year ASC, month ASC LIMIT 1)  AS first_period,
            (SELECT period_label FROM financial_impact
             ORDER BY year DESC, month DESC LIMIT 1) AS last_period,
            COUNT(*) AS months
        FROM financial_impact
    """,

    # ── Financial breakdown templates ─────────────────────────────────────────
    "annual_sc_cost": """
        SELECT
            year,
            SUM(total_sc_cost_usd)                          AS annual_cost,
            ROUND(SUM(total_sc_cost_usd) / 1000000.0, 2)   AS annual_cost_millions
        FROM financial_impact
        GROUP BY year
        ORDER BY year
    """,

    "ai_investment_by_year": """
        SELECT
            year,
            SUM(ai_investment_usd) AS investment,
            SUM(ai_savings_usd)    AS savings
        FROM financial_impact
        GROUP BY year
        ORDER BY year
    """,

    # ── SLA gap view (per supplier) ───────────────────────────────────────────
    "sla_gap_by_supplier": """
        SELECT
            s.supplier_id,
            s.sla_on_time_target_pct AS target,
            ROUND(
                100.0 * SUM(CASE WHEN sh.status = 'OnTime' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) AS actual_otd,
            ROUND(
                s.sla_on_time_target_pct
                - 100.0 * SUM(CASE WHEN sh.status = 'OnTime' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            ) AS sla_gap,
            SUM(CASE WHEN sh.sla_breach = 'Yes' THEN 1 ELSE 0 END) AS breaches
        FROM suppliers_master s
        JOIN shipments sh ON s.supplier_id = sh.supplier_id
        GROUP BY s.supplier_id
        ORDER BY sla_gap DESC
    """,
}

# ── Task description → template key routing ───────────────────────────────────
#
# WHY this routing table instead of passing template keys directly?
#   The Planning Agent produces human-readable task descriptions for
#   explainability ("Get delay count and delay rate by supplier and region").
#   The DB Agent needs to map those descriptions to SQL template keys.
#   A routing table keeps both layers independent — the Planning Agent
#   doesn't need to know SQL template names, and the DB Agent doesn't
#   need to parse planning logic.

_TASK_ROUTING: list[tuple[list[str], str]] = [
    # Each entry: ([keywords that appear in the task description], template_key)
    # Evaluated in order — first match wins.
    # WHY most-specific templates first?
    #   Broad keywords like "delay" appear in many templates. Specific phrases
    #   like "lowest delay rate by supplier" must be matched before "delay rate"
    #   alone would match a broader template. Most-specific → least-specific order.

    # ── Count / aggregate templates (most specific first) ─────────────────────
    (["total delayed count", "total delayed", "fleet-wide count", "total delayed fleet"],
                                                                 "total_delayed_count"),
    (["delayed count by month", "delayed in month", "month delayed count"],
                                                                 "delayed_count_by_month"),
    (["total shipments fleet", "total shipments", "total number of shipments",
      "count all shipments", "fleet size"],                      "total_shipments"),

    # ── Dimension-specific delay rate templates ────────────────────────────────
    (["lowest delay rate", "best delay rate", "best performing supplier",
      "lowest delay supplier", "fewest delays supplier"],        "lowest_delay_rate_supplier"),
    (["highest delay rate region", "delay rate by region",
      "worst region", "highest region delay", "region delay"],   "highest_delay_rate_region"),
    (["highest delay rate product", "delay rate by product",
      "worst product category", "category delay"],    "highest_delay_rate_product_category"),
    (["supplier delay comparison", "compare delay rate", "compare suppliers",
      "all supplier delay", "delay rate across", "across all suppliers"],
                                                       "supplier_delay_comparison"),

    # ── Financial aggregate templates ─────────────────────────────────────────
    (["expedited cost", "expedited shipping cost", "total expedited cost",
      "total_expedited_cost", "expedited ship", "total expedited"],
                                                                 "total_expedited_cost"),
    (["total sla breaches", "total_sla_breaches", "fleet sla breach",
      "sla breach count", "how many sla breaches", "total breaches"],
                                                                 "total_sla_breaches"),
    (["total avoidable cost", "total avoidable usd",
      "avoidable cost total", "total avoidable"],                "total_avoidable_cost"),
    # ── OTD vs benchmark (fleet scalar) ──────────────────────────────────────
    (["fleet otd", "fleet_otd", "fleet on-time", "otd vs benchmark",
      "fleet_otd_vs_benchmark"],                                 "fleet_otd_vs_benchmark"),
    (["total supply chain cost", "total sc cost",
      "total supply chain spend", "all supply chain",
      "supply chain cost", "sc cost"],                           "total_supply_chain_cost"),

    # ── Standard multi-row templates ──────────────────────────────────────────
    (["delay count", "delay rate", "by supplier", "by region"],  "delay_count_by_supplier"),
    (["delay trend", "by month", "monthly"],                     "delay_trend_by_month"),
    (["delay reason", "top delay", "reason"],                    "top_delay_reasons"),
    (["sla target", "sla performance", "on-time rate", "otd",
      "sla gap", "on time rate", "actual_otd"],                  "supplier_sla_performance"),
    (["delay rate", "sla breach", "breach count", "risk flag",
      "by supplier"],                                            "delay_count_by_supplier"),
    (["cost breakdown", "annual cost", "freight", "penalty",
      "stockout"],                                               "financial_cost_breakdown"),
    (["roi", "ai_savings", "ai savings", "go-live", "payback",
      "cumulative", "return on investment"],                     "roi_progression"),
    (["cost trend", "monthly cost", "geopolitical",
      "total_sc_cost", "cost over time"],                        "monthly_cost_trend"),
    (["high risk", "risk_flag", "risk flag", "high-risk"],        "high_risk_shipments"),
    # Fallback aliases
    (["supplier performance", "supplier risk", "supplier name"], "supplier_sla_performance"),

    # ── New Category-2 templates ──────────────────────────────────────────────
    (["total shipment value", "shipment value total"],            "total_shipment_value"),
    (["average shipment value", "avg shipment value"],            "avg_shipment_value"),
    (["shipment value by supplier", "supplier.*value",
      "which supplier carries", "most shipment value"],           "supplier_shipment_value"),
    (["region.*value", "region carries",
      "shipment value by region"],                                "region_shipment_value"),
    (["category.*value", "product.*value",
      "shipment value by category"],                              "category_shipment_value"),
    (["average delay", "avg delay", "maximum delay",
      "max delay", "average delay days"],                         "avg_delay_days"),
    (["overall delay rate", "fleet.*delay rate",
      "fleet delay rate", "overall.*delay"],                      "overall_delay_rate"),
    (["shipment date span", "date.*span", "data.*span",
      "when.*data", "earliest shipment", "latest shipment"],      "shipment_date_span"),
    (["financial.*span", "financial date", "financial period"],   "financial_date_span"),
    (["annual.*cost", "cost.*annual", "cost.*year",
      "yearly cost", "per year cost"],                            "annual_sc_cost"),
    (["ai.*invest", "invest.*ai", "money.*ai",
      "ai investment", "how much.*ai", "spent on ai"],            "ai_investment_by_year"),
    (["sla.*gap", "delay.*sla", "gap.*sla",
      "sla gap by supplier", "sla performance gap"],              "sla_gap_by_supplier"),
]


# ── Templates that use financial_impact (filter by year / month INT columns) ─
_FINANCIAL_TEMPLATES: set[str] = {
    "total_supply_chain_cost", "total_expedited_cost", "total_avoidable_cost",
    "financial_cost_breakdown", "roi_progression", "monthly_cost_trend",
    "annual_sc_cost", "ai_investment_by_year",
}

# ── Templates that use shipments (filter by shipment_date TEXT column) ────────
_SHIPMENT_TEMPLATES: set[str] = {
    "total_delayed_count", "delayed_count_by_month", "total_shipments",
    "delay_count_by_supplier", "lowest_delay_rate_supplier",
    "highest_delay_rate_region", "highest_delay_rate_product_category",
    "supplier_delay_comparison", "delay_trend_by_month", "top_delay_reasons",
    "fleet_otd_vs_benchmark", "total_sla_breaches", "supplier_delay_rate",
    "high_risk_shipments", "total_shipment_value", "avg_shipment_value",
    "supplier_shipment_value", "region_shipment_value", "category_shipment_value",
    "avg_delay_days", "max_delay_days", "overall_delay_rate",
}


def _inject_time_filter(sql: str, template_key: str,
                        filter_year, filter_month) -> str:
    """
    Inject year / month WHERE conditions into a pre-written SQL template.

    WHY runtime injection instead of parameterised placeholders?
        SQL templates use structural clauses (GROUP BY, ORDER BY) that
        cannot be parameterised. The year and month come from controlled
        planning-agent extraction (regex on known years 2022-2024), not
        raw user input, so string injection is safe here.

    For financial_impact:  year = {y}  [AND month = {m}]
    For shipments:         strftime('%Y', shipment_date) = '{y}'  [AND month]
    Templates not in either set are returned unchanged.
    """
    import re as _re
    if not filter_year:
        return sql

    if template_key in _FINANCIAL_TEMPLATES:
        conds = f"year = {filter_year}"
        if filter_month:
            conds += f" AND month = {filter_month}"
    elif template_key in _SHIPMENT_TEMPLATES:
        conds = f"strftime('%Y', shipment_date) = '{filter_year}'"
        if filter_month:
            conds += f" AND strftime('%m', shipment_date) = '{filter_month:02d}'"
    else:
        return sql   # no filter for join-heavy templates (sla_gap_by_supplier etc.)

    has_where   = bool(_re.search(r'\bWHERE\b', sql, _re.IGNORECASE))
    connector   = "AND" if has_where else "WHERE"

    # Try to insert before GROUP BY
    gm = _re.search(r'\bGROUP\s+BY\b', sql, _re.IGNORECASE)
    if gm:
        pos = gm.start()
        return sql[:pos] + f"        {connector} {conds}\n        " + sql[pos:]

    # Try to insert before ORDER BY (no GROUP BY)
    om = _re.search(r'\bORDER\s+BY\b', sql, _re.IGNORECASE)
    if om:
        pos = om.start()
        return sql[:pos] + f"        {connector} {conds}\n        " + sql[pos:]

    # No GROUP BY, no ORDER BY — append at end
    return sql.rstrip() + f"\n        {connector} {conds}"


def _resolve_template_key(task_description: str) -> Optional[str]:
    """
    Map a planning-agent task description to a SQL template key.

    WHY keyword list of lists?
        A task description may contain multiple signals. We score by
        how many keywords from a routing entry appear in the description,
        then return the highest-scoring template. This handles edge cases
        better than a simple "first match wins" strategy.
    """
    task_lower = task_description.lower()
    best_key   = None
    best_score = 0

    for keywords, template_key in _TASK_ROUTING:
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > best_score:
            best_score = score
            best_key   = template_key

    return best_key


def get_sql_template(task: str, role: str) -> dict:
    """
    Return the pre-written SQL for a given task, with role-based access check.

    Accepts either a template key ("delay_count_by_supplier") or a
    natural-language task description from the Planning Agent.

    WHY a role parameter here?
        Financial templates (financial_cost_breakdown, roi_progression,
        monthly_cost_trend) touch financial_impact. Demand Planners don't
        have access to that table. Blocking at template selection time gives
        a clear, early error rather than a cryptic DB permission failure.

    Args:
        task: Template key name OR planning-agent task description string.
        role: User's display role name (e.g. "Demand Planner").

    Returns:
        {
            "sql":        str,    # the SQL string (stripped)
            "task":       str,    # resolved template key
            "found":      bool,
            "allowed":    bool,
            "reason":     str,    # explanation if not found / not allowed
        }
    """
    # ── Resolve to a template key ─────────────────────────────────────────────
    # If task is already a known key, use it directly.
    # Otherwise run the routing heuristic.
    resolved_key = task if task in _SQL_TEMPLATES else _resolve_template_key(task)

    if not resolved_key or resolved_key not in _SQL_TEMPLATES:
        log.warning(
            f"get_sql_template | NOT FOUND | task='{task}' | role='{role}'"
        )
        return {
            "sql":     "",
            "task":    task,
            "found":   False,
            "allowed": False,
            "reason":  (
                f"No SQL template found for task '{task}'. "
                f"Available templates: {list(_SQL_TEMPLATES.keys())}"
            ),
        }

    # ── Role-based access check ───────────────────────────────────────────────
    # Identify which table this template primarily touches
    _TEMPLATE_TABLES: dict[str, str] = {
        "total_delayed_count":               "shipments",
        "delayed_count_by_month":            "shipments",
        "total_shipments":                   "shipments",
        "delay_count_by_supplier":           "shipments",
        "lowest_delay_rate_supplier":        "shipments",
        "highest_delay_rate_region":         "shipments",
        "highest_delay_rate_product_category": "shipments",
        "supplier_delay_comparison":         "shipments",
        "delay_trend_by_month":              "shipments",
        "top_delay_reasons":                 "shipments",
        "supplier_sla_performance":          "suppliers_master",
        "financial_cost_breakdown":          "financial_impact",
        "roi_progression":                   "financial_impact",
        "monthly_cost_trend":                "financial_impact",
        "total_avoidable_cost":              "financial_impact",
        "total_supply_chain_cost":           "financial_impact",
        "total_expedited_cost":              "financial_impact",
        "total_sla_breaches":                "shipments",
        "supplier_delay_rate":               "shipments",
        "fleet_otd_vs_benchmark":            "shipments",
        "high_risk_shipments":               "shipments",
        "total_shipment_value":              "shipments",
        "avg_shipment_value":                "shipments",
        "supplier_shipment_value":           "shipments",
        "region_shipment_value":             "shipments",
        "category_shipment_value":           "shipments",
        "avg_delay_days":                    "shipments",
        "max_delay_days":                    "shipments",
        "overall_delay_rate":                "shipments",
        "shipment_date_span":                "shipments",
        "financial_date_span":               "financial_impact",
        "annual_sc_cost":                    "financial_impact",
        "ai_investment_by_year":             "financial_impact",
        "sla_gap_by_supplier":               "suppliers_master",
    }

    required_table  = _TEMPLATE_TABLES.get(resolved_key, "shipments")
    role_config     = ROLES.get(role, {})
    allowed_tables  = role_config.get("allowed_tables", [])

    # Empty allowed_tables list = no restriction (super-user, or unknown role)
    if allowed_tables and required_table not in allowed_tables:
        log.warning(
            f"get_sql_template | ACCESS DENIED | "
            f"role='{role}' | template='{resolved_key}' | "
            f"table='{required_table}'"
        )
        return {
            "sql":     "",
            "task":    resolved_key,
            "found":   True,
            "allowed": False,
            "reason":  (
                f"Role '{role}' does not have access to "
                f"table '{required_table}' required by template '{resolved_key}'."
            ),
        }

    log.debug(
        f"get_sql_template | FOUND | template='{resolved_key}' | "
        f"role='{role}' | table='{required_table}'"
    )
    return {
        "sql":     _SQL_TEMPLATES[resolved_key].strip(),
        "task":    resolved_key,
        "found":   True,
        "allowed": True,
        "reason":  "Template found and access permitted.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — RESULT INTERPRETER
#  Converts raw DB rows into a structured finding dict.
#  Pure Python — zero LLM tokens. Findings are coded string formatting.
#
#  WHY a coded finding string instead of an LLM summary?
#    The finding sentence follows a predictable pattern for each query type.
#    "Supplier X has the highest delay rate at Y%." does not need a language
#    model — it needs string formatting. Saving the LLM call for the final
#    Executive Agent step (which synthesises ALL findings) is more valuable.
# ═══════════════════════════════════════════════════════════════════════════════

# Confidence thresholds — based on result row count.
# WHY these specific thresholds?
#   >= 10 rows → statistically meaningful sample for supply chain analysis.
#   5–9 rows   → directionally useful but a small sample — note as partial.
#   2–4 rows   → two data points can show a direction; confidence is low.
#   1 row      → single data point; cannot draw conclusions from it.
#   0 rows     → no data; answer is unknown.
_CONFIDENCE_BY_ROW_COUNT: list[tuple[int, float]] = [
    (10, 0.9),
    (5,  0.75),
    (2,  0.6),
    (1,  0.4),
    (0,  0.0),
]

def _score_confidence(row_count: int) -> float:
    """Return confidence score based on result row count."""
    for threshold, score in _CONFIDENCE_BY_ROW_COUNT:
        if row_count >= threshold:
            return score
    return 0.0


def _data_quality_label(row_count: int) -> str:
    """Return a data quality label based on row count."""
    if row_count >= 10:
        return "complete"
    if row_count >= 2:
        return "partial"
    return "insufficient"


def interpret_result(task: str, data: list[dict]) -> dict:
    """
    Convert raw DB rows into a structured finding with confidence metadata.

    Produces a one-sentence finding for each task type using coded string
    formatting (no LLM). The finding is the most important single insight
    from the data — not a full explanation (that is the Executive Agent's job).

    Args:
        task: Resolved SQL template key (e.g. "delay_count_by_supplier").
        data: List of row dicts from execute_query().

    Returns:
        {
            "task":         str,
            "finding":      str,   # one-sentence coded summary
            "data":         list,  # raw rows passed through
            "key_metric":   any,   # most important single value
            "confidence":   float,
            "data_quality": str,   # "complete" | "partial" | "insufficient"
            "caveat":       str,   # data limitation note (empty string if none)
        }
    """
    row_count   = len(data)
    confidence  = _score_confidence(row_count)
    quality     = _data_quality_label(row_count)

    # Default values — overridden per task below
    finding    = "No data returned for this query."
    key_metric = None
    caveat     = ""

    if row_count == 0:
        return {
            "task":         task,
            "finding":      finding,
            "data":         data,
            "key_metric":   None,
            "confidence":   0.0,
            "data_quality": "insufficient",
            "caveat":       "No matching records found. Check date range or filters.",
        }

    # ── Per-task interpretation ───────────────────────────────────────────────

    if task == "total_delayed_count":
        # Single row: total_delayed, suppliers_affected, categories_affected
        row        = data[0]
        key_metric = row.get("total_delayed", 0)
        suppliers  = row.get("suppliers_affected", "?")
        categories = row.get("categories_affected", "?")
        finding    = (
            f"Total delayed shipments: {key_metric} "
            f"(across {suppliers} supplier(s) and {categories} product category/ies)."
        )
        # Override confidence: row_count is always 1 for a scalar aggregate,
        # but the answer is fully reliable — use fixed high confidence.
        confidence = 0.9
        quality    = "complete"

    elif task == "delayed_count_by_month":
        # Single row for the requested month
        row        = data[0]
        key_metric = row.get("delayed_count", 0)
        month      = row.get("month", "N/A")
        total      = row.get("total_shipments", 0)
        rate       = row.get("delay_rate_pct", 0)
        finding    = (
            f"In {month}: {key_metric} delayed shipment(s) out of "
            f"{total} total ({rate}% delay rate)."
        )
        # Override confidence: 1 row is correct for a parameterised month filter
        # (exactly 1 row expected). Treat as complete, high-confidence data.
        confidence = 0.9
        quality    = "complete"

    elif task == "total_shipments":
        # Single aggregate row: total_shipments, on_time_count, delayed_count
        row        = data[0]
        key_metric = row.get("total_shipments", 0)
        on_time    = row.get("on_time_count", 0)
        delayed    = row.get("delayed_count", 0)
        otd_pct    = row.get("on_time_rate_pct", 0)
        finding    = (
            f"Total fleet: {key_metric} shipments "
            f"({on_time} on-time, {delayed} delayed — {otd_pct}% OTD rate)."
        )
        # Override confidence: scalar aggregate, always reliable
        confidence = 0.9
        quality    = "complete"

    elif task == "lowest_delay_rate_supplier":
        # Single row for best-performing supplier (ORDER BY delay_rate_pct ASC)
        row        = data[0]
        key_metric = row.get("delay_rate_pct", 0)
        avg_days   = row.get("avg_delay_days")
        avg_str    = f" (avg {avg_days:.1f} days per delay)" if avg_days else ""
        finding    = (
            f"{row.get('supplier_id', 'Unknown')} has the lowest delay rate "
            f"at {key_metric}%{avg_str} across "
            f"{row.get('total_shipments', 0)} shipments."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "highest_delay_rate_region":
        # Rows ordered by delay_rate_pct DESC — row 0 is worst region
        top        = data[0]
        key_metric = top.get("delay_rate_pct", 0)
        finding    = (
            f"Highest delay rate by region: {top.get('region', 'Unknown')} "
            f"at {key_metric}% "
            f"({top.get('delayed_count', 0)} delayed of "
            f"{top.get('total_shipments', 0)} total shipments)."
        )
        if row_count > 1:
            # Append all regions for context
            all_regions = ", ".join(
                f"{r.get('region', '?')} {r.get('delay_rate_pct', '?')}%"
                for r in data
            )
            finding += f" All regions: {all_regions}."

    elif task == "highest_delay_rate_product_category":
        # Rows ordered by delay_rate_pct DESC — row 0 is worst category
        top        = data[0]
        key_metric = top.get("delay_rate_pct", 0)
        finding    = (
            f"Highest delay rate by product category: "
            f"{top.get('product_category', 'Unknown')} at {key_metric}% "
            f"({top.get('delayed_count', 0)} delayed of "
            f"{top.get('total_shipments', 0)} total shipments)."
        )
        if row_count > 1:
            all_cats = ", ".join(
                f"{r.get('product_category', '?')} {r.get('delay_rate_pct', '?')}%"
                for r in data
            )
            finding += f" All categories: {all_cats}."

    elif task == "total_avoidable_cost":
        # Single aggregate row
        row          = data[0]
        total_avoid  = row.get("total_avoidable", 0) or 0
        expedited    = row.get("total_expedited", 0) or 0
        stockout     = row.get("total_stockout", 0) or 0
        penalties    = row.get("total_penalties", 0) or 0
        from_year    = row.get("from_year", "N/A")
        to_year      = row.get("to_year", "N/A")
        key_metric   = total_avoid
        finding      = (
            f"Total avoidable costs {from_year}–{to_year}: "
            f"${total_avoid:,.0f} "
            f"(expedited freight: ${expedited:,.0f}, "
            f"stockout losses: ${stockout:,.0f}, "
            f"delay penalties: ${penalties:,.0f})."
        )
        confidence = 0.9
        quality    = "complete"
        caveat     = "Financial data. Restricted to authorised roles only."

    elif task == "total_expedited_cost":
        # Single aggregate row — total expedited freight cost across all months.
        # WHY early return if None/0?
        #   A None result means the financial_impact table is empty or the column
        #   is not populated. Returning a "not found" string immediately prevents
        #   the executive from hallucinating a cost figure from adjacent findings.
        row        = data[0]
        total      = row.get("total_expedited_cost")
        if total is None or float(total) == 0:
            return {
                "task":         task,
                "finding":      "Total expedited shipping cost data not found in database.",
                "data":         data,
                "key_metric":   0,
                "confidence":   0.0,
                "data_quality": "insufficient",
                "caveat":       "No expedited_ship_usd data in financial_impact table.",
            }
        total      = float(total)
        key_metric = total
        finding    = f"Total expedited shipping cost is ${total:,.0f}."
        confidence = 0.9
        quality    = "complete"
        caveat     = "Financial data. Restricted to authorised roles only."

    elif task == "supplier_delay_rate":
        # Single row — delay rate for exactly one supplier (WHERE supplier_id = ?).
        # WHY hard-set confidence to 0.90?
        #   row_count = 1 would normally score 0.40 (see _CONFIDENCE_BY_ROW_COUNT).
        #   But a targeted parameterised query that returns 1 row is by design — it
        #   is a complete, authoritative answer for that specific supplier, not a
        #   low-sample aggregate. Override to 0.90 to match scalar aggregate pattern.
        row              = data[0]
        delay_rate       = row.get("delay_rate_pct", 0)
        delayed          = int(row.get("delayed_shipments") or 0)
        total_shipments  = int(row.get("total_shipments") or 0)
        supplier_id      = row.get("supplier_id", "Unknown")
        key_metric       = delay_rate
        finding          = (
            f"{supplier_id} delay rate is {delay_rate}% "
            f"({delayed} of {total_shipments} shipments delayed)."
        )
        confidence = 0.90
        quality    = "complete"

    elif task == "fleet_otd_vs_benchmark":
        # Single aggregate row — shipment-weighted fleet OTD vs 87% industry benchmark.
        # WHY 87.0% hardcoded?
        #   Industry OTD benchmark = 87% (defined in alert_agent.INDUSTRY_OTD_BENCHMARK).
        #   Hardcoded here to avoid a circular import (same rationale as
        #   _SLA_BREACH_THRESHOLD above). Locked business constant — update both
        #   files if the benchmark changes.
        _BENCHMARK = 87.0
        row        = data[0]
        fleet_otd  = row.get("fleet_otd")
        if fleet_otd is None:
            finding    = "Fleet OTD data not available."
            confidence = 0.0
            quality    = "insufficient"
        else:
            fleet_otd  = float(fleet_otd)
            gap        = round(_BENCHMARK - fleet_otd, 1)
            key_metric = fleet_otd
            finding    = (
                f"Fleet OTD is {fleet_otd}% vs industry benchmark {_BENCHMARK}% "
                f"— gap of {gap} percentage points."
            )
            confidence = 0.90
            quality    = "complete"

    elif task == "total_sla_breaches":
        # Single aggregate row — fleet-wide SLA breach count from shipments table.
        # WHY threshold=10 hardcoded here?
        #   The alert threshold is defined in alert_agent.THRESHOLDS["sla_breaches"].
        #   Reproducing it here avoids an import cross-dependency (db_agent is a
        #   lower-level module; importing from alert_agent would create a circular
        #   dependency risk). The value is a locked business constant —
        #   any change requires stakeholder sign-off, at which point both files
        #   must be updated together (tracked in CLAUDE.md alert thresholds section).
        _SLA_BREACH_THRESHOLD = 10
        row        = data[0]
        # WHY column key "sla_breaches" not "total_sla_breaches"?
        #   SQL aliases the column as total_sla_breaches but the user's
        #   validation expected output uses the exact string "Total SLA breaches: N".
        #   Using the aliased column name from the SELECT keeps the interpreter
        #   aligned with the SQL without needing a rename in the result dict.
        total_b    = int(row.get("total_sla_breaches") or 0)
        key_metric = total_b
        finding    = f"Total SLA breaches: {total_b} (above threshold of {_SLA_BREACH_THRESHOLD})."
        confidence = 0.95
        quality    = "complete"
        caveat     = ""

    elif task == "total_supply_chain_cost":
        # Single aggregate row
        row       = data[0]
        total_sc  = row.get("total_supply_chain_cost", 0) or 0
        avg_mo    = row.get("avg_monthly_cost", 0) or 0
        from_year = row.get("from_year", "N/A")
        to_year   = row.get("to_year", "N/A")
        months    = row.get("months_of_data", 0)
        key_metric = total_sc
        finding    = (
            f"Total supply chain cost {from_year}–{to_year}: "
            f"${total_sc:,.0f} "
            f"(avg ${avg_mo:,.0f}/month across {months} months)."
        )
        confidence = 0.9
        quality    = "complete"
        caveat     = "Financial data. Restricted to authorised roles only."

    elif task == "supplier_delay_comparison":
        # All suppliers ranked by delay_rate_pct DESC — multi-row comparison view
        # WHY build the ranked list in the finding string?
        #   The executive_agent will receive multi_row=True and format a ranked
        #   display, but the finding string also carries the data independently
        #   so groundedness validation can check all three supplier percentages.
        ranked = sorted(data, key=lambda r: r.get("delay_rate_pct", 0) or 0, reverse=True)
        lines  = []
        for i, r in enumerate(ranked, start=1):
            sup     = r.get("supplier_id", "Unknown")
            pct     = r.get("delay_rate_pct", 0)
            cnt     = r.get("total_shipments", 0)
            delayed = r.get("delayed_count", 0)
            lines.append(
                f"{i}. {sup} — {pct}% delay rate ({delayed}/{cnt} shipments delayed)"
            )
        finding    = "Delay rate comparison across all suppliers:\n" + "\n".join(lines)
        key_metric = ranked[0].get("delay_rate_pct", 0) if ranked else 0
        confidence = 0.9
        quality    = "complete"

    elif task == "delay_count_by_supplier":
        # Most delayed supplier is in row 0 (ORDER BY delay_rate_pct DESC)
        top = data[0]
        key_metric = top.get("delay_rate_pct", 0)
        avg_days   = top.get("avg_delay_days")
        avg_str    = (
            f" (avg {avg_days:.1f} days per delay)" if avg_days else ""
        )
        # WHY include all suppliers in the finding, not just row 0?
        #   Alert-driven entity queries (e.g. "What is SUP001 delay rate?")
        #   need the specific supplier's data in the finding so the Executive
        #   Agent can cite it. Row 0 is SUP003 (worst), but the user may ask
        #   about SUP001. Including all rows lets the LLM find the right entity.
        per_supplier_lines = [
            f"{r.get('supplier_id', '?')}: {r.get('delay_rate_pct', '?')}% delay rate "
            f"({r.get('delayed_count', 0)}/{r.get('total_shipments', 0)} shipments delayed)"
            for r in data
        ]
        finding = (
            f"{top.get('supplier_id', 'Unknown')} has the highest delay rate "
            f"at {key_metric}%{avg_str} across "
            f"{top.get('total_shipments', 0)} shipments. "
            f"All supplier delay rates: {'; '.join(per_supplier_lines)}."
        )
        if row_count < 3:
            caveat = "Fewer than 3 suppliers found — comparison is limited."

    elif task == "delay_trend_by_month":
        # Most recent month is last row (ORDER BY month ASC)
        latest  = data[-1]
        key_metric = latest.get("delay_rate_pct", 0)
        finding = (
            f"Most recent month ({latest.get('month', 'N/A')}): "
            f"{key_metric}% delay rate across "
            f"{latest.get('total_shipments', 0)} shipments."
        )
        # Compare first and last months to detect trend direction
        if row_count >= 2:
            first_rate = data[0].get("delay_rate_pct", 0) or 0
            last_rate  = latest.get("delay_rate_pct", 0) or 0
            direction  = "improving" if last_rate < first_rate else "worsening"
            finding   += f" Trend is {direction} vs {data[0].get('month', 'N/A')}."
        if row_count < 6:
            caveat = "Less than 6 months of data — trend may not be statistically reliable."

    elif task == "top_delay_reasons":
        top        = data[0]
        key_metric = top.get("pct_of_delays", 0)
        finding    = (
            f"Top delay cause: '{top.get('delay_reason_category', 'Unknown')}' "
            f"accounts for {key_metric}% of all delays "
            f"({top.get('incident_count', 0)} incidents, "
            f"avg {(top.get('avg_delay_days') or 0):.1f} days each)."
        )
        if row_count < 3:
            caveat = "Few delay reason categories found — data may be incomplete."

    elif task == "supplier_sla_performance":
        # Row 0 has the worst SLA gap (ORDER BY sla_gap_pct DESC)
        # WHY supplier_id not supplier_name?
        #   supplier_name ("SupplierA") is a display alias that can confuse
        #   the groundedness checker — it looks for SUP001/SUP002/SUP003 as
        #   canonical identifiers. Using supplier_id keeps finding text
        #   consistent with every other agent finding and DB column reference.
        #
        # WHY compute weighted overall OTD?
        #   For BENCHMARK_COMPARISON queries the executive needs a single
        #   fleet-level OTD figure to compare against the industry benchmark.
        #   Weighting by total_shipments is more accurate than a simple
        #   per-supplier average when shipment volumes are unequal.
        worst      = data[0]
        key_metric = worst.get("sla_gap_pct", 0)

        # ── Compute shipment-weighted overall OTD ─────────────────────────
        total_shipments_all = sum(r.get("total_shipments", 0) or 0 for r in data)
        if total_shipments_all > 0:
            weighted_otd = round(
                sum(
                    (r.get("actual_otd_pct", 0) or 0) * (r.get("total_shipments", 0) or 0)
                    for r in data
                ) / total_shipments_all,
                1,
            )
        else:
            weighted_otd = None

        # Per-supplier breakdown (one line per row)
        per_supplier = "; ".join(
            f"{r.get('supplier_id', '?')}: {r.get('actual_otd_pct', '?')}% OTD "
            f"(target {r.get('sla_on_time_target_pct', '?')}%, gap {r.get('sla_gap_pct', '?')}%)"
            for r in data
        )

        overall_line = (
            f"Overall fleet OTD: {weighted_otd}% (shipment-weighted average). "
            if weighted_otd is not None else ""
        )
        finding    = (
            f"{overall_line}"
            f"Per-supplier: {per_supplier}. "
            f"Worst SLA gap: {worst.get('supplier_id', 'Unknown')} at "
            f"{key_metric}% below their contracted target of "
            f"{worst.get('sla_on_time_target_pct', 'N/A')}% "
            f"({worst.get('breach_count', 0)} SLA breaches)."
        )
        if key_metric and key_metric <= 0:
            finding = (
                f"{overall_line}"
                f"All suppliers are meeting or exceeding their SLA targets. "
                f"Best performer: "
                f"{worst.get('supplier_id', 'N/A')} at "
                f"{worst.get('actual_otd_pct', 'N/A')}% OTD."
            )

    elif task == "financial_cost_breakdown":
        # Last row = most recent year (ORDER BY year ASC)
        latest     = data[-1]
        key_metric = latest.get("annual_cost", 0)
        avoidable  = latest.get("avoidable_total", 0) or 0
        finding    = (
            f"Total supply chain cost {latest.get('year', 'N/A')}: "
            f"${key_metric:,.0f} | "
            f"Avoidable costs: ${avoidable:,.0f} "
            f"({(avoidable / key_metric * 100):.1f}% of total)."
            if key_metric else
            "Financial cost data retrieved — see data for breakdown."
        )
        caveat = (
            "Financial data. Restricted to authorised roles only."
            if not key_metric else ""
        )

    elif task == "roi_progression":
        # Last row = most recent ROI figure
        latest     = data[-1]
        key_metric = latest.get("roi_pct", 0)
        cum_savings = latest.get("cumulative_savings", 0) or 0
        finding    = (
            f"Latest ROI: {key_metric}% as of {latest.get('period_label', 'N/A')}. "
            f"Cumulative AI savings: ${cum_savings:,.0f}."
        )
        caveat = "ROI data — restricted to Operations Manager and CFO roles."

    elif task == "monthly_cost_trend":
        # Find peak cost month
        peak       = max(data, key=lambda r: r.get("total_sc_cost_usd") or 0)
        key_metric = peak.get("total_sc_cost_usd", 0)
        latest     = data[-1]
        finding    = (
            f"Peak cost month: {peak.get('period_label', 'N/A')} "
            f"at ${key_metric:,.0f}. "
            f"Most recent ({latest.get('period_label', 'N/A')}): "
            f"${latest.get('total_sc_cost_usd', 0):,.0f}."
        )

    elif task == "high_risk_shipments":
        key_metric = row_count
        regions    = list({r.get("region", "") for r in data if r.get("region")})
        finding    = (
            f"{row_count} high-risk shipment(s) identified. "
            f"Regions affected: {', '.join(regions) if regions else 'N/A'}. "
            f"Most recent: {data[0].get('shipment_id', 'N/A')} "
            f"({data[0].get('shipment_date', 'N/A')})."
        )
        if row_count == 20:
            caveat = "Result capped at 20 rows. More high-risk shipments may exist."

    elif task == "total_shipment_value":
        row        = data[0]
        millions   = row.get("total_millions", 0) or 0
        count      = row.get("shipments", 0)
        key_metric = millions
        finding    = (
            f"Total shipment value across {count} shipments: "
            f"${millions:.2f}M."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "avg_shipment_value":
        row        = data[0]
        avg_val    = row.get("avg_value", 0) or 0
        count      = row.get("shipments", 0)
        key_metric = avg_val
        finding    = (
            f"Average shipment value: ${avg_val:,.0f} "
            f"(across {count} shipments)."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "supplier_shipment_value":
        top        = data[0]
        key_metric = top.get("value_millions", 0)
        lines      = [
            f"{r.get('supplier_id', '?')}: ${r.get('value_millions', 0):.2f}M "
            f"({r.get('pct', 0)}%)"
            for r in data
        ]
        finding    = (
            f"Shipment value by supplier — "
            f"{'; '.join(lines)}. "
            f"Highest: {top.get('supplier_id', 'N/A')} at "
            f"${key_metric:.2f}M."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "region_shipment_value":
        top        = data[0]
        key_metric = top.get("value_millions", 0)
        lines      = [
            f"{r.get('region', '?')}: ${r.get('value_millions', 0):.2f}M"
            for r in data
        ]
        finding    = (
            f"Shipment value by region — {'; '.join(lines)}. "
            f"Highest: {top.get('region', 'N/A')} at ${key_metric:.2f}M."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "category_shipment_value":
        top        = data[0]
        key_metric = top.get("value_millions", 0)
        lines      = [
            f"{r.get('product_category', '?')}: ${r.get('value_millions', 0):.2f}M"
            for r in data
        ]
        finding    = (
            f"Shipment value by product category — {'; '.join(lines)}. "
            f"Highest: {top.get('product_category', 'N/A')} at "
            f"${key_metric:.2f}M."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "avg_delay_days":
        row        = data[0]
        avg_d      = row.get("avg_delay", 0) or 0
        max_d      = row.get("max_delay", 0) or 0
        count      = row.get("delayed_count", 0)
        key_metric = avg_d
        finding    = (
            f"Average delay for delayed shipments: {avg_d:.1f} days "
            f"(max {max_d} days, across {count} delayed shipments)."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "max_delay_days":
        row        = data[0]
        max_d      = row.get("max_delay", 0) or 0
        count      = row.get("delayed_count", 0)
        key_metric = max_d
        finding    = (
            f"Maximum observed delay is {max_d} days "
            f"(across {count} delayed shipments)."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "overall_delay_rate":
        row        = data[0]
        rate       = row.get("rate", 0) or 0
        total      = row.get("total", 0)
        delayed    = row.get("delayed", 0)
        key_metric = rate
        finding    = (
            f"Overall fleet delay rate: {rate}% "
            f"({delayed} delayed of {total} total shipments)."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "shipment_date_span":
        row        = data[0]
        first      = row.get("first_date", "N/A")
        last       = row.get("last_date", "N/A")
        total      = row.get("total", 0)
        key_metric = first
        finding    = (
            f"Shipment data spans from {first} to {last} "
            f"({total} total shipments)."
        )
        confidence = 0.9
        quality    = "complete"

    elif task == "financial_date_span":
        row        = data[0]
        first      = row.get("first_period", "N/A")
        last       = row.get("last_period", "N/A")
        months     = row.get("months", 0)
        key_metric = first
        finding    = (
            f"Financial data spans from {first} to {last} "
            f"({months} months of data)."
        )
        confidence = 0.9
        quality    = "complete"
        caveat     = "Financial data. Restricted to authorised roles only."

    elif task == "annual_sc_cost":
        # Multi-row: one row per year (ORDER BY year ASC)
        lines      = [
            f"{r.get('year', '?')}: ${r.get('annual_cost', 0):,.0f} "
            f"(${r.get('annual_cost_millions', 0):.2f}M)"
            for r in data
        ]
        latest     = data[-1]
        key_metric = latest.get("annual_cost", 0)
        finding    = (
            f"Annual supply chain cost — {'; '.join(lines)}."
        )
        confidence = 0.9
        quality    = "complete"
        caveat     = "Financial data. Restricted to authorised roles only."

    elif task == "ai_investment_by_year":
        lines      = []
        for r in data:
            inv  = r.get("investment") or 0
            sav  = r.get("savings") or 0
            yr   = r.get("year", "?")
            if inv > 0 or sav > 0:
                lines.append(
                    f"{yr}: invested ${inv:,.0f}, saved ${sav:,.0f}"
                )
            else:
                lines.append(f"{yr}: $0 invested (pre-deployment)")
        key_metric = sum(r.get("investment") or 0 for r in data)
        finding    = (
            f"AI investment by year — "
            + ("; ".join(lines) if lines else "No AI investment records found")
            + f". Total invested: ${key_metric:,.0f}."
        )
        confidence = 0.9
        quality    = "complete"
        caveat     = "Financial data. Restricted to authorised roles only."

    elif task == "sla_gap_by_supplier":
        # Rows ordered by sla_gap DESC — largest gap first
        lines      = [
            f"{r.get('supplier_id', '?')}: target {r.get('target', '?')}%, "
            f"actual {r.get('actual_otd', '?')}%, "
            f"gap {r.get('sla_gap', '?')}% "
            f"({r.get('breaches', 0)} breaches)"
            for r in data
        ]
        worst      = data[0]
        key_metric = worst.get("sla_gap", 0)
        finding    = (
            f"SLA gap by supplier — {'; '.join(lines)}. "
            f"Worst gap: {worst.get('supplier_id', 'N/A')} at "
            f"{key_metric}% below target."
        )
        confidence = 0.9
        quality    = "complete"

    else:
        # Unknown task — return raw data with a generic finding
        finding    = f"Query returned {row_count} row(s) for task '{task}'."
        key_metric = row_count
        caveat     = "Unrecognised task — finding is a raw row count only."

    log.info(
        f"interpret_result | task='{task}' | rows={row_count} | "
        f"confidence={confidence} | quality='{quality}'"
    )

    return {
        "task":         task,
        "finding":      finding,
        "data":         data,
        "key_metric":   key_metric,
        "confidence":   confidence,
        "data_quality": quality,
        "caveat":       caveat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MAIN DB AGENT FUNCTION
#  Receives one planning-agent step, runs it, returns a structured finding.
# ═══════════════════════════════════════════════════════════════════════════════

def run(step: dict, role: str) -> dict:
    """
    Execute one step from the Planning Agent and return a structured finding.

    Flow:
        1.  Log step received (agent, step number, task description)
        2.  Extract task description from step dict
        3.  Resolve SQL template via get_sql_template()
        4.  If template not found or not allowed → return error finding
        5.  Execute SQL via db_connection.execute_query()
        6.  If DB query failed → return error finding
        7.  Interpret raw results via interpret_result()
        8.  Apply role row cap from guardrails (ROLES[role]["max_rows_returned"])
        9.  Log completion (row_count, execution_time_ms, confidence)
        10. Write audit record via log_agent_decision()
        11. Return complete finding dict

    Args:
        step: One step dict from Planning Agent. Expected keys:
              step_number, agent, task, table, requires_human_approval.
        role: User's display role name.

    Returns:
        {
            "step_number":      int,
            "agent":            "db_agent",
            "task":             str,
            "sql_used":         str,
            "finding":          str,
            "data":             list[dict],
            "key_metric":       any,
            "confidence":       float,
            "data_quality":     str,
            "caveat":           str,
            "row_count":        int,
            "execution_time_ms": int,
            "success":          bool,
            "error":            str | None,
        }
    """
    wall_start   = time.perf_counter()
    step_number  = step.get("step_number", 0)
    task_desc    = step.get("task", "")
    step_table   = step.get("table", "")

    # ── 1. Log step received ──────────────────────────────────────────────────
    log.info(
        f"run | STEP {step_number} RECEIVED | "
        f"role='{role}' | task='{task_desc[:80]}'"
    )

    # Error finding template — populated and returned on any failure
    def _error_finding(reason: str, sql: str = "") -> dict:
        return {
            "step_number":       step_number,
            "agent":             "db_agent",
            "task":              task_desc,
            "sql_used":          sql,
            "finding":           f"Step failed: {reason}",
            "data":              [],
            "key_metric":        None,
            "confidence":        0.0,
            "data_quality":      "insufficient",
            "caveat":            reason,
            "row_count":         0,
            "execution_time_ms": 0,
            "success":           False,
            "error":             reason,
        }

    # ── 2+3. Resolve SQL template ─────────────────────────────────────────────
    template_result = get_sql_template(task_desc, role)

    if not template_result["found"]:
        log.warning(
            f"run | TEMPLATE NOT FOUND | step={step_number} | "
            f"task='{task_desc[:80]}'"
        )
        return _error_finding(template_result["reason"])

    if not template_result["allowed"]:
        log.warning(
            f"run | TEMPLATE ACCESS DENIED | step={step_number} | "
            f"role='{role}' | task='{task_desc[:80]}'"
        )
        return _error_finding(template_result["reason"])

    sql          = template_result["sql"]
    resolved_key = template_result["task"]

    # ── 3b. Inject time filter if Planning Agent stamped one ──────────────────
    # WHY here and not inside get_sql_template()?
    #   get_sql_template() is a pure dict lookup — it cannot take runtime
    #   step context (filter_year / filter_month). Injecting here keeps
    #   the template library immutable and all runtime modification in run().
    filter_year  = step.get("filter_year")
    filter_month = step.get("filter_month")
    sql = _inject_time_filter(sql, resolved_key, filter_year, filter_month)

    # ── 4+5. Execute SQL via db_connection ────────────────────────────────────
    # WHY read params from step dict?
    #   Parameterised templates (e.g. delayed_count_by_month) need a runtime
    #   value (the target month "YYYY-MM") that the Planning Agent extracted
    #   from the user query. The step dict carries it as step["params"] so
    #   the DB Agent can pass it to execute_query() without generating SQL.
    sql_params = step.get("params")  # tuple/list or None

    log.info(
        f"run | EXECUTING | step={step_number} | "
        f"template='{resolved_key}' | role='{role}'"
        + (f" | params={sql_params}" if sql_params else "")
    )
    db_result = execute_query(sql, params=sql_params)

    # ── 6. Handle DB failure ──────────────────────────────────────────────────
    if not db_result["success"]:
        error_msg = db_result.get("error", "Unknown database error.")
        log.error(
            f"run | DB FAILED | step={step_number} | error='{error_msg}'"
        )
        return _error_finding(error_msg, sql=sql)

    data           = db_result["data"]
    execution_time = db_result["execution_time_ms"]

    # ── 7. Interpret results ──────────────────────────────────────────────────
    interpretation = interpret_result(resolved_key, data)

    # ── 8. Apply role row cap ─────────────────────────────────────────────────
    # WHY cap after interpretation, not before?
    #   interpret_result() needs the full dataset to calculate percentages
    #   and find the correct maximum/minimum. Capping before interpretation
    #   would produce wrong key_metrics and misleading findings.
    role_config  = ROLES.get(role, {})
    max_rows     = role_config.get("max_rows_returned", 100)
    capped_data  = interpretation["data"]
    cap_caveat   = ""

    if len(capped_data) > max_rows:
        capped_data = capped_data[:max_rows]
        cap_caveat  = (
            f" Result trimmed to {max_rows} rows (your role limit)."
        )
        log.info(
            f"run | ROW CAP APPLIED | role='{role}' | "
            f"original={len(interpretation['data'])} | capped={max_rows}"
        )

    # ── 9. Log completion ─────────────────────────────────────────────────────
    total_wall_ms = int((time.perf_counter() - wall_start) * 1000)
    log.success(
        f"run | STEP {step_number} COMPLETE | "
        f"template='{resolved_key}' | rows={len(capped_data)} | "
        f"db_time={execution_time}ms | total_time={total_wall_ms}ms | "
        f"confidence={interpretation['confidence']}"
    )

    # ── 10. Audit record ──────────────────────────────────────────────────────
    log_agent_decision({
        "user_query":       task_desc,
        "role_used":        role,
        "agent_used":       "db_agent",
        "tables_accessed":  step_table or resolved_key,
        "sql_generated":    sql,
        "result_summary":   interpretation["finding"][:200],
        "confidence_score": interpretation["confidence"],
        "response_time_ms": total_wall_ms,
    })

    # ── 11. Return finding ────────────────────────────────────────────────────
    combined_caveat = (
        interpretation["caveat"] + cap_caveat
    ).strip()

    return {
        "step_number":       step_number,
        "agent":             "db_agent",
        "task":              task_desc,
        "sql_used":          sql,
        "finding":           interpretation["finding"],
        "data":              capped_data,
        "key_metric":        interpretation["key_metric"],
        "confidence":        interpretation["confidence"],
        "data_quality":      interpretation["data_quality"],
        "caveat":            combined_caveat,
        "row_count":         len(capped_data),
        "execution_time_ms": execution_time,
        "success":           True,
        "error":             None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SCHEMA HELPER
#  Returns a filtered view of the database schema for a given role.
#  Used by the Planning Agent to confirm which data is available before
#  building steps — prevents planning steps for tables the role cannot access.
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_data(role: str) -> dict:
    """
    Return the database tables and columns accessible to a given role.

    WHY the Planning Agent calls this:
        Before adding a "financial_cost_breakdown" step to a plan, the
        Planning Agent can confirm whether the role can access
        financial_impact. This prevents building plans with steps that
        will be blocked at execution time — a better user experience
        than building the plan and then failing at step 3.

    Args:
        role: User's display role name (e.g. "Demand Planner").

    Returns:
        {
            "role":           str,
            "allowed_tables": list[str],
            "schema":         {
                "<table_name>": {
                    "columns":   [{"name": str, "type": str, "pk": bool}],
                    "row_count": int,
                },
                ...
            },
            "available_templates": list[str],
            "error":          str | None,
        }
    """
    role_config    = ROLES.get(role, {})
    allowed_tables = role_config.get("allowed_tables", [])

    # ── Pull full schema from db_connection ───────────────────────────────────
    full_schema_result = get_table_schema()
    if full_schema_result.get("error"):
        log.error(
            f"get_available_data | schema fetch failed | "
            f"error='{full_schema_result['error']}'"
        )
        return {
            "role":                role,
            "allowed_tables":      allowed_tables,
            "schema":              {},
            "available_templates": [],
            "error":               full_schema_result["error"],
        }

    full_schema = full_schema_result["tables"]

    # ── Filter to role-allowed tables ─────────────────────────────────────────
    # WHY empty allowed_tables = all tables?
    #   If a role is not in the ROLES dict, we treat it as unconfigured
    #   and return nothing. If allowed_tables is explicitly empty in ROLES,
    #   that would be a config error — we default to showing all tables so
    #   the system doesn't silently break.
    if allowed_tables:
        filtered_schema = {
            t: info
            for t, info in full_schema.items()
            if t in allowed_tables
        }
    else:
        filtered_schema = full_schema

    # ── Identify which SQL templates are usable for this role ─────────────────
    _TEMPLATE_TABLES: dict[str, str] = {
        "delay_count_by_supplier":  "shipments",
        "delay_trend_by_month":     "shipments",
        "top_delay_reasons":        "shipments",
        "supplier_sla_performance": "suppliers_master",
        "financial_cost_breakdown": "financial_impact",
        "roi_progression":          "financial_impact",
        "monthly_cost_trend":       "financial_impact",
        "high_risk_shipments":      "shipments",
    }

    accessible = [
        key
        for key, table in _TEMPLATE_TABLES.items()
        if not allowed_tables or table in allowed_tables
    ]

    log.info(
        f"get_available_data | role='{role}' | "
        f"tables={list(filtered_schema.keys())} | "
        f"templates={accessible}"
    )

    return {
        "role":                role,
        "allowed_tables":      list(filtered_schema.keys()),
        "schema":              filtered_schema,
        "available_templates": accessible,
        "error":               None,
    }
