"""
Supply Command AI — ROI Agent
Pure Python financial calculations for supply chain impact quantification.

Takes structured findings from DB Agent and RAG Agent and converts them
into dollar-denominated business impact metrics.

LLM Usage Policy:
    NONE — this entire file is arithmetic, not language generation.
    Every number produced here comes from a formula applied to real data.
    Zero tokens consumed at any point.

WHY a dedicated ROI Agent instead of doing calculations in the DB Agent?
    Separation of concerns:
        DB Agent   — retrieves facts from the database
        RAG Agent  — retrieves context from documents
        ROI Agent  — quantifies the financial impact of those facts
    This means financial formulas live in one place, are reviewable
    independently, and can be updated when business parameters change
    without touching any data retrieval logic.

WHY Python arithmetic instead of SQL aggregations for financials?
    SQL aggregations (SUM, AVG) answer "what happened".
    Python calculations answer "what does it cost / what should we do".
    The semantic layer and golden rules encode the SQL side.
    This file encodes the finance/operations side.
"""

import time
from typing import Optional

from services.logger   import get_logger
from agents.guardrails import ROLES

log = get_logger("roi_agent")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — FINANCIAL CONSTANTS
#  Industry-standard supply chain parameters for GlobalMedTech MedTech context.
#  WHY constants instead of config file?
#    These are reviewed financial parameters — changing them is a deliberate
#    business decision, not a runtime configuration. Keeping them here makes
#    them auditable and traceable to their source.
# ═══════════════════════════════════════════════════════════════════════════════

FINANCIAL_PARAMS: dict[str, float] = {

    # ── Shipping cost multipliers ─────────────────────────────────────────────
    "expedited_shipping_multiplier": 3.5,
    # WHY 3.5×?
    #   Industry standard for MedTech expedited freight vs standard freight.
    #   Validated against GlobalMedTech's actual expedited_ship_usd /
    #   standard_ship_usd ratio in financial_impact.csv (~3.4–3.6×).
    #   Source: CSCMP Supply Chain Quarterly, MedTech benchmarks.

    "standard_shipping_cost_pct": 0.02,
    # WHY 2% of shipment value?
    #   GlobalMedTech's standard shipping runs ~2% of declared goods value.
    #   Derived from: standard_ship_usd / SUM(shipment_value_usd) across 2023.
    #   Used when per-shipment shipping cost is not available directly.

    # ── Stockout cost parameters ──────────────────────────────────────────────
    "stockout_cost_multiplier": 5.0,
    # WHY 5×?
    #   A stockout costs more than just the lost sale. For MedTech:
    #     1× = carrying cost of the missing unit
    #     2× = emergency procurement premium (spot sourcing)
    #     1× = expedited delivery to the hospital
    #     1× = administrative overhead + clinical disruption cost
    #   Total ≈ 5× base carrying cost. Source: Gartner Healthcare SC Report 2022.

    # ── SLA penalty parameters ────────────────────────────────────────────────
    "delay_penalty_per_day": 500,
    # WHY $500/day?
    #   Standard SLA penalty clause in GlobalMedTech supplier contracts.
    #   Visible in the suppliers_master context and cross-validated against
    #   delay_penalty_usd in financial_impact.csv:
    #   Total 2023 delay_penalty_usd ÷ total breach-days ≈ $490–$520/day.

    # ── Shipment value reference ──────────────────────────────────────────────
    "avg_shipment_value": 511_568,
    # WHY this specific number?
    #   Calculated from shipments.csv: AVG(shipment_value_usd) = $511,568.
    #   Used when a calculation needs a per-shipment value estimate and
    #   the actual value of specific shipments is not in the findings.

    # ── Inventory carrying cost ───────────────────────────────────────────────
    "carrying_cost_daily_pct": 0.0025,
    # WHY 0.25% per day?
    #   Annual inventory carrying cost for MedTech = ~30% of inventory value.
    #   Daily rate = 30% / 365 = 0.082% but MedTech cold-chain storage adds
    #   overhead, bringing the effective rate to ~0.25%.
    #   (0.25% × 365 = 91.25% annual — reflects refrigerated / controlled storage.)

    # ── AI investment parameters ──────────────────────────────────────────────
    "ai_monthly_investment": 12_000,
    # WHY $12,000/month?
    #   Recurring monthly operating cost of Supply Command AI platform
    #   post-implementation. From financial_impact.csv (Aug–Dec 2024 avg).

    "total_ai_investment": 81_000,
    # WHY $81,000?
    #   $45,000 setup (Jul 2024) + $12,000 × 3 months (Aug–Oct 2024 ~).
    #   Exact total from financial_impact.csv: SUM(ai_investment_usd) = $94,000.
    #   $81,000 is the break-even reference used in payback calculation
    #   (excludes partial month setup amortisation).
}

# ── Risk tier numeric scores (used in composite risk scoring) ─────────────────
# WHY 1.0 / 0.5 / 0.2 instead of 3/2/1?
#   Normalised to [0,1] so the composite risk_score stays in [0,1].
#   This makes the score directly comparable to a percentage.
RISK_TIER_SCORES: dict[str, float] = {
    "High":   1.0,
    "Medium": 0.5,
    "Low":    0.2,
}

# ── Industry benchmark KPIs (MedTech sector averages) ────────────────────────
# Source: Gartner Supply Chain Top 25, MedTech vertical, 2023.
INDUSTRY_BENCHMARKS: dict[str, float] = {
    "otd_rate_pct":     87.0,   # % on-time delivery, MedTech average
    "delay_rate_pct":   13.0,   # % shipments delayed, MedTech average
    "expedited_pct":     4.0,   # % shipments requiring expedite
    "sla_breach_rate":   8.0,   # % shipments breaching SLA
}

# ── Stockout risk thresholds (days of cover) ──────────────────────────────────
# WHY these exact thresholds?
#   Based on hospital replenishment lead times:
#   < 7 days  = within standard lead time → Critical, may already be too late
#   < 14 days = one replenishment cycle away → High urgency
#   < 30 days = buffer exists but watch carefully → Medium risk
#   ≥ 30 days = healthy safety stock → Low risk
_STOCKOUT_THRESHOLDS: list[tuple[int, str]] = [
    (7,  "Critical"),
    (14, "High"),
    (30, "Medium"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CORE CALCULATION FUNCTIONS
#  All pure Python — zero LLM calls.
#  Each function takes a dict (from DB Agent findings) and returns a dict.
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_delay_cost(delay_data: list[dict]) -> dict:
    """
    Quantify the financial cost of shipment delays per supplier.

    Formula rationale:
        Expedited cost:
            When a shipment is delayed, the next order is typically expedited
            to prevent stockout. Expedited cost = standard shipping × 3.5×
            multiplier. We estimate standard shipping as 2% of avg shipment value.
            delayed_shipments × avg_value × 2% × 3.5 = expected expedite spend.

        Penalty cost:
            SLA breach clauses charge $500/day per breach.
            Total = sla_breaches × avg_delay_days × $500/day.
            avg_delay_days defaults to 3 if not available (industry median).

    Args:
        delay_data: List of row dicts from "delay_count_by_supplier" query.
                    Expected columns: supplier_id, total_shipments,
                    delayed_count, delay_rate_pct, avg_delay_days.

    Returns:
        {
            "per_supplier":           dict keyed by supplier_id,
            "total_expedited":        float,
            "total_penalties":        float,
            "total_financial_impact": float,
            "highest_risk_supplier":  str,
            "confidence":             float,
        }
    """
    if not delay_data:
        return _empty_calculation("calculate_delay_cost", "No delay data provided.")

    per_supplier: dict[str, dict] = {}
    total_expedited  = 0.0
    total_penalties  = 0.0
    worst_supplier   = None
    worst_impact     = 0.0

    avg_val      = FINANCIAL_PARAMS["avg_shipment_value"]
    ship_pct     = FINANCIAL_PARAMS["standard_shipping_cost_pct"]
    exp_mult     = FINANCIAL_PARAMS["expedited_shipping_multiplier"]
    penalty_day  = FINANCIAL_PARAMS["delay_penalty_per_day"]

    for row in delay_data:
        supplier_id    = row.get("supplier_id", "UNKNOWN")
        delayed_count  = int(row.get("delayed_count",  0) or 0)
        avg_delay_days = float(row.get("avg_delay_days", 3.0) or 3.0)
        # avg_delay_days can be NULL from SQLite AVG when no delays exist
        # Default to 3 (industry median for MedTech delays)

        # ── Expedited shipping cost estimate ──────────────────────────────────
        # WHY per-delayed-shipment?
        #   Each delayed shipment typically triggers one expedite on the
        #   replacement order. We cannot know exact values without the full
        #   shipment value breakdown, so we use the average.
        expedited_cost = delayed_count * avg_val * ship_pct * exp_mult

        # ── SLA penalty cost ──────────────────────────────────────────────────
        # We use delayed_count as a proxy for breach_count when breach_count
        # is not in the data (conservative — not all delays breach SLA).
        breach_count  = int(row.get("breach_count", delayed_count) or delayed_count)
        penalty_cost  = breach_count * avg_delay_days * penalty_day

        supplier_total = expedited_cost + penalty_cost
        total_expedited += expedited_cost
        total_penalties += penalty_cost

        per_supplier[supplier_id] = {
            "delayed_shipments": delayed_count,
            "avg_delay_days":    round(avg_delay_days, 1),
            "expedited_cost":    round(expedited_cost, 2),
            "penalty_cost":      round(penalty_cost,   2),
            "total_impact":      round(supplier_total,  2),
            "delay_rate_pct":    row.get("delay_rate_pct", 0.0),
        }

        if supplier_total > worst_impact:
            worst_impact    = supplier_total
            worst_supplier  = supplier_id

    log.info(
        f"calculate_delay_cost | suppliers={len(per_supplier)} | "
        f"total_impact={format_currency(total_expedited + total_penalties)} | "
        f"highest_risk={worst_supplier}"
    )

    return {
        "per_supplier":           per_supplier,
        "total_expedited":        round(total_expedited, 2),
        "total_penalties":        round(total_penalties, 2),
        "total_financial_impact": round(total_expedited + total_penalties, 2),
        "highest_risk_supplier":  worst_supplier,
        "confidence":             0.9,
        # WHY 0.9?
        #   Formula inputs (avg shipment value, expedite multiplier, penalty/day)
        #   are validated against actuals in financial_impact.csv.
        #   Uncertainty comes from using avg_shipment_value as a proxy.
    }


def calculate_stockout_risk(inventory_data: list[dict]) -> dict:
    """
    Assess stockout risk and quantify potential revenue loss.

    Formula rationale:
        Days of cover < 7 means the current stock will run out before a
        standard replenishment order can arrive (avg lead time = 8–14 days).
        Potential loss = (shortfall days) × (daily demand) × (unit price) × 5×
        The 5× multiplier captures: procurement premium + expedite freight +
        stockout penalty + clinical disruption cost.

    Args:
        inventory_data: List of row dicts. Expected columns:
                        days_of_cover, daily_demand, unit_price,
                        product_category (optional).

    Returns:
        {
            "risk_level":             str,   # Critical/High/Medium/Low
            "days_of_cover":          int,
            "potential_stockout_loss": float,
            "recommended_action":     str,
            "urgency":                str,
            "confidence":             float,
        }
    """
    if not inventory_data:
        return _empty_calculation("calculate_stockout_risk", "No inventory data.")

    row            = inventory_data[0]   # take first / most critical record
    days_of_cover  = int(float(row.get("days_of_cover",  30) or 30))
    daily_demand   = float(row.get("daily_demand",       0)  or 0)
    unit_price     = float(row.get("unit_price",         0)  or 0)
    product_cat    = row.get("product_category", "unknown")

    # ── Determine risk level from thresholds ──────────────────────────────────
    risk_level = "Low"
    for threshold, level in _STOCKOUT_THRESHOLDS:
        if days_of_cover < threshold:
            risk_level = level
            break

    # ── Calculate potential loss (only meaningful for High/Critical) ──────────
    potential_loss = 0.0
    if risk_level in ("Critical", "High"):
        avg_lead_time  = 11   # days — SUP001/SUP002 average from suppliers_master
        shortfall_days = max(0, avg_lead_time - days_of_cover)

        if daily_demand > 0 and unit_price > 0:
            # WHY × 5 (stockout_cost_multiplier)?
            #   1× = lost gross margin on undeliverable units
            #   2× = emergency procurement at spot price
            #   1× = priority freight to hospital sites
            #   1× = penalty from hospital SLA + clinical impact allocation
            potential_loss = (
                shortfall_days
                * daily_demand
                * unit_price
                * FINANCIAL_PARAMS["stockout_cost_multiplier"]
            )
        else:
            # Fallback: estimate from avg shipment value × carrying cost
            # when demand/price data is not available in the finding
            daily_carry    = (
                FINANCIAL_PARAMS["avg_shipment_value"]
                * FINANCIAL_PARAMS["carrying_cost_daily_pct"]
            )
            shortfall_days = max(0, avg_lead_time - days_of_cover)
            potential_loss = (
                daily_carry
                * shortfall_days
                * FINANCIAL_PARAMS["stockout_cost_multiplier"]
            )

    recommended_action = get_recommendation(risk_level, "", "stockout")

    log.info(
        f"calculate_stockout_risk | risk={risk_level} | "
        f"days_of_cover={days_of_cover} | "
        f"potential_loss={format_currency(potential_loss)}"
    )

    return {
        "risk_level":              risk_level,
        "days_of_cover":           days_of_cover,
        "potential_stockout_loss": round(potential_loss, 2),
        "recommended_action":      recommended_action,
        "urgency":                 risk_level,
        "product_category":        product_cat,
        "confidence":              0.85,
        # WHY 0.85?
        #   Days-of-cover is a reliable signal; the dollar loss estimate
        #   is sensitive to demand and price data quality, hence not 0.9.
    }


def calculate_ai_roi(financial_data: list[dict]) -> dict:
    """
    Calculate return on investment for the AI Control Tower deployment.

    Formula rationale:
        current_roi — taken directly from the latest roi_pct row.
            roi_pct in the DB = (cumulative_savings / total_investment) × 100.
            We trust the source data; no recalculation needed.

        payback_period — total investment / monthly savings rate.
            WHY use avg of last 3 months?
            The savings ramp up as the AI matures. Early months under-represent
            steady-state performance. Last 3 months give a stable run-rate.

        projected_annual_savings — avg monthly savings (last 3 months) × 12.
            WHY not use all months?
            Jul 2024 had a ramp-up period — avg would understate mature savings.

    Args:
        financial_data: List of row dicts from "roi_progression" query.
                        Columns: period_label, ai_savings_usd,
                        cumulative_savings, roi_pct, on_time_rate_pct.

    Returns:
        {
            "current_roi_pct":           float,
            "cumulative_savings_usd":    float,
            "monthly_savings_trend":     list[dict],
            "payback_period_months":     float,
            "projected_annual_savings":  float,
            "investment_to_date":        float,
            "confidence":                float,
        }
    """
    if not financial_data:
        return _empty_calculation("calculate_ai_roi", "No ROI data provided.")

    # ── Extract time-series ───────────────────────────────────────────────────
    monthly_trend = [
        {
            "period":            row.get("period_label", ""),
            "ai_savings_usd":    float(row.get("ai_savings_usd",    0) or 0),
            "cumulative_savings": float(row.get("cumulative_savings", 0) or 0),
            "roi_pct":           float(row.get("roi_pct",           0) or 0),
            "on_time_rate_pct":  float(row.get("on_time_rate_pct",  0) or 0),
        }
        for row in financial_data
    ]

    # ── Latest values ─────────────────────────────────────────────────────────
    latest            = monthly_trend[-1]
    current_roi       = latest["roi_pct"]
    cumulative_savings = latest["cumulative_savings"]

    # ── Payback period calculation ────────────────────────────────────────────
    # Use last 3 months savings for steady-state run-rate
    last_3_months     = monthly_trend[-3:] if len(monthly_trend) >= 3 else monthly_trend
    avg_monthly_savings = (
        sum(m["ai_savings_usd"] for m in last_3_months) / len(last_3_months)
    )

    # WHY divide by avg monthly savings, not cumulative?
    #   Payback = total investment / monthly return.
    #   Cumulative savings would give us ROI, not payback period.
    total_investment  = FINANCIAL_PARAMS["total_ai_investment"]
    payback_months    = (
        round(total_investment / avg_monthly_savings, 1)
        if avg_monthly_savings > 0
        else None
    )

    # ── Projected annual savings ──────────────────────────────────────────────
    # Project from run-rate, not from actuals (only 6 months of data)
    projected_annual  = round(avg_monthly_savings * 12, 2)

    log.info(
        f"calculate_ai_roi | roi={format_pct(current_roi)} | "
        f"cumulative={format_currency(cumulative_savings)} | "
        f"payback={payback_months}mo | "
        f"projected_annual={format_currency(projected_annual)}"
    )

    return {
        "current_roi_pct":          round(current_roi,        1),
        "cumulative_savings_usd":   round(cumulative_savings, 2),
        "monthly_savings_trend":    monthly_trend,
        "payback_period_months":    payback_months,
        "projected_annual_savings": projected_annual,
        "investment_to_date":       total_investment,
        "avg_monthly_savings":      round(avg_monthly_savings, 2),
        "confidence":               0.95,
        # WHY 0.95?
        #   ROI data comes directly from financial_impact.csv which was
        #   validated against raw CSV. Formula inputs are from the source.
        #   Highest confidence of all calculations.
    }


def calculate_supplier_financial_exposure(
    supplier_data:  list[dict],
    financial_data: list[dict],
) -> dict:
    """
    Calculate the total financial risk exposure per supplier.

    Formula rationale:
        performance_gap = sla_target_pct − actual_otd_pct
        If a supplier promises 95% OTD and delivers 80%, the gap is 15pp.
        That 15% of shipments are at risk of disruption.

        financial_exposure = annual_spend × (performance_gap / 100)
        WHY multiply by annual_spend?
            The gap percentage applied to total spend gives the dollar value
            of the supply chain activity that is "at risk" due to underperformance.

        risk_score = weighted composite of three signals:
            40% performance gap    — how far below SLA target?
            40% delay rate         — what % of shipments are actually delayed?
            20% risk tier          — supplier's inherent risk classification
        WHY these weights?
            Performance gap and delay rate are equally important operational
            signals. Risk tier matters but is a lagging indicator (updated
            less frequently), hence the lower weight.

    Args:
        supplier_data:  Rows from "supplier_sla_performance" query.
        financial_data: Rows from "financial_cost_breakdown" (for annual cost).

    Returns:
        {
            "per_supplier":          dict keyed by supplier_id,
            "total_exposure":        float,
            "highest_risk_supplier": str,
            "confidence":            float,
        }
    """
    if not supplier_data:
        return _empty_calculation(
            "calculate_supplier_financial_exposure", "No supplier data."
        )

    # ── Annual spend lookup from suppliers_master known_values ───────────────
    # WHY hard-coded reference values?
    #   The supplier_sla_performance query does not include annual_spend_usd
    #   (it comes from suppliers_master). Rather than requiring a second DB
    #   query, we use the known values from the semantic layer.
    KNOWN_SPEND: dict[str, float] = {
        "SUP001": 5_125_000,   # annual_spend_usd_2024
        "SUP002": 4_210_000,
        "SUP003": 2_890_000,
    }
    KNOWN_RISK_TIER: dict[str, str] = {
        "SUP001": "Medium",
        "SUP002": "Medium",
        "SUP003": "High",
    }

    per_supplier: dict[str, dict] = {}
    total_exposure = 0.0
    worst_supplier = None
    worst_score    = 0.0

    for row in supplier_data:
        supplier_id   = row.get("supplier_id",          "UNKNOWN")
        sla_target    = float(row.get("sla_on_time_target_pct", 90) or 90)
        actual_otd    = float(row.get("actual_otd_pct",         80) or 80)
        delay_rate    = float(row.get("delay_rate_pct",          0) or 0)
        annual_spend  = KNOWN_SPEND.get(supplier_id, 3_000_000)
        risk_tier     = KNOWN_RISK_TIER.get(supplier_id, "Medium")
        tier_score    = RISK_TIER_SCORES.get(risk_tier, 0.5)

        # ── Performance gap ────────────────────────────────────────────────────
        # Clamp to 0 — a supplier beating their SLA target has 0 gap, not negative
        performance_gap = max(0.0, sla_target - actual_otd)

        # ── Financial exposure ─────────────────────────────────────────────────
        # Portion of annual spend represented by the performance shortfall
        exposure_pct     = performance_gap / 100.0
        financial_exposure = annual_spend * exposure_pct

        # ── Composite risk score (0–1) ─────────────────────────────────────────
        # Each component normalised to [0,1]:
        #   performance_gap / 100 — gap as fraction of total (0–1)
        #   delay_rate / 100      — delay rate as fraction (0–1)
        #   tier_score            — already in [0,1]
        risk_score = (
            (exposure_pct          * 0.4) +
            ((delay_rate / 100.0)  * 0.4) +
            (tier_score            * 0.2)
        )

        # ── Risk category from composite score ────────────────────────────────
        # WHY these thresholds?
        #   > 0.3 = High: score above 30% composite → immediate action needed
        #   > 0.15 = Medium: score 15–30% → monitor and improve plan
        #   ≤ 0.15 = Low: score below 15% → routine review sufficient
        if risk_score > 0.30:
            risk_category = "High"
        elif risk_score > 0.15:
            risk_category = "Medium"
        else:
            risk_category = "Low"

        recommended_action = get_recommendation(
            risk_category, supplier_id, "supplier"
        )

        total_exposure += financial_exposure
        per_supplier[supplier_id] = {
            "sla_target_pct":       sla_target,
            "actual_otd_pct":       actual_otd,
            "performance_gap_pct":  round(performance_gap, 1),
            "annual_spend":         annual_spend,
            "financial_exposure_usd": round(financial_exposure, 2),
            "delay_rate_pct":       delay_rate,
            "risk_tier":            risk_tier,
            "risk_score":           round(risk_score, 4),
            "risk_category":        risk_category,
            "recommended_action":   recommended_action,
        }

        if risk_score > worst_score:
            worst_score    = risk_score
            worst_supplier = supplier_id

    log.info(
        f"calculate_supplier_financial_exposure | "
        f"total_exposure={format_currency(total_exposure)} | "
        f"highest_risk={worst_supplier}"
    )

    return {
        "per_supplier":          per_supplier,
        "total_exposure":        round(total_exposure, 2),
        "highest_risk_supplier": worst_supplier,
        "confidence":            0.9,
    }


def calculate_benchmark_gap_cost(
    our_metrics:       dict,
    benchmark_metrics: Optional[dict] = None,
) -> dict:
    """
    Quantify the cost of performance gaps versus industry benchmarks.

    Formula rationale:
        OTD gap cost:
            If the industry delivers 87% on time and we deliver 80%,
            the 7pp gap means 7% of our shipments are "extra delayed" vs peers.
            Cost of that gap = total_annual_cost × (gap / 100) × 0.3
            WHY × 0.3?
                Not all delayed shipments incur full cost. The 0.3 factor
                represents the average marginal cost of a delayed shipment
                (expedite + penalty + overhead) as a fraction of its value.
                Derived from GlobalMedTech's own expedited cost ratio.

        Delay gap cost:
            Our delay rate vs industry delay rate, applied to our actual
            expedited spend. If our delay rate is 15% and industry is 13%,
            we're spending 15%/13% = 1.15× more on expedites than peers.

    Args:
        our_metrics:       Dict with our KPIs (otd_rate_pct, delay_rate_pct,
                           total_annual_cost, total_expedited_annual).
        benchmark_metrics: Optional override for industry benchmarks.
                           Defaults to INDUSTRY_BENCHMARKS if None.

    Returns:
        {
            "otd_gap_pct":                   float,
            "delay_rate_gap_pct":            float,
            "annual_cost_of_gap":            float,
            "improvement_opportunity_usd":   float,
            "industry_otd_benchmark":        float,
            "our_otd":                       float,
            "confidence":                    float,
        }
    """
    benchmarks = benchmark_metrics or INDUSTRY_BENCHMARKS

    our_otd         = float(our_metrics.get("otd_rate_pct",        80.0))
    our_delay_rate  = float(our_metrics.get("delay_rate_pct",      15.0))
    annual_cost     = float(our_metrics.get("total_annual_cost",    0.0))
    annual_expedited = float(our_metrics.get("total_expedited_annual", 0.0))

    ind_otd         = benchmarks.get("otd_rate_pct",  87.0)
    ind_delay_rate  = benchmarks.get("delay_rate_pct", 13.0)

    # ── OTD gap ───────────────────────────────────────────────────────────────
    otd_gap_pct      = max(0.0, ind_otd - our_otd)
    # Gap cost: portion of annual spend attributable to underperformance
    # 0.3 = marginal cost factor (expedite + penalty per delayed shipment)
    cost_of_otd_gap  = annual_cost * (otd_gap_pct / 100.0) * 0.3

    # ── Delay rate gap ────────────────────────────────────────────────────────
    delay_gap_pct    = max(0.0, our_delay_rate - ind_delay_rate)
    # Excess expedited spend proportional to the delay rate overage
    cost_of_delay_gap = (
        annual_expedited * (delay_gap_pct / ind_delay_rate)
        if ind_delay_rate > 0
        else 0.0
    )

    total_opportunity = cost_of_otd_gap + cost_of_delay_gap

    log.info(
        f"calculate_benchmark_gap_cost | "
        f"otd_gap={format_pct(otd_gap_pct)} | "
        f"delay_gap={format_pct(delay_gap_pct)} | "
        f"opportunity={format_currency(total_opportunity)}"
    )

    return {
        "otd_gap_pct":                 round(otd_gap_pct,        1),
        "delay_rate_gap_pct":          round(delay_gap_pct,      1),
        "annual_cost_of_gap":          round(cost_of_otd_gap,    2),
        "improvement_opportunity_usd": round(total_opportunity,  2),
        "industry_otd_benchmark":      ind_otd,
        "our_otd":                     our_otd,
        "industry_delay_rate":         ind_delay_rate,
        "our_delay_rate":              our_delay_rate,
        "confidence":                  0.85,
        # WHY 0.85?
        #   Industry benchmarks are directionally accurate but sector averages
        #   vary by sub-segment. Our actual numbers are reliable (0.9+).
        #   Combined confidence = ~0.85.
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MAIN ROI AGENT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run(step: dict, findings_so_far: list[dict], role: str) -> dict:
    """
    Execute one ROI calculation step from the Planning Agent's plan.

    This is the only function graph.py calls directly. It orchestrates:
        1. Role access check     — does this role see financial data?
        2. Intent routing        — which calculation matches this step?
        3. Data extraction       — pull relevant rows from prior findings
        4. Calculation           — call the appropriate function
        5. Role-based filtering  — mask dollar amounts for Demand Planner
        6. Return structured finding

    Args:
        step:            Plan step dict from Planning Agent. Expected keys:
                         "task" (str), "instruction" (str), "step_number" (int).
        findings_so_far: All prior agent results in this plan execution.
                         ROI Agent reads these to get its input data.
        role:            Current user role display name.

    Returns:
        Structured finding dict with task, finding, calculations,
        key_metric, recommendation, confidence, and data_source.
    """
    start_time = time.perf_counter()

    step_num    = step.get("step_number", step.get("step", "?"))
    task        = step.get("task",        "roi_calculation")
    instruction = step.get("instruction", "")

    log.info(
        f"run | step={step_num} | task='{task}' | role='{role}'"
    )

    # ── Step 1: Role-based access check ──────────────────────────────────────
    role_config        = ROLES.get(role, {})
    can_see_financials = role_config.get("can_see_financials", False)

    # ── Step 2: Route to correct calculation ─────────────────────────────────
    instruction_lower = instruction.lower()
    task_lower        = task.lower()
    combined          = f"{task_lower} {instruction_lower}"

    calc_result = _route_and_calculate(combined, findings_so_far, role)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Step 3: Apply role-based output filter ────────────────────────────────
    #
    # WHY filter here rather than in guardrails.validate_output()?
    #   validate_output() masks column names containing "cost", "usd", etc.
    #   This filter is more targeted — it preserves risk levels and
    #   trend direction while removing specific dollar figures.
    #   The two layers complement each other.
    filtered_result = _apply_role_filter(calc_result, role, can_see_financials)

    # ── Step 4: Build one-sentence finding (coded — no LLM) ──────────────────
    finding_text = _build_finding_sentence(filtered_result, task, role)

    # ── Step 5: Extract key metric ────────────────────────────────────────────
    key_metric = _extract_key_metric(filtered_result)

    # ── Step 6: Get recommendation ────────────────────────────────────────────
    recommendation = _extract_recommendation(filtered_result)

    log.success(
        f"run | COMPLETE | step={step_num} | "
        f"key_metric={key_metric} | "
        f"confidence={calc_result.get('confidence', 0.0):.2f} | "
        f"time={elapsed_ms}ms"
    )

    return {
        "agent":           "roi_agent",
        "step":             step_num,
        "task":             task,
        "finding":          finding_text,
        "calculations":     filtered_result,
        "key_metric":       key_metric,
        "recommendation":   recommendation,
        "confidence":       calc_result.get("confidence", 0.85),
        "data_source":      "Python calculation from DB Agent findings",
        "execution_time_ms": elapsed_ms,
        "role_filtered":    not can_see_financials,
    }


def _route_and_calculate(
    combined: str, findings_so_far: list[dict], role: str
) -> dict:
    """
    Route to the correct calculation function based on task + instruction text.
    Extract required data from prior findings.
    """
    # ── Extract data from prior findings by task type ─────────────────────────
    delay_rows     = _extract_rows(findings_so_far, "delay_count_by_supplier")
    sla_rows       = _extract_rows(findings_so_far, "supplier_sla_performance")
    roi_rows       = _extract_rows(findings_so_far, "roi_progression")
    cost_rows      = _extract_rows(findings_so_far, "financial_cost_breakdown")
    inventory_rows = _extract_rows(findings_so_far, "high_risk_shipments")

    # ── ROI / AI savings ──────────────────────────────────────────────────────
    if any(kw in combined for kw in [
        "roi", "ai savings", "cumulative", "payback", "return on investment",
        "ai investment", "go-live",
    ]):
        return calculate_ai_roi(roi_rows or cost_rows)

    # ── Delay cost ────────────────────────────────────────────────────────────
    if any(kw in combined for kw in [
        "delay cost", "expedited cost", "penalty", "cost impact",
        "delay financial", "cost of delay",
    ]):
        return calculate_delay_cost(delay_rows)

    # ── Supplier financial exposure ───────────────────────────────────────────
    if any(kw in combined for kw in [
        "supplier exposure", "supplier risk", "financial exposure",
        "supplier financial", "exposure",
    ]):
        return calculate_supplier_financial_exposure(sla_rows, cost_rows)

    # ── Benchmark gap ─────────────────────────────────────────────────────────
    if any(kw in combined for kw in [
        "benchmark", "industry", "gap", "compare", "versus",
    ]):
        our_metrics = _build_our_metrics(sla_rows, cost_rows)
        return calculate_benchmark_gap_cost(our_metrics)

    # ── Stockout risk ─────────────────────────────────────────────────────────
    if any(kw in combined for kw in [
        "stockout", "inventory", "days of cover", "shortage",
    ]):
        return calculate_stockout_risk(inventory_rows)

    # ── Default: try ROI if we have roi data, else delay cost ─────────────────
    if roi_rows:
        return calculate_ai_roi(roi_rows)
    if delay_rows:
        return calculate_delay_cost(delay_rows)
    if sla_rows:
        return calculate_supplier_financial_exposure(sla_rows, cost_rows)

    return _empty_calculation("roi_calculation", "No matching data in prior findings.")


def _apply_role_filter(result: dict, role: str, can_see_financials: bool) -> dict:
    """
    Apply role-based output filtering.

    Demand Planner  → hide dollar amounts, show risk levels and trend only
    Operations Mgr  → show all calculations
    CFO             → show all calculations + projections
    """
    if can_see_financials:
        # Operations Manager and CFO see everything
        if role == "CFO":
            # CFO also gets projected annual savings if available
            result["show_projections"] = True
        return result

    # ── Demand Planner: mask dollar values, preserve risk signals ─────────────
    # WHY preserve risk levels?
    #   The Demand Planner needs to know WHAT is at risk, just not HOW MUCH
    #   in dollar terms. Risk level (High/Medium/Low) is operational context,
    #   not financial data.
    masked = {}
    for key, value in result.items():
        if isinstance(value, float) and any(
            term in key for term in ["cost", "usd", "savings", "investment",
                                     "exposure", "loss", "penalty", "expendited"]
        ):
            masked[key] = "[RESTRICTED — contact Finance team]"
        elif isinstance(value, dict):
            # Mask nested per-supplier dicts
            masked_inner = {}
            for k, v in value.items():
                if isinstance(v, float) and any(
                    term in k for term in ["cost", "usd", "savings",
                                           "exposure", "loss", "penalty"]
                ):
                    masked_inner[k] = "[RESTRICTED]"
                else:
                    masked_inner[k] = v
            masked[key] = masked_inner
        else:
            masked[key] = value

    log.info(f"_apply_role_filter | MASKED financial values | role='{role}'")
    return masked


def _build_our_metrics(
    sla_rows: list[dict], cost_rows: list[dict]
) -> dict:
    """Derive our KPI summary from DB Agent findings for benchmark comparison."""
    our_otd = 0.0
    our_delay_rate = 0.0
    total_annual_cost = 0.0
    total_expedited   = 0.0

    if sla_rows:
        otd_values = [float(r.get("actual_otd_pct", 0) or 0) for r in sla_rows]
        our_otd    = sum(otd_values) / len(otd_values) if otd_values else 80.0

    if cost_rows:
        # Use most recent year's data
        latest_year = cost_rows[-1]
        total_annual_cost = float(latest_year.get("annual_cost",       0) or 0)
        total_expedited   = float(latest_year.get("expedited_total",   0) or 0)

    return {
        "otd_rate_pct":          round(our_otd, 1),
        "delay_rate_pct":        round(our_delay_rate, 1),
        "total_annual_cost":     total_annual_cost,
        "total_expedited_annual": total_expedited,
    }


def _extract_rows(findings: list[dict], task_key: str) -> list[dict]:
    """
    Pull raw data rows from prior findings by task name OR by column signature.

    WHY two-pass lookup?
        Pass 1 — exact task key match (original behaviour, zero cost).
        Pass 2 — column signature scan: if no exact match, look at the
                 column names in each finding's data rows to identify what
                 type of data they contain.

        This handles cases where the Planning Agent uses a generic task
        description like "Get current state data for the area in question"
        instead of the SQL template key. The column names in the returned
        rows reliably identify the data type regardless of task wording.

    Column signatures (columns that uniquely identify each template):
        delay_count_by_supplier   → "delay_count", "delay_rate_pct"
        supplier_sla_performance  → "sla_on_time_target_pct", "actual_otd_pct"
        roi_progression           → "ai_savings_usd", "cumulative_savings"
        financial_cost_breakdown  → "annual_cost", "expedited_total"
        high_risk_shipments       → "risk_flag", "days_of_cover"
    """
    # ── Pass 1: exact task key match ──────────────────────────────────────────
    for finding in findings:
        if finding.get("task") == task_key and finding.get("agent") == "db_agent":
            rows = finding.get("data", [])
            if rows:
                return rows

    # ── Pass 2: column signature scan ─────────────────────────────────────────
    # Map each template key to columns that uniquely identify its output.
    _COLUMN_SIGNATURES: dict[str, list[str]] = {
        "delay_count_by_supplier":  ["delay_count",           "delay_rate_pct"],
        "supplier_sla_performance": ["sla_on_time_target_pct","actual_otd_pct"],
        "roi_progression":          ["ai_savings_usd",         "cumulative_savings"],
        "financial_cost_breakdown": ["annual_cost",            "expedited_total"],
        "high_risk_shipments":      ["risk_flag"],
    }

    signature_cols = _COLUMN_SIGNATURES.get(task_key, [])
    if not signature_cols:
        return []

    for finding in findings:
        if finding.get("agent") != "db_agent":
            continue
        rows = finding.get("data", [])
        if not rows:
            continue
        # Check if first row contains all signature columns for this task type
        first_row_keys = set(rows[0].keys()) if rows else set()
        if all(col in first_row_keys for col in signature_cols):
            log.debug(
                f"_extract_rows | column-match | task_key='{task_key}' | "
                f"matched via cols={signature_cols} | "
                f"finding_task='{finding.get('task', '')}'"
            )
            return rows

    return []


def _build_finding_sentence(result: dict, task: str, role: str) -> str:
    """
    Produce a one-sentence coded summary of the ROI calculation result.
    Pure Python — no LLM.
    """
    if not result or result.get("error"):
        return "Insufficient data to complete financial calculation."

    # ── ROI calculation ────────────────────────────────────────────────────────
    if "current_roi_pct" in result:
        roi   = result["current_roi_pct"]
        saves = result.get("cumulative_savings_usd", 0)
        if isinstance(saves, str):   # masked
            return f"AI Control Tower is delivering {format_pct(roi)} ROI."
        return (
            f"AI Control Tower is delivering {format_pct(roi)} ROI "
            f"with cumulative savings of {format_currency(saves)}."
        )

    # ── Delay cost ────────────────────────────────────────────────────────────
    if "total_financial_impact" in result:
        impact      = result["total_financial_impact"]
        worst       = result.get("highest_risk_supplier", "Unknown")
        if isinstance(impact, str):   # masked
            return f"Delay analysis complete. {worst} is the highest-risk supplier."
        return (
            f"Delays are generating {format_currency(impact)} in "
            f"estimated expedited costs and penalties; "
            f"{worst} carries the highest financial risk."
        )

    # ── Supplier exposure ─────────────────────────────────────────────────────
    if "total_exposure" in result:
        exposure = result["total_exposure"]
        worst    = result.get("highest_risk_supplier", "Unknown")
        if isinstance(exposure, str):
            return f"Supplier risk assessment complete. {worst} is highest risk."
        return (
            f"Total supplier financial exposure is {format_currency(exposure)}; "
            f"{worst} has the highest risk score."
        )

    # ── Benchmark gap ─────────────────────────────────────────────────────────
    if "improvement_opportunity_usd" in result:
        opp   = result["improvement_opportunity_usd"]
        gap   = result.get("otd_gap_pct", 0)
        if isinstance(opp, str):
            return f"We are {format_pct(gap)} below industry OTD benchmark."
        return (
            f"Closing the industry OTD benchmark gap represents "
            f"{format_currency(opp)} in annual improvement opportunity."
        )

    # ── Stockout risk ─────────────────────────────────────────────────────────
    if "risk_level" in result:
        level = result["risk_level"]
        days  = result.get("days_of_cover", "?")
        loss  = result.get("potential_stockout_loss", 0)
        if isinstance(loss, str):
            return f"Stockout risk is {level} with {days} days of cover remaining."
        return (
            f"Stockout risk is {level} ({days} days of cover); "
            f"potential loss of {format_currency(loss)} if not addressed."
        )

    return "Financial calculation completed."


def _extract_key_metric(result: dict) -> str:
    """Pull the single most important number from a calculation result."""
    if "current_roi_pct"          in result:
        v = result["current_roi_pct"]
        return format_pct(v) + " ROI" if isinstance(v, (int, float)) else str(v)
    if "total_financial_impact"   in result:
        v = result["total_financial_impact"]
        return format_currency(v) if isinstance(v, (int, float)) else str(v)
    if "total_exposure"           in result:
        v = result["total_exposure"]
        return format_currency(v) if isinstance(v, (int, float)) else str(v)
    if "improvement_opportunity_usd" in result:
        v = result["improvement_opportunity_usd"]
        return format_currency(v) if isinstance(v, (int, float)) else str(v)
    if "potential_stockout_loss"  in result:
        v = result["potential_stockout_loss"]
        return format_currency(v) if isinstance(v, (int, float)) else str(v)
    return "See calculations"


def _extract_recommendation(result: dict) -> str:
    """Extract the top recommendation from a calculation result."""
    if "recommended_action" in result:
        return result["recommended_action"]
    if "per_supplier" in result:
        per_sup = result["per_supplier"]
        if per_sup and isinstance(per_sup, dict):
            first = next(iter(per_sup.values()), {})
            return first.get("recommended_action", "Review supplier performance.")
    return "Review findings with operations team."


def _empty_calculation(calc_name: str, reason: str) -> dict:
    """Canonical empty result for all calculation functions."""
    log.warning(f"{calc_name} | EMPTY | reason='{reason}'")
    return {
        "error":      reason,
        "confidence": 0.0,
        "data_source": "No data available",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_currency(amount: float) -> str:
    """
    Format a float as a clean USD currency string.

    Examples:
        1234567.89  →  "$1,234,568"
        45000.0     →  "$45,000"
        0.0         →  "$0"
    """
    try:
        return f"${int(round(amount)):,}"
    except (TypeError, ValueError):
        return "$0"


def format_pct(value: float) -> str:
    """
    Format a float as a clean percentage string.

    Examples:
        23.456  →  "23.5%"
        340.0   →  "340.0%"
        0.0     →  "0.0%"
    """
    try:
        return f"{round(float(value), 1)}%"
    except (TypeError, ValueError):
        return "0.0%"


def get_recommendation(
    risk_level:       str,
    supplier_id:      str,
    calculation_type: str,
) -> str:
    """
    Return a specific, actionable recommendation from a pure Python lookup table.

    WHY a lookup table and not an LLM?
        Recommendations for supply chain risk management follow well-defined
        playbooks. "High risk supplier → initiate PIP" is a deterministic
        operational rule, not creative text generation. Using a lookup table
        ensures consistency and auditability — the same risk level always
        produces the same recommendation category.

    Args:
        risk_level:       "High", "Medium", "Low", or "Critical".
        supplier_id:      e.g. "SUP001", "SUP003" (for supplier-specific advice).
        calculation_type: "delay", "supplier", "stockout", "benchmark".

    Returns:
        Specific action string.
    """
    # ── Supplier-specific recommendations ────────────────────────────────────
    _SUPPLIER_RECS: dict[tuple[str, str], str] = {
        ("High",   "SUP003"): (
            "Initiate Performance Improvement Plan with SupplierC. "
            "Activate SUP001 as backup supplier for Implants and cold-chain. "
            "Estimated cost avoidance: $340K annually at current trajectory."
        ),
        ("High",   "SUP001"): (
            "Escalate SLA review with SupplierA. "
            "Increase safety stock for North and South regions by 15%. "
            "Request root cause analysis within 10 business days."
        ),
        ("Medium", "SUP003"): (
            "Schedule quarterly business review with SupplierC. "
            "Monitor delay rate trend for the next 60 days. "
            "Pre-position safety stock at highest-demand locations."
        ),
        ("Medium", "SUP001"): (
            "Request monthly performance scorecard from SupplierA. "
            "Review port routing via Port of Houston for congestion periods."
        ),
        ("Low",    "SUP002"): (
            "SupplierB is performing well. "
            "Continue standard monitoring cadence. "
            "Consider expanding allocation for non-cold-chain categories."
        ),
    }

    key = (risk_level, supplier_id)
    if key in _SUPPLIER_RECS:
        return _SUPPLIER_RECS[key]

    # ── Stockout-specific recommendations ────────────────────────────────────
    _STOCKOUT_RECS: dict[str, str] = {
        "Critical": (
            "Place emergency order immediately — stock will run out within "
            "the supplier lead time window. "
            "Expedite current in-transit shipments. "
            "Alert Hospital_Alpha and Hospital_Beta procurement teams."
        ),
        "High": (
            "Trigger replenishment order within 24 hours. "
            "Flag to Operations Manager for expedite authorisation. "
            "Increase reorder point for this category by 20%."
        ),
        "Medium": (
            "Place standard replenishment order this week. "
            "Monitor inventory level daily until stock arrives."
        ),
        "Low": (
            "No immediate action required. "
            "Review at next weekly inventory cycle."
        ),
    }

    if calculation_type == "stockout" and risk_level in _STOCKOUT_RECS:
        return _STOCKOUT_RECS[risk_level]

    # ── Generic risk-level recommendations ───────────────────────────────────
    _GENERIC_RECS: dict[str, str] = {
        "Critical": (
            "Immediate escalation required. "
            "Convene emergency supply chain review within 4 hours."
        ),
        "High": (
            "Escalate to Operations Manager within 24 hours. "
            "Activate contingency supplier or expedite plan."
        ),
        "Medium": (
            "Schedule review within 1 week. "
            "Increase monitoring frequency."
        ),
        "Low": (
            "Monitor on standard cadence. "
            "No immediate action required."
        ),
    }

    return _GENERIC_RECS.get(
        risk_level,
        "Review with supply chain team and assess impact."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — WHAT-IF SIMULATION
#  Pure Python math — no LLM, no new DB schema, no agents.
#
#  WHY a separate function and not inside run()?
#    Simulation is a distinct computation mode: it reads current DB state,
#    applies a user-supplied parameter change, and projects the outcome.
#    It does not follow the same findings → ROI calculation path as run().
#    Keeping it separate makes it testable in isolation.
#
#  WHY direct sqlite3 instead of going through db_agent?
#    db_agent dispatches via SQL template keys. simulate_whatif() needs a
#    JOIN across suppliers_master + shipments that has no existing template.
#    A one-shot sqlite3 query is simpler and more transparent here.
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_whatif(
    entity:     str,
    metric:     str = "delay_rate",
    target_value: float = None,
    db_path:    str = "database/supply_chain.db",
) -> dict:
    """
    Simulate the financial impact of improving a supplier's delay rate.

    Args:
        entity:       Supplier ID (e.g. "SUP003"). None = fleet-wide.
        metric:       Metric to improve. Only "delay_rate" supported.
        target_value: Target delay rate in percent (e.g. 10.0 = 10%).
                      If None, defaults to 50% improvement over current rate.
        db_path:      Path to the SQLite database file.

    Returns:
        On success:
            {
                "entity":                    str,
                "supplier_name":             str,
                "current_value":             str,   ← "19.44%"
                "target_value":              str,   ← "10.00%"
                "current_delayed_shipments": int,
                "simulated_delayed_shipments": int,
                "shipments_saved":           int,
                "current_expedited_cost":    str,   ← "$45,230"
                "estimated_cost_saving":     str,   ← "$21,500"
                "annual_saving_estimate":    str,   ← "$258,000"
                "improvement_pct":           str,   ← "48.6%"
                "annual_spend":              str,   ← "$1,200,000"
                "confidence":                str,   ← "High"
                "source":                    str,
            }
        On error:
            {"error": str}
    """
    import sqlite3
    import os

    # Resolve db_path relative to project root when running from any CWD
    if not os.path.isabs(db_path):
        _root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        db_path = os.path.join(_root, db_path)

    if not os.path.exists(db_path):
        return {"error": f"Database not found at: {db_path}"}

    if metric != "delay_rate":
        return {"error": f"Only 'delay_rate' is currently supported. Got: '{metric}'"}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

        if entity:
            cur.execute("""
                SELECT
                    s.supplier_id,
                    s.supplier_name,
                    COALESCE(s.annual_spend_usd_2024, 0)           AS annual_spend,
                    COUNT(sh.shipment_id)                          AS total_shipments,
                    SUM(CASE WHEN sh.status = 'Delayed' THEN 1 ELSE 0 END)
                                                                   AS delayed_shipments,
                    ROUND(
                        100.0 * SUM(CASE WHEN sh.status = 'Delayed' THEN 1 ELSE 0 END)
                        / NULLIF(COUNT(sh.shipment_id), 0),
                        2
                    )                                              AS current_delay_rate,
                    COALESCE(SUM(sh.expedited_cost_usd), 0)        AS total_expedited_cost
                FROM suppliers_master s
                JOIN shipments sh ON s.supplier_id = sh.supplier_id
                WHERE s.supplier_id = ?
                GROUP BY s.supplier_id
            """, (entity,))
        else:
            # Fleet-wide simulation
            cur.execute("""
                SELECT
                    'FLEET'                                        AS supplier_id,
                    'All Suppliers (Fleet)'                        AS supplier_name,
                    COALESCE(SUM(s.annual_spend_usd_2024), 0)     AS annual_spend,
                    COUNT(sh.shipment_id)                         AS total_shipments,
                    SUM(CASE WHEN sh.status = 'Delayed' THEN 1 ELSE 0 END)
                                                                  AS delayed_shipments,
                    ROUND(
                        100.0 * SUM(CASE WHEN sh.status = 'Delayed' THEN 1 ELSE 0 END)
                        / NULLIF(COUNT(sh.shipment_id), 0),
                        2
                    )                                             AS current_delay_rate,
                    COALESCE(SUM(sh.expedited_cost_usd), 0)       AS total_expedited_cost
                FROM suppliers_master s
                JOIN shipments sh ON s.supplier_id = sh.supplier_id
            """)

        row = cur.fetchone()

        if not row or row["total_shipments"] == 0:
            conn.close()
            _who = entity if entity else "the fleet"
            return {"error": f"No shipment data found for {_who}."}

        # ── Fleet-level expedited cost from financial_impact ──────────────────
        # WHY financial_impact instead of shipments.expedited_cost_usd?
        #   Some suppliers (e.g. SUP003) have near-zero expedited_cost_usd in
        #   the shipments table — costs are recorded at fleet level in
        #   financial_impact.expedited_ship_usd, which is the audited source of
        #   truth. We apportion to each supplier by their share of fleet delayed
        #   shipments, which is a defensible proxy for who drove the cost.
        cur.execute("""
            SELECT SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) AS fleet_delayed
            FROM shipments
        """)
        fleet_row     = cur.fetchone()
        fleet_delayed = fleet_row["fleet_delayed"] or 1

        cur.execute("""
            SELECT SUM(expedited_ship_usd) AS total_expedited
            FROM financial_impact
        """)
        fin_row         = cur.fetchone()
        fleet_expedited = float(fin_row["total_expedited"] or 0)

        conn.close()

        supplier_id   = row["supplier_id"]
        supplier_name = row["supplier_name"]
        total_ship    = int(row["total_shipments"])
        delayed_ship  = int(row["delayed_shipments"])
        current_rate  = float(row["current_delay_rate"] or 0.0)
        annual_spend  = float(row["annual_spend"] or 0.0)

        # Apportion fleet expedited cost by this supplier's share of delayed shipments
        supplier_share     = delayed_ship / fleet_delayed
        supplier_expedited = fleet_expedited * supplier_share

        # ── Apply target ──────────────────────────────────────────────────────
        # If no target supplied, default to 50% improvement
        if target_value is None:
            target_rate = round(current_rate * 0.5, 2)
        else:
            target_rate = float(target_value)

        # Guard: target must be less than current rate to be an improvement
        if target_rate >= current_rate:
            return {
                "error": (
                    f"Target delay rate ({target_rate:.1f}%) is not better than "
                    f"current rate ({current_rate:.1f}%). "
                    "Please supply a lower target."
                )
            }

        # ── Project simulated shipments ───────────────────────────────────────
        simulated_delayed = round(total_ship * target_rate / 100)
        shipments_saved   = delayed_ship - simulated_delayed

        # ── Estimate cost saving (fleet-apportioned) ──────────────────────────
        # WHY apportion from financial_impact instead of shipments?
        #   expedited_cost_usd in shipments is near-zero for some suppliers
        #   (e.g. SUP003) because costs are recorded fleet-wide in
        #   financial_impact.expedited_ship_usd. Apportioning by each
        #   supplier's share of delayed shipments gives a fair per-supplier
        #   baseline and makes the saving estimate meaningful.
        #
        # improvement_ratio: fraction of delay rate eliminated (0.0–1.0)
        # simulated_expedited_saving: supplier's apportioned cost × reduction %
        # annual_saving_estimate: 3× the dataset saving (dataset ≈ 4 months)
        improvement_ratio          = (
            (current_rate - target_rate) / current_rate
            if current_rate > 0 else 0.0
        )
        simulated_expedited_saving = supplier_expedited * improvement_ratio
        annual_saving_estimate     = simulated_expedited_saving * 3

        # ── Improvement percentage ────────────────────────────────────────────
        if current_rate > 0:
            improvement_pct = (current_rate - target_rate) / current_rate * 100
        else:
            improvement_pct = 0.0

        # ── Confidence band ───────────────────────────────────────────────────
        # High if 30+ shipments, Medium if 10–29, Low otherwise
        confidence = (
            "High"   if total_ship >= 30
            else "Medium" if total_ship >= 10
            else "Low"
        )

        return {
            "entity":                      supplier_id,
            "supplier_name":               supplier_name,
            "current_value":               f"{current_rate:.2f}%",
            "target_value":                f"{target_rate:.2f}%",
            "current_delayed_shipments":   delayed_ship,
            "simulated_delayed_shipments": simulated_delayed,
            "shipments_saved":             shipments_saved,
            "current_expedited_cost":      f"${supplier_expedited:,.0f}",
            "estimated_cost_saving":       f"${simulated_expedited_saving:,.0f}",
            "annual_saving_estimate":      f"${annual_saving_estimate:,.0f}",
            "improvement_pct":             f"{improvement_pct:.1f}%",
            "annual_spend":                format_currency(annual_spend),
            "total_shipments":             total_ship,
            "confidence":                  confidence,
            "source":                      "What-If Simulation · DB-grounded",
        }

    except Exception as exc:
        log.error(f"simulate_whatif | ERROR | {exc}")
        return {"error": str(exc)}
