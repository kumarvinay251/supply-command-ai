"""
Supply Command AI — Data Health Agent
agents/data_health_agent.py

Runs a set of lightweight SQL-based checks against supply_chain.db and
returns a structured health report consumed by the dashboard header badge.

LLM Usage Policy:
    NONE — every check is a deterministic SQL query or Python arithmetic.

WHY a dedicated health agent?
    Data quality problems (missing values, impossible dates, orphaned IDs)
    silently corrupt AI answers. Surfacing them in the dashboard header gives
    the user instant visibility before they ask a question — not after they
    receive a wrong answer. A low health score triggers a caveat in chat
    responses so users know to treat numbers with caution.

Return contract (run_health_checks):
    {
        "health_score":  int,        # 0–100 (100 = no issues)
        "high_count":    int,        # number of HIGH severity issues
        "medium_count":  int,        # number of MEDIUM severity issues
        "issues": [
            {
                "severity":       "HIGH" | "MEDIUM",
                "check":          str,   # short check name
                "finding":        str,   # what was found
                "recommendation": str,   # what to do
            },
            ...
        ]
    }
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from database.db_connection import execute_query
from services.logger        import get_logger

log = get_logger("data_health_agent")


# ── Penalty weights ───────────────────────────────────────────────────────────
# HIGH issues cost 15 points each; MEDIUM cost 5 points.
# Starting from 100, the score floors at 0.
_PENALTY: dict[str, int] = {"HIGH": 15, "MEDIUM": 5}


# ═══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL CHECKS
#  Each returns None (pass) or a dict {severity, check, finding, recommendation}
# ═══════════════════════════════════════════════════════════════════════════════

def _check_null_supplier_ids() -> "dict | None":
    """Shipments with no supplier_id — orphaned rows break JOIN queries."""
    res = execute_query(
        "SELECT COUNT(*) AS cnt FROM shipments WHERE supplier_id IS NULL OR supplier_id = ''"
    )
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "HIGH",
        "check":          "Null supplier_id in shipments",
        "finding":        f"{cnt} shipment(s) have no supplier_id — JOINs to suppliers_master will drop these rows.",
        "recommendation": "Backfill supplier_id or exclude NULL rows from aggregation queries.",
    }


def _check_null_shipment_dates() -> "dict | None":
    """Shipments with no shipment_date — breaks all time-series queries."""
    res = execute_query(
        "SELECT COUNT(*) AS cnt FROM shipments WHERE shipment_date IS NULL OR shipment_date = ''"
    )
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "HIGH",
        "check":          "Null shipment_date in shipments",
        "finding":        f"{cnt} shipment(s) have no shipment_date — trend and month-over-month queries will be incomplete.",
        "recommendation": "Populate shipment_date from source system or remove rows from active analysis.",
    }


def _check_future_shipment_dates() -> "dict | None":
    """Shipments with a date in the future — likely data entry errors."""
    res = execute_query(
        "SELECT COUNT(*) AS cnt FROM shipments WHERE shipment_date > date('now')"
    )
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    severity = "HIGH" if cnt > 5 else "MEDIUM"
    return {
        "severity":       severity,
        "check":          "Future shipment dates",
        "finding":        f"{cnt} shipment(s) have a shipment_date in the future — likely data entry errors.",
        "recommendation": "Review and correct dates; future-dated rows inflate 'active shipment' counts.",
    }


def _check_negative_shipment_values() -> "dict | None":
    """Shipments with negative shipment_value_usd — impossible for goods value."""
    res = execute_query(
        "SELECT COUNT(*) AS cnt FROM shipments WHERE shipment_value_usd < 0"
    )
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "HIGH",
        "check":          "Negative shipment values",
        "finding":        f"{cnt} shipment(s) have negative shipment_value_usd — distorts cost aggregations.",
        "recommendation": "Replace negatives with ABS() or NULL; investigate source system sign convention.",
    }


def _check_orphaned_shipments() -> "dict | None":
    """Shipments whose supplier_id has no matching row in suppliers_master."""
    res = execute_query("""
        SELECT COUNT(*) AS cnt
        FROM shipments sh
        LEFT JOIN suppliers_master s ON sh.supplier_id = s.supplier_id
        WHERE sh.supplier_id IS NOT NULL AND s.supplier_id IS NULL
    """)
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "HIGH",
        "check":          "Orphaned shipments (unknown supplier)",
        "finding":        f"{cnt} shipment(s) reference a supplier_id not in suppliers_master.",
        "recommendation": "Add missing supplier rows or correct supplier_id values in shipments.",
    }


def _check_missing_financial_months() -> "dict | None":
    """Gaps in financial_impact monthly records — breaks ROI trend analysis."""
    res = execute_query("""
        SELECT COUNT(DISTINCT year || '-' || printf('%02d', month)) AS months_present
        FROM financial_impact
    """)
    if not res.get("success") or not res.get("data"):
        return None
    months_present = int(res["data"][0].get("months_present", 0) or 0)
    # Expect at least 24 months (2023–2024) for full trend analysis
    if months_present >= 24:
        return None
    missing = 24 - months_present
    severity = "HIGH" if missing > 6 else "MEDIUM"
    return {
        "severity":       severity,
        "check":          "Incomplete financial_impact history",
        "finding":        f"Only {months_present}/24 expected monthly records present ({missing} gap(s)).",
        "recommendation": "Back-populate missing months or adjust trend queries to available date range.",
    }


def _check_zero_annual_spend() -> "dict | None":
    """Suppliers with zero annual_spend_usd_2024 — breaks ROI apportionment."""
    res = execute_query("""
        SELECT COUNT(*) AS cnt
        FROM suppliers_master
        WHERE annual_spend_usd_2024 IS NULL OR annual_spend_usd_2024 = 0
    """)
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "MEDIUM",
        "check":          "Zero or null annual spend in suppliers_master",
        "finding":        f"{cnt} supplier(s) have no annual_spend_usd_2024 — What-If cost apportionment will be inaccurate.",
        "recommendation": "Populate annual_spend_usd_2024 from procurement system.",
    }


def _check_sla_target_range() -> "dict | None":
    """SLA targets outside 0–100% are impossible percentages."""
    res = execute_query("""
        SELECT COUNT(*) AS cnt
        FROM suppliers_master
        WHERE sla_on_time_target_pct < 0 OR sla_on_time_target_pct > 100
    """)
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "MEDIUM",
        "check":          "Invalid SLA target percentages",
        "finding":        f"{cnt} supplier(s) have sla_on_time_target_pct outside 0–100.",
        "recommendation": "Correct SLA targets; benchmark comparisons will be misleading until fixed.",
    }


def _check_status_vs_delay_days() -> "dict | None":
    """
    Shipments marked OnTime but with delay_days > 0.

    WHY this matters:
        The pipeline counts delayed shipments using status='Delayed'.
        If some delayed rows were coded as 'OnTime' (no space variant) in the
        source system, the reported delay count (15) is understated — the true
        figure (from delay_days) may be significantly higher. This directly
        affects OTD rate, SLA breach count, and every supplier delay ranking.
    """
    res = execute_query(
        "SELECT COUNT(*) AS cnt FROM shipments WHERE status = 'OnTime' AND delay_days > 0"
    )
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "HIGH",
        "check":          "Status vs Delay Days",
        "finding":        (
            f"{cnt} shipment(s) marked OnTime but have delay_days > 0 — "
            f"true delayed count may be higher than reported."
        ),
        "recommendation": "Reconcile status field with delay_days at source before production use.",
    }


def _check_delivery_date_vs_status() -> "dict | None":
    """
    Shipments where actual_delivery_date > expected_delivery_date but status is OnTime.

    WHY this matters:
        OTD rate is calculated from the status field. If rows are marked OnTime
        despite arriving after the expected date, the OTD figure is overstated.
        This can mask supplier performance problems and invalidate SLA reporting.
    """
    res = execute_query("""
        SELECT COUNT(*) AS cnt
        FROM shipments
        WHERE actual_delivery_date > expected_delivery_date
          AND status = 'OnTime'
          AND actual_delivery_date IS NOT NULL
    """)
    if not res.get("success") or not res.get("data"):
        return None
    cnt = int(res["data"][0].get("cnt", 0) or 0)
    if cnt == 0:
        return None
    return {
        "severity":       "HIGH",
        "check":          "Delivery Date vs Status",
        "finding":        (
            f"{cnt} shipment(s) delivered after expected date but marked OnTime — "
            f"OTD rate may be overstated."
        ),
        "recommendation": "Review SLA grace period definition with operations team.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered list of all checks to run — add new checks here to include them.
_CHECKS = [
    _check_null_supplier_ids,
    _check_null_shipment_dates,
    _check_future_shipment_dates,
    _check_negative_shipment_values,
    _check_orphaned_shipments,
    _check_missing_financial_months,
    _check_zero_annual_spend,
    _check_sla_target_range,
    _check_status_vs_delay_days,
    _check_delivery_date_vs_status,
]


def run_health_checks() -> dict:
    """
    Run all registered data quality checks and return a health report.

    Returns:
        {
            "health_score":  int,   # 0–100
            "high_count":    int,
            "medium_count":  int,
            "issues":        list[dict],
        }
    """
    log.info("run_health_checks | START")

    issues: list[dict] = []

    for check_fn in _CHECKS:
        try:
            result = check_fn()
            if result is not None:
                issues.append(result)
                log.warning(
                    f"run_health_checks | {result['severity']} | "
                    f"{result['check']} | {result['finding'][:80]}"
                )
        except Exception as exc:
            log.error(f"run_health_checks | ERROR in {check_fn.__name__} | {exc}")

    high_count   = sum(1 for i in issues if i["severity"] == "HIGH")
    medium_count = sum(1 for i in issues if i["severity"] == "MEDIUM")

    penalty      = high_count * _PENALTY["HIGH"] + medium_count * _PENALTY["MEDIUM"]
    health_score = max(0, 100 - penalty)

    log.info(
        f"run_health_checks | COMPLETE | "
        f"score={health_score} | high={high_count} | medium={medium_count}"
    )

    return {
        "health_score": health_score,
        "high_count":   high_count,
        "medium_count": medium_count,
        "issues":       issues,
    }
