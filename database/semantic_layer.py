"""
Supply Command AI — Semantic Layer
Complete data dictionary and business context for the system.

WHY this file exists:
    LLMs generate SQL by pattern-matching on column names they THINK exist.
    Without a semantic layer they hallucinate column names, use wrong
    allowed values, ignore NULL rules, and produce subtly wrong aggregations.

    This file is the single source of truth handed to the LLM BEFORE it
    writes any SQL. It tells the LLM:
        • exactly which columns exist and their types
        • allowed values for every categorical column
        • business rules (e.g. delay_days is always 0 when OnTime)
        • which joins are safe vs dangerous
        • golden SQL rules that must never be broken

LLM Usage:
    NONE — this file is pure Python data structures.
    It is READ by other modules and injected into LLM prompts.
    It never calls an LLM itself.

Usage:
    from database.semantic_layer import build_llm_context, get_golden_rules_text

    # Before SQL generation:
    context = build_llm_context(["shipments", "suppliers_master"])
    prompt  = f"{context}\\n\\nQuestion: {user_query}"
"""

from services.logger import get_logger

log = get_logger("planning_agent")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TABLE DEFINITIONS
#  Full data dictionary for every table in supply_chain.db.
#  WHY so detailed? The more context the LLM has, the less it guesses.
#  Every allowed_values list eliminates an entire class of hallucination.
# ═══════════════════════════════════════════════════════════════════════════════

SEMANTIC_LAYER: dict[str, dict] = {

    # ── TABLE: shipments ──────────────────────────────────────────────────────
    "shipments": {
        "description": (
            "Core transactional table. One row per shipment dispatched "
            "by a GlobalMedTech supplier. Covers Jan 2023 – Dec 2024. "
            "The primary source for delay analysis, SLA performance, "
            "expedite tracking, and AI intervention measurement."
        ),
        "row_count":   100,
        "time_range":  "2023-01-01 to 2024-12-31",
        "grain":       "one row per shipment",

        "columns": {

            "shipment_id": {
                "type":        "TEXT",
                "description": "Unique shipment identifier.",
                "example":     "SHP0001",
                "never_null":  True,
                "primary_key": True,
            },

            "shipment_date": {
                "type":       "DATE",
                "format":     "YYYY-MM-DD",
                "range":      "2023-01-01 to 2024-12-31",
                "never_null": True,
                "sql_note":   (
                    "Use strftime('%Y', shipment_date) for year filtering. "
                    "SQLite does NOT support YEAR() function."
                ),
            },

            "expected_delivery_date": {
                "type":   "DATE",
                "format": "YYYY-MM-DD",
            },

            "actual_delivery_date": {
                "type":        "DATE",
                "format":      "YYYY-MM-DD",
                "business_rule": "NULL for In_Transit and Cancelled shipments.",
            },

            "supplier_id": {
                "type":           "TEXT",
                "allowed_values": ["SUP001", "SUP002", "SUP003"],
                "joins_to":       "suppliers_master.supplier_id",
                "never_null":     True,
                "sql_note":       (
                    "ALWAYS use supplier_id in WHERE clauses. "
                    "JOIN to suppliers_master to get supplier_name."
                ),
            },

            "product_id": {
                "type":    "TEXT",
                "example": "PRD011",
            },

            "location_id": {
                "type":    "TEXT",
                "example": "LOC002",
            },

            "region": {
                "type":           "TEXT",
                "allowed_values": ["North", "South", "West"],
                "never_null":     True,
            },

            "product_category": {
                "type":           "TEXT",
                "allowed_values": [
                    "Surgical Instruments",
                    "Diagnostic Equipment",
                    "Consumables",
                    "Implants",
                    "PPE",
                ],
                "never_null": True,
            },

            "quantity_units": {
                "type": "INTEGER",
            },

            "shipment_value_usd": {
                "type":        "REAL",
                "description": "Total declared value of the shipment in USD.",
            },

            "status": {
                "type":           "TEXT",
                "allowed_values": ["OnTime", "Delayed", "Cancelled", "In_Transit"],
                "never_null":     True,
                "business_rule":  (
                    "delay_days is ALWAYS 0 when status = 'OnTime'. "
                    "delay_reason_category is ALWAYS NULL when status = 'OnTime'."
                ),
            },

            "delay_days": {
                "type":          "INTEGER",
                "description":   "Number of calendar days beyond expected delivery.",
                "range":         "0 to 12",
                "business_rule": (
                    "ALWAYS 0 when status = 'OnTime' or 'Cancelled'. "
                    "Only meaningful when status = 'Delayed'."
                ),
            },

            "delay_reason": {
                "type":          "TEXT",
                "description":   "Free-text description of delay cause.",
                "business_rule": "NULL when status = 'OnTime'.",
            },

            "delay_reason_category": {
                "type":           "TEXT",
                "allowed_values": [
                    "Supplier_Dispatch",
                    "Port_Congestion",
                    "Customs_Hold",
                    "Weather",
                    "Transport_Breakdown",
                    "Documentation_Error",
                ],
                "business_rule": (
                    "NULL when status = 'OnTime'. "
                    "ALWAYS filter WHERE status = 'Delayed' before "
                    "grouping by delay_reason_category to avoid NULL rows."
                ),
            },

            "inventory_level_at_dispatch": {
                "type":           "TEXT",
                "allowed_values": ["Low", "Medium", "High"],
            },

            "weather_condition": {
                "type":           "TEXT",
                "allowed_values": ["Clear", "Rain", "Snow", "Storm"],
            },

            "port_condition": {
                "type":           "TEXT",
                "allowed_values": ["Clear", "Congested", "Closed"],
            },

            "customs_status": {
                "type":           "TEXT",
                "allowed_values": ["Cleared", "Held", "Pending"],
            },

            "sla_breach": {
                "type":           "TEXT",
                "allowed_values": ["Yes", "No"],
                "never_null":     True,
                "business_rule":  (
                    "'Yes' when actual delivery exceeded the supplier's "
                    "contracted SLA target. Calculated at load time."
                ),
            },

            "expedited_flag": {
                "type":           "TEXT",
                "allowed_values": ["Yes", "No"],
                "never_null":     True,
            },

            "expedited_cost_usd": {
                "type":          "REAL",
                "business_rule": (
                    "ALWAYS 0.0 when expedited_flag = 'No'. "
                    "Only non-zero when expedited_flag = 'Yes'."
                ),
            },

            "risk_flag": {
                "type":           "TEXT",
                "allowed_values": ["High", "Medium", "Low"],
                "never_null":     True,
            },

            "ai_intervention_applied": {
                "type":           "TEXT",
                "allowed_values": ["Yes", "No"],
                "never_null":     True,
                "business_rule":  (
                    "ALWAYS 'No' for shipment_date < '2024-07-01'. "
                    "ALWAYS 'Yes' for shipment_date >= '2024-07-01'. "
                    "WHY: AI Control Tower went live on 2024-07-01. "
                    "Do NOT use this column to filter pre/post AI — "
                    "use the date boundary instead."
                ),
            },

            "root_cause_category": {
                "type":          "TEXT",
                "business_rule": "NULL when no investigation was performed.",
            },

            "recommended_action": {
                "type":          "TEXT",
                "description":   "AI or analyst recommended remediation step.",
                "business_rule": "NULL when no recommendation was generated.",
            },

            "recovery_status": {
                "type":           "TEXT",
                "allowed_values": ["Resolved", "InProgress", "Escalated", "NA"],
            },
        },
    },

    # ── TABLE: financial_impact ───────────────────────────────────────────────
    "financial_impact": {
        "description": (
            "Monthly financial summary for GlobalMedTech supply chain. "
            "Covers Jan 2022 – Dec 2024 (36 months). "
            "Used for cost trend analysis, ROI measurement, and AI savings "
            "quantification. Contains both actuals and AI-period data."
        ),
        "row_count":  36,
        "time_range": "2022-01-01 to 2024-12-31",
        "grain":      "one row per calendar month",

        "columns": {

            "period_id": {
                "type":        "TEXT",
                "description": "Surrogate key.",
                "example":     "FIN-2023-07",
                "primary_key": True,
            },

            "period_label": {
                "type":        "TEXT",
                "format":      "Mon-YYYY",
                "example":     "Jan-2022",
                "description": "Human-readable month label. Use for display.",
            },

            "month": {
                "type":  "INTEGER",
                "range": "1 to 12",
            },

            "year": {
                "type":           "INTEGER",
                "allowed_values": [2022, 2023, 2024],
            },

            "quarter": {
                "type":    "TEXT",
                "example": "Q1-2023",
            },

            "total_sc_cost_usd": {
                "type":        "REAL",
                "description": "Total monthly supply chain cost (all components).",
                "range":       "$385,000 to $528,000",
                "sql_note":    (
                    "Use SUM(total_sc_cost_usd) for annual totals. "
                    "Use GROUP BY year, ORDER BY year, month for trends."
                ),
            },

            "standard_ship_usd": {
                "type":        "REAL",
                "description": "Standard (non-expedited) shipping spend.",
            },

            "expedited_ship_usd": {
                "type":          "REAL",
                "description":   "Expedited / emergency shipping premium.",
                "business_rule": (
                    "Spikes during disruption months "
                    "(e.g. Red Sea crisis, port congestion). "
                    "Check the 'notes' column for context on spikes."
                ),
            },

            "freight_cost_usd": {
                "type":        "REAL",
                "description": "Total freight (standard + expedited).",
            },

            "insurance_cost_usd": {
                "type":        "REAL",
                "description": "Cargo insurance premiums.",
            },

            "stockout_loss_usd": {
                "type":        "REAL",
                "description": "Revenue lost due to product stockouts.",
            },

            "excess_inv_cost_usd": {
                "type":        "REAL",
                "description": "Cost of carrying excess safety stock.",
            },

            "delay_penalty_usd": {
                "type":        "REAL",
                "description": "SLA breach penalty charges paid to customers.",
            },

            "quality_reject_usd": {
                "type":        "REAL",
                "description": "Cost of quality failures and returns.",
            },

            "total_avoidable_usd": {
                "type":          "REAL",
                "description":   "Total cost that could have been prevented.",
                "business_rule": (
                    "Equals: expedited_ship_usd + stockout_loss_usd + "
                    "excess_inv_cost_usd + delay_penalty_usd + quality_reject_usd. "
                    "This is the primary metric for AI savings potential."
                ),
            },

            "ai_investment_usd": {
                "type":          "REAL",
                "description":   "Monthly spend on AI Control Tower platform.",
                "business_rule": (
                    "ALWAYS 0 before July 2024. "
                    "Includes $45,000 setup cost in July 2024, "
                    "then ~$7,000–$12,000/month ongoing."
                ),
            },

            "ai_savings_usd": {
                "type":          "REAL",
                "description":   "Documented cost savings attributed to AI.",
                "business_rule": (
                    "ALWAYS 0 before July 2024. "
                    "Grows each month as AI matures. "
                    "Reached $110,000 by December 2024."
                ),
            },

            "cumulative_savings": {
                "type":          "REAL",
                "description":   "Running total of ai_savings_usd from July 2024.",
                "business_rule": (
                    "ALWAYS 0 before July 2024. "
                    "Reaches $401,000 by December 2024. "
                    "Use MAX(cumulative_savings) to get total AI savings — "
                    "NEVER SUM(cumulative_savings) as it double-counts."
                ),
            },

            "roi_pct": {
                "type":            "REAL",
                "description":     "Monthly ROI percentage for AI investment.",
                "business_rule":   (
                    "ALWAYS 0 before July 2024. "
                    "NEVER SUM(roi_pct) — it is not additive across months. "
                    "Use the value for individual months or the latest month."
                ),
                "range_post_ai":   "40 to 340",
            },

            "on_time_rate_pct": {
                "type":             "REAL",
                "description":      "Monthly OTD (On-Time Delivery) rate.",
                "range_2023":       "78 to 84",
                "range_2024_h2":    "84 to 93",
                "sql_note":         (
                    "Use AVG(on_time_rate_pct) for period averages. "
                    "Do not SUM — it is a percentage, not a count."
                ),
            },

            "perfect_order_pct": {
                "type":        "REAL",
                "description": "Percentage of orders with zero defects/issues.",
            },

            "inventory_turnover": {
                "type":        "REAL",
                "description": "Inventory turns per month.",
            },

            "geopolitical_cost": {
                "type":          "REAL",
                "description":   "Extra costs from geopolitical disruptions.",
                "business_rule": (
                    "0 when no event. Non-zero during Red Sea crisis, "
                    "Russia-Ukraine war impact, and similar events. "
                    "See 'notes' column for event labels."
                ),
            },

            "notes": {
                "type":        "TEXT",
                "description": "Business event tag for this month.",
                "examples":    [
                    "Baseline_2022",
                    "Russia_Ukraine_War_Begins",
                    "Shanghai_COVID_Lockdown",
                    "Red_Sea_Houthi_Crisis",
                    "Port_LA_Congestion_Peak",
                    "AI_Control_Tower_GoLive",
                ],
                "sql_note": (
                    "Useful for contextualising cost spikes. "
                    "Include in SELECT when explaining anomalies."
                ),
            },
        },
    },

    # ── TABLE: suppliers_master ───────────────────────────────────────────────
    "suppliers_master": {
        "description": (
            "Reference / dimension table. One row per active supplier. "
            "JOIN to shipments on supplier_id to enrich transactional data "
            "with supplier name, SLA targets, and risk classification."
        ),
        "row_count": 3,
        "grain":     "one row per supplier",

        "columns": {

            "supplier_id": {
                "type":           "TEXT",
                "allowed_values": ["SUP001", "SUP002", "SUP003"],
                "primary_key":    True,
                "never_null":     True,
            },

            "supplier_name": {
                "type":           "TEXT",
                "allowed_values": ["SupplierA", "SupplierB", "SupplierC"],
                "never_null":     True,
                "sql_note":       (
                    "NEVER hardcode supplier names in WHERE clauses. "
                    "Always filter on supplier_id and JOIN for the name."
                ),
            },

            "supplier_type": {
                "type":        "TEXT",
                "description": "Specialisation category of the supplier.",
            },

            "country": {
                "type":    "TEXT",
                "example": "USA",
            },

            "region_served": {
                "type":    "TEXT",
                "example": "North|South",
            },

            "primary_port": {
                "type":    "TEXT",
                "example": "Port of Houston",
            },

            "risk_tier": {
                "type":           "TEXT",
                "allowed_values": ["Low", "Medium", "High"],
                "known_values": {
                    "SUP001": "Medium",
                    "SUP002": "Medium",
                    "SUP003": "High",       # elevated — highest delay rate (20%)
                },
            },

            "contract_start_date": {
                "type":   "DATE",
                "format": "YYYY-MM-DD",
            },

            "sla_on_time_target_pct": {
                "type":         "REAL",
                "description":  "Contracted minimum on-time delivery percentage.",
                "known_values": {
                    "SUP001": 92.0,
                    "SUP002": 95.0,
                    "SUP003": 88.0,
                },
                "sql_note": (
                    "Compare against actual OTD from shipments table to "
                    "calculate SLA gap."
                ),
            },

            "avg_lead_time_days": {
                "type":         "INTEGER",
                "description":  "Contracted average lead time in calendar days.",
                "known_values": {
                    "SUP001": 11,
                    "SUP002": 8,
                    "SUP003": 14,
                },
            },

            "payment_terms_days": {
                "type":        "INTEGER",
                "description": "Payment terms agreed in contract.",
            },

            "cold_chain_capability": {
                "type":           "TEXT",
                "allowed_values": ["Yes", "No"],
                "known_values": {
                    "SUP001": "Yes",
                    "SUP002": "No",
                    "SUP003": "Yes",
                },
                "business_rule": (
                    "Only SUP001 and SUP003 can handle cold-chain products "
                    "(e.g. certain Implants and Diagnostic Equipment). "
                    "SUP002 must NOT be used for cold-chain shipments."
                ),
            },

            "backup_supplier_id": {
                "type":          "TEXT",
                "business_rule": (
                    "No supplier backs up itself. "
                    "Use this column to identify failover options."
                ),
            },

            "annual_spend_usd_2023": {
                "type":        "REAL",
                "description": "Total spend with this supplier in 2023.",
                "known_values": {
                    "SUP001": 4_850_000,
                    "SUP002": 3_980_000,
                    "SUP003": 2_640_000,
                },
            },

            "annual_spend_usd_2024": {
                "type":        "REAL",
                "description": "Total spend with this supplier in 2024.",
                "known_values": {
                    "SUP001": 5_125_000,
                    "SUP002": 4_210_000,
                    "SUP003": 2_890_000,
                },
            },

            "preferred_carrier": {
                "type":    "TEXT",
                "example": "FedEx Freight",
            },

            "quality_certification": {
                "type":    "TEXT",
                "example": "ISO 13485; FDA Registered",
            },

            "account_manager_name": {
                "type":    "TEXT",
                "example": "Olivia Carter",
            },
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — RELATIONSHIP MAP
#  Defines which JOINs are safe and which are dangerous.
#  WHY? LLMs frequently try to JOIN unrelated tables on mismatched keys,
#  producing silently wrong results. This map is checked before SQL generation.
# ═══════════════════════════════════════════════════════════════════════════════

RELATIONSHIPS: list[dict] = [
    {
        "from_table":   "shipments",
        "from_column":  "supplier_id",
        "to_table":     "suppliers_master",
        "to_column":    "supplier_id",
        "join_type":    "INNER JOIN",
        "safe":         True,
        "description":  (
            "Standard dimension join. Use to enrich shipment rows with "
            "supplier_name, SLA targets, and risk tier."
        ),
        "example_sql":  (
            "SELECT s.supplier_name, COUNT(*) AS total_shipments "
            "FROM shipments sh "
            "JOIN suppliers_master s ON sh.supplier_id = s.supplier_id "
            "GROUP BY s.supplier_name"
        ),
    },
    {
        "from_table":  "financial_impact",
        "to_table":    "shipments",
        "join_type":   "NONE — no direct foreign key",
        "safe":        False,
        "note":        (
            "These tables share year and month columns but have no "
            "direct FK relationship. Link via time period only "
            "(year, strftime('%Y', shipment_date))."
        ),
        "warning":     (
            "NEVER JOIN financial_impact to shipments directly. "
            "Run separate queries and combine results in Python. "
            "A direct JOIN will produce a cartesian product."
        ),
    },
    {
        "from_table":  "financial_impact",
        "to_table":    "suppliers_master",
        "join_type":   "NONE — no direct foreign key",
        "safe":        False,
        "note":        "No shared key. Query each table independently.",
        "warning":     "NEVER JOIN financial_impact to suppliers_master.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — GOLDEN RULES FOR SQL
#  These rules must be injected into every LLM prompt before SQL generation.
#  Each rule exists because a naive LLM would get it wrong without guidance.
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_RULES: list[dict] = [
    {
        "rule_id":     "RULE_01",
        "title":       "Delay Rate Calculation",
        "description": (
            "ALWAYS calculate delay rate using conditional SUM, not COUNT filter."
        ),
        "correct_sql": (
            "SUM(CASE WHEN status = 'Delayed' THEN 1 ELSE 0 END) "
            "* 100.0 / COUNT(*)"
        ),
        "wrong_sql":   "COUNT(status = 'Delayed') / COUNT(*)",
        "why":         (
            "COUNT() in SQLite ignores the condition inside it — it counts "
            "all non-NULL rows regardless of the boolean expression. "
            "This produces 100% for everything."
        ),
    },
    {
        "rule_id":     "RULE_02",
        "title":       "Pre-AI vs Post-AI Period Split",
        "description": (
            "Use date boundaries for pre/post AI comparison. "
            "Never filter on the ai_intervention_applied column directly."
        ),
        "correct_sql": (
            "Pre-AI:  WHERE shipment_date < '2024-07-01'\\n"
            "Post-AI: WHERE shipment_date >= '2024-07-01'"
        ),
        "wrong_sql":   "WHERE ai_intervention_applied = 'Yes'",
        "why":         (
            "ai_intervention_applied was populated from the date boundary — "
            "both approaches yield the same rows. But using the date is more "
            "explicit, readable, and auditable. The column is a derived flag, "
            "not an independent source of truth."
        ),
    },
    {
        "rule_id":     "RULE_03",
        "title":       "Supplier Name Lookup",
        "description": (
            "ALWAYS JOIN to suppliers_master for supplier_name. "
            "ALWAYS filter on supplier_id, not supplier_name."
        ),
        "correct_sql": (
            "JOIN suppliers_master s ON sh.supplier_id = s.supplier_id "
            "WHERE sh.supplier_id = 'SUP003'"
        ),
        "wrong_sql":   "WHERE supplier_name = 'SupplierC'",
        "why":         (
            "supplier_name does not exist in the shipments table. "
            "Filtering on it would silently return 0 rows with no error."
        ),
    },
    {
        "rule_id":     "RULE_04",
        "title":       "Financial Totals and ROI Aggregation",
        "description": (
            "Annual totals: GROUP BY year. "
            "Monthly trends: ORDER BY year, month. "
            "Total AI savings: MAX(cumulative_savings). "
            "NEVER SUM(roi_pct) or SUM(cumulative_savings)."
        ),
        "correct_sql": (
            "SELECT MAX(cumulative_savings) AS total_ai_savings "
            "FROM financial_impact"
        ),
        "wrong_sql":   "SELECT SUM(roi_pct) FROM financial_impact",
        "why":         (
            "roi_pct is a percentage calculated per month — summing it is "
            "mathematically meaningless. cumulative_savings is already a "
            "running total — summing it double-counts every month's savings."
        ),
    },
    {
        "rule_id":     "RULE_05",
        "title":       "NULL Handling for Delay Reasons",
        "description": (
            "ALWAYS filter WHERE status = 'Delayed' before analysing "
            "delay_reason_category."
        ),
        "correct_sql": (
            "WHERE status = 'Delayed' "
            "AND delay_reason_category IS NOT NULL"
        ),
        "wrong_sql":   "GROUP BY delay_reason_category",
        "why":         (
            "delay_reason_category is NULL for all OnTime shipments (80% of rows). "
            "Without the filter, NULL appears as the largest 'category', "
            "which produces a misleading result."
        ),
    },
    {
        "rule_id":     "RULE_06",
        "title":       "Date Filtering in SQLite",
        "description": (
            "ALWAYS use strftime() for date part extraction. "
            "SQLite does NOT support YEAR(), MONTH(), or DATE_PART()."
        ),
        "correct_sql": (
            "strftime('%Y', shipment_date) = '2024'\\n"
            "strftime('%Y-%m', shipment_date) for monthly grouping"
        ),
        "wrong_sql":   "YEAR(shipment_date) = 2024",
        "why":         (
            "SQLite has no YEAR() function. Calling it returns NULL silently, "
            "causing the entire WHERE clause to match nothing."
        ),
    },
    {
        "rule_id":     "RULE_07",
        "title":       "OTD Rate Aggregation",
        "description": (
            "Use AVG(on_time_rate_pct) for period averages. "
            "NEVER SUM(on_time_rate_pct)."
        ),
        "correct_sql": "SELECT AVG(on_time_rate_pct) FROM financial_impact WHERE year = 2023",
        "wrong_sql":   "SELECT SUM(on_time_rate_pct) FROM financial_impact",
        "why":         (
            "on_time_rate_pct is a percentage (0–100), not a count. "
            "Summing percentages produces a number > 100, which is nonsensical."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HELPER FUNCTIONS
#  These are called by planning_agent.py and db_agent.py to build LLM prompts.
#  All pure Python — zero LLM calls.
# ═══════════════════════════════════════════════════════════════════════════════

def get_table_context(table_name: str) -> dict:
    """
    Return the full semantic definition for a single table.

    WHY: Agents request only the tables relevant to their step —
    injecting all tables into every prompt wastes tokens and confuses the LLM.

    Args:
        table_name: One of 'shipments', 'financial_impact', 'suppliers_master'.

    Returns:
        Full dict from SEMANTIC_LAYER, or empty dict if table not found.
    """
    if table_name not in SEMANTIC_LAYER:
        log.warning(
            f"get_table_context | Unknown table: '{table_name}'. "
            f"Valid tables: {list(SEMANTIC_LAYER.keys())}"
        )
        return {}

    log.debug(f"get_table_context | retrieved: {table_name}")
    return SEMANTIC_LAYER[table_name]


def get_column_info(table_name: str, column_name: str) -> dict:
    """
    Return the semantic definition for a single column.

    WHY: When an agent needs to validate one specific field
    (e.g. checking allowed values before writing a WHERE clause),
    it should not have to load the entire table definition.

    Args:
        table_name:  Table the column belongs to.
        column_name: Exact column name as it exists in SQLite.

    Returns:
        Column definition dict, or empty dict if not found.
    """
    table = SEMANTIC_LAYER.get(table_name, {})
    columns = table.get("columns", {})

    if column_name not in columns:
        log.warning(
            f"get_column_info | '{column_name}' not found in '{table_name}'. "
            f"Available columns: {list(columns.keys())}"
        )
        return {}

    log.debug(f"get_column_info | {table_name}.{column_name} retrieved")
    return columns[column_name]


def build_llm_context(tables_needed: list[str]) -> str:
    """
    Build the complete context string injected into the LLM prompt
    BEFORE SQL generation.

    WHY: This is the core hallucination-prevention mechanism.
    By telling the LLM exactly what columns exist, what values are allowed,
    and what rules must be followed, we eliminate the most common SQL errors.

    Structure of output:
        1. Table summaries (description, row count, grain)
        2. Column definitions (type, allowed_values, business_rules)
        3. Relevant relationships (safe joins only)
        4. All golden SQL rules

    Args:
        tables_needed: List of table names required for this query.
                       E.g. ["shipments", "suppliers_master"]

    Returns:
        Formatted multi-line string ready to prepend to an LLM prompt.
    """
    lines: list[str] = []

    lines.append("=" * 65)
    lines.append("DATABASE CONTEXT — READ BEFORE WRITING ANY SQL")
    lines.append("=" * 65)
    lines.append("")

    # ── 1. Table + column definitions ────────────────────────────────────────
    for table_name in tables_needed:
        table = SEMANTIC_LAYER.get(table_name)
        if not table:
            log.warning(f"build_llm_context | Unknown table '{table_name}' skipped")
            continue

        lines.append(f"TABLE: {table_name.upper()}")
        lines.append(f"  Description : {table['description']}")
        lines.append(f"  Rows        : {table['row_count']}")
        if "time_range" in table:
            lines.append(f"  Time range  : {table['time_range']}")
        lines.append(f"  Grain       : {table['grain']}")
        lines.append("")
        lines.append("  COLUMNS:")

        for col_name, col_def in table["columns"].items():
            col_line = f"    {col_name} ({col_def.get('type', 'TEXT')})"

            if col_def.get("never_null"):
                col_line += " [NOT NULL]"
            if col_def.get("primary_key"):
                col_line += " [PK]"

            lines.append(col_line)

            if "description" in col_def:
                lines.append(f"        → {col_def['description']}")
            if "allowed_values" in col_def:
                lines.append(
                    f"        → Allowed values: {col_def['allowed_values']}"
                )
            if "known_values" in col_def:
                for k, v in col_def["known_values"].items():
                    lines.append(f"        → {k}: {v}")
            if "business_rule" in col_def:
                lines.append(f"        ⚠ RULE: {col_def['business_rule']}")
            if "sql_note" in col_def:
                lines.append(f"        ✎ SQL:  {col_def['sql_note']}")

        lines.append("")

    # ── 2. Safe relationships for these tables ────────────────────────────────
    relevant_rels = [
        r for r in RELATIONSHIPS
        if r["from_table"] in tables_needed or r.get("to_table") in tables_needed
    ]
    if relevant_rels:
        lines.append("RELATIONSHIPS:")
        for rel in relevant_rels:
            if rel["safe"]:
                lines.append(
                    f"  ✅ SAFE JOIN: {rel['from_table']}.{rel.get('from_column','')} "
                    f"→ {rel['to_table']}.{rel.get('to_column','')}"
                )
                if "example_sql" in rel:
                    lines.append(f"     Example: {rel['example_sql']}")
            else:
                lines.append(
                    f"  ❌ UNSAFE: {rel['from_table']} ↔ {rel['to_table']} "
                    f"— {rel.get('warning', rel.get('note', ''))}"
                )
        lines.append("")

    # ── 3. Golden SQL rules ───────────────────────────────────────────────────
    lines.append("GOLDEN SQL RULES — MUST FOLLOW:")
    for rule in GOLDEN_RULES:
        lines.append(f"  [{rule['rule_id']}] {rule['title']}")
        lines.append(f"    ✅ CORRECT: {rule['correct_sql']}")
        lines.append(f"    ❌ WRONG:   {rule['wrong_sql']}")
        lines.append(f"    WHY: {rule['why']}")
        lines.append("")

    lines.append("=" * 65)
    lines.append("END OF DATABASE CONTEXT")
    lines.append("=" * 65)

    context = "\n".join(lines)
    log.debug(
        f"build_llm_context | built for tables={tables_needed} | "
        f"{len(context)} chars"
    )
    return context


def get_golden_rules_text() -> str:
    """
    Return all golden SQL rules as a compact, numbered plain-text string
    for injection into LLM prompts.

    WHY a separate function?
        build_llm_context() includes full table definitions — expensive for
        simple formatting prompts. This returns only the rules, suitable
        for the Executive Agent's answer-formatting prompt.

    Returns:
        Numbered list of golden rules as a plain string.
    """
    lines: list[str] = ["GOLDEN SQL RULES:"]
    for i, rule in enumerate(GOLDEN_RULES, start=1):
        lines.append(f"  {i}. [{rule['rule_id']}] {rule['title']}: {rule['description']}")
    return "\n".join(lines)
