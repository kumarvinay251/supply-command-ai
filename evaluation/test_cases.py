# evaluation/test_cases.py
# LOCKED EXPECTED ANSWERS — do not change without updating the system
# Format: (query, role, expected_contains, expected_not_contains, category)
#
# expected_contains:     ALL strings must appear in the answer (case-insensitive)
# expected_not_contains: NONE of these strings may appear in the answer
#
# WHY a separate locked file?
#   Keeping expected answers in their own module prevents accidental drift.
#   Changing run_eval.py logic must not touch expected values, and vice-versa.
#   Any change here requires a deliberate "update golden set" commit message.

TEST_CASES = [
    # ── CATEGORY: basic_metrics ──────────────────────────────────────────────
    # Scalar lookup queries — fastest path, SIMPLE_COUNT or SIMPLE_METRIC.
    # All expect exact numbers that come directly from the database.

    ("how many shipments are delayed",
     "Operations Manager",
     ["15"],
     [],
     "basic_metrics"),

    ("what is overall delay rate",
     "Operations Manager",
     ["15.0%"],
     [],
     "basic_metrics"),

    ("what is total number of shipments",
     "Operations Manager",
     ["100"],
     [],
     "basic_metrics"),

    ("what is maximum delay observed",
     "Operations Manager",
     ["12"],
     ["6.0", "average"],
     "basic_metrics"),

    ("what is average delay for delayed shipments",
     "Operations Manager",
     ["6.0"],
     [],
     "basic_metrics"),

    ("what is total shipment value",
     "Operations Manager",
     ["51"],
     [],
     "basic_metrics"),

    ("what is average shipment value",
     "Operations Manager",
     ["510"],
     [],
     "basic_metrics"),

    ("what is the shipment date span",
     "Operations Manager",
     ["2023", "2024"],
     [],
     "basic_metrics"),

    # ── CATEGORY: supplier_analysis ──────────────────────────────────────────
    # Per-supplier queries — delay rates, SLA targets, comparisons.

    ("which supplier has highest delay rate",
     "Operations Manager",
     ["SUP003", "20.0%"],
     [],
     "supplier_analysis"),

    ("which supplier has lowest delay rate",
     "Operations Manager",
     ["SUP002", "5.9%"],
     [],
     "supplier_analysis"),

    ("which supplier has highest shipment value",
     "Operations Manager",
     ["SUP001"],
     [],
     "supplier_analysis"),

    ("what is delay rate for sup003",
     "Operations Manager",
     ["20.0%", "SUP003"],
     [],
     "supplier_analysis"),

    ("what is sla target for sup003",
     "Operations Manager",
     ["88", "SUP003"],
     [],
     "supplier_analysis"),

    ("compare delay rate across all suppliers",
     "Operations Manager",
     ["SUP001", "SUP002", "SUP003"],
     [],
     "supplier_analysis"),

    # ── CATEGORY: financial ──────────────────────────────────────────────────
    # Financial queries — restricted to CFO / Operations Manager.
    # Dollar amounts must match the DB to within rounding.

    ("what is total supply chain cost",
     "CFO",
     ["17,443,000"],
     [],
     "financial"),

    ("what is total sc cost for 2022",
     "CFO",
     ["5,763,000", "5,7"],
     [],
     "financial"),

    ("what is 2024 expedited shipping cost",
     "CFO",
     ["39,797"],
     [],
     "financial"),

    ("what is total avoidable cost in 2024",
     "CFO",
     ["623"],
     [],
     "financial"),

    ("what is 2024 ai roi",
     "CFO",
     ["340%"],
     [],
     "financial"),

    ("how much did ai save in 2024",
     "CFO",
     ["401,000", "401"],
     [],
     "financial"),

    ("what is roi of ai investment",
     "CFO",
     ["340%", "401,000"],
     [],
     "financial"),

    # ── CATEGORY: region_category ────────────────────────────────────────────
    # Geographic and product-category delay analysis.

    ("which region has highest delay rate",
     "Operations Manager",
     ["South", "23.7%"],
     [],
     "region_category"),

    ("which region contributes most to delays",
     "Operations Manager",
     ["South"],
     ["Supplier_Dispatch", "dispatch"],
     "region_category"),

    ("which category has highest delay rate",
     "Operations Manager",
     ["PPE", "21.1%"],
     [],
     "region_category"),

    ("what is total shipment value by region",
     "Operations Manager",
     ["North", "South", "West"],
     [],
     "region_category"),

    # ── CATEGORY: rbac ───────────────────────────────────────────────────────
    # Role-Based Access Control — Demand Planner must be blocked from financial data.
    # Expected answer must include an access-denied message.

    ("what is total supply chain cost",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac"),

    ("what is 2024 ai roi",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac"),

    # ── CATEGORY: whatif ─────────────────────────────────────────────────────
    # ROI simulation — bidirectional (improvement and worsening scenarios).

    ("what if sup003 delay rate drops to 10%",
     "Operations Manager",
     ["75,999", "50%"],
     [],
     "whatif"),

    ("what if delay rate increases to 30%",
     "Operations Manager",
     ["increase", "exposure", "30.00%"],
     [],
     "whatif"),

    # ── CATEGORY: decision ───────────────────────────────────────────────────
    # Decision queries must trigger the human approval gate.

    ("should we terminate sup003",
     "Operations Manager",
     ["approval", "human", "SUP003"],
     [],
     "decision"),

    # ── CATEGORY: formatting ─────────────────────────────────────────────────
    # These check for absence of known broken formatting artefacts.
    # expected_contains may be empty; expected_not_contains is the key signal.

    ("which supplier has highest shipment value",
     "Operations Manager",
     [],
     ["30.31M(59.4", "\n,\n", "expeditedfreight"],
     "formatting"),

    ("what is total avoidable cost in 2024",
     "CFO",
     ["$623", "623,297"],
     ["623,297(expedited", "\n,\n"],
     "formatting"),
]
