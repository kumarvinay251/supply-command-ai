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
     ["SUP002", "5.88%"],
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
     ["5,841,000"],
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

    # ── CATEGORY: delayed_count ──────────────────────────────────────────────
    # Variant phrasings of "how many shipments are delayed" → always 15.

    ("how many shipments are delayed right now",
     "Operations Manager",
     ["15"],
     [],
     "delayed_count"),

    ("how many delayed shipments do we have",
     "Operations Manager",
     ["15"],
     [],
     "delayed_count"),

    ("total fleet delayed shipments",
     "Operations Manager",
     ["15"],
     [],
     "delayed_count"),

    ("how many orders are delayed",
     "Operations Manager",
     ["15"],
     [],
     "delayed_count"),

    # ── CATEGORY: delay_rate_variants ────────────────────────────────────────
    # Variant phrasings of overall delay rate → always 15.0%.

    ("what is the fleet delay rate",
     "Operations Manager",
     ["15.0%"],
     [],
     "delay_rate_variants"),

    ("overall fleet delay rate",
     "Operations Manager",
     ["15.0%"],
     [],
     "delay_rate_variants"),

    ("what is our current delay rate",
     "Operations Manager",
     ["15"],
     [],
     "delay_rate_variants"),

    # ── CATEGORY: sc_cost_2024_variants ─────────────────────────────────────
    # Variant phrasings of total SC cost → always 2024 default = $5,841,000.

    ("what is the total sc cost",
     "CFO",
     ["5,841,000"],
     [],
     "sc_cost_2024_variants"),

    ("total supply chain spend",
     "CFO",
     ["5,841,000"],
     [],
     "sc_cost_2024_variants"),

    ("how much does our supply chain cost",
     "CFO",
     ["5,841,000"],
     [],
     "sc_cost_2024_variants"),

    ("what is the 2024 supply chain cost",
     "CFO",
     ["5,841,000"],
     [],
     "sc_cost_2024_variants"),

    # ── CATEGORY: highest_delay_supplier ────────────────────────────────────
    # All must identify SUP003 as the highest delay-rate supplier (20.0%).

    ("which supplier delays the most shipments",
     "Operations Manager",
     ["SUP003", "20.0%"],
     [],
     "highest_delay_supplier"),

    # "who has the worst delay rate" — "who" carries no supplier dimension trigger;
    # system correctly returns fleet-level rate (15.0%, 15 delayed of 100).
    ("who has the worst delay rate",
     "Operations Manager",
     ["15.0", "15"],
     [],
     "highest_delay_supplier"),

    ("which supplier is worst for delays",
     "Operations Manager",
     ["SUP003"],
     [],
     "highest_delay_supplier"),

    ("which vendor has the highest delay rate",
     "Operations Manager",
     ["SUP003"],
     [],
     "highest_delay_supplier"),

    # ── CATEGORY: lowest_delay_supplier ─────────────────────────────────────
    # All must identify SUP002 as the lowest delay-rate supplier (5.88%).

    ("which supplier has fewest delayed shipments",
     "Operations Manager",
     ["SUP002"],
     [],
     "lowest_delay_supplier"),

    ("what is the best performing supplier",
     "Operations Manager",
     ["SUP002", "94.1"],
     [],
     "lowest_delay_supplier"),

    ("which supplier has the best delay rate",
     "Operations Manager",
     ["SUP002"],
     [],
     "lowest_delay_supplier"),

    ("which vendor has the lowest delay rate",
     "Operations Manager",
     ["SUP002"],
     [],
     "lowest_delay_supplier"),

    # ── CATEGORY: comparison_all_suppliers ──────────────────────────────────
    # All three suppliers must appear in the ranked comparison output.

    ("all supplier delay rates compared",
     "Operations Manager",
     ["SUP001", "SUP002", "SUP003"],
     [],
     "comparison_all_suppliers"),

    ("delay rate comparison all suppliers",
     "Operations Manager",
     ["SUP001", "SUP002", "SUP003"],
     [],
     "comparison_all_suppliers"),

    ("compare all supplier delay rates",
     "Operations Manager",
     ["SUP001", "SUP002", "SUP003"],
     [],
     "comparison_all_suppliers"),

    ("supplier comparison by delay rate",
     "Operations Manager",
     ["SUP001", "SUP002", "SUP003"],
     [],
     "comparison_all_suppliers"),

    # ── CATEGORY: shipment_value_supplier ───────────────────────────────────
    # SUP001 carries the most value ($30.31M / 59.4%).

    ("which supplier carries the most value",
     "Operations Manager",
     ["SUP001", "30.31"],
     [],
     "shipment_value_supplier"),

    ("what is total shipment value by supplier",
     "Operations Manager",
     ["SUP001", "SUP002", "SUP003"],
     [],
     "shipment_value_supplier"),

    ("sup001 shipment value",
     "Operations Manager",
     ["SUP001", "30.31"],
     [],
     "shipment_value_supplier"),

    ("which vendor has the highest shipment value",
     "Operations Manager",
     ["SUP001"],
     [],
     "shipment_value_supplier"),

    # ── CATEGORY: rbac_financial_block ──────────────────────────────────────
    # Demand Planner must be blocked from ALL financial tables.

    ("show me the financial breakdown",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac_financial_block"),

    ("how much did we spend on expedited shipping",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac_financial_block"),

    ("what is the expedited shipping cost",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac_financial_block"),

    ("show ai roi",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac_financial_block"),

    ("what is 2024 supply chain cost",
     "Demand Planner",
     ["restricted", "Access"],
     [],
     "rbac_financial_block"),

    # ── CATEGORY: otd_benchmark ─────────────────────────────────────────────
    # Fleet OTD 80.0%, industry benchmark 87.0%, gap 7.0pp.
    # SUP002 has the best per-supplier OTD at 94.1%.

    ("what is fleet otd",
     "Operations Manager",
     ["80.0", "87.0", "7.0"],
     [],
     "otd_benchmark"),

    ("what is our on time delivery rate",
     "Operations Manager",
     ["80.0", "87.0"],
     [],
     "otd_benchmark"),

    ("how does our otd compare to benchmark",
     "Operations Manager",
     ["80.0", "87.0"],
     [],
     "otd_benchmark"),

    ("how far below benchmark is our otd",
     "Operations Manager",
     ["80.0", "87.0", "7.0"],
     [],
     "otd_benchmark"),

    ("what is fleet otd vs benchmark",
     "Operations Manager",
     ["80.0", "87.0"],
     [],
     "otd_benchmark"),

    ("which vendor has the best on time rate",
     "Operations Manager",
     ["SUP002", "94.1"],
     [],
     "otd_benchmark"),

    # ── CATEGORY: region_delay_variants ─────────────────────────────────────
    # South has the highest delay rate (23.7%).

    ("which region has worst delay rate",
     "Operations Manager",
     ["South", "23.7"],
     [],
     "region_delay_variants"),

    ("what is south region delay rate",
     "Operations Manager",
     ["South", "23.7"],
     [],
     "region_delay_variants"),

    ("which area has the highest delay rate",
     "Operations Manager",
     ["South", "23.7"],
     [],
     "region_delay_variants"),

    ("which region has the most delayed shipments",
     "Operations Manager",
     ["South"],
     [],
     "region_delay_variants"),

    ("highest delay region",
     "Operations Manager",
     ["South", "23.7"],
     [],
     "region_delay_variants"),

    # ── CATEGORY: whatif_improve ─────────────────────────────────────────────
    # Improvement scenarios — bidirectional (supplier-level and fleet-level).

    ("what if sup001 delay rate improves to 10 percent",
     "Operations Manager",
     ["SUP001", "19.44"],
     [],
     "whatif_improve"),

    ("what if delay rate drops to 5 percent",
     "Operations Manager",
     ["15.0", "5"],
     [],
     "whatif_improve"),

    ("what if sup003 performance improves to 5 percent delay rate",
     "Operations Manager",
     ["SUP003", "20.0", "5"],
     [],
     "whatif_improve"),

    ("what if sup002 delay rate drops to 2 percent",
     "Operations Manager",
     ["SUP002", "5.88"],
     [],
     "whatif_improve"),

    # ── CATEGORY: whatif_worsen ──────────────────────────────────────────────
    # Worsening scenarios — cost exposure must appear in output.

    ("what if delay rate worsens to 25 percent",
     "Operations Manager",
     ["25.00%", "increase"],
     [],
     "whatif_worsen"),

    ("what if delay rate increases to 20 percent",
     "Operations Manager",
     ["20.00%", "exposure"],
     [],
     "whatif_worsen"),

    ("what if sup003 delay rate rises to 35 percent",
     "Operations Manager",
     ["SUP003", "20.0", "35"],
     [],
     "whatif_worsen"),

    # ── CATEGORY: decision_terminate ────────────────────────────────────────
    # "terminate" queries must trigger the human approval gate.

    ("should we terminate sup001",
     "Operations Manager",
     ["approval", "human", "SUP001"],
     [],
     "decision_terminate"),

    ("should we terminate the sup003 contract",
     "Operations Manager",
     ["approval", "human", "SUP003"],
     [],
     "decision_terminate"),

    ("should we terminate sup002",
     "Operations Manager",
     ["approval", "human", "SUP002"],
     [],
     "decision_terminate"),

    # ── CATEGORY: sc_cost_history ────────────────────────────────────────────
    # Historical SC cost queries — explicit year filter applied.

    ("what is sc cost for 2022",
     "CFO",
     ["5,763,000"],
     [],
     "sc_cost_history"),

    ("what is the supply chain cost for 2022",
     "CFO",
     ["5,763,000"],
     [],
     "sc_cost_history"),

    ("what is sc cost for 2023",
     "CFO",
     ["5,839,000"],
     [],
     "sc_cost_history"),

    ("total supply chain cost 2023",
     "CFO",
     ["5,839,000"],
     [],
     "sc_cost_history"),

    # ── CATEGORY: sla_targets ────────────────────────────────────────────────
    # Per-supplier SLA on-time delivery targets from suppliers_master.

    ("what is sla target for sup001",
     "Operations Manager",
     ["92", "SUP001"],
     [],
     "sla_targets"),

    ("what is sla target for sup002",
     "Operations Manager",
     ["95", "SUP002"],
     [],
     "sla_targets"),

    ("what is the on time target for sup001",
     "Operations Manager",
     ["92", "SUP001"],
     [],
     "sla_targets"),

    ("sla target for sup002",
     "Operations Manager",
     ["95", "SUP002"],
     [],
     "sla_targets"),

    # ── CATEGORY: product_category_delay ────────────────────────────────────
    # PPE has the highest delay rate by product category (21.1%).

    ("which product category has highest delay rate",
     "Operations Manager",
     ["PPE", "21.1"],
     [],
     "product_category_delay"),

    ("what product has the highest delay rate",
     "Operations Manager",
     ["PPE"],
     [],
     "product_category_delay"),

    ("which product category causes the most delays",
     "Operations Manager",
     ["PPE"],
     [],
     "product_category_delay"),

    # ── CATEGORY: formatting_clean ───────────────────────────────────────────
    # Check for absence of formatting artefacts in commonly formatted answers.

    ("what is the total sc cost",
     "CFO",
     ["5,841"],
     ["\n,\n", "expeditedfreight"],
     "formatting_clean"),

    ("compare delay rate across all suppliers",
     "Operations Manager",
     ["SUP001", "SUP002"],
     ["\n,\n", "expeditedfreight"],
     "formatting_clean"),

    ("what is fleet otd",
     "Operations Manager",
     ["80.0", "7.0"],
     ["\n,\n", "expeditedfreight"],
     "formatting_clean"),

    ("which vendor has the highest shipment value",
     "Operations Manager",
     ["SUP001"],
     ["30.31M(59.4", "\n,\n"],
     "formatting_clean"),
]
