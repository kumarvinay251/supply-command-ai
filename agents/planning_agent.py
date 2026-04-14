"""
Supply Command AI — Planning Agent

This is the brain of the system. It reads a user question and produces a
structured, step-by-step investigation plan BEFORE any data is fetched.

WHY a planning layer at all?
    Without a plan, agents fetch data ad-hoc, leading to redundant DB calls,
    inconsistent answers, and no audit trail of reasoning. The planner
    converts a free-text question into a deterministic, logged sequence of
    specialist agent calls — so every answer is reproducible and explainable.

LLM Usage Policy (STRICTLY ENFORCED):
    NO   classify_intent()     — keyword scoring, pure Python
    NO   get_plan_template()   — lookup table, pure Python
    NO   create_plan()         — orchestration logic, pure Python
    NO   explain_plan()        — string formatting, pure Python
    YES  Executive Agent only  — called ONCE at the end, by the caller
                                  of this plan, to format the final answer

Every plan is logged. Every guardrail check is logged. If the plan is
blocked at any stage, the reason is returned — never silently swallowed.
"""

import re
from datetime import datetime, timezone
from typing import Optional

from services.logger import get_logger
from agents.guardrails import (
    validate_role,
    check_query_access,
    detect_prompt_injection,
    check_human_approval_needed,
    ROLES,
    FINANCIAL_IMPACT_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)

log = get_logger("planning_agent")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INTENT CLASSIFICATION
#  Pure Python keyword matching. Zero LLM tokens.
#
#  WHY keyword matching instead of an LLM classifier?
#  1. Speed — a dict lookup is microseconds; an LLM call is 1–3 seconds.
#  2. Cost  — every plan creation would cost tokens; this is free.
#  3. Auditability — "keyword 'delay' matched DELAY_ANALYSIS" is a clear,
#     explainable decision. An LLM classification is a black box.
#  4. Reliability — keywords don't hallucinate or drift across model versions.
# ═══════════════════════════════════════════════════════════════════════════════

# Each intent maps to a list of trigger keywords.
# WHY multi-word phrases in the list?
#   Single words like "why" would fire too broadly. Multi-word phrases
#   ("root cause", "what caused") are more precise signals.
# WHY these specific intents?
#   They map 1:1 to the types of questions a supply chain operator asks:
#   What broke? Who caused it? What did it cost? What should I do?

INTENT_KEYWORDS: dict[str, list[str]] = {

    "DELAY_ANALYSIS": [
        # WHY these keywords?
        #   All terms that indicate a question about delivery timing or SLA.
        #   "OTD" = On-Time Delivery, a standard supply chain KPI.
        #   "breach" captures SLA breach questions without needing the
        #   full phrase.
        "delay", "delayed", "late", "shipment", "on-time", "otd",
        "breach", "behind schedule", "overdue", "missed delivery",
    ],

    "SUPPLIER_RISK": [
        # WHY include supplier IDs (SUP001-003)?
        #   Users often ask about a specific supplier by code. Without these,
        #   "show me SUP002 performance" would match DELAY_ANALYSIS
        #   instead of SUPPLIER_RISK.
        "supplier", "vendor", "sup001", "sup002", "sup003",
        "performance", "risk", "sla", "tier", "account manager",
        "reliability", "on time rate",
    ],

    "INVENTORY_RISK": [
        # WHY "days of cover" as one phrase?
        #   It is a specific inventory metric — matching it as a phrase
        #   prevents false positives from "days" alone.
        "stock", "inventory", "stockout", "days of cover", "reorder",
        "shortage", "excess", "overstock", "buffer", "safety stock",
        "inventory level", "dispatch",
    ],

    "FINANCIAL_IMPACT": [
        # WHY "roi" in lower-case?
        #   The query is lowercased before matching — "ROI" and "roi" and
        #   "RoI" all match the same keyword.
        "cost", "spend", "roi", "savings", "expedited cost",
        "penalty", "loss", "financial", "budget", "revenue",
        "avoidable", "freight", "insurance", "stockout cost",
        "total cost", "supply chain cost",
    ],

    "BENCHMARK_COMPARISON": [
        # WHY "vs" and "versus"?
        #   Both are common comparison signals in natural language.
        #   "how do we rank" covers the implicit benchmark question:
        #   "Are we better or worse than industry?"
        "benchmark", "industry", "compare", "vs", "versus",
        "better than", "worse than", "how do we rank", "industry average",
        "best in class", "world class", "peers",
    ],

    "ROOT_CAUSE": [
        # WHY "why" as a standalone keyword?
        #   "Why" almost always signals an explanatory investigation.
        #   It scores low alone (1 hit) but combines with other words
        #   to boost confidence. A bare "why" with nothing else falls
        #   back to DELAY_ANALYSIS or RECOMMENDATION.
        "why", "root cause", "reason", "what caused", "explain",
        "investigate", "analysis", "because", "contributing factor",
        "caused by",
    ],

    "RECOMMENDATION": [
        # WHY action-oriented phrases?
        #   These keywords signal the user wants a decision, not just data.
        #   This intent triggers the HUMAN CHECKPOINT step — because
        #   recommendations have real-world consequences.
        # WHY "switch" and "terminate" and "change" here?
        #   These are irreversible supply chain actions — switching a supplier,
        #   terminating a contract, changing a routing. Adding them to
        #   RECOMMENDATION ensures these queries are classified as DECISION_QUERY
        #   (not METRIC_QUERY or SUPPLIER_RISK), which is the prerequisite for
        #   the Stage 8 human-approval gate to fire.
        "what should", "recommend", "action", "fix", "improve",
        "what to do", "next steps", "how can we", "how to reduce",
        "suggestion", "advice", "priority", "should we",
        "switch", "switch from", "terminate", "change supplier",
        "expedite", "cancel order",
    ],
}

# Fallback when no keywords match at all
DEFAULT_INTENT = "DELAY_ANALYSIS"


def classify_intent(query: str) -> dict:
    """
    Classify a natural-language query into one or two supply chain intents.

    Algorithm:
        1. Lowercase the query for case-insensitive matching.
        2. For each intent, count how many of its keywords appear in the query.
        3. Primary intent   = highest keyword hit count.
        4. Secondary intent = second highest (only if hit count > 0).
        5. Confidence       = primary_hits / total_keywords_in_that_intent.
           WHY? If an intent has 10 keywords and 7 matched, we are very
           confident. If only 1 of 10 matched, we are less certain.

    Returns:
        {
            "primary_intent":    str,
            "secondary_intent":  str or None,
            "confidence":        float (0.0 – 1.0),
            "keywords_matched":  list[str],
        }
    """
    q_lower  = query.lower().strip()
    scores:  dict[str, int]       = {}
    matched: dict[str, list[str]] = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in q_lower]
        if hits:
            scores[intent]  = len(hits)
            matched[intent] = hits

    if not scores:
        # No keywords matched at all — default to DELAY_ANALYSIS
        # WHY DELAY_ANALYSIS as default?
        #   It is the most common query type in supply chain operations.
        #   An ambiguous question is more likely to be about delivery
        #   than about benchmarks or financial modelling.
        log.debug(
            f"classify_intent | no keywords matched — "
            f"defaulting to {DEFAULT_INTENT}"
        )
        return {
            "primary_intent":   DEFAULT_INTENT,
            "secondary_intent": None,
            "confidence":       0.3,    # low confidence — honest about uncertainty
            "keywords_matched": [],
        }

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    primary_intent    = ranked[0][0]
    primary_hits      = ranked[0][1]
    primary_keywords  = INTENT_KEYWORDS[primary_intent]

    # Confidence = fraction of that intent's keywords that fired
    confidence = round(min(primary_hits / len(primary_keywords), 1.0), 2)

    # Secondary intent: only assign if it has at least 1 hit
    secondary_intent = ranked[1][0] if len(ranked) > 1 else None

    all_matched = matched.get(primary_intent, [])
    if secondary_intent:
        all_matched = all_matched + matched.get(secondary_intent, [])

    log.info(
        f"classify_intent | "
        f"primary='{primary_intent}' ({primary_hits} hits, conf={confidence}) | "
        f"secondary='{secondary_intent}' | "
        f"matched={all_matched}"
    )

    return {
        "primary_intent":   primary_intent,
        "secondary_intent": secondary_intent,
        "confidence":       confidence,
        "keywords_matched": all_matched,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PLAN TEMPLATES
#  Pure Python lookup table. Zero LLM tokens.
#
#  WHY fixed templates instead of dynamic plan generation?
#  1. Predictability — the same question always produces the same plan.
#  2. Auditability   — each template was designed by a human and reviewed.
#  3. Cost           — LLM plan generation costs tokens every time.
#  4. Safety         — a fixed template cannot accidentally route a question
#     to an agent that has no business seeing that data.
#
#  Each step has:
#    step_number            — execution order (1 = first)
#    agent                  — which specialist handles this step
#    task                   — human-readable description of the work
#    table                  — which DB table is primarily accessed (or None)
#    requires_human_approval — pre-flagged for high-risk steps
# ═══════════════════════════════════════════════════════════════════════════════

def get_plan_template(intent: str) -> list[dict]:
    """
    Return the ordered investigation steps for a given intent.

    WHY each intent maps to its template:

    DELAY_ANALYSIS:
        A delay question always needs: the count (who/where), the trend
        (when it got worse), the reason (why), the SOP (what to do), and
        the cost (how much it matters). Skipping any step leaves the
        answer incomplete.

    SUPPLIER_RISK:
        Supplier risk requires comparing actual performance vs SLA targets,
        then quantifying financial exposure, so the ops team can prioritise
        which supplier to address first.

    FINANCIAL_IMPACT:
        Cost questions need both the breakdown (where money is going) and
        the ROI view (is the AI investment paying off?). The RAG step adds
        benchmark context so costs are interpreted relative to industry norms.

    BENCHMARK_COMPARISON:
        Must retrieve external benchmark FIRST (from the Annual Report PDF)
        before pulling internal KPIs, so the comparison is meaningful.
        RAG → DB order is intentional.

    ROOT_CAUSE:
        Two DB steps (data + correlations) before RAG because we need
        specific numbers before searching for matching scenarios in
        the decision framework. Evidence before explanation.

    RECOMMENDATION:
        Contains a mandatory HUMAN CHECKPOINT at step 4.
        WHY? A recommendation implies action in the real world.
        Before the system formats a recommendation, a human must
        confirm the impact estimate is acceptable. No autonomous action.

    Args:
        intent: One of the INTENT_KEYWORDS keys or a fallback string.

    Returns:
        List of step dicts ordered by step_number.
    """

    templates: dict[str, list[dict]] = {

        # ── DELAY_ANALYSIS ────────────────────────────────────────────────────
        # WHY 5 steps before the executive summary?
        #   Delay questions need both breadth (who/where/when) and depth
        #   (why it happened, what it cost). Skipping any step produces
        #   an incomplete picture that could lead to wrong fixes.
        "DELAY_ANALYSIS": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get delay count and delay rate by supplier and region",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "DB Agent",
                "task":                    "Get delay trend by month (last 24 months)",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "DB Agent",
                "task":                    "Get top delay reasons ranked by frequency",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             4,
                "agent":                   "RAG Agent",
                "task":                    "Search knowledge base for relevant SOP or escalation guideline",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             5,
                "agent":                   "ROI Agent",
                "task":                    "Calculate cost impact of delays (penalties + expedited costs)",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             6,
                "agent":                   "Executive Agent",
                "task":                    "Format and narrate findings for the user's role",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],

        # ── SUPPLIER_RISK ─────────────────────────────────────────────────────
        # WHY performance vs SLA targets first?
        #   The most actionable supplier risk signal is the gap between
        #   contracted SLA (in suppliers_master) and actual delivery rate
        #   (in shipments). The financial exposure step quantifies whether
        #   switching or escalating is worth the effort.
        "SUPPLIER_RISK": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get supplier on-time rate vs SLA target from suppliers_master",
                "table":                   "suppliers_master",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "DB Agent",
                "task":                    "Get delay rate, SLA breach count, and risk flag by supplier",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "RAG Agent",
                "task":                    "Search knowledge base for supplier evaluation best practices",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             4,
                "agent":                   "ROI Agent",
                "task":                    "Calculate financial exposure from underperforming suppliers",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             5,
                "agent":                   "Executive Agent",
                "task":                    "Format supplier risk summary for the user's role",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],

        # ── INVENTORY_RISK ────────────────────────────────────────────────────
        # WHY a separate inventory template?
        #   Inventory questions focus on stock levels at dispatch and
        #   stockout events — different columns and logic from delay analysis,
        #   even though both use the shipments table.
        "INVENTORY_RISK": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get inventory level at dispatch grouped by product category and region",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "DB Agent",
                "task":                    "Identify low inventory events and stockout risk shipments",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "RAG Agent",
                "task":                    "Search knowledge base for reorder policy and safety stock guidelines",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             4,
                "agent":                   "ROI Agent",
                "task":                    "Calculate stockout loss and excess inventory cost from financial_impact",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             5,
                "agent":                   "Executive Agent",
                "task":                    "Format inventory risk summary for the user's role",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],

        # ── FINANCIAL_IMPACT ──────────────────────────────────────────────────
        # WHY cost breakdown before ROI?
        #   The user needs to understand where money is going before the
        #   ROI calculation is meaningful. Showing ROI first without context
        #   of the cost structure is misleading.
        "FINANCIAL_IMPACT": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get cost breakdown by month: freight, insurance, penalty, stockout",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "DB Agent",
                "task":                    "Calculate ROI: ai_savings vs ai_investment since go-live",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "DB Agent",
                "task":                    "Identify avoidable costs: total_avoidable_usd trend over time",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             4,
                "agent":                   "RAG Agent",
                "task":                    "Retrieve industry benchmark cost figures from Annual Report PDF",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             5,
                "agent":                   "Executive Agent",
                "task":                    "Format financial narrative with ROI progression for the user's role",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],

        # ── BENCHMARK_COMPARISON ──────────────────────────────────────────────
        # WHY DB first, then RAG?
        #   We pull our own numbers first so the Executive Agent has a
        #   concrete internal figure to compare against the PDF benchmark.
        #   RAG retrieves the industry standard (87% OTD from Annual Report).
        #   The gap is then self-evident from the two findings side-by-side.
        #
        # WHY NO ROI Agent?
        #   The ROI Agent has no benchmark gap calculation logic — it only
        #   knows how to calculate delay cost, stockout risk, AI ROI, and
        #   supplier financial exposure. Without a matching calculation
        #   function it returns $0, which is both wrong and misleading.
        #   The Executive Agent can calculate the gap (87% − 81.4% = 5.6pp)
        #   directly from the two DB and RAG findings without a separate step.
        "BENCHMARK_COMPARISON": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get actual OTD performance vs SLA target from suppliers_master — actual_otd_pct, sla_target_pct, sla_gap for each supplier",
                "table":                   "suppliers_master",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "RAG Agent",
                "task":                    "Retrieve industry benchmark KPIs from the GlobalMedTech Annual Report PDF",
                # WHY separate instruction from task here?
                #   rag_agent.run() uses step.get("instruction", step.get("task", ""))
                #   as the FAISS search query. "Retrieve industry benchmark KPIs from
                #   the Annual Report PDF" scores 0.81 against the document header chunk
                #   (which has no benchmark numbers in its first 200 chars).
                #   "industry benchmark on-time delivery OTD performance" scores 0.86
                #   against the APPENDIX chunk which contains the exact line:
                #   "OTD 81.4% vs industry benchmark of 87%, gap 5.6pp".
                "instruction":             "industry benchmark on-time delivery OTD performance",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "Executive Agent",
                "task":                    "Format benchmark comparison with gap analysis for the user's role",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],

        # ── ROOT_CAUSE ────────────────────────────────────────────────────────
        # WHY two DB steps before RAG?
        #   Root cause analysis requires facts before interpretation.
        #   Step 1 gathers the evidence. Step 2 finds statistical
        #   correlations in that evidence. Only THEN does RAG search
        #   the decision framework for matching patterns — so it receives
        #   specific numbers, not just a vague question.
        "ROOT_CAUSE": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get all relevant data for the problem area (delays, statuses, flags)",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "DB Agent",
                "task":                    "Find correlations: delay vs weather, port condition, customs, supplier",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "RAG Agent",
                "task":                    "Match observed pattern to scenarios in decision framework or troubleshooting guide",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             4,
                "agent":                   "ROI Agent",
                "task":                    "Quantify total cost impact of the root cause identified",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             5,
                "agent":                   "Executive Agent",
                "task":                    "Format root cause narrative with evidence chain for the user's role",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],

        # ── RECOMMENDATION ────────────────────────────────────────────────────
        # WHY a HUMAN CHECKPOINT at step 4?
        #   Recommendations trigger real-world actions: cancelling a PO,
        #   expediting a shipment, switching a supplier. These are
        #   irreversible or costly. A human MUST review the impact
        #   estimate before the system narrates a recommendation.
        #   This is the only template with a mandatory checkpoint.
        "RECOMMENDATION": [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "Get current state data for the area in question",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "RAG Agent",
                "task":                    "Retrieve best-practice recommendation from decision framework",
                "table":                   None,
                "requires_human_approval": False,
            },
            {
                "step_number":             3,
                "agent":                   "ROI Agent",
                "task":                    "Estimate improvement potential and cost of recommended action",
                "table":                   "financial_impact",
                "requires_human_approval": False,
            },
            {
                "step_number":             4,
                "agent":                   "HUMAN CHECKPOINT",
                "task":                    "Review impact estimate — approve before recommendation is presented",
                "table":                   None,
                # WHY always True for this step?
                #   Because recommendations have real consequences and we
                #   never know the financial impact until step 3 completes.
                #   Making this checkpoint unconditional prevents the system
                #   from ever auto-approving an action recommendation.
                "requires_human_approval": True,
            },
            {
                "step_number":             5,
                "agent":                   "Executive Agent",
                "task":                    "Format actionable recommendation with confidence and caveats",
                "table":                   None,
                "requires_human_approval": False,
            },
        ],
    }

    # Fallback to DELAY_ANALYSIS if an unrecognised intent is passed
    # WHY not raise an exception?
    #   A planning failure should degrade gracefully — better to return
    #   a generic plan than to crash and return nothing to the user.
    template = templates.get(intent)
    if template is None:
        log.warning(
            f"get_plan_template | unknown intent='{intent}' — "
            f"falling back to DELAY_ANALYSIS template"
        )
        template = templates["DELAY_ANALYSIS"]

    return [step.copy() for step in template]   # return copies, not references


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MAIN PLANNING FUNCTION
#  Orchestrates guardrails → classification → template → output.
#  Zero LLM tokens.
# ═══════════════════════════════════════════════════════════════════════════════

def create_plan(query: str, role: str) -> dict:
    """
    Convert a natural-language question + role into a structured execution plan.

    Execution order (each stage can abort early):
        1. validate_role        — does this role exist?
        2. check_query_access   — can this role ask this question?
        3. detect_prompt_injection — is this a legitimate supply chain query?
        4. classify_intent      — what kind of investigation is needed?
        5. get_plan_template    — which steps answer this type of question?
        6. Merge secondary intent steps (if applicable)
        7. Filter steps by role access (remove financial steps for restricted roles)
        8. Flag human checkpoints
        9. Log the final plan
       10. Return complete plan dict

    WHY abort early at each stage?
        Each stage builds on the previous one. If we can't trust the role,
        we can't build a safe plan. If we can't trust the query, we shouldn't
        touch the database. Failing fast with a clear reason is better than
        producing a plan we can't execute safely.

    Args:
        query: Natural-language question from the user.
        role:  Display role name (e.g. "Demand Planner", "CFO").

    Returns:
        On success:
            {
                "status":              "ok",
                "query":               str,
                "role":                str,
                "intent":              str,
                "secondary_intent":    str or None,
                "confidence":          float,
                "keywords_matched":    list[str],
                "steps":               list[dict],
                "total_steps":         int,
                "estimated_llm_calls": int,
                "human_checkpoints":   list[int],
                "created_at":          str (ISO timestamp),
            }

        On failure (any guardrail block):
            {
                "status":  "blocked" | "access_denied" | "injection_detected"
                           | "invalid_role",
                "reason":  str,
                "query":   str,
                "role":    str,
            }
    """
    log.info(
        f"create_plan | START | role='{role}' | query='{query[:80]}'"
    )

    # ── Stage 1: Role Validation ──────────────────────────────────────────────
    role_check = validate_role(role)
    if not role_check["valid"]:
        log.warning(
            f"create_plan | INVALID ROLE | role='{role}' | "
            f"reason='{role_check['reason']}'"
        )
        return {
            "status": "invalid_role",
            "reason": role_check["reason"],
            "query":  query,
            "role":   role,
        }

    # ── Stage 2: Query Access Check ───────────────────────────────────────────
    access_check = check_query_access(query, role)
    if not access_check["allowed"]:
        log.warning(
            f"create_plan | ACCESS DENIED | role='{role}' | "
            f"blocked_term='{access_check['blocked_term']}'"
        )
        return {
            "status":       "access_denied",
            "reason":       access_check["reason"],
            "blocked_term": access_check["blocked_term"],
            "query":        query,
            "role":         role,
        }

    # ── Stage 3: Prompt Injection Check ──────────────────────────────────────
    injection_check = detect_prompt_injection(query)
    if not injection_check["safe"]:
        log.warning(
            f"create_plan | INJECTION DETECTED | role='{role}' | "
            f"reason='{injection_check['reason']}'"
        )
        return {
            "status": "injection_detected",
            "reason": injection_check["reason"],
            "query":  query,
            "role":   role,
        }

    # ── Stage 4: Intent Classification ───────────────────────────────────────
    intent_result = classify_intent(query)
    primary_intent   = intent_result["primary_intent"]
    secondary_intent = intent_result["secondary_intent"]
    confidence       = intent_result["confidence"]

    # ── Stage 4b: Query Type Classification ──────────────────────────────────
    # WHY three query types?
    #   METRIC_QUERY:      User wants a factual number (DB-only answer).
    #                      RAG is irrelevant — PDF benchmarks don't override SQL.
    #   DECISION_QUERY:    User wants an action recommendation (may need RAG).
    #                      Only these trigger human approval if action keywords appear.
    #   EXPLANATION_QUERY: User wants to understand why something happened.
    #                      RAG context enriches the root-cause narrative.
    _METRIC_INTENTS      = {"FINANCIAL_IMPACT", "DELAY_ANALYSIS", "SUPPLIER_RISK",
                            "INVENTORY_RISK"}
    _DECISION_INTENTS    = {"RECOMMENDATION"}
    # WHY BENCHMARK_COMPARISON is EXPLANATION_QUERY, not METRIC_QUERY?
    #   A benchmark comparison is not asking for a single factual number —
    #   it is asking for an interpretation of two numbers side-by-side
    #   (our OTD vs industry standard). That requires RAG context (the PDF
    #   benchmark figure) AND DB data (our actual figure). Classifying it
    #   as METRIC_QUERY would strip the RAG step and leave the executive
    #   agent with only our internal number and nothing to compare it to.
    _EXPLANATION_INTENTS = {"ROOT_CAUSE", "BENCHMARK_COMPARISON"}

    if primary_intent in _METRIC_INTENTS:
        query_type = "METRIC_QUERY"
    elif primary_intent in _DECISION_INTENTS:
        query_type = "DECISION_QUERY"
    else:
        query_type = "EXPLANATION_QUERY"

    # ── Stage 4c: Metric Disambiguation ──────────────────────────────────────
    # WHY negative-polarity words only ("worst", "bottom", "weakest")?
    #   Positive superlatives ("best", "highest") appear in queries that
    #   already name the metric explicitly — "highest delay rate" is
    #   unambiguous. Negative-polarity words in supplier queries are the
    #   only genuinely ambiguous case: "which supplier is worst?" leaves the
    #   metric undefined. We resolve it here so the Executive Agent receives
    #   a concrete column definition rather than guessing.
    # WHY only fire when there is NO explicit metric noun in the query?
    #   If the user already said "worst delay rate" or "weakest ROI", the
    #   metric is stated — disambiguation would be redundant and potentially
    #   wrong. The has_explicit_metric guard prevents overwriting a clear intent.
    ranking_words = ["worst", "bottom", "weakest"]
    metric_nouns = [
        "delay rate", "delay", "otd", "cost", "spend", "sla",
        "on-time", "financial", "roi", "savings",
        "penalty", "inventory", "stockout"
    ]

    query_lower = query.lower()
    is_supplier_query  = "supplier" in query_lower or "vendor" in query_lower
    has_ranking_word   = any(w in query_lower for w in ranking_words)
    has_explicit_metric = any(m in query_lower for m in metric_nouns)

    if is_supplier_query and has_ranking_word and not has_explicit_metric:
        detected_ranking_word = next(
            (w for w in ranking_words if w in query_lower),
            "worst"
        )
        metric_definition = (
            f"Defining '{detected_ranking_word} supplier' as the highest "
            f"delay rate based on shipment performance data."
        )
    else:
        metric_definition = ""

    log.info(
        f"create_plan | query_type='{query_type}' | "
        f"metric_definition='{metric_definition[:60] if metric_definition else 'none'}'"
    )

    # ── Stage 4d1: Metric and Dimension Detection (FIX 1+2) ──────────────────
    # WHY detect metric and dimension explicitly?
    #   Keyword routing in db_agent._TASK_ROUTING is powerful but cannot
    #   distinguish "highest delay rate by supplier" from "highest delay rate
    #   by region" when both come from the same plan template step.
    #   By detecting (metric, dimension, polarity) here we can override the
    #   step task string to a precise routing target that maps 1:1 to a
    #   specific SQL template — no ambiguity, no wrong template selected.
    #
    # WHY polarity (highest vs lowest)?
    #   "Which supplier is worst?" vs "Which supplier is best?" need opposite
    #   ORDER BY directions. Same metric, same dimension, different templates.
    #   Polarity detection resolves the ambiguity before the DB step runs.

    _POLARITY_HIGHEST = ["highest", "worst", "most", "top", "maximum", "max"]
    _POLARITY_LOWEST  = ["lowest", "best", "least", "minimum", "min",
                         "fewest", "bottom"]

    _METRIC_TRIGGERS: dict[str, list[str]] = {
        "delay_rate":    ["delay rate", "delay percentage"],
        "otd":           ["otd", "on-time delivery", "on time delivery",
                          "on-time rate", "on time rate"],
        "delayed_count": ["delayed shipments", "delayed count"],
        "total_count":   ["total shipments", "total number of shipments",
                          "number of shipments", "how many shipments"],
        # expedited_cost MUST come before "cost" — the loop breaks on first match,
        # so "expedited shipping cost" would otherwise match the generic "cost" entry
        # and STAGE 5c would route to total_supply_chain_cost instead.
        "expedited_cost": ["expedited cost", "expedited shipping",
                           "expedited ship", "expedited freight"],
        "cost":           ["avoidable cost", "supply chain cost", "total cost",
                           "cost", "spend"],
        "roi":            ["roi", "return on investment"],
    }

    _DIMENSION_TRIGGERS: dict[str, list[str]] = {
        "supplier":         ["supplier", "vendor",
                             "sup001", "sup002", "sup003"],
        "region":           ["region"],
        "product_category": ["product category", "category", "product"],
    }

    detected_metric    = ""
    detected_dimension = "fleet"   # default: no dimension filter
    detected_polarity  = "highest" # default: highest/worst

    for metric_key, triggers in _METRIC_TRIGGERS.items():
        if any(t in query_lower for t in triggers):
            detected_metric = metric_key
            break

    for dim_key, triggers in _DIMENSION_TRIGGERS.items():
        if any(t in query_lower for t in triggers):
            detected_dimension = dim_key
            break

    if any(w in query_lower for w in _POLARITY_LOWEST):
        detected_polarity = "lowest"
    elif any(w in query_lower for w in _POLARITY_HIGHEST):
        detected_polarity = "highest"

    log.info(
        f"create_plan | detected_metric='{detected_metric}' | "
        f"detected_dimension='{detected_dimension}' | "
        f"detected_polarity='{detected_polarity}'"
    )

    # ── Stage 4d: Count Query Detection ──────────────────────────────────────
    # WHY detect count queries before building the plan?
    #   "How many shipments are delayed?" is a scalar count question.
    #   The full DELAY_ANALYSIS template (6 steps: supplier breakdown, trend,
    #   reasons, RAG, ROI, executive) over-answers it — the user wants one
    #   number, not a root-cause investigation. We detect count queries here
    #   and replace the plan with a minimal 2-step version that returns exactly
    #   the right answer ("There are 15 delayed shipments.").
    #
    # WHY "how many" as the trigger phrase?
    #   It is the canonical English form for a count question. Related forms
    #   ("how many", "total number of", "count of") are covered below.
    #   Bare "delayed" alone does not trigger count mode — that is a normal
    #   DELAY_ANALYSIS question ("why are shipments delayed?").
    #
    # WHY also detect month/period mentions?
    #   "How many shipments were delayed in December 2024?" is a count question
    #   with a time scope. We extract the YYYY-MM string and pass it as a SQL
    #   parameter via step["params"] — no SQL generated at runtime.

    _COUNT_TRIGGERS = ["how many", "total number of", "count of", "number of delayed"]
    is_count_query  = any(t in query_lower for t in _COUNT_TRIGGERS)

    # Does the count question specifically ask about DELAYED shipments?
    # WHY this distinction?
    #   "How many shipments are delayed?" → total_delayed_count template.
    #   "What is the total number of shipments?" → total_shipments template.
    #   Without this check both would use total_delayed_count and return
    #   a misleadingly low number for the second question.
    _DELAY_TERMS    = ["delay", "delayed", "late", "overdue", "behind schedule"]
    is_delayed_count = any(t in query_lower for t in _DELAY_TERMS)

    # Month-name → month-number mapping for extraction
    _MONTH_MAP = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    # Time-period words that indicate the user wants a filtered (not fleet-wide) count
    _TIME_WORDS = [
        "last month", "this month", "last week", "this week",
        "in january", "in february", "in march", "in april", "in may",
        "in june", "in july", "in august", "in september", "in october",
        "in november", "in december",
        "january", "february", "march", "april",
        "june", "july", "august", "september", "october", "november", "december",
    ]
    has_time_period = any(tw in query_lower for tw in _TIME_WORDS)

    # Extract "YYYY-MM" from queries like "in December 2024"
    period_param: Optional[str] = None
    if has_time_period:
        import re as _re_month
        # Match "month_name YYYY" e.g. "December 2024"
        m = _re_month.search(
            r'(january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+(\d{4})',
            query_lower,
        )
        if m:
            month_num = _MONTH_MAP.get(m.group(1), "01")
            year_str  = m.group(2)
            period_param = f"{year_str}-{month_num}"

    # ── Stage 5: Primary Plan Template ───────────────────────────────────────
    steps = get_plan_template(primary_intent)

    # ── Stage 5a: Override plan for count queries ────────────────────────────
    # WHY override instead of adding a new intent?
    #   Count questions ("how many shipments are delayed?") match DELAY_ANALYSIS
    #   intent correctly — they ARE delay analysis questions. We don't need a
    #   new intent; we just need a simpler plan. Overriding here keeps the intent
    #   classification clean and avoids polluting the keyword lists.
    #
    # SLA BREACH EARLY-EXIT: "How many total SLA breaches do we have?" is a count
    # query that also triggers DELAY_ANALYSIS, but it MUST route to the
    # total_sla_breaches SQL template (not total_shipments). We check for SLA
    # breach terms BEFORE the generic count-query fork so it gets its own 1-step
    # plan instead of being swallowed by the SIMPLE_COUNT path.
    _SLA_BREACH_TERMS = [
        "sla breach", "sla breaches", "breach count", "total breaches",
        "total sla", "how many sla", "number of sla",
    ]
    _is_sla_breach_query = any(t in query_lower for t in _SLA_BREACH_TERMS)
    if _is_sla_breach_query:
        steps = [
            {
                "step_number":             1,
                "agent":                   "DB Agent",
                "task":                    "total_sla_breaches",
                "table":                   "shipments",
                "requires_human_approval": False,
            },
            {
                "step_number":             2,
                "agent":                   "Executive Agent",
                "task":                    "State the total SLA breach count in one sentence",
                "table":                   None,
                "requires_human_approval": False,
            },
        ]
        metric_definition = "SIMPLE_COUNT"
        log.info("create_plan | SLA BREACH QUERY — using total_sla_breaches plan")
    elif is_count_query and primary_intent == "DELAY_ANALYSIS":
        if has_time_period and period_param and is_delayed_count:
            # Month-specific delayed count: use parameterised template
            steps = [
                {
                    "step_number":             1,
                    "agent":                   "DB Agent",
                    "task":                    "delayed count by month",
                    "table":                   "shipments",
                    "requires_human_approval": False,
                    "params":                  (period_param,),
                },
                {
                    "step_number":             2,
                    "agent":                   "Executive Agent",
                    "task":                    "State the delayed shipment count for the requested month",
                    "table":                   None,
                    "requires_human_approval": False,
                },
            ]
            # Set metric_definition to signal 1-sentence response
            metric_definition = "SIMPLE_COUNT"
            log.info(
                f"create_plan | COUNT QUERY (month={period_param}) — "
                f"using delayed_count_by_month plan"
            )
        elif is_delayed_count:
            # Fleet-wide delayed count: no params
            steps = [
                {
                    "step_number":             1,
                    "agent":                   "DB Agent",
                    "task":                    "total delayed count fleet-wide",
                    "table":                   "shipments",
                    "requires_human_approval": False,
                },
                {
                    "step_number":             2,
                    "agent":                   "Executive Agent",
                    "task":                    "State the total delayed shipment count in one sentence",
                    "table":                   None,
                    "requires_human_approval": False,
                },
            ]
            metric_definition = "SIMPLE_COUNT"
            log.info(
                "create_plan | COUNT QUERY (delayed fleet-wide) — "
                "using total_delayed_count plan"
            )
        else:
            # Total shipments (not specifically delayed) — e.g. "total number of shipments"
            steps = [
                {
                    "step_number":             1,
                    "agent":                   "DB Agent",
                    "task":                    "total shipments fleet-wide",
                    "table":                   "shipments",
                    "requires_human_approval": False,
                },
                {
                    "step_number":             2,
                    "agent":                   "Executive Agent",
                    "task":                    "State the total shipment count in one sentence",
                    "table":                   None,
                    "requires_human_approval": False,
                },
            ]
            metric_definition = "SIMPLE_COUNT"
            log.info(
                "create_plan | COUNT QUERY (total shipments fleet-wide) — "
                "using total_shipments plan"
            )

    # ── Stage 5b: Strip RAG Agent from METRIC_QUERY plans ────────────────────
    # WHY? For factual metric questions ("which supplier has the highest delay
    # rate?"), the PDF Annual Report adds no value and can dilute the answer
    # with qualitative narrative. DB numbers are the ground truth here.
    if query_type == "METRIC_QUERY":
        rag_removed = [s for s in steps if s["agent"] == "RAG Agent"]
        steps = [s for s in steps if s["agent"] != "RAG Agent"]
        if rag_removed:
            log.info(
                f"create_plan | METRIC_QUERY — removed {len(rag_removed)} "
                f"RAG Agent step(s) from plan"
            )

    # ── Stage 5c: Override DB Step Task based on (metric, dimension, polarity) ─
    # WHY override here, not in get_plan_template()?
    #   Plan templates are static lookups defined at startup — they cannot
    #   know the user's specific metric or dimension at definition time.
    #   This stage applies runtime routing: the first DB Agent step's task
    #   string is replaced with a precise routing target that maps 1:1 to a
    #   SQL template in db_agent._TASK_ROUTING.
    # WHY only the first DB Agent step?
    #   The first step is always the primary data fetch. Later DB steps
    #   (trend, reasons) are supplementary and use their own fixed tasks.
    # WHY not override count-query plans?
    #   Count query plans (Stage 5a) already set exact tasks — no override needed.
    # WHY use exact SQL template key names (e.g. "lowest_delay_rate_supplier")?
    #   In db_agent.get_sql_template(), if the task string IS a key in
    #   _SQL_TEMPLATES, it is used directly without going through the keyword
    #   routing heuristic. This avoids score collisions where a broader template
    #   (e.g. "delay_count_by_supplier") outscores a specific one. Using exact
    #   template key names is unambiguous and deterministic.
    _METRIC_DIM_TASK_MAP: dict[tuple, str] = {
        # (detected_metric, detected_dimension, detected_polarity) → template key
        ("delay_rate", "supplier",         "highest"): "delay_count_by_supplier",
        ("delay_rate", "supplier",         "lowest"):  "lowest_delay_rate_supplier",
        ("delay_rate", "region",           "highest"): "highest_delay_rate_region",
        ("delay_rate", "region",           "lowest"):  "highest_delay_rate_region",
        ("delay_rate", "product_category", "highest"): "highest_delay_rate_product_category",
        ("delay_rate", "product_category", "lowest"):  "highest_delay_rate_product_category",
        ("delay_rate", "fleet",            "highest"): "delay_count_by_supplier",
        ("total_count", "fleet",           "highest"): "total_shipments",
        ("total_count", "supplier",        "highest"): "total_shipments",
        ("otd",            "fleet",           "highest"): "fleet_otd_vs_benchmark",
        ("cost",           "fleet",           "highest"): "total_supply_chain_cost",
        ("roi",            "fleet",           "highest"): "roi_progression",
        ("expedited_cost", "fleet",           "highest"): "total_expedited_cost",
        ("expedited_cost", "supplier",        "highest"): "total_expedited_cost",
    }

    # "avoidable cost" always overrides to total_avoidable_cost regardless of polarity
    _avoidable_override = detected_metric == "cost" and "avoidable" in query_lower

    # Do NOT apply in count-query mode (Stage 5a already set exact tasks)
    _skip_5c = metric_definition == "SIMPLE_COUNT"

    if not _skip_5c and detected_metric:
        override_task = (
            "total_avoidable_cost"
            if _avoidable_override
            else _METRIC_DIM_TASK_MAP.get(
                (detected_metric, detected_dimension, detected_polarity)
            )
        )

        if override_task:
            # Override only the FIRST DB Agent step
            for _s in steps:
                if _s["agent"] == "DB Agent":
                    _old = _s["task"]
                    _s["task"] = override_task
                    log.info(
                        f"create_plan | STAGE 5c OVERRIDE | "
                        f"metric='{detected_metric}' dim='{detected_dimension}' "
                        f"polarity='{detected_polarity}' | "
                        f"'{_old[:60]}' → '{override_task}'"
                    )
                    break

    # ── Stage 5d: Comparison Query Override (NEW FIX 1) ──────────────────────
    # WHY a separate stage after 5c?
    #   "Compare delay rate across all suppliers" wants ALL suppliers ranked,
    #   not just the top/bottom one. Stage 5c only maps to single-result
    #   templates. Comparison queries need a dedicated template that returns
    #   all rows, formatted as a numbered list by the executive agent.
    # WHY check detected_dimension == "supplier"?
    #   Comparison makes most sense at supplier level in the current data model.
    #   Region / product category comparisons use the multi-row results that
    #   highest_delay_rate_region already returns.
    _COMPARISON_KEYWORDS = [
        "compare", "comparison", "across all", "all suppliers",
        "each supplier", "supplier breakdown", "supplier by supplier",
    ]
    is_comparison = any(kw in query_lower for kw in _COMPARISON_KEYWORDS)
    multi_row     = False

    if is_comparison and detected_dimension == "supplier" and not _skip_5c:
        for _s in steps:
            if _s["agent"] == "DB Agent":
                _old_task = _s["task"]
                _s["task"] = "supplier_delay_comparison"
                log.info(
                    f"create_plan | STAGE 5d COMPARISON OVERRIDE | "
                    f"'{_old_task}' → 'supplier_delay_comparison' | multi_row=True"
                )
                break
        multi_row = True

    # ── Stage 6: Merge Secondary Intent Steps ────────────────────────────────
    # WHY add secondary steps instead of replacing?
    #   A question like "why is SupplierA causing delays and what is the cost?"
    #   has primary=ROOT_CAUSE and secondary=FINANCIAL_IMPACT. The financial
    #   steps enrich the primary plan — they don't replace it.
    # WHY deduplicate by agent+task?
    #   Some templates share steps (e.g. both DELAY and FINANCIAL have a
    #   ROI Agent step). Merging without deduplication would run the same
    #   query twice and confuse the executive summary.
    # WHY skip secondary merge for BENCHMARK_COMPARISON?
    #   The benchmark template is self-contained: DB (our KPIs) → RAG (PDF
    #   benchmark) → Executive. Adding DELAY_ANALYSIS secondary steps via
    #   "otd" keyword match floods the plan with 4 extra steps that produce
    #   delay rate findings — not the OTD comparison the user asked for.
    #   The executive receives the wrong data and cannot construct a
    #   meaningful industry comparison. For BENCHMARK_COMPARISON the
    #   primary template is always sufficient.
    # SLA breach queries must also skip the secondary merge — the SLA count is
    # a single fleet-wide number. Merging SUPPLIER_RISK steps gives the LLM
    # supplier-level figures it then uses to hallucinate an incorrect total.
    _NO_SECONDARY_MERGE = {"BENCHMARK_COMPARISON"}
    if _is_sla_breach_query:
        _NO_SECONDARY_MERGE = _NO_SECONDARY_MERGE | {primary_intent}

    if secondary_intent and secondary_intent != primary_intent \
            and primary_intent not in _NO_SECONDARY_MERGE:
        secondary_steps = get_plan_template(secondary_intent)
        existing_tasks  = {(s["agent"], s["task"]) for s in steps}

        # Insert secondary steps before the final Executive Agent step
        # WHY before executive? The Executive Agent always synthesises last.
        executive_steps  = [s for s in steps if s["agent"] == "Executive Agent"]
        non_exec_steps   = [s for s in steps if s["agent"] != "Executive Agent"]

        for sec_step in secondary_steps:
            key = (sec_step["agent"], sec_step["task"])
            if key not in existing_tasks and sec_step["agent"] != "Executive Agent":
                non_exec_steps.append(sec_step)
                existing_tasks.add(key)

        steps = non_exec_steps + executive_steps

        # Re-number steps sequentially after merge
        for i, step in enumerate(steps, start=1):
            step["step_number"] = i

        log.info(
            f"create_plan | MERGED secondary='{secondary_intent}' | "
            f"total steps after merge: {len(steps)}"
        )

    # ── Stage 7: Filter Steps by Role Permissions ────────────────────────────
    # WHY filter after template selection instead of before?
    #   We want to log which steps were removed for audit purposes.
    #   Filtering after also allows us to warn the user that some
    #   analysis was skipped, rather than silently returning a shorter plan.
    role_config       = ROLES.get(role, {})
    can_see_financials = role_config.get("can_see_financials", True)
    allowed_tables    = set(role_config.get("allowed_tables", []))

    filtered_steps: list[dict] = []
    skipped_steps:  list[str]  = []

    for step in steps:
        step_table = step.get("table")

        # Skip steps that touch blocked tables for this role
        if step_table and allowed_tables and step_table not in allowed_tables:
            skipped_steps.append(
                f"Step {step['step_number']} ({step['agent']}) — "
                f"table '{step_table}' not accessible for role '{role}'"
            )
            log.info(
                f"create_plan | STEP FILTERED | role='{role}' | "
                f"step={step['step_number']} | table='{step_table}'"
            )
            continue

        # Flag ROI Agent steps for non-financial roles
        # WHY flag rather than remove?
        #   The ROI Agent may still be able to return a partial result
        #   (e.g. shipment counts without cost figures). Flagging lets
        #   the ROI Agent decide at runtime what it can safely return.
        if step["agent"] == "ROI Agent" and not can_see_financials:
            step = {**step, "requires_human_approval": True}
            log.info(
                f"create_plan | ROI STEP FLAGGED for role='{role}' | "
                f"step={step['step_number']}"
            )

        filtered_steps.append(step)

    # Re-number after filtering
    for i, step in enumerate(filtered_steps, start=1):
        step["step_number"] = i

    if skipped_steps:
        log.info(
            f"create_plan | {len(skipped_steps)} step(s) filtered: "
            f"{skipped_steps}"
        )

    # ── Stage 8: Identify Human Checkpoints ──────────────────────────────────
    # WHY replace the confidence < 0.6 rule with ACTION_KEYWORDS?
    #   The old rule triggered human approval for every low-confidence query,
    #   including read-only analytical questions like "which supplier has the
    #   highest delay rate?" — these carry zero action risk.
    #   The new rule only triggers for DECISION_QUERY questions that contain
    #   specific irreversible action keywords (terminate, expedite, etc.).
    #   This eliminates false positives while preserving the safety gate for
    #   queries that genuinely require human sign-off.
    _ACTION_KEYWORDS = [
        "terminate", "change supplier", "expedite", "increase inventory",
        "stop", "cancel", "switch supplier", "switch", "place order",
        "reallocate", "reroute", "halt", "approve",
    ]
    # WHY also check for supplier IDs in DECISION_QUERY?
    #   "What should we do about SUP003?" contains no classic action keyword
    #   but is clearly an action-oriented decision query — the supplier ID IS
    #   the action target. Recognising supplier IDs as implicit action signals
    #   ensures human approval fires for these queries without needing the
    #   user to phrase them with explicit verbs like "terminate" or "switch".
    _SUPPLIER_IDS  = ["sup001", "sup002", "sup003"]
    q_lower_action = query.lower()
    is_action_decision = (
        query_type == "DECISION_QUERY"
        and (
            any(kw  in q_lower_action for kw  in _ACTION_KEYWORDS)
            or  any(sid in q_lower_action for sid in _SUPPLIER_IDS)
        )
    )

    if not is_action_decision:
        # Strip HUMAN CHECKPOINT agent steps (from RECOMMENDATION template)
        # and clear any requires_human_approval flags set by role filtering.
        # WHY strip rather than keep? Read-only metric/explanation queries
        # do not benefit from approval gates — they slow the pipeline for
        # zero safety gain.
        filtered_steps = [
            s for s in filtered_steps if s["agent"] != "HUMAN CHECKPOINT"
        ]
        for step in filtered_steps:
            step["requires_human_approval"] = False
        log.info(
            f"create_plan | No action decision detected — "
            f"HUMAN CHECKPOINT steps stripped (query_type='{query_type}')"
        )
    else:
        log.warning(
            f"create_plan | ACTION DECISION detected — "
            f"human checkpoints retained (query_type='{query_type}')"
        )

    # Re-number after potential HUMAN CHECKPOINT removal
    for i, step in enumerate(filtered_steps, start=1):
        step["step_number"] = i

    human_checkpoints = [
        s["step_number"]
        for s in filtered_steps
        if s.get("requires_human_approval")
    ]

    # ── Stage 9: Count LLM calls ──────────────────────────────────────────────
    # WHY track this?
    #   Displayed in the UI so the user understands the system's cost.
    #   It builds trust and reinforces the "minimal LLM" architecture.
    #   1 call = Executive Agent only. RAG Agent also uses 1 embedding
    #   call (not a text generation call), excluded from this count.
    estimated_llm_calls = sum(
        1 for s in filtered_steps if s["agent"] == "Executive Agent"
    )

    # ── Stage 10: Log and Return ──────────────────────────────────────────────
    plan = {
        "status":             "ok",
        "query":              query,
        "role":               role,
        "intent":             primary_intent,
        "secondary_intent":   secondary_intent,
        "query_type":         query_type,
        "metric_definition":  metric_definition,
        "detected_metric":    detected_metric,
        "detected_dimension": detected_dimension,
        "detected_polarity":  detected_polarity,
        "multi_row":          multi_row,
        "confidence":         confidence,
        "keywords_matched":   intent_result["keywords_matched"],
        "steps":              filtered_steps,
        "total_steps":        len(filtered_steps),
        "estimated_llm_calls": estimated_llm_calls,
        "human_checkpoints":  sorted(human_checkpoints),
        "skipped_steps":      skipped_steps,
        "created_at":         datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        f"create_plan | COMPLETE | "
        f"intent='{primary_intent}' | "
        f"steps={len(filtered_steps)} | "
        f"checkpoints={human_checkpoints} | "
        f"llm_calls={estimated_llm_calls} | "
        f"confidence={confidence}"
    )

    return plan


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — EXPLAINABILITY
#  Converts the machine-readable plan into a human-readable preview.
#
#  WHY show the plan to the user before executing?
#  1. Transparency — the user sees exactly what data will be accessed.
#  2. Trust        — no black box. The user can spot if the intent was
#                    misclassified before any DB calls are made.
#  3. Consent      — especially important for HUMAN CHECKPOINT steps,
#                    where the user needs to know they will be asked to
#                    approve an action.
# ═══════════════════════════════════════════════════════════════════════════════

# Agent name → short display label for the explanation string
_AGENT_LABELS: dict[str, str] = {
    "DB Agent":          "Query the shipments database",
    "RAG Agent":         "Search the knowledge base",
    "ROI Agent":         "Calculate financial impact",
    "Executive Agent":   "Format findings for your role",
    "HUMAN CHECKPOINT":  "⚠️  PAUSE — human approval required",
}

# Intent → brief plain-English description shown at the top of the explanation
_INTENT_DESCRIPTIONS: dict[str, str] = {
    "DELAY_ANALYSIS":        "investigating delivery delays",
    "SUPPLIER_RISK":         "assessing supplier risk",
    "INVENTORY_RISK":        "analysing inventory levels",
    "FINANCIAL_IMPACT":      "calculating financial impact",
    "BENCHMARK_COMPARISON":  "comparing performance to industry benchmarks",
    "ROOT_CAUSE":            "identifying root causes",
    "RECOMMENDATION":        "generating an action recommendation",
}


def explain_plan(plan: dict) -> str:
    """
    Return a human-readable preview of what the system is about to do.

    Shown in the UI BEFORE execution so the user can understand and
    optionally cancel before any data is accessed.

    Args:
        plan: Output of create_plan().

    Returns:
        Multi-line string formatted for display in the Streamlit UI.

    Example output:
        "I will investigate this question by investigating delivery delays.
         This will take 6 steps with 1 LLM call (for final formatting only).

         Step 1: Query the shipments database
                 → Get delay count and delay rate by supplier and region

         Step 2: Query the shipments database
                 → Get delay trend by month (last 24 months)

         Step 3: Query the shipments database
                 → Get top delay reasons ranked by frequency

         Step 4: Search the knowledge base
                 → Search for relevant SOP or escalation guideline

         Step 5: Calculate financial impact
                 → Calculate cost impact of delays (penalties + expedited costs)

         ⚠️  Step 6 requires human approval before proceeding.

         Step 6: Format findings for your role
                 → Format and narrate findings for the user's role"
    """
    if plan.get("status") != "ok":
        # Return a clear blocked message instead of a plan
        return (
            f"This query could not be planned.\n"
            f"Reason: {plan.get('reason', 'Unknown error.')}\n"
            f"Status: {plan.get('status', 'error').replace('_', ' ').title()}"
        )

    intent      = plan.get("intent", "")
    steps       = plan.get("steps", [])
    total       = plan.get("total_steps", len(steps))
    llm_calls   = plan.get("estimated_llm_calls", 1)
    checkpoints = set(plan.get("human_checkpoints", []))
    intent_desc = _INTENT_DESCRIPTIONS.get(intent, intent.lower().replace("_", " "))

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(
        f"I will answer your question by {intent_desc}."
    )
    lines.append(
        f"This will take {total} step{'s' if total != 1 else ''} "
        f"with {llm_calls} LLM call{'s' if llm_calls != 1 else ''} "
        f"(used only for final answer formatting)."
    )

    if checkpoints:
        checkpoint_steps = ", ".join(f"Step {n}" for n in sorted(checkpoints))
        lines.append(
            f"⚠️  Human approval required at: {checkpoint_steps}"
        )

    lines.append("")

    # ── Step-by-step breakdown ────────────────────────────────────────────────
    for step in steps:
        n           = step["step_number"]
        agent       = step["agent"]
        task        = step["task"]
        label       = _AGENT_LABELS.get(agent, agent)
        needs_approval = step.get("requires_human_approval", False)

        if needs_approval and agent != "HUMAN CHECKPOINT":
            lines.append(f"  ⚠️  Step {n} requires human approval before proceeding.")

        lines.append(f"  Step {n}: {label}")
        lines.append(f"          → {task}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    skipped = plan.get("skipped_steps", [])
    if skipped:
        lines.append(
            f"  ℹ️  {len(skipped)} step(s) were skipped — "
            f"data not accessible at your current role level."
        )

    confidence = plan.get("confidence", 1.0)
    if confidence < CONFIDENCE_THRESHOLD:
        lines.append(
            f"  ⚠️  Intent confidence is {confidence:.0%} — "
            f"the system is not fully certain this plan matches your question. "
            f"Human review of the final answer is recommended."
        )

    return "\n".join(lines)
