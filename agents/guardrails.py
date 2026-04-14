"""
Supply Command AI — Guardrails Layer
Ethics, safety, and access control for every agent request and response.

WHY this file exists as a separate layer:
    Agents focus on answering questions. Guardrails focus on whether they
    *should* answer, and whether the answer is *safe to show*. Keeping
    them separate means we can tighten security rules without touching
    agent logic, and vice versa.

Three responsibilities (in order of execution):
    1. INPUT  — validate role, check access, detect prompt injection
               (runs BEFORE the agent touches the database)
    2. OUTPUT — validate results, mask restricted data, check confidence
               (runs AFTER the agent returns, BEFORE the UI renders)
    3. AUDIT  — write every guardrail event to rbac_audit_log
               (runs whenever INPUT blocks or OUTPUT modifies)

LLM Usage:
    NONE — every decision in this file is a coded rule.
    WHY? A guardrail that can be talked out of blocking something by the
    LLM it is supposed to guard is not a guardrail — it is a suggestion.
"""

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from services.logger import get_logger

log = get_logger("guardrails")

# ── DB path for direct audit writes ───────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "database" / "supply_chain.db"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — RBAC CONFIGURATION
#  Source of truth for every role's permissions in the system.
#  All other guardrail functions read from this dict — never hard-code
#  permissions inside functions.
# ═══════════════════════════════════════════════════════════════════════════════

ROLES: dict[str, dict] = {

    "Demand Planner": {
        # WHY only shipments + suppliers_master?
        #   Demand Planners need to track deliveries and evaluate supplier
        #   reliability. They have no business need for financial data and
        #   exposing it creates regulatory and competitive risk.
        "allowed_tables":    ["shipments", "suppliers_master"],
        "blocked_tables":    ["financial_impact"],
        "blocked_keywords":  [
            "cost", "spend", "revenue", "roi", "profit",
            "salary", "contract value", "annual spend",
            # FIX 2: Expanded financial keywords so "financial impact of delays"
            # and similar queries are caught before the planning agent runs.
            "financial", "impact", "penalty", "avoidable",
            "expenditure", "saving",
        ],
        "can_see_financials": False,
        # WHY 50 rows?
        #   Enough for an operational view; prevents bulk data exports
        #   that could circumvent data governance policies.
        "max_rows_returned":  50,
        "persona":           "operational",
    },

    "Operations Manager": {
        # WHY full table access but no keyword blocks?
        #   Ops Managers make cross-functional decisions — they need cost
        #   data to prioritise expedites. Blocking keywords here would
        #   make the tool useless for their actual job.
        "allowed_tables":    ["shipments", "suppliers_master", "financial_impact"],
        "blocked_tables":    [],
        "blocked_keywords":  [],
        "can_see_financials": True,
        # WHY 100 rows?
        #   Managers need broader views for trend analysis, but still
        #   not unlimited — prevents accidental full-table dumps in the UI.
        "max_rows_returned":  100,
        "persona":           "operational",
    },

    "CFO": {
        # WHY no restrictions?
        #   The CFO owns the P&L and AI investment portfolio. They need
        #   visibility into everything to make strategic decisions.
        #   Restricting the CFO would undermine the system's value.
        "allowed_tables":    ["financial_impact", "suppliers_master", "shipments"],
        "blocked_tables":    [],
        "blocked_keywords":  [],
        "can_see_financials": True,
        # WHY 200 rows?
        #   CFO may pull full financial history for board reporting.
        #   We still cap at 200 to keep the UI responsive.
        "max_rows_returned":  200,
        "persona":           "financial",
    },
}

# ── Column names that contain financial / sensitive data ──────────────────────
# Used by validate_output() to mask values for roles with can_see_financials=False.
# WHY a list and not a regex? Lists are faster to maintain and audit.
FINANCIAL_COLUMN_PATTERNS: list[str] = [
    "cost", "spend", "revenue", "roi", "profit", "salary",
    "value", "savings", "investment", "penalty", "insurance",
    "freight", "stockout", "avoidable", "usd", "amount", "price",
]

# ── Prompt injection trigger phrases ─────────────────────────────────────────
# Pre-compiled regex for speed — checked on every user input.
_INJECTION_PHRASES: list[str] = [
    r"ignore\s+(previous|prior|all)\s+instructions?",
    r"you\s+are\s+now",
    r"pretend\s+you\s+are",
    r"forget\s+your\s+rules?",
    r"act\s+as",
    r"disregard\s+(all|your|the)\s+",
    r"override\s+(your|the)\s+(rules?|instructions?|guidelines?)",
    r"new\s+persona",
    r"jailbreak",
    r"dan\s+mode",
]
_INJECTION_PATTERN = re.compile(
    "|".join(_INJECTION_PHRASES),
    re.IGNORECASE | re.DOTALL,
)

# WHY 500 chars as the secondary injection signal?
#   Legitimate supply chain questions are short and specific.
#   A 500+ char query is statistically unusual and warrants extra scrutiny.
#   Combined with ANY of the behavioral keywords above, it is blocked.
INJECTION_LENGTH_THRESHOLD = 500

# ── Human-in-the-loop thresholds ─────────────────────────────────────────────
# WHY $50,000?
#   GlobalMedTech's procurement policy requires manager sign-off on any
#   single decision affecting more than $50K. This mirrors that policy.
FINANCIAL_IMPACT_THRESHOLD = 50_000

# WHY 0.6?
#   Below 60% confidence, the system is essentially guessing. A human
#   must review before any action is taken based on that guess.
CONFIDENCE_THRESHOLD = 0.6

# WHY 10 shipments?
#   A single mis-routed recommendation affecting 10+ shipments could
#   create a cascade failure across multiple hospital sites.
SHIPMENT_COUNT_THRESHOLD = 10

# High-risk action words that could cause irreversible supply chain events
HIGH_RISK_ACTIONS: list[str] = ["expedite", "cancel", "terminate"]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — INPUT GUARDRAIL FUNCTIONS
#  Called BEFORE any agent touches the database.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_role(role_name: str) -> dict:
    """
    Confirm the role_name exists in the ROLES configuration.

    WHY check this first?
        An unknown role means we cannot apply any access rules — the
        safest response is to block everything rather than guess at
        permissions. Fail closed, never fail open.

    Args:
        role_name: Display name from the UI (e.g. "Demand Planner").

    Returns:
        {"valid": True,  "reason": "Role recognised."}
        {"valid": False, "reason": "Unknown role: '<name>'. ..."}
    """
    if role_name in ROLES:
        log.debug(f"validate_role | PASS | role='{role_name}'")
        return {"valid": True, "reason": "Role recognised."}

    reason = (
        f"Unknown role: '{role_name}'. "
        f"Valid roles: {list(ROLES.keys())}."
    )
    log.warning(f"validate_role | FAIL | {reason}")
    return {"valid": False, "reason": reason}


def check_query_access(query: str, role_name: str) -> dict:
    """
    Verify a natural-language query does not violate the role's access rules.

    Two checks (both must pass):
        1. Keyword check  — query must not contain any blocked_keywords
        2. Table check    — query must not explicitly name a blocked_table

    WHY keyword check on natural language, not just on generated SQL?
        The SQL guardrail in db_connection.py handles SQL-level protection.
        This check happens earlier — at the question stage — so we can
        give the user a clear "you don't have access to financial data"
        message before wasting time generating SQL that would be blocked.

    WHY table name check in natural language?
        If a user types "show me financial_impact data", we should catch
        that immediately rather than letting it reach SQL generation.

    Args:
        query:     Raw natural-language question from the user.
        role_name: The user's current role.

    Returns:
        {"allowed": True,  "reason": "...", "blocked_term": None}
        {"allowed": False, "reason": "...", "blocked_term": "<term>"}
    """
    role_check = validate_role(role_name)
    if not role_check["valid"]:
        return {
            "allowed":      False,
            "reason":       role_check["reason"],
            "blocked_term": None,
        }

    role      = ROLES[role_name]
    q_lower   = query.lower()

    # ── Check 1: Blocked keywords ─────────────────────────────────────────────
    for keyword in role.get("blocked_keywords", []):
        if keyword.lower() in q_lower:
            # FIX 6: Specific message for financial data blocks so the UI
            # shows the exact prescribed message rather than a generic one.
            # WHY check can_see_financials instead of matching role name?
            #   Future-proof: if a new role without financial access is added,
            #   it automatically gets the same clear message without code changes.
            if not role.get("can_see_financials", True):
                reason = (
                    "Access restricted. Financial data requires "
                    "Operations Manager or CFO role."
                )
            else:
                reason = (
                    f"Your role ('{role_name}') does not have access to "
                    f"'{keyword}' information."
                )
            log.warning(
                f"check_query_access | BLOCKED | role='{role_name}' | "
                f"keyword='{keyword}' | query='{query[:80]}'"
            )
            return {
                "allowed":      False,
                "reason":       reason,
                "blocked_term": keyword,
            }

    # ── Check 2: Blocked table names ─────────────────────────────────────────
    for table in role.get("blocked_tables", []):
        if table.lower() in q_lower:
            # FIX 6: Consistent financial restriction message for blocked tables
            if not role.get("can_see_financials", True):
                reason = (
                    "Access restricted. Financial data requires "
                    "Operations Manager or CFO role."
                )
            else:
                reason = (
                    f"Your role ('{role_name}') does not have access to "
                    f"the '{table}' table."
                )
            log.warning(
                f"check_query_access | BLOCKED | role='{role_name}' | "
                f"table='{table}' | query='{query[:80]}'"
            )
            return {
                "allowed":      False,
                "reason":       reason,
                "blocked_term": table,
            }

    log.debug(
        f"check_query_access | PASS | role='{role_name}' | "
        f"query='{query[:80]}'"
    )
    return {
        "allowed":      True,
        "reason":       "Query permitted for this role.",
        "blocked_term": None,
    }


def pre_planning_rbac_check(query: str, role_name: str) -> dict:
    """
    Fast RBAC keyword check that runs BEFORE the Planning Agent touches the DB.

    WHY a separate function from check_query_access()?
        check_query_access() is comprehensive — it validates the role, scans
        all blocked_keywords, and checks table names. This function is a thin
        wrapper that returns a simpler {"blocked": bool, "reason": str} dict
        matched to the shape expected by graph.py's input_guardrails_node.
        It makes the "Check 0" call site in the graph readable without
        duplicating the blocking logic.

    Args:
        query:     Raw user question.
        role_name: Authenticated role name.

    Returns:
        {"blocked": False, "reason": ""}                  — query is permitted
        {"blocked": True,  "reason": "<clean message>"}   — query is blocked;
            reason is the exact message to show the user (no "Access denied:" prefix).
    """
    access = check_query_access(query, role_name)
    if not access["allowed"]:
        return {"blocked": True, "reason": access["reason"]}
    return {"blocked": False, "reason": ""}


def detect_prompt_injection(query: str) -> dict:
    """
    Scan the query for adversarial patterns designed to manipulate agents.

    WHY do we need this?
        Supply Command AI is a multi-agent system where user input
        eventually reaches an LLM. A malicious user could craft a query
        that tricks the LLM into ignoring its system prompt, revealing
        data it shouldn't, or generating harmful recommendations.
        This check intercepts such attempts before they reach any agent.

    Patterns blocked:
        • Classic jailbreak phrases ("ignore previous instructions", etc.)
        • Role impersonation ("you are now", "pretend you are", "act as")
        • Rule removal attempts ("forget your rules", "override instructions")
        • Long queries (>500 chars) that also contain behavioral keywords
          — legitimate supply chain questions are concise.

    Args:
        query: Raw user input string.

    Returns:
        {"safe": True,  "reason": "No injection patterns detected."}
        {"safe": False, "reason": "<specific threat description>"}
    """
    if not query or not query.strip():
        return {"safe": False, "reason": "Empty query rejected."}

    q_lower = query.lower().strip()

    # ── Check 1: Known injection phrases ─────────────────────────────────────
    match = _INJECTION_PATTERN.search(q_lower)
    if match:
        detected = match.group(0).strip()
        reason = (
            f"Prompt injection detected: pattern '{detected}' "
            f"is not permitted in supply chain queries."
        )
        log.warning(
            f"detect_prompt_injection | INJECTION DETECTED | "
            f"pattern='{detected}' | query='{query[:80]}'"
        )
        return {"safe": False, "reason": reason}

    # ── Check 2: Suspiciously long query with behavioral language ─────────────
    # WHY length + keywords together, not length alone?
    #   A detailed operational question can legitimately be long.
    #   Length only becomes a signal when paired with system-manipulation
    #   language — then it suggests a sophisticated multi-step attack.
    if len(query) > INJECTION_LENGTH_THRESHOLD:
        behavioral_words = [
            "system", "prompt", "instruction", "rule", "persona",
            "assistant", "ai", "model", "gpt", "claude",
        ]
        if any(w in q_lower for w in behavioral_words):
            reason = (
                f"Query exceeds {INJECTION_LENGTH_THRESHOLD} characters "
                f"and contains system-behavioral language. "
                f"Please rephrase as a specific supply chain question."
            )
            log.warning(
                f"detect_prompt_injection | LONG+BEHAVIORAL | "
                f"length={len(query)} | query='{query[:80]}'"
            )
            return {"safe": False, "reason": reason}

    log.debug(
        f"detect_prompt_injection | SAFE | length={len(query)}"
    )
    return {"safe": True, "reason": "No injection patterns detected."}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — OUTPUT GUARDRAIL FUNCTIONS
#  Called AFTER the agent returns, BEFORE the UI renders the answer.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_output(result: dict, role_name: str) -> dict:
    """
    Inspect and potentially modify an agent result before it reaches the UI.

    Three output checks:
        1. Empty result   — flag if no data returned
        2. Low confidence — add warning if confidence_score < 0.6
        3. Financial mask — replace sensitive numeric values with [RESTRICTED]
                            for roles where can_see_financials = False

    WHY mask at output rather than filtering at query time?
        Sometimes a query returns a mixed result — e.g. a shipment record
        that incidentally includes cost columns. We cannot always predict
        this at query-build time. Output masking is the safety net.

    WHY mask numbers, not entire rows?
        Removing entire rows could change counts and mislead the user.
        Masking specific column values preserves the shape of the data
        while still protecting the sensitive figures.

    Args:
        result:    The dict returned by execute_query() or an agent.
                   Expected keys: success, data (list of dicts),
                   row_count, confidence_score (optional).
        role_name: The current user's role.

    Returns:
        {
            "approved":  bool,
            "result":    dict,          # possibly modified copy
            "warnings":  list[str],
        }
    """
    warnings: list[str] = []
    role       = ROLES.get(role_name, {})
    data       = result.get("data", [])
    confidence = result.get("confidence_score", 1.0)   # default 1.0 if not set

    # ── Check 1: Empty result ─────────────────────────────────────────────────
    if not result.get("success") or not data:
        warnings.append(
            "No data was found for this query. "
            "Try broadening your search criteria."
        )
        log.info(
            f"validate_output | EMPTY RESULT | role='{role_name}'"
        )

    # ── Check 2: Low confidence warning ──────────────────────────────────────
    # WHY warn rather than block?
    #   A low-confidence answer is still an answer — the user deserves to
    #   see it. But they need to know it may not be reliable so they can
    #   apply their own judgement before acting on it.
    if confidence < CONFIDENCE_THRESHOLD:
        warnings.append(
            f"⚠️ Confidence score is {confidence:.0%} — below the "
            f"{CONFIDENCE_THRESHOLD:.0%} threshold. "
            f"Treat this answer with caution and seek human verification."
        )
        log.warning(
            f"validate_output | LOW CONFIDENCE | "
            f"score={confidence:.2f} | role='{role_name}'"
        )

    # ── Check 3: Financial data masking ──────────────────────────────────────
    if not role.get("can_see_financials", True) and data:
        masked_data   = []
        masked_fields = set()

        for row in data:
            masked_row = {}
            for col, val in row.items():
                col_lower = col.lower()

                # Check if this column name matches any financial pattern
                is_financial = any(
                    pattern in col_lower
                    for pattern in FINANCIAL_COLUMN_PATTERNS
                )

                # Only mask actual numeric values — strings like "N/A"
                # don't need masking and masking them looks strange.
                if is_financial and isinstance(val, (int, float)):
                    masked_row[col] = "[RESTRICTED]"
                    masked_fields.add(col)
                else:
                    masked_row[col] = val

            masked_data.append(masked_row)

        if masked_fields:
            warnings.append(
                f"Some financial columns have been restricted "
                f"for your role ('{role_name}'): "
                f"{sorted(masked_fields)}."
            )
            log.info(
                f"validate_output | MASKED | role='{role_name}' | "
                f"fields={sorted(masked_fields)}"
            )
            result = {**result, "data": masked_data}

    # Enforce row cap for this role
    max_rows = role.get("max_rows_returned", 100)
    if len(result.get("data", [])) > max_rows:
        trimmed = result["data"][:max_rows]
        warnings.append(
            f"Result trimmed to {max_rows} rows (your role limit). "
            f"Use filters to narrow your query."
        )
        result = {**result, "data": trimmed, "row_count": max_rows}
        log.info(
            f"validate_output | ROW CAP | role='{role_name}' | "
            f"capped_at={max_rows}"
        )

    approved = result.get("success", False)

    return {
        "approved": approved,
        "result":   result,
        "warnings": warnings,
    }


def check_human_approval_needed(result: dict) -> dict:
    """
    Determine whether a human must review the result before any action is taken.

    WHY human-in-the-loop?
        Supply Command AI advises — it does not act autonomously.
        For high-impact decisions, a human must confirm before the
        recommended action is executed. This is a hard architectural
        principle, not a suggestion.

    Triggers (ANY ONE is sufficient to require approval):
        1. Financial impact > $50,000   — procurement policy threshold
        2. Recommendation contains      — these words imply irreversible
           "expedite", "cancel",          supply chain actions
           or "terminate"
        3. Confidence score < 0.6       — system is not certain enough
                                          to act without human review
        4. Result affects > 10          — cascade risk across multiple
           shipments                      hospital delivery sites

    Args:
        result: Agent result dict. Relevant keys:
                data (list of dicts), confidence_score, row_count,
                recommended_action (str, optional).

    Returns:
        {
            "needs_approval":  bool,
            "reason":          str,    # first trigger found, or "None"
            "impact_summary":  str,    # human-readable summary
        }
    """
    reasons: list[str]  = []
    impacts: list[str]  = []
    data                = result.get("data", [])
    confidence          = result.get("confidence_score", 1.0)
    row_count           = result.get("row_count", len(data))
    recommendation      = result.get("recommended_action", "") or ""

    # ── Trigger 1: Large financial impact — DISABLED ─────────────────────────
    # WHY disabled?
    #   The financial amount scan fires on EVERY query that returns cost data,
    #   including read-only analytical questions like "what is the total cost?"
    #   or "show the ROI progression". These queries carry zero execution risk —
    #   the user is asking for information, not authorising a spending decision.
    #   Triggering human approval on the dollar amount in the data conflates
    #   "the answer mentions a large number" with "a large action is being taken".
    #
    #   Human approval is now exclusively governed by Trigger 4 (action keyword
    #   + DECISION_QUERY in planning_agent) and Trigger 2 (irreversible action
    #   words in the recommendation field). Those two gates are sufficient to
    #   protect against autonomous high-impact decisions without creating
    #   approval fatigue on legitimate read-only financial queries.
    #
    # Retained for reference (do not delete — may be re-enabled with a
    # DECISION_QUERY guard if procurement policy requires it):
    #
    #   total_financial_exposure = 0.0
    #   for row in data:
    #       for col, val in row.items():
    #           if isinstance(val, (int, float)):
    #               col_lower = col.lower()
    #               if any(p in col_lower for p in FINANCIAL_COLUMN_PATTERNS):
    #                   total_financial_exposure += float(val)
    #   if total_financial_exposure > FINANCIAL_IMPACT_THRESHOLD:
    #       reasons.append(...)
    #       impacts.append(...)
    total_financial_exposure = 0.0   # calculated but not used as a trigger

    # ── Trigger 2: High-risk action keywords in recommendation ───────────────
    # WHY check the recommendation field AND the result data?
    #   The recommendation might come from the agent's suggested action,
    #   or it might appear in a "recommended_action" column in the DB.
    all_text = recommendation.lower()
    for row in data:
        for col, val in row.items():
            if "recommend" in col.lower() and isinstance(val, str):
                all_text += " " + val.lower()

    for action_word in HIGH_RISK_ACTIONS:
        if action_word in all_text:
            reasons.append(
                f"High-risk action '{action_word}' found in recommendation. "
                f"Irreversible actions require human sign-off."
            )
            impacts.append(f"Action: '{action_word}'")
            break   # one high-risk word is enough — no need to list all

    # ── Trigger 3: Low confidence ─────────────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Confidence score {confidence:.0%} is below the "
            f"{CONFIDENCE_THRESHOLD:.0%} minimum for autonomous action."
        )
        impacts.append(f"Confidence: {confidence:.0%}")

    # ── Trigger 4: High shipment count — ACTION queries only ─────────────────
    # WHY only trigger on action queries?
    #   A read-only analytical query like "which supplier has the highest
    #   delay rate?" may return many rows but carries zero execution risk —
    #   the user is asking for information, not authorising a change.
    #   Requiring human approval for every analytical query defeats the
    #   purpose of the system and creates alert fatigue.
    #   We only require approval when the query contains keywords that
    #   imply an irreversible supply chain action is being requested.
    _ACTION_KEYWORDS = [
        "expedite", "terminate", "cancel", "switch supplier",
        "pause", "halt", "approve", "execute", "place order",
        "reroute", "reallocate",
    ]
    query_text   = result.get("recommended_action", "") or ""
    query_lower  = query_text.lower()
    is_action_query = any(kw in query_lower for kw in _ACTION_KEYWORDS)

    if is_action_query and row_count > SHIPMENT_COUNT_THRESHOLD:
        reasons.append(
            f"Result affects {row_count} shipments "
            f"(threshold: {SHIPMENT_COUNT_THRESHOLD}). "
            f"Broad-impact decisions require human review."
        )
        impacts.append(f"Shipments affected: {row_count}")

    needs_approval = len(reasons) > 0

    if needs_approval:
        log.warning(
            f"check_human_approval_needed | APPROVAL REQUIRED | "
            f"triggers={len(reasons)} | reasons={reasons}"
        )
    else:
        log.debug("check_human_approval_needed | AUTO-APPROVED")

    return {
        "needs_approval": needs_approval,
        "reason":         reasons[0] if reasons else "No approval triggers met.",
        "impact_summary": " | ".join(impacts) if impacts else "Within safe thresholds.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — AUDIT FUNCTION
#  Every guardrail event is written to rbac_audit_log.
#  WHY? Compliance, forensics, and continuous improvement.
#  If the system blocks something it shouldn't (false positive), we can
#  trace exactly what triggered it. If a breach occurs, we have the audit trail.
# ═══════════════════════════════════════════════════════════════════════════════

# Valid event types for the audit log
EVENT_TYPES = {
    "INPUT_BLOCKED",            # query rejected before agents ran
    "OUTPUT_MASKED",            # result modified before UI rendered
    "HUMAN_APPROVAL_REQUIRED",  # result flagged for human sign-off
    "INJECTION_DETECTED",       # prompt injection attempt caught
}


def log_guardrail_event(
    event_type:  str,
    role:        str,
    query:       str,
    outcome:     str,
    reason:      str,
) -> bool:
    """
    Write a guardrail event to rbac_audit_log.

    WHY write directly to SQLite here rather than via execute_query()?
        execute_query() blocks all INSERT statements via validate_sql() —
        that is correct for agent-generated SQL. Audit writes are trusted
        internal operations that must bypass that guardrail. We use
        get_connection() directly but ONLY for the audit table.

    WHY not use log_agent_decision()?
        That function writes to ai_decisions_log (agent answers).
        This writes to rbac_audit_log (security events). They are
        separate concerns with separate schemas.

    Args:
        event_type: One of INPUT_BLOCKED / OUTPUT_MASKED /
                    HUMAN_APPROVAL_REQUIRED / INJECTION_DETECTED.
        role:       The role that triggered the event.
        query:      The original user query (may be truncated for storage).
        outcome:    "blocked", "masked", "flagged", "detected".
        reason:     Human-readable explanation of why this event fired.

    Returns:
        True on successful write, False on failure.
    """
    if event_type not in EVENT_TYPES:
        log.warning(
            f"log_guardrail_event | Unknown event_type='{event_type}' "
            f"— writing anyway with caution."
        )

    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))

        conn.execute(
            """
            INSERT INTO rbac_audit_log (
                timestamp,
                role_id,
                query_attempted,
                access_granted,
                blocked_reason,
                tables_requested
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                role,
                query[:500],           # cap at 500 chars to protect storage
                outcome,
                f"[{event_type}] {reason}",
                "",                    # populated by calling agent if known
            ),
        )
        conn.commit()

        log.info(
            f"log_guardrail_event | SAVED | event='{event_type}' | "
            f"role='{role}' | outcome='{outcome}'"
        )
        return True

    except sqlite3.Error as exc:
        # WHY not re-raise?
        #   A failed audit write must NEVER block the user from receiving
        #   their (already-validated) response. Security events are
        #   important but not more important than system availability.
        log.error(
            f"log_guardrail_event | FAILED to write audit record: {exc}"
        )
        return False

    finally:
        if conn:
            conn.close()
