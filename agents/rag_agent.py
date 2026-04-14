"""
Supply Command AI — RAG Agent
Searches the knowledge base (FAISS vector store) and returns
cited evidence to the Planning Agent.

Sources indexed:
    • sample_data/documents/  — GlobalMedTech Annual Report 2023 (PDF)
    • knowledge_base/         — Operational SOPs, decision frameworks (MD)

LLM Usage Policy (strictly enforced):
    NO   search_knowledge_base() — pure Python + FAISS lookup
    NO   format_rag_finding()    — pure Python string formatting
    NO   get_available_sections() — pure Python dict lookup
    NO   run()                   — orchestration only, no LLM
    YES  Only the Executive Agent (graph.py) may call LLM to format
         the final combined answer AFTER this agent returns findings.

Hallucination Prevention Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  If similarity < 0.75:                              │
    │    → return {"found": False}                        │
    │    → LLM is NEVER called with this query            │
    │    → LLM cannot hallucinate an answer               │
    │    → because we never ask it the question           │
    │                                                     │
    │  This is hallucination prevention at the            │
    │  ARCHITECTURE level — not the prompt level.         │
    │  A prompt instruction can be overridden.            │
    │  A code branch that never calls the LLM cannot.     │
    └─────────────────────────────────────────────────────┘
"""

import time
from pathlib import Path
from typing import Optional

from services.logger    import get_logger
from services.memory    import search as vector_search
from agents.guardrails  import (
    ROLES,
    validate_role,
    check_query_access,
    log_guardrail_event,
    FINANCIAL_COLUMN_PATTERNS,
)

log = get_logger("rag_agent")

# ── Similarity threshold (mirrors memory.py — kept here for readable logging) ─
SIMILARITY_THRESHOLD = 0.75

# ── Section-to-role access map ────────────────────────────────────────────────
# Defines which document sections each role is permitted to read.
# WHY map sections to roles here rather than in guardrails.py?
#   guardrails.py owns table-level RBAC for the database.
#   This file owns document-section-level RBAC for the knowledge base.
#   Keeping them co-located with their respective agents makes each
#   access rule easier to audit, test, and update independently.

# Sections that contain financial data — blocked for Demand Planner
FINANCIAL_SECTIONS: set[str] = {
    "FINANCIAL_PERFORMANCE",
    "FINANCIAL_IMPACT",
    "ROI_ANALYSIS",
    "COST_BREAKDOWN",
    "AI_INVESTMENT",
    "REVENUE",
    "BUDGET",
    "EXPENDITURE",
    "PROFIT_LOSS",
    "ANNUAL_SPEND",
}

# All known document sections (built from [SECTION:] tags in indexed docs)
ALL_SECTIONS: list[str] = [
    "SUPPLIER_PERFORMANCE",
    "GEOPOLITICAL_RISK",
    "SUPPLY_CHAIN_RESILIENCE",
    "SLA_COMPLIANCE",
    "COLD_CHAIN_POLICY",
    "DELAY_ANALYSIS",
    "RISK_MANAGEMENT",
    "PROCUREMENT_POLICY",
    "INVENTORY_MANAGEMENT",
    "REGULATORY_COMPLIANCE",
    "FINANCIAL_PERFORMANCE",
    "FINANCIAL_IMPACT",
    "ROI_ANALYSIS",
    "COST_BREAKDOWN",
    "AI_INVESTMENT",
    "EXECUTIVE_SUMMARY",
    "OPERATIONAL_REVIEW",
    "RECOMMENDATIONS",
]

# Roles that can see financial document sections
_FINANCIAL_SECTION_ROLES: set[str] = {"Operations Manager", "CFO"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NOT-FOUND RESPONSE TEMPLATE
#  Reused across functions — single definition prevents divergence.
# ═══════════════════════════════════════════════════════════════════════════════

def _not_found_result(reason: str, query: str = "") -> dict:
    """
    Canonical not-found response.

    WHY a helper function for this?
        Every caller that returns not-found must look identical so the
        Executive Agent and UI can handle them uniformly without special cases.
    """
    return {
        "found":      False,
        "reason":     reason,
        "answer":     None,
        "source":     None,
        "chunks":     [],
        "sections":   [],
        "sources":    [],
        "similarity_scores": [],
        "confidence": 0.0,
        "query":      query,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SEARCH KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

def search_knowledge_base(
    query:  str,
    role:   str,
    top_k:  int = 3,
) -> dict:
    """
    Search the FAISS vector store for chunks relevant to the query.

    WHY check role BEFORE searching?
        We must not embed and search a query that the role is not permitted
        to ask about. Even though the vector store itself is not sensitive,
        a blocked query should never consume an LLM embedding token.

    WHY not check financial sections here?
        Section filtering happens in format_rag_finding() and
        get_available_sections() — after retrieval, before formatting.
        This keeps search() a clean, single-responsibility operation.

    Args:
        query:  Natural language question from the Planning Agent.
        role:   Current user role (must exist in guardrails.ROLES).
        top_k:  Maximum number of chunks to retrieve if above threshold.

    Returns:
        On success:
            {
                "found":             True,
                "chunks":            list[dict],   # raw chunk dicts from FAISS
                "sources":           list[str],    # unique source filenames
                "sections":          list[str],    # section tags per chunk
                "similarity_scores": list[float],
                "confidence":        float,        # highest similarity score
                "query":             str,
            }

        On guardrail block or not-found:
            {
                "found":      False,
                "reason":     str,
                "answer":     None,
                "source":     None,
                "confidence": 0.0,
                ...
            }
    """
    log.info(
        f"search_knowledge_base | role='{role}' | "
        f"query='{query[:80]}{'...' if len(query) > 80 else ''}'"
    )

    # ── Step 1: Validate role exists ──────────────────────────────────────────
    role_check = validate_role(role)
    if not role_check["valid"]:
        log.warning(
            f"search_knowledge_base | INVALID ROLE | role='{role}'"
        )
        log_guardrail_event(
            event_type = "INPUT_BLOCKED",
            role       = role,
            query      = query,
            outcome    = "blocked",
            reason     = role_check["reason"],
        )
        return _not_found_result(role_check["reason"], query)

    # ── Step 2: Check query access (keyword / topic guardrail) ────────────────
    # WHY check this for RAG, not just DB queries?
    #   A blocked keyword (e.g. "roi" for Demand Planner) should not reach
    #   the vector store even if the relevant chunk is in the annual report.
    #   Access control applies to the question, not just the answer.
    access = check_query_access(query, role)
    if not access["allowed"]:
        log.warning(
            f"search_knowledge_base | ACCESS DENIED | "
            f"role='{role}' | term='{access['blocked_term']}'"
        )
        log_guardrail_event(
            event_type = "INPUT_BLOCKED",
            role       = role,
            query      = query,
            outcome    = "blocked",
            reason     = access["reason"],
        )
        return _not_found_result(access["reason"], query)

    # ── Step 3: Search FAISS vector store ────────────────────────────────────
    # memory.search() handles the embedding call (1 LLM token spend)
    # and applies the 0.75 similarity guardrail internally.
    raw = vector_search(query, top_k=top_k)

    # ── If similarity < 0.75 ──────────────────────────────────────────────────
    # memory.search() already blocked this — we honour its decision.
    #
    # If similarity < 0.75 we return not-found.
    # The LLM never sees this query.
    # It cannot hallucinate an answer
    # because we never ask it the question.
    # This is hallucination prevention at
    # the architecture level not prompt level.
    if not raw.get("found"):
        reason = raw.get("reason", "No relevant content found.")
        score  = raw.get("best_score", 0.0)
        log.warning(
            f"search_knowledge_base | NOT FOUND | "
            f"best_score={score:.4f} | threshold={SIMILARITY_THRESHOLD}"
        )
        return _not_found_result(
            f"{reason} (best similarity: {score:.2f}, "
            f"threshold: {SIMILARITY_THRESHOLD})",
            query,
        )

    # ── Step 4: Filter chunks by role's section access ────────────────────────
    # WHY filter AFTER retrieval, not before?
    #   FAISS cannot filter by metadata — it only knows vectors.
    #   We retrieve liberally, then restrict by role before returning.
    chunks          = raw["results"]
    permitted_sections = get_available_sections(role)

    filtered_chunks = []
    redacted_count  = 0

    for chunk in chunks:
        section = chunk.get("section", "GENERAL").upper()
        if section in FINANCIAL_SECTIONS and section not in permitted_sections:
            redacted_count += 1
            log.info(
                f"search_knowledge_base | SECTION REDACTED | "
                f"section='{section}' | role='{role}'"
            )
        else:
            filtered_chunks.append(chunk)

    # If ALL chunks were redacted, the result is effectively not-found
    if not filtered_chunks:
        reason = (
            f"The most relevant document sections "
            f"({redacted_count} chunk(s) redacted) "
            f"are restricted for your role ('{role}')."
        )
        log.warning(f"search_knowledge_base | ALL CHUNKS REDACTED | role='{role}'")
        log_guardrail_event(
            event_type = "OUTPUT_MASKED",
            role       = role,
            query      = query,
            outcome    = "blocked",
            reason     = reason,
        )
        return _not_found_result(reason, query)

    # ── Step 5: Build clean return structure ──────────────────────────────────
    sources    = list(dict.fromkeys(c["source"]  for c in filtered_chunks))
    sections   = [c.get("section", "GENERAL")    for c in filtered_chunks]
    scores     = [c["similarity_score"]           for c in filtered_chunks]
    confidence = max(scores) if scores else 0.0

    log.success(
        f"search_knowledge_base | FOUND | "
        f"chunks={len(filtered_chunks)} | "
        f"confidence={confidence:.4f} | "
        f"sources={sources}"
    )

    return {
        "found":             True,
        "chunks":            filtered_chunks,
        "sources":           sources,
        "sections":          sections,
        "similarity_scores": scores,
        "confidence":        round(confidence, 4),
        "query":             query,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. FORMAT RAG FINDING
# ═══════════════════════════════════════════════════════════════════════════════

def format_rag_finding(search_result: dict, query: str) -> dict:
    """
    Convert a raw search result into a structured finding for the graph.

    WHY pure Python, no LLM here?
        The finding is intermediate — it feeds into the Executive Agent
        which makes the ONE allowed LLM call to format the final answer.
        Adding an LLM call here would violate the "one LLM call total" rule
        and double the cost and latency of every RAG step.

    Finding summary is built deterministically:
        • First 40 words of the top chunk — always factual, always grounded
        • Section tag — tells the user WHERE the answer came from
        • Source filename — enables the user to go read the original
        • Similarity score — transparent confidence signal

    Args:
        search_result: Output of search_knowledge_base() with found=True.
        query:         Original query string (for citation context).

    Returns:
        {
            "task":       "knowledge_base_search",
            "finding":    str,   # one-sentence summary, coded string format
            "evidence":   str,   # full text of the top matching chunk
            "source":     str,   # source document filename
            "section":    str,   # section tag e.g. GEOPOLITICAL_RISK
            "confidence": float, # similarity score of best match
            "citation":   str,   # formatted citation string for UI
            "all_chunks": list,  # all retrieved chunks (for Executive Agent)
        }
    """
    if not search_result.get("found"):
        # Propagate not-found cleanly — caller should have caught this
        return {
            "task":       "knowledge_base_search",
            "finding":    "No relevant content found in knowledge base.",
            "evidence":   None,
            "source":     None,
            "section":    None,
            "confidence": 0.0,
            "citation":   "No citation available.",
            "all_chunks": [],
        }

    # ── Top chunk is the highest-scoring result ───────────────────────────────
    chunks     = search_result["chunks"]
    top_chunk  = chunks[0]

    top_text   = top_chunk.get("text",    "")
    top_section = top_chunk.get("section", "GENERAL").strip().upper()
    top_source  = top_chunk.get("source",  "Unknown source")
    top_score   = top_chunk.get("similarity_score", 0.0)

    # ── Build one-sentence summary from chunk text — pure Python ─────────────
    # WHY first 40 words?
    #   40 words = ~2 sentences on average. Enough to give the Executive Agent
    #   grounded material; short enough to force precision over verbosity.
    words   = top_text.split()
    preview = " ".join(words[:40]) + ("..." if len(words) > 40 else "")

    # Strip the filename extension for cleaner display
    source_clean = Path(top_source).stem.replace("_", " ").title()

    # ── Readable section label (convert SNAKE_CASE to Title Case) ────────────
    section_label = top_section.replace("_", " ").title()

    # ── Build citation string ─────────────────────────────────────────────────
    citation = f"Source: {source_clean} | Section: {section_label}"

    # ── One-sentence finding — coded template, not LLM ───────────────────────
    finding = (
        f"The knowledge base contains relevant information in the "
        f"'{section_label}' section of {source_clean}: \"{preview}\""
    )

    log.info(
        f"format_rag_finding | section='{top_section}' | "
        f"source='{top_source}' | confidence={top_score:.4f}"
    )

    return {
        "task":       "knowledge_base_search",
        "finding":    finding,
        "evidence":   top_text,
        "source":     top_source,
        "section":    top_section,
        "confidence": round(top_score, 4),
        "citation":   citation,
        "all_chunks": chunks,           # full list for Executive Agent context
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. RUN — Main entry point called by graph.py
# ═══════════════════════════════════════════════════════════════════════════════

def run(step: dict, role: str) -> dict:
    """
    Execute one RAG step from the Planning Agent's plan.

    This is the only function graph.py calls directly. All other functions
    in this file are internal helpers — they are tested independently but
    not invoked by the orchestration layer.

    Flow:
        1. Log step received
        2. Extract query from step dict
        3. search_knowledge_base()
           → If not found: return honest not-found (never fabricate)
        4. format_rag_finding()
        5. Log completion with confidence
        6. Return structured finding

    WHY "never fabricate" matters here:
        The RAG agent is the only agent that searches unstructured text.
        Structured data (DB Agent) has a schema — wrong answers are
        immediately visible. Unstructured answers can *sound* right while
        being completely fabricated. The similarity threshold + honest
        not-found response is the only reliable defence.

    Args:
        step: Plan step dict from Planning Agent. Expected keys:
              "task" (str), "instruction" (str, used as query),
              "step_number" (int).
        role: Current user role string.

    Returns:
        Structured finding dict from format_rag_finding(), or a
        not-found dict if similarity threshold not met.
    """
    start_time = time.perf_counter()

    step_num   = step.get("step_number", step.get("step", "?"))
    task       = step.get("task", "knowledge_base_search")
    # Use 'instruction' as the search query — Planning Agent populates this
    query      = step.get("instruction", step.get("task", ""))

    log.info(
        f"run | step={step_num} | task='{task}' | "
        f"role='{role}' | query='{query[:60]}{'...' if len(query) > 60 else ''}'"
    )

    # ── Execute search ────────────────────────────────────────────────────────
    search_result = search_knowledge_base(query=query, role=role, top_k=3)

    # ── If similarity < 0.75: return honest not-found, never fabricate ────────
    #
    # WHY is "never fabricate" its own explicit code branch?
    #   An LLM asked "what does the policy say about X?" will generate an
    #   answer whether or not X appears in its context. The only way to
    #   guarantee it doesn't is to never call it.
    #   This branch ensures we return a structured signal the UI can display
    #   ("I couldn't find this in the knowledge base") rather than a
    #   confident-sounding hallucination.
    if not search_result.get("found"):
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        reason     = search_result.get("reason", "Not found in knowledge base.")

        log.warning(
            f"run | NOT FOUND | step={step_num} | "
            f"reason='{reason}' | time={elapsed_ms}ms"
        )

        return {
            "agent":              "rag_agent",
            "step":               step_num,
            "task":               task,
            "found":              False,
            "finding":            (
                f"No relevant content found in the knowledge base for: "
                f"'{query[:100]}'. {reason}"
            ),
            "evidence":           None,
            "source":             None,
            "section":            None,
            "citation":           "No citation — content not found.",
            "confidence":         0.0,
            "execution_time_ms":  elapsed_ms,
            "all_chunks":         [],
        }

    # ── Format the finding ────────────────────────────────────────────────────
    finding    = format_rag_finding(search_result, query)
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    log.success(
        f"run | COMPLETE | step={step_num} | "
        f"confidence={finding['confidence']:.4f} | "
        f"section='{finding['section']}' | "
        f"source='{finding['source']}' | "
        f"time={elapsed_ms}ms"
    )

    # ── Return enriched finding with agent metadata ───────────────────────────
    return {
        "agent":             "rag_agent",
        "step":              step_num,
        "task":              task,
        "found":             True,
        "finding":           finding["finding"],
        "evidence":          finding["evidence"],
        "source":            finding["source"],
        "section":           finding["section"],
        "citation":          finding["citation"],
        "confidence":        finding["confidence"],
        "similarity_scores": search_result["similarity_scores"],
        "all_chunks":        finding["all_chunks"],
        "sources":           search_result["sources"],
        "execution_time_ms": elapsed_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  4. GET AVAILABLE SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_sections(role: str) -> set[str]:
    """
    Return the set of document sections this role is permitted to access.

    WHY the Planning Agent needs this:
        Before adding a RAG step to a plan, the Planning Agent should know
        whether the role can actually see the sections likely to contain
        the answer. If not, it can skip the RAG step and save a FAISS
        embedding call.

    Access rules:
        Demand Planner   → Operational sections only (no financial)
        Operations Manager → All sections
        CFO              → All sections

    Args:
        role: Role display name as defined in guardrails.ROLES.

    Returns:
        Set of section name strings the role may read.
        Returns all sections if role is unknown (fail open for read-only).
    """
    role_config = ROLES.get(role, {})

    # Roles with full financial visibility get all sections
    if role_config.get("can_see_financials", True):
        log.debug(
            f"get_available_sections | role='{role}' | access=ALL "
            f"({len(ALL_SECTIONS)} sections)"
        )
        return set(ALL_SECTIONS)

    # Demand Planner: return only non-financial sections
    permitted = set(ALL_SECTIONS) - FINANCIAL_SECTIONS
    log.debug(
        f"get_available_sections | role='{role}' | "
        f"access=OPERATIONAL ({len(permitted)} sections, "
        f"{len(FINANCIAL_SECTIONS)} financial sections blocked)"
    )
    return permitted
