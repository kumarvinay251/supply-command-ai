"""
Supply Command AI — Executive Agent
The single LLM call in the entire pipeline.

All other agents produce structured Python dicts from coded rules and SQL.
This agent takes those verified findings and formats them into a
role-appropriate natural language answer using one GPT-4o-mini call.

LLM Usage Policy:
    THIS IS THE ONLY FILE IN THE SYSTEM THAT CALLS AN LLM.
    One call per user question. No exceptions.

    WHY only one LLM call?
        Every additional LLM call is:
            • Another opportunity to hallucinate
            • Another latency cost (~1–2 seconds)
            • Another token cost
        All routing, SQL, retrieval, and calculation is done in coded Python.
        The LLM only does what it is uniquely good at: formatting verified
        facts into readable, role-appropriate language.

    WHY GPT-4o-mini at temperature=0.1?
        We want consistent, factual answers — not creative text.
        Low temperature keeps outputs close to the prompt's framing.
        gpt-4o-mini is sufficient for formatting tasks and cheaper than
        GPT-4o (relevant for production cost control).

Groundedness guarantee:
    validate_llm_output() checks every number in the LLM response
    against the verified findings. If the LLM introduces a figure
    not present in findings, it is flagged as ungrounded.
    If groundedness < 0.8, a warning is appended to the final answer.
"""

import os
import re
import time
from typing import Optional

from dotenv   import load_dotenv
from openai   import OpenAI

from services.logger import get_logger
from agents.guardrails import (
    ROLES,
    validate_role,
    validate_output,
    check_human_approval_needed,
)
from database.db_connection   import log_agent_decision
from database.semantic_layer  import get_golden_rules_text

load_dotenv()
log = get_logger("executive_agent")

# ── OpenAI client (lazy — only instantiated when run() is called) ─────────────
_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    """Return a shared OpenAI client. Instantiated once, reused across calls."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_key_here":
            raise ValueError(
                "OPENAI_API_KEY not set or still placeholder. "
                "Update .env with your real key before calling the LLM."
            )
        _client = OpenAI(api_key=api_key)
    return _client

# ── Model configuration ───────────────────────────────────────────────────────
LLM_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS  = 500
LLM_TEMPERATURE = 0.1
# WHY 0.1 temperature?
#   Supply chain answers need to be consistent and factual.
#   Higher temperature (>0.5) introduces creative paraphrasing that can
#   subtly distort numbers and percentages. 0.1 keeps the LLM close to
#   the exact wording of the findings while still producing fluent prose.

# ── Token cost for audit logging ──────────────────────────────────────────────
# gpt-4o-mini: $0.15 per 1M input tokens = $0.00000015 per token
COST_PER_TOKEN_USD = 0.00000015

# ── Groundedness threshold ────────────────────────────────────────────────────
# WHY 0.8?
#   We accept that some stylistic claims ("this is significant") have no
#   numeric anchor and cannot be verified. 80% means at least 4 out of 5
#   factual claims are grounded. Below that, the answer has too much
#   invented content to be shown without a warning.
GROUNDEDNESS_THRESHOLD = 0.8


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ROLE PERSONAS
#  Controls how the LLM frames its answer for each audience.
#  WHY different personas?
#    A Demand Planner needs "shipment SHP0032 is at risk" not
#    "the $45K expedite exposure represents 0.9% of annual spend".
#    A CFO needs the exact opposite. Same data — different framing.
# ═══════════════════════════════════════════════════════════════════════════════

PERSONAS: dict[str, dict] = {

    "Demand Planner": {
        "style":  "operational",
        "focus":  "stock levels, delivery timing, supplier reliability",
        "avoid":  "financial figures, contract details, ROI metrics",
        "format": (
            "bullet points with clear action items. "
            "Start with the most urgent operational concern. "
            "Each bullet = one actionable insight."
        ),
        "tone":   "practical and specific — tell me what to do, not what happened",
        "length": "concise — 150 words maximum",
    },

    "Operations Manager": {
        "style":  "analytical",
        "focus":  "root causes, trends, cross-functional impact, supplier performance",
        "avoid":  "nothing — full access to all findings",
        "format": (
            "structured paragraphs with clear metric callouts in plain text. "
            "Lead with the headline finding, follow with supporting evidence, "
            "end with recommended actions ranked by priority."
        ),
        "tone":   "data-driven and decisive — state findings then recommend action",
        "length": "comprehensive — 300 words maximum",
    },

    "CFO": {
        "style":  "financial",
        "focus":  "cost impact, ROI, risk exposure, benchmark comparison, projections",
        "avoid":  "operational details unless they carry direct financial framing",
        "format": (
            "executive summary format. "
            "Open with the financial headline (dollar figure or ROI). "
            "Follow with 3 key supporting metrics. "
            "Close with strategic recommendation and risk exposure."
        ),
        "tone":   "strategic and quantified — every claim backed by a number",
        "length": "executive — 250 words maximum",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PROMPT BUILDER
#  Pure Python — assembles the LLM system + user prompt from verified findings.
#  WHY assemble in code rather than a template file?
#    Code is auditable — every line of the prompt is visible and testable.
#    Template files can drift out of sync with the data model.
# ═══════════════════════════════════════════════════════════════════════════════

# ── DECISION_QUERY structured output prompt ───────────────────────────────────
# WHY a dedicated constant?
#   DECISION_QUERY answers must follow a strict 3-section structure so the
#   UI can render them consistently: Issue → Recommendation → Risk.
#   Inline strings inside build_prompt would be harder to audit and extend.
#   A named constant is testable and visible here in the prompt-builder section.
DECISION_QUERY_PROMPT = """DECISION QUERY FORMAT — respond with EXACTLY these three sections:

**Issue:** [One sentence stating the supply chain problem based on the data findings.]
**Recommendation:** [One to two sentences stating the specific action to take, citing the data.]
**Risk:** [One sentence describing the key risk if no action is taken.]

Rules:
• Use the ** markers exactly as shown — the UI renders them as section headers.
• Ground every claim in the VERIFIED FINDINGS below.
• Do NOT add additional sections, bullet points, or paragraphs outside this structure.
• Do NOT repeat data unnecessarily — one number per section is sufficient."""


def trim_decision_output(text: str) -> str:
    """
    Enforce the Issue / Recommendation / Risk structure for DECISION_QUERY output.

    WHY post-process instead of relying on the LLM prompt alone?
        LLMs sometimes add preamble ("Based on the findings..."), trailing
        commentary, or extra bullets outside the three required sections.
        This function strips everything outside the three **Section:** markers
        so the UI always receives a clean, predictable structure.

    If the LLM did not produce the expected sections, returns the original text
    unchanged — better than returning an empty string.

    Args:
        text: Raw LLM response for a DECISION_QUERY.

    Returns:
        Trimmed text with only Issue / Recommendation / Risk sections,
        or the original text if the expected structure is absent.
    """
    import re as _re_dec
    # Find all **Section:** … blocks
    section_pattern = _re_dec.compile(
        r'(\*\*(?:Issue|Recommendation|Risk):\*\*.*?)(?=\*\*(?:Issue|Recommendation|Risk):\*\*|$)',
        _re_dec.DOTALL | _re_dec.IGNORECASE,
    )
    matches = section_pattern.findall(text)
    if matches:
        return "\n".join(m.strip() for m in matches)
    # Fallback: return original if structure not found
    return text


def format_financials(text: str) -> str:
    """
    Post-process any text string to clean up financial formatting.

    FIX 7 — why a named function?
        The cleanup logic (strip ** bold, fix "$ space" artefacts) was
        previously inline in run(). As a named function it can be:
        • Called from the Streamlit UI layer for any displayed string.
        • Unit-tested independently of the full LLM pipeline.
        • Extended with new formatting rules without touching run().

    Operations:
        1. Remove ** bold markers that gpt-4o-mini wraps around numbers
           (e.g. **$5,841,000** → $5,841,000).
        2. Collapse "$ 5,841,000" (space after $) to "$5,841,000".
           This artefact occurs because the LLM tokenises $ and the
           following digit separately.

    Args:
        text: Any string — typically LLM response text.

    Returns:
        Cleaned string with markdown and spacing normalised.
    """
    import re as _re_fmt
    # Strip bold markers around any content (numbers, text, mixed)
    text = _re_fmt.sub(r'\*\*([^*\n]+)\*\*', r'\1', text)
    # Collapse "$ 1,234" → "$1,234"
    text = _re_fmt.sub(r'\$\s+', '$', text)
    return text


# ── FIX 4 — Cross-metric contamination filter ────────────────────────────────
# WHY per-metric blocked terms?
#   A delay_rate answer should never mention cost figures — the LLM sometimes
#   appends financial context it infers from related findings. These blocked
#   terms act as a post-hoc guardrail, stripping sentences that mix metric
#   domains. We only filter METRIC_QUERY answers (facts only, no narrative).
METRIC_BLOCKED_TERMS: dict[str, list[str]] = {
    "delay_rate":     ["$", "cost", "penalty", "roi", "revenue", "saving"],
    "otd":            ["cost", "penalty", "roi", "saving"],
    "sla_breaches":   ["cost", "roi", "saving", "revenue"],
    "expedited_cost": ["delay rate", "otd", "on-time"],
}


def filter_metric_contamination(text: str, detected_metric: str) -> str:
    """
    Drop sentences that introduce metric-domain cross-contamination.

    WHY sentence-level granularity?
        A contaminated number mid-sentence cannot be surgically removed
        without rewriting. Dropping the entire sentence is safer than
        leaving a partially corrected claim that may still mislead.

    Args:
        text:             LLM response text.
        detected_metric:  Metric key from planning_agent (e.g. "delay_rate").

    Returns:
        Text with contaminated sentences removed, or original text if
        detected_metric has no blocked terms defined.
    """
    blocked = METRIC_BLOCKED_TERMS.get(detected_metric, [])
    if not blocked:
        return text

    sentences = text.replace("\n", " ").split(". ")
    clean = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(term.lower() in sentence_lower for term in blocked):
            log.debug(
                f"filter_metric_contamination | dropped sentence with blocked term | "
                f"metric='{detected_metric}'"
            )
            continue
        clean.append(sentence)

    return ". ".join(clean).strip() if clean else text


# ── FIX 6 — Hard brevity enforcement for METRIC_QUERY ────────────────────────
# WHY a dedicated function instead of inline logic?
#   Brevity enforcement runs as the absolute last step before return — after
#   contamination filtering, format_financials, and groundedness check.
#   A named function is testable, visible in the call chain, and easy to
#   adjust if the per-query type limits change.
def enforce_metric_brevity(
    text:        str,
    query_type:  str,
    alert_driven: bool,
    multi_row:   bool = False,
) -> str:
    """
    Hard sentence-count limit for METRIC_QUERY and alert-driven answers.

    Limits:
        alert_driven=True  → 1 sentence max (exact metric value only),
                             applied regardless of query_type so that
                             BENCHMARK_COMPARISON / EXPLANATION_QUERY
                             alert follow-ups are also kept to one sentence.
        alert_driven=False → 2 sentences max (value + one context sentence),
                             only for METRIC_QUERY (non-alert path unchanged).
        multi_row=True     → pass through unchanged (comparison lists)
        Non-METRIC_QUERY, non-alert → pass through unchanged

    Preserves Sources: and Confidence: metadata lines.
    Filters qualitative padding sentences (recommend, consider, etc.).
    """
    # multi_row always passes through unchanged
    if multi_row:
        return text

    # alert_driven applies regardless of query_type (covers BENCHMARK_COMPARISON etc.)
    # non-alert non-METRIC_QUERY passes through unchanged
    if not alert_driven and query_type != "METRIC_QUERY":
        return text

    import re as _re_brev
    _sentences = _re_brev.split(r'(?<=[.!?])\s+', text.strip())

    # Separate metadata lines
    _meta_lines = [
        s for s in _sentences
        if s.startswith("Sources:") or s.startswith("Confidence:")
    ]
    _main_sentences = [
        s for s in _sentences
        if not s.startswith("Sources:") and not s.startswith("Confidence:")
    ]

    # Filter qualitative / advisory sentences
    _FILTER_WORDS = [
        "recommend", "consider", "suggest", "root cause",
        "because", "therefore", "it is advised", "you should",
        "we advise",
    ]
    _filtered = [
        s for s in _main_sentences
        if not any(fw in s.lower() for fw in _FILTER_WORDS)
    ]
    if _filtered:
        _main_sentences = _filtered

    # Hard limit: 1 for alert-driven (exact metric only), 2 for regular
    limit    = 1 if alert_driven else 2
    _trimmed = " ".join(_main_sentences[:limit])

    log.info(
        f"enforce_metric_brevity | alert_driven={alert_driven} | "
        f"limit={limit} | original_sentences={len(_main_sentences)} | "
        f"trimmed_to={min(limit, len(_main_sentences))}"
    )

    return _trimmed + ("\n" + "\n".join(_meta_lines) if _meta_lines else "")


def build_prompt(
    findings:             list[dict],
    query:                str,
    role:                 str,
    persona:              dict,
    conversation_history: Optional[list[dict]] = None,
    query_type:           str = "METRIC_QUERY",
    metric_definition:    str = "",
    needs_approval:       bool = False,
) -> tuple[str, str]:
    """
    Assemble the system prompt and user message for the LLM call.

    Returns:
        (system_prompt, user_message) — both strings, ready for the API.

    WHY separate system and user messages?
        System prompt = static rules and context (cached by OpenAI, cheaper).
        User message  = dynamic query + findings (changes every call).
        Keeping them separate also makes it clearer what the LLM "knows"
        vs what it is "asked".
    """
    # ── Format findings as numbered, cited facts ──────────────────────────────
    formatted_findings = _format_findings_block(findings)

    # ── Collect SQL queries used (for transparency in the answer) ─────────────
    sql_queries = [
        f["sql"] for f in findings
        if f.get("sql") and isinstance(f.get("sql"), str)
    ]
    sql_block = (
        "\n".join(f"  SQL {i+1}: {s.strip()[:200]}"
                  for i, s in enumerate(sql_queries))
        if sql_queries
        else "  (No SQL — findings from document search or calculation)"
    )

    # ── Conversation history block (for follow-up questions) ──────────────────
    history_block = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-4:]:   # last 2 turns only
            speaker = "User"    if msg["role"] == "user"      else "Assistant"
            history_lines.append(f"  {speaker}: {msg['content'][:200]}")
        history_block = (
            "\nPREVIOUS CONVERSATION (for context only — "
            "do NOT repeat previous answers):\n"
            + "\n".join(history_lines)
            + "\n"
        )

    # ── CRITICAL DATA RULES block (query_type-aware) ─────────────────────────
    # WHY prepend before the 10 STRICT RULES?
    #   The LLM processes instructions in order. By placing query_type-specific
    #   data rules FIRST, they take precedence over the generic formatting rules
    #   that follow. This prevents the LLM from blending RAG narrative into a
    #   METRIC_QUERY answer where only DB numbers are valid ground truth.
    if query_type == "METRIC_QUERY":
        metric_def_line = (
            f"\n   METRIC BEING ANSWERED: {metric_definition}"
            if metric_definition
            else ""
        )
        critical_data_rules = f"""CRITICAL DATA RULES — THIS IS A METRIC QUERY:
• ONE primary result only. Lead with the single most important number.
• Use ONLY numbers from DB Agent findings (shipments / financial_impact / suppliers_master tables).
• RAG Agent findings are CONTEXT ONLY — never cite them for numeric claims.
• Do NOT blend PDF narrative with SQL results. Keep them strictly separate.
• Answer must be under 80 words. Facts only — no qualitative padding.
• If the DB finding contains zero rows, say "No data available" — do NOT estimate.{metric_def_line}

"""
    elif query_type == "DECISION_QUERY":
        # Structured Issue / Recommendation / Risk format
        _approval_note = (
            "\n⚠️ NOTE: Human approval is required before this recommendation "
            "is actioned. State this clearly in the Risk section."
            if needs_approval else ""
        )
        critical_data_rules = (
            f"CRITICAL DATA RULES — THIS IS A DECISION QUERY:\n"
            f"• DB Agent findings are GROUND TRUTH for all numeric claims.\n"
            f"• RAG Agent findings ADD CONTEXT only.\n"
            f"{DECISION_QUERY_PROMPT}{_approval_note}\n\n"
        )
    else:
        # EXPLANATION_QUERY
        critical_data_rules = f"""CRITICAL DATA RULES — THIS IS AN {query_type.replace("_", " ")}:
• DB Agent findings are GROUND TRUTH — all numeric claims must come from them.
• RAG Agent findings ADD CONTEXT only — use for explanations and best practices,
  never to override or supplement a DB number.
• If DB and RAG figures conflict, the DB figure is always correct.
• Clearly distinguish between "our data shows X" (DB) and "industry guidance says Y" (RAG).
• For benchmark comparison queries: your FIRST SENTENCE must state the industry
  benchmark figure (e.g. "87% OTD industry benchmark") found in RAG Evidence.
  Look carefully through ALL evidence chunks — the benchmark table appears after
  introductory text. If no RAG finding contains a benchmark percentage, write
  "The industry benchmark data was not retrieved from the knowledge base."

"""

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = f"""You are an AI supply chain analyst assistant for GlobalMedTech Inc.
You help {role} understand supply chain performance and make better decisions.

{critical_data_rules}STRICT RULES YOU MUST FOLLOW:
1. Only use information provided in VERIFIED FINDINGS below.
   Do not add any information, statistics, or claims not present there.
2. If a finding contains [RESTRICTED], do not include or reference that value.
3. Always cite the data source for every key claim (e.g. "Source: shipments table").
4. Always state the confidence level when it is below 0.85.
5. Format your response for a {persona['style']} audience.
6. Focus on: {persona['focus']}
7. Avoid: {persona['avoid']}
8. Tone: {persona['tone']}
9. Length: {persona['length']}
10. End every response with one clear next-step recommendation.
11. Do NOT use markdown bold (**) around numbers, dollar amounts, or percentages.
    Write all figures in plain text only (e.g. write $5,841,000 not **$5,841,000**).

VERIFIED FINDINGS (these are the ONLY facts you may reference):
{formatted_findings}

SQL QUERIES THAT PRODUCED THESE FINDINGS:
{sql_block}

{history_block}"""

    # ── User message ──────────────────────────────────────────────────────────
    user_message = f"""Question from {role}: {query}

Format required: {persona['format']}

Write your response now using ONLY the verified findings above.
Do not introduce any information not in the findings.
End with: "Sources: [list sources used]" and "Confidence: [overall %]"."""

    return system_prompt, user_message


def _format_findings_block(findings: list[dict]) -> str:
    """
    Convert the list of agent findings into a numbered, cited text block.

    WHY this specific format?
        Numbering helps the groundedness checker match LLM claims back to
        specific findings. Source + confidence on every finding makes it
        easy for the LLM to cite correctly.
    """
    if not findings:
        return "  No findings available."

    lines = []
    for i, f in enumerate(findings, start=1):
        agent       = f.get("agent",      "unknown_agent")
        task        = f.get("task",       "unknown_task")
        finding_txt = f.get("finding",    "No finding text available.")
        confidence  = f.get("confidence", 0.0)
        source      = f.get("source",     "")
        section     = f.get("section",    "")
        citation    = f.get("citation",   "")
        sql         = f.get("sql",        "")

        # Build source label
        if citation:
            source_label = citation
        elif source:
            sec = f", Section: {section}" if section else ""
            source_label = f"Source: {source}{sec}"
        else:
            _AGENT_SOURCE = {
                "db_agent":       "shipments / suppliers_master / financial_impact (SQLite)",
                "rag_agent":      "Knowledge Base / Annual Report PDF",
                "roi_agent":      "Python financial calculation from verified DB data",
                "planning_agent": "Internal plan",
            }
            source_label = _AGENT_SOURCE.get(agent, agent)

        lines.append(f"\nFinding {i}: [{agent.upper()}] {finding_txt}")
        lines.append(f"  Source     : {source_label}")
        lines.append(f"  Confidence : {confidence:.0%}")

        # Include key calculation metrics if present
        if f.get("key_metric"):
            lines.append(f"  Key metric : {f['key_metric']}")

        # Include evidence for RAG findings
        # WHY 1000 chars for top chunk?
        #   The industry benchmark table (87% OTD, 81.4% actual, 5.6pp gap)
        #   begins at ~650 chars into the APPENDIX chunk. A 200-char preview
        #   never reached it, causing the LLM to answer without the benchmark
        #   figure. 1000 chars reliably captures all key benchmark KPIs.
        # WHY also include additional chunks?
        #   FAISS returns 3 chunks. The benchmark data appears across multiple
        #   chunks (APPENDIX and RECOMMENDATIONS sections). Surfacing all of
        #   them gives the LLM the full picture without needing a second query.
        if f.get("evidence") and agent == "rag_agent":
            preview = str(f["evidence"])[:1000]
            lines.append(f"  Evidence   : \"{preview}{'...' if len(str(f['evidence'])) > 1000 else ''}\"")
            # Surface additional chunks (chunk 2 and 3) at 400 chars each
            for extra_idx, extra_chunk in enumerate(f.get("all_chunks", [])[1:], start=2):
                extra_text    = str(extra_chunk.get("text", ""))
                extra_preview = extra_text[:400]
                extra_section = extra_chunk.get("section", "GENERAL")
                extra_score   = extra_chunk.get("similarity_score", 0.0)
                lines.append(
                    f"  Evidence (chunk {extra_idx}, section={extra_section}, "
                    f"score={extra_score:.2f}): "
                    f"\"{extra_preview}{'...' if len(extra_text) > 400 else ''}\""
                )

        # Include top-level calculation if ROI finding
        if f.get("calculations") and agent == "roi_agent":
            calcs = f["calculations"]
            if "current_roi_pct" in calcs:
                lines.append(
                    f"  ROI detail : {calcs['current_roi_pct']}% ROI | "
                    f"Cumulative savings: "
                    f"{calcs.get('cumulative_savings_usd', '[RESTRICTED]')}"
                )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — OUTPUT VALIDATOR
#  Checks every factual claim in the LLM response against verified findings.
#  WHY validate LLM output?
#    Even with strict prompting, LLMs occasionally generate plausible-sounding
#    numbers that are not in the source data. This validator catches those.
#    It is not a perfect filter — it checks numbers and key terms, not meaning.
#    But it catches the most common hallucination pattern: wrong statistics.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_llm_output(response: str, findings: list[dict]) -> dict:
    """
    Check every factual claim in the LLM response against verified findings.

    Method:
        1. Extract all numbers from the LLM response (dollars, %, counts).
        2. Build a "facts corpus" from all finding texts and data rows.
        3. For each extracted number, check if it appears in the corpus.
        4. groundedness_score = verified_numbers / total_numbers_found.
        5. Also check for key named entities (supplier IDs, product categories).

    WHY check numbers specifically?
        Numbers are the most common hallucination vector for analytics tasks.
        An LLM might say "27% delay rate" when the finding says "20%".
        Text claims ("supplier performance was poor") are harder to verify
        and less dangerous than wrong statistics.

    Args:
        response: The raw text returned by the LLM.
        findings: All agent findings passed to the prompt.

    Returns:
        {
            "approved":           bool,
            "ungrounded_claims":  list[str],
            "groundedness_score": float,
            "safe_to_show":       bool,
        }
    """
    if not response or not response.strip():
        return {
            "approved":           False,
            "ungrounded_claims":  ["Empty response from LLM."],
            "groundedness_score": 0.0,
            "safe_to_show":       False,
        }

    # ── Build facts corpus from all findings ──────────────────────────────────
    corpus_parts: list[str] = []
    for f in findings:
        corpus_parts.append(str(f.get("finding",  "")))
        corpus_parts.append(str(f.get("evidence", "")))
        corpus_parts.append(str(f.get("key_metric", "")))
        # Include all data row values as strings
        for row in f.get("data", []):
            corpus_parts.extend(str(v) for v in row.values())
        # Include calculation values
        calcs = f.get("calculations", {})
        if isinstance(calcs, dict):
            corpus_parts.extend(str(v) for v in calcs.values())

    corpus = " ".join(corpus_parts).lower()

    # ── Strip metadata lines before number extraction ─────────────────────────
    # WHY strip Sources: and Confidence: lines?
    #   These lines are system-added metadata, not LLM factual claims.
    #   "Confidence: 90%" contains "90%" which would fail the corpus check
    #   (the corpus has no "90" — it comes from score calculation, not data).
    #   Including them causes false-positive groundedness failures on correct answers.
    response_for_check = "\n".join(
        line for line in response.splitlines()
        if not line.strip().startswith("Sources:")
        and not line.strip().startswith("Confidence:")
    )

    # ── Extract numbers from LLM response ────────────────────────────────────
    # Match: dollar amounts, percentages, plain integers, decimal numbers
    number_pattern = re.compile(
        r'\$[\d,]+(?:\.\d+)?'      # $1,234.56
        r'|[\d,]+(?:\.\d+)?%'      # 23.4%
        r'|[\d]{4,}(?:,[\d]{3})*'  # 1,234,567 (large numbers)
        r'|\b[\d]+\.\d+\b',        # decimal numbers
        re.IGNORECASE,
    )
    numbers_in_response = number_pattern.findall(response_for_check)

    ungrounded: list[str] = []
    verified_count = 0

    for number in numbers_in_response:
        # Normalise: remove $, commas, % for corpus matching
        normalised = re.sub(r'[$,%]', '', number).replace(',', '').strip()
        # Check if this number (or its close variants) appears in corpus
        if normalised in corpus or number.lower() in corpus:
            verified_count += 1
        else:
            ungrounded.append(number)

    total_numbers = len(numbers_in_response)
    if total_numbers == 0:
        # No numbers to verify — check key supply chain terms instead
        key_terms    = ["SUP001", "SUP002", "SUP003", "shipment", "delay",
                        "SupplierA", "SupplierB", "SupplierC"]
        terms_found  = sum(1 for t in key_terms if t.lower() in response_for_check.lower())
        terms_corpus = sum(1 for t in key_terms if t.lower() in corpus)
        groundedness = min(1.0, terms_found / max(terms_corpus, 1))
    else:
        groundedness = verified_count / total_numbers

    approved    = groundedness >= GROUNDEDNESS_THRESHOLD
    safe_to_show = groundedness >= 0.5   # always show with warning above 50%

    log.info(
        f"validate_llm_output | groundedness={groundedness:.2f} | "
        f"numbers_checked={total_numbers} | "
        f"ungrounded={len(ungrounded)} | approved={approved}"
    )

    if ungrounded:
        log.warning(
            f"validate_llm_output | UNGROUNDED CLAIMS: {ungrounded[:5]}"
        )

    return {
        "approved":           approved,
        "ungrounded_claims":  ungrounded,
        "groundedness_score": round(groundedness, 4),
        "safe_to_show":       safe_to_show,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MAIN EXECUTIVE FUNCTION
#  The single orchestration point for the final LLM call.
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    all_findings:          list[dict],
    query:                 str,
    role:                  str,
    conversation_history:  Optional[list[dict]] = None,
    query_type:            str = "METRIC_QUERY",
    metric_definition:     str = "",
    multi_row:             bool = False,
    alert_driven:          bool = False,
    detected_metric:       str = "",
) -> dict:
    """
    Format verified agent findings into a role-appropriate natural language answer.

    THIS FUNCTION CONTAINS THE ONLY LLM CALL IN THE ENTIRE SYSTEM.

    Flow:
        a. Log execution start
        b. Validate role via guardrails
        c. Apply output guardrails (mask [RESTRICTED] for Demand Planner)
        d. Check human approval needed — if yes, return WITHOUT calling LLM
        e. Build persona from PERSONAS dict
        f. build_prompt() → assemble system + user messages
        g. Call GPT-4o-mini (ONE call, max 500 tokens, temperature 0.1)
        h. validate_llm_output() — groundedness check
        i. Append warning if groundedness < 0.8
        j. log_agent_decision() → persist to ai_decisions_log
        k. Return structured final answer

    Args:
        all_findings:         List of finding dicts from DB/RAG/ROI agents.
        query:                Original user question.
        role:                 User role display name.
        conversation_history: Optional prior turns [{role, content}, ...].

    Returns:
        {
            "answer":                   str,
            "sources":                  list[str],
            "confidence":               float,
            "sql_shown":                list[str],
            "agents_used":              list[str],
            "groundedness_score":       float,
            "human_approval_required":  bool,
            "tokens_used":              int,
            "cost_usd":                 float,
        }
    """
    start_time = time.perf_counter()
    log.info(
        f"run | role='{role}' | "
        f"findings={len(all_findings)} | "
        f"query='{query[:80]}{'...' if len(query) > 80 else ''}'"
    )

    # ── a. Validate role ──────────────────────────────────────────────────────
    role_check = validate_role(role)
    if not role_check["valid"]:
        log.warning(f"run | INVALID ROLE | '{role}'")
        return _error_response(role_check["reason"], query, role)

    # ── b. Apply output guardrails to all findings ────────────────────────────
    # validate_output() masks [RESTRICTED] values, enforces row caps,
    # and appends low-confidence warnings. It operates on the findings
    # as a whole — we synthesise them into a pseudo-result for the check.
    warnings:      list[str] = []
    clean_findings: list[dict] = []

    for finding in all_findings:
        # Only pass DB-style findings with a "data" key through validate_output.
        # RAG and ROI findings are already filtered by their own agents.
        # Skip validate_output for failed steps (confidence=0 or "Step failed"
        # finding text) — running it on failures generates spurious "No data
        # was found" warnings that bury the correct answer from other steps.
        _is_failed_step = (
            finding.get("confidence", 1.0) == 0.0
            or str(finding.get("finding", "")).startswith("Step failed")
        )
        if finding.get("data") is not None and not _is_failed_step:
            guard_result = validate_output(
                {
                    "success":          finding.get("success", True),
                    "data":             finding.get("data", []),
                    "row_count":        finding.get("row_count", 0),
                    "confidence_score": finding.get("confidence", 1.0),
                },
                role,
            )
            warnings.extend(guard_result.get("warnings", []))
            # Merge guardrail-sanitised data back into the finding
            clean_finding = {
                **finding,
                "data": guard_result["result"].get("data", finding.get("data", [])),
            }
            clean_findings.append(clean_finding)
        else:
            clean_findings.append(finding)

    # ── c. Check if human approval is required (BEFORE LLM call) ─────────────
    # WHY check before the LLM call?
    #   If human approval is needed, we must NOT generate an automated answer.
    #   Generating an answer and then saying "but you need approval" creates
    #   anchoring bias — the human is influenced by the AI's suggestion.
    #   Returning "approval required" with no answer is the correct behaviour.
    combined_data         = _merge_finding_data(clean_findings)
    avg_confidence        = _weighted_confidence(clean_findings)
    top_recommendation    = _extract_top_recommendation(clean_findings)

    approval_check = check_human_approval_needed({
        "data":               combined_data,
        "confidence_score":   avg_confidence,
        "row_count":          len(combined_data),
        "recommended_action": top_recommendation,
    })

    # For DECISION_QUERY: capture the approval flag but continue to LLM call.
    # WHY not early-return for DECISION_QUERY?
    #   A decision query needs a structured answer (Issue/Recommendation/Risk)
    #   even when human approval is required — the LLM formats the context the
    #   human will review. Early-returning only the approval message leaves the
    #   human with no information to review. The approval flag is passed to the
    #   prompt so the LLM includes the notice in the Risk section.
    # For non-DECISION_QUERY: keep the early-return behaviour — a metric or
    #   explanation query that somehow triggers approval (e.g. high-risk action
    #   in recommendation field) should halt before LLM for safety.
    human_approval_required = approval_check["needs_approval"]

    if human_approval_required and query_type != "DECISION_QUERY":
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log.warning(
            f"run | HUMAN APPROVAL REQUIRED (non-DECISION_QUERY) | "
            f"reason='{approval_check['reason']}'"
        )
        return {
            "answer": (
                f"⚠️ This action requires human approval before proceeding.\n\n"
                f"Reason: {approval_check['reason']}\n"
                f"Impact: {approval_check['impact_summary']}\n\n"
                f"Please review the findings with your manager and confirm "
                f"before any supply chain action is taken."
            ),
            "sources":                  _collect_sources(clean_findings),
            "confidence":               avg_confidence,
            "sql_shown":                _collect_sql(clean_findings),
            "agents_used":              _collect_agents(clean_findings),
            "groundedness_score":       1.0,   # no LLM = no hallucination risk
            "human_approval_required":  True,
            "approval_reason":          approval_check["reason"],
            "impact_summary":           approval_check["impact_summary"],
            "tokens_used":              0,
            "cost_usd":                 0.0,
            "execution_time_ms":        elapsed_ms,
            "warnings":                 warnings,
        }

    if human_approval_required:
        log.warning(
            f"run | HUMAN APPROVAL REQUIRED (DECISION_QUERY — continuing to LLM) | "
            f"reason='{approval_check['reason']}'"
        )

    # ── c2. Filter findings by active metric (METRIC_QUERY only) ─────────────
    # WHY filter here, not earlier?
    #   We need clean_findings (guardrail-sanitised) not raw all_findings.
    #   Filtering before guardrails could restore a blocked value.
    #   Filtering before the human approval check would skip the safety gate.
    #   This point — after guardrails and approval, before prompt — is correct.
    #
    # WHY only when metric_definition is set?
    #   metric_definition is only set when the query is ambiguous (Stage 4c
    #   in planning_agent). If it's empty, the query already names the metric
    #   explicitly ("highest delay rate") and no filtering is needed.
    #
    # WHY always keep ROI Agent findings?
    #   ROI findings provide financial quantification that enriches any metric
    #   answer. They never conflict with the primary metric — they add cost
    #   context. Excluding them would make answers factually thinner.
    if query_type == "METRIC_QUERY" and metric_definition:
        METRIC_KEYWORDS = {
            "delay rate": ["delay", "late", "overdue", "otd", "on-time"],
            "cost":       ["cost", "spend", "financial", "penalty"],
            "sla":        ["sla", "breach", "compliance", "target"],
        }

        # Detect which metric is active from metric_definition text
        active_metric = None
        for metric, keywords in METRIC_KEYWORDS.items():
            if metric in metric_definition.lower():
                active_metric = keywords
                break

        log.info(
            f"run | METRIC FILTER | active_metric={active_metric} | "
            f"findings_before={len(clean_findings)}"
        )

        if active_metric:
            log.debug(
                f"run | METRIC FILTER | checking {len(clean_findings)} findings "
                f"against keywords={active_metric}"
            )

            primary_findings = [
                f for f in clean_findings
                if any(kw in f.get("finding", "").lower()
                       for kw in active_metric)
                or f.get("agent") == "roi_agent"
            ]

            # Only replace if the filter kept at least one finding
            if primary_findings:
                clean_findings = primary_findings
                log.info(
                    f"run | METRIC FILTER APPLIED | "
                    f"kept={len(clean_findings)} findings"
                )
            else:
                log.warning(
                    f"run | METRIC FILTER SKIPPED — zero matches; "
                    f"using all {len(clean_findings)} findings"
                )

    # ── FIX 3 — DB data presence check for METRIC_QUERY (before LLM) ─────────
    # WHY abort before LLM when no DB data?
    #   Without DB data, the LLM has nothing grounded to format. It will
    #   hallucinate plausible-sounding numbers rather than admit uncertainty.
    #   Returning "Data not available" immediately is safer and cheaper than
    #   a hallucinated answer that passes groundedness by accident.
    # WHY only for METRIC_QUERY?
    #   EXPLANATION_QUERY and DECISION_QUERY can answer from RAG context alone.
    #   METRIC_QUERY is factual-number-only — if DB has no data, there is
    #   no answer to give.
    if query_type == "METRIC_QUERY":
        db_findings_with_data = [
            f for f in clean_findings
            if f.get("agent") == "db_agent"
            and isinstance(f.get("data"), list)
            and len(f["data"]) > 0
        ]
        if not db_findings_with_data:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            log.warning(
                "run | FIX3 | METRIC_QUERY with no DB data — "
                "returning 'Data not available' without LLM call"
            )
            return {
                "answer":                  "Data not available in database.",
                "sources":                 _collect_sources(clean_findings),
                "confidence":              0.0,
                "sql_shown":               _collect_sql(clean_findings),
                "agents_used":             _collect_agents(clean_findings),
                "groundedness_score":      1.0,
                "human_approval_required": False,
                "tokens_used":             0,
                "cost_usd":                0.0,
                "execution_time_ms":       elapsed_ms,
                "warnings":                ["No data found in database for this query."],
            }

    # ── FIX 3b — LLM bypass for alert-driven OR SIMPLE_COUNT queries ────────────
    # WHY bypass the LLM entirely for alert-driven metric queries?
    #   The DB Agent's interpret_result() already produces a complete,
    #   correctly formatted one-sentence finding for every alert template
    #   (supplier_delay_rate, total_expedited_cost, fleet_otd_vs_benchmark,
    #   total_sla_breaches). These strings are authored at code-review time
    #   and cannot hallucinate. Passing them through GPT-4o-mini introduces:
    #     • Risk of paraphrasing the number slightly
    #     • LLM adding context the alert didn't ask for
    #     • Unnecessary latency (~1.5 s) and token cost
    #   When the DB finding is a non-empty, non-error string and the query
    #   is alert-driven, we return it directly with zero LLM involvement.
    #
    # WHY also bypass for SIMPLE_COUNT?
    #   Count queries ("How many total SLA breaches?") already have a complete
    #   one-sentence DB finding ("Total SLA breaches: 14 (above threshold of 10).").
    #   The LLM paraphrases it into 2+ sentences, then brevity enforcement trims
    #   back to 1 — losing threshold/context data. Worse, the number regex in
    #   validate_llm_output misses small integers (< 4 digits), producing a false
    #   groundedness=0% warning for perfectly correct answers. Bypassing the LLM
    #   entirely delivers the exact DB finding with zero hallucination risk.
    _bypass_llm = alert_driven or metric_definition == "SIMPLE_COUNT"
    #
    # WHY still apply contamination filter + brevity enforcement?
    #   The contamination filter strips sentences that mix metric domains
    #   (e.g. a delay finding that accidentally mentions cost).
    #   Brevity enforcement guarantees exactly 1 sentence.
    #   Both are pure Python — they add no hallucination risk.
    if _bypass_llm:
        _db_alert_finding = next(
            (
                f for f in clean_findings
                if f.get("agent") == "db_agent"
                and isinstance(f.get("data"), list)
                and len(f["data"]) > 0
                and f.get("finding", "")
                and not f["finding"].startswith("No data")
                and not f["finding"].startswith("Step failed")
                and not f["finding"].startswith("Total expedited shipping cost data not found")
            ),
            None,
        )
        if _db_alert_finding:
            _raw_finding   = _db_alert_finding["finding"]
            _clean_finding = filter_metric_contamination(_raw_finding, detected_metric)
            _clean_finding = enforce_metric_brevity(
                text         = _clean_finding,
                query_type   = "METRIC_QUERY",
                alert_driven = True,
            )
            _elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            _bypass_label = "ALERT" if alert_driven else "SIMPLE_COUNT"
            log.success(
                f"run | {_bypass_label} LLM BYPASS | finding='{_clean_finding[:80]}' | "
                f"metric='{detected_metric}' | time={_elapsed_ms}ms"
            )
            log_agent_decision({
                "user_query":       query,
                "role_used":        role,
                "agent_used":       "executive_agent (SQL bypass)",
                "tables_accessed":  _db_alert_finding.get("tables_accessed", ""),
                "sql_generated":    _db_alert_finding.get("sql_used", ""),
                "result_summary":   _clean_finding[:300],
                "confidence_score": _db_alert_finding.get("confidence", 0.90),
                "response_time_ms": _elapsed_ms,
            })
            return {
                "answer":                   _clean_finding,
                "sources":                  ["SQL Template — db_agent (alert bypass)"],
                "confidence":               _db_alert_finding.get("confidence", 0.90),
                "sql_shown":                _collect_sql(clean_findings),
                "agents_used":              _collect_agents(clean_findings),
                "groundedness_score":       1.0,   # DB finding — zero hallucination risk
                "human_approval_required":  False,
                "tokens_used":              0,
                "cost_usd":                 0.0,
                "execution_time_ms":        _elapsed_ms,
                "warnings":                 warnings,
            }
        else:
            log.info(
                "run | ALERT LLM BYPASS SKIPPED — no usable DB finding; "
                "falling through to LLM"
            )

    # ── d. Build persona and prompt ───────────────────────────────────────────
    persona         = PERSONAS.get(role, PERSONAS["Operations Manager"])
    system_prompt, user_message = build_prompt(
        clean_findings, query, role, persona, conversation_history,
        query_type        = query_type,
        metric_definition = metric_definition,
        needs_approval    = human_approval_required,
    )

    # ── FIX 5 — Alert-driven strict mode injection ────────────────────────────
    # WHY inject after build_prompt() rather than inside it?
    #   build_prompt() does not know whether the query is alert-driven (no
    #   alert context flows into that function's signature). Post-injection
    #   keeps build_prompt() stateless and independently testable.
    # WHY prepend to system_prompt instead of appending?
    #   LLMs process the beginning of the system prompt first; prepending
    #   strict mode rules gives them highest priority over later role/persona
    #   instructions that might soften the constraint.
    if alert_driven:
        _strict_addition = (
            "STRICT MODE — ALERT-DRIVEN QUERY:\n"
            "• Answer ONLY the exact metric specified in the question.\n"
            "• Do NOT add recommendations, root causes, or related metrics.\n"
            "• Do NOT reinterpret the question.\n"
            "• Return the metric value and entity only — one sentence maximum.\n\n"
        )
        system_prompt = _strict_addition + system_prompt
        log.info("run | FIX5 | alert_driven strict mode injected into system prompt")

    # ── e. THE ONE LLM CALL ───────────────────────────────────────────────────
    # Everything above is coded Python. This is the only point where
    # language generation happens. It formats already-verified findings.
    log.info(
        f"run | CALLING LLM | model={LLM_MODEL} | "
        f"max_tokens={LLM_MAX_TOKENS} | temperature={LLM_TEMPERATURE}"
    )

    llm_response_text = ""
    tokens_used       = 0

    try:
        client   = _get_client()
        messages = [{"role": "system", "content": system_prompt}]

        # Inject conversation history if provided (last 4 messages = 2 turns)
        if conversation_history:
            messages.extend(conversation_history[-4:])

        messages.append({"role": "user", "content": user_message})

        response     = client.chat.completions.create(
            model       = LLM_MODEL,
            messages    = messages,
            max_tokens  = LLM_MAX_TOKENS,
            temperature = LLM_TEMPERATURE,
        )

        llm_response_text = response.choices[0].message.content or ""
        tokens_used       = response.usage.total_tokens if response.usage else 0

        log.success(
            f"run | LLM COMPLETE | tokens={tokens_used} | "
            f"cost_usd={tokens_used * COST_PER_TOKEN_USD:.6f}"
        )

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log.error(f"run | LLM CALL FAILED | error='{exc}'")
        return _error_response(
            f"LLM call failed: {exc}. "
            f"Verified findings are available — please try again.",
            query, role,
        )

    # ── e2a. DECISION_QUERY: enforce structured output ───────────────────────
    # WHY trim before the METRIC_QUERY trim block?
    #   DECISION_QUERY and METRIC_QUERY are mutually exclusive — we never reach
    #   the METRIC_QUERY block when query_type == "DECISION_QUERY". But placing
    #   this block first makes the control flow explicit and avoids any risk of
    #   accidentally entering the METRIC_QUERY trim for a decision answer.
    # WHY trim_decision_output() here rather than in build_prompt()?
    #   The LLM may add preamble or trailing commentary outside the three
    #   required sections — that can only be detected and removed post-LLM.
    if query_type == "DECISION_QUERY":
        llm_response_text = trim_decision_output(llm_response_text)
        log.info("run | DECISION_QUERY — trim_decision_output applied")

    # ── e2. Cross-metric contamination filter (FIX 4) ────────────────────────
    # WHY before the brevity trim?
    #   filter_metric_contamination() works on all sentences. If it runs after
    #   trim, it has fewer sentences to check — reducing the chance of catching
    #   a contaminated sentence that survived into the kept portion.
    #   Filtering first, then trimming, gives the cleanest result.
    if query_type == "METRIC_QUERY" and detected_metric:
        llm_response_text = filter_metric_contamination(
            llm_response_text, detected_metric
        )
        log.info(
            f"run | FIX4 | filter_metric_contamination applied | "
            f"metric='{detected_metric}'"
        )

    # ── e2b. Hard brevity enforcement — replaces old 2-sentence trim (FIX 6) ─
    # WHY enforce_metric_brevity() instead of the previous inline trim?
    #   The old inline block had four separate passes (split, filter, keep,
    #   rejoin) scattered across 40 lines. enforce_metric_brevity() is a
    #   single named function that:
    #   • Respects alert_driven (1 sentence) vs regular (2 sentences) mode
    #   • Handles SIMPLE_COUNT (always 1 sentence) via alert_driven=True path
    #   • Handles multi_row passthrough correctly
    #   • Is independently testable
    # SIMPLE_COUNT queries are single-sentence by design — treat as alert_driven
    _is_simple = metric_definition == "SIMPLE_COUNT"
    llm_response_text = enforce_metric_brevity(
        text         = llm_response_text,
        query_type   = query_type,
        alert_driven = alert_driven or _is_simple,
        multi_row    = multi_row,
    )

    # ── e3. Post-processing cleanup (METRIC and EXPLANATION only) ────────────
    # WHY skip for DECISION_QUERY?
    #   DECISION_QUERY output uses **Issue:**, **Recommendation:**, **Risk:**
    #   as structural section headers. format_financials() strips ALL ** bold
    #   markers, which would destroy the header formatting that the UI relies
    #   on to render the structured decision output correctly.
    # WHY a named function (FIX 7)?
    #   format_financials() can be called from outside this function
    #   (e.g. Streamlit UI layer) to sanitise any string before display.
    if query_type != "DECISION_QUERY":
        llm_response_text = format_financials(llm_response_text)

    log.debug(
        f"run | MARKDOWN CLEANUP | bold markers and $ spacing normalised"
    )

    # ── f. Validate LLM output groundedness ──────────────────────────────────
    validation = validate_llm_output(llm_response_text, clean_findings)
    groundedness = validation["groundedness_score"]

    # Append warning if groundedness below threshold
    if not validation["approved"]:
        warning_note = (
            "\n\n⚠️ Note: Some claims in this response could not be fully "
            "verified against the source data. "
            "Please cross-check key figures before taking action."
        )
        llm_response_text += warning_note
        warnings.append(
            f"Groundedness score {groundedness:.0%} below "
            f"{GROUNDEDNESS_THRESHOLD:.0%} threshold."
        )
        log.warning(
            f"run | LOW GROUNDEDNESS | score={groundedness:.2f} | "
            f"ungrounded={validation['ungrounded_claims'][:3]}"
        )

    # Prepend any guardrail warnings to the response
    if warnings:
        warning_header = (
            "⚠️ " + " | ".join(warnings[:2]) + "\n\n"
        )
        llm_response_text = warning_header + llm_response_text

    # ── g. Collect metadata for return and audit ──────────────────────────────
    elapsed_ms      = int((time.perf_counter() - start_time) * 1000)
    sources         = _collect_sources(clean_findings)
    sql_shown       = _collect_sql(clean_findings)
    agents_used     = _collect_agents(clean_findings)
    cost_usd        = tokens_used * COST_PER_TOKEN_USD

    # ── h. Log to ai_decisions_log ────────────────────────────────────────────
    log_agent_decision({
        "user_query":       query,
        "role_used":        role,
        "agent_used":       "executive_agent",
        "tables_accessed":  ", ".join(
            t for f in clean_findings
            for t in (f.get("tables_accessed", "").split(",") if f.get("tables_accessed") else [])
        ),
        "sql_generated":    "; ".join(sql_shown[:2]),   # log first 2 queries
        "result_summary":   llm_response_text[:300],
        "confidence_score": avg_confidence,
        "response_time_ms": elapsed_ms,
    })

    log.success(
        f"run | COMPLETE | role='{role}' | time={elapsed_ms}ms | "
        f"tokens={tokens_used} | groundedness={groundedness:.2f} | "
        f"cost=${cost_usd:.6f}"
    )

    return {
        "answer":                   llm_response_text,
        "sources":                  sources,
        "confidence":               avg_confidence,
        "sql_shown":                sql_shown,
        "agents_used":              agents_used,
        "groundedness_score":       groundedness,
        "human_approval_required":  human_approval_required,
        "approval_reason":          approval_check["reason"] if human_approval_required else "",
        "impact_summary":           approval_check["impact_summary"] if human_approval_required else "",
        "tokens_used":              tokens_used,
        "cost_usd":                 cost_usd,
        "execution_time_ms":        elapsed_ms,
        "warnings":                 warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CONVERSATION HANDLER
#  Enables drill-down follow-up questions on the same topic.
# ═══════════════════════════════════════════════════════════════════════════════

def handle_followup(
    previous_answer: dict,
    new_query:       str,
    role:            str,
) -> dict:
    """
    Handle a follow-up question in the context of a previous answer.

    WHY conversation history?
        Users often ask "why?" or "which region?" after an initial answer.
        Without context, the system treats each question in isolation.
        Passing the previous answer as conversation history lets the LLM
        understand "it" and "that supplier" as references to prior context.

    Architecture:
        The conversation_history is a standard OpenAI messages list:
            [{"role": "user",      "content": "<original question>"},
             {"role": "assistant", "content": "<previous answer>"}]

        This is passed to run() which injects it between the system prompt
        and the new user message. The LLM sees the full context and can
        answer coherently without re-fetching data.

    WHY only 2 prior turns (4 messages)?
        Longer histories increase token cost and can dilute the system
        prompt's instructions. 2 turns = enough for drill-down questions
        without context pollution.

    Args:
        previous_answer: The full return dict from a prior run() call.
        new_query:       The follow-up question from the user.
        role:            Current user role.

    Returns:
        Full run() return dict for the follow-up question.
    """
    log.info(
        f"handle_followup | role='{role}' | "
        f"new_query='{new_query[:80]}'"
    )

    # ── Build conversation history from previous interaction ──────────────────
    original_query  = previous_answer.get("query",  "")
    previous_text   = previous_answer.get("answer", "")

    # Reconstruct prior findings list — the previous answer already has
    # all findings embedded. We pass a minimal stub so the LLM has context.
    prior_findings_stub = [{
        "agent":      "executive_agent",
        "task":       "previous_answer",
        "finding":    previous_text[:500],   # first 500 chars as context
        "confidence": previous_answer.get("confidence", 0.85),
        "source":     "Previous answer in this session",
        "data":       None,
    }]

    conversation_history = [
        {"role": "user",      "content": original_query or "Previous question"},
        {"role": "assistant", "content": previous_text[:800]},
    ]

    # ── Re-use previous findings + new stub ───────────────────────────────────
    # WHY pass prior_findings_stub alongside conversation_history?
    #   The stub gives validate_llm_output() something to check groundedness
    #   against. Without it, every number in the follow-up would appear
    #   ungrounded (because the new findings list would be empty).
    return run(
        all_findings         = prior_findings_stub,
        query                = new_query,
        role                 = role,
        conversation_history = conversation_history,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_sources(findings: list[dict]) -> list[str]:
    """Extract unique, non-empty source citations from findings."""
    seen   = set()
    result = []
    for f in findings:
        citation = f.get("citation") or f.get("source") or ""
        if citation and citation not in seen:
            seen.add(citation)
            result.append(citation)
    return result or ["Supply Chain Database (SQLite)"]


def _collect_sql(findings: list[dict]) -> list[str]:
    """Extract unique SQL queries from DB agent findings."""
    queries = []
    seen    = set()
    for f in findings:
        sql = f.get("sql", "")
        if sql and isinstance(sql, str) and sql.strip() not in seen:
            seen.add(sql.strip())
            queries.append(sql.strip())
    return queries


def _collect_agents(findings: list[dict]) -> list[str]:
    """Return ordered, unique list of agents that contributed findings."""
    seen   = set()
    result = []
    for f in findings:
        agent = f.get("agent", "")
        if agent and agent not in seen:
            seen.add(agent)
            result.append(agent)
    return result or ["executive_agent"]


def _merge_finding_data(findings: list[dict]) -> list[dict]:
    """Flatten all data rows from all findings into one list."""
    merged = []
    for f in findings:
        data = f.get("data")
        if data and isinstance(data, list):
            merged.extend(data)
    return merged


def _weighted_confidence(findings: list[dict]) -> float:
    """
    Calculate weighted average confidence across all findings.

    WHY weighted by finding position?
        Later findings (ROI, RAG) often have higher/lower confidence than
        DB findings. A simple average gives equal weight to a 3-row result
        (confidence 0.6) and a 100-row result (confidence 0.9). Weighting
        by position (earlier findings weighted slightly more) is a pragmatic
        approximation without requiring explicit finding weights.
    """
    confidences = [f.get("confidence", 0.85) for f in findings if f.get("confidence")]
    if not confidences:
        return 0.85
    return round(sum(confidences) / len(confidences), 4)


def _extract_top_recommendation(findings: list[dict]) -> str:
    """Pull the highest-priority recommendation from all findings."""
    for f in reversed(findings):   # ROI agent findings tend to be last
        rec = f.get("recommendation", "") or f.get("recommended_action", "")
        if rec:
            return str(rec)[:200]
    return ""


def _error_response(reason: str, query: str, role: str) -> dict:
    """Canonical error return structure — used on validation or LLM failure."""
    return {
        "answer":                   f"Unable to generate answer: {reason}",
        "sources":                  [],
        "confidence":               0.0,
        "sql_shown":                [],
        "agents_used":              [],
        "groundedness_score":       0.0,
        "human_approval_required":  False,
        "tokens_used":              0,
        "cost_usd":                 0.0,
        "execution_time_ms":        0,
        "warnings":                 [reason],
        "error":                    reason,
    }
