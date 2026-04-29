#!/usr/bin/env python3
# evaluation/dynamic_test_run.py
#
# Diagnostic script — runs 20 dynamic queries through the full production
# pipeline and reports routing, template, answer, and pass/fail details.
#
# PURPOSE: Diagnostic only. Do NOT fix anything based on this script.
#
# Usage:
#   python3 evaluation/dynamic_test_run.py
#   python3 evaluation/dynamic_test_run.py --verbose
#
# Exit code 0  = all 20 tests passed
# Exit code 1  = one or more failures
#
# WHY separate from run_eval.py?
#   run_eval.py is the locked golden-set harness. This file is a diagnostic
#   probe for query routing, template resolution, and answer content — it
#   captures richer per-query metadata (query_type, agents_used, template_used)
#   and classifies failures by category (wrong_dimension, formatting_error, etc.)
#   without touching or modifying any production code.

import sys
import os
import time
import json
import argparse
from datetime import datetime

# ── Path bootstrap — must run from repo root or evaluation/ ──────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Pre-import pipeline and silence loguru so output stays readable ───────────
from services.graph import run_pipeline as _run_pipeline_import  # noqa: F401
try:
    from loguru import logger as _log
    _log.remove()
    _log.add(sys.stderr, level="CRITICAL")
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TEST CASE DEFINITIONS
#  Each case has: id, query, role, must_contain, must_not_contain
#  No expected category — failure_category is inferred dynamically from the
#  content of must_not_contain violations and missing terms.
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": "D01",
        "query": "What is the average delay in days for delayed shipments?",
        "role": "Operations Manager",
        "must_contain": ["6.0", "days"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D02",
        "query": "How many SLA breaches do we have?",
        "role": "Operations Manager",
        "must_contain": ["SLA", "breach"],
        "must_not_contain": ["</div>"],
    },
    {
        "id": "D03",
        "query": "What is the maximum delay observed?",
        "role": "Operations Manager",
        "must_contain": ["12"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D04",
        "query": "What category has the highest delay rate?",
        "role": "Operations Manager",
        "must_contain": ["PPE", "21.1"],
        "must_not_contain": ["South", "Supplier_Dispatch", "</div>"],
    },
    {
        "id": "D05",
        "query": "What is the total shipment value?",
        "role": "Operations Manager",
        "must_contain": ["51"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D06",
        "query": "What is the AI return on investment for 2024?",
        "role": "CFO",
        "must_contain": ["340"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D07",
        "query": "How much did we spend on expedited shipping?",
        "role": "CFO",
        "must_contain": ["379,997"],
        "must_not_contain": ["</div>", "\n,\n", "379997("],
    },
    {
        "id": "D08",
        "query": "How many shipments are in transit?",
        "role": "Operations Manager",
        "must_contain": ["transit"],
        "must_not_contain": ["</div>"],
    },
    {
        "id": "D09",
        "query": "Which supplier has the most SLA breaches?",
        "role": "Operations Manager",
        "must_contain": ["SUP001"],
        "must_not_contain": ["</div>", "Supplier_Dispatch"],
    },
    {
        "id": "D10",
        "query": "What is the delay rate for the North region?",
        "role": "Operations Manager",
        "must_contain": ["North", "11.1"],
        "must_not_contain": ["South", "Supplier_Dispatch", "</div>"],
    },
    {
        "id": "D11",
        "query": "What if Supplier A delay rate drops to 15%?",
        "role": "Operations Manager",
        "must_contain": ["SUP001", "15"],
        "must_not_contain": ["</div>", "\n,\n", "75,999"],
    },
    {
        "id": "D12",
        "query": "What is the cumulative AI savings?",
        "role": "CFO",
        "must_contain": ["401,000"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D13",
        "query": "Should we expedite all SUP001 shipments?",
        "role": "Operations Manager",
        "must_contain": ["approval", "human"],
        "must_not_contain": ["</div>"],
    },
    {
        "id": "D14",
        "query": "What is the delay rate for PPE category?",
        "role": "Operations Manager",
        "must_contain": ["PPE", "21.1"],
        "must_not_contain": ["South", "Supplier_Dispatch", "</div>"],
    },
    {
        "id": "D15",
        "query": "How many shipments does SUP002 handle?",
        "role": "Operations Manager",
        "must_contain": ["34", "SUP002"],
        "must_not_contain": ["</div>"],
    },
    {
        "id": "D16",
        "query": "What is the on-time delivery rate for Supplier C?",
        "role": "Operations Manager",
        "must_contain": ["SUP003", "80"],
        "must_not_contain": ["</div>", "Supplier_Dispatch"],
    },
    {
        "id": "D17",
        "query": "Show me shipments delayed in December 2024",
        "role": "Operations Manager",
        "must_contain": ["1", "December"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D18",
        "query": "What is the supply chain cost trend from 2022 to 2024?",
        "role": "CFO",
        "must_contain": ["2022", "2024", "5,763", "5,841"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D19",
        "query": "Can you show AI investment for 2024?",
        "role": "CFO",
        "must_contain": ["94,000"],
        "must_not_contain": ["</div>", "\n,\n"],
    },
    {
        "id": "D20",
        "query": "Which supplier should we prioritize for performance review?",
        "role": "Operations Manager",
        "must_contain": ["SUP001"],
        "must_not_contain": ["</div>"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one(query: str, role: str) -> dict:
    """
    Run a single query through the full production pipeline and return the raw
    result dict enriched with diagnostic fields:
        query_type     str   — METRIC_QUERY / EXPLANATION_QUERY / DECISION_QUERY /
                               WHATIF_QUERY
        agents_used    list  — specialist agents that contributed
        template_used  str   — first db_agent SQL template key (or "" if none)
        answer         str   — final formatted answer
        confidence     float — pipeline confidence (0.0–1.0)
        elapsed_ms     int   — wall-clock time in ms
    """
    from services.graph import run_pipeline

    t0     = time.perf_counter()
    result = run_pipeline(query, role)
    elapsed = int((time.perf_counter() - t0) * 1000)

    # ── Extract db_agent template key from findings ───────────────────────────
    # WHY all_findings?
    #   Each finding dict has an "agent" field and a "task" field.
    #   For db_agent findings, "task" is the resolved SQL template key
    #   (e.g. "avg_delay_days", "highest_delay_rate_region").
    #   We take the first db_agent finding's task as the "primary template".
    template_used = ""
    for f in result.get("all_findings", []):
        if isinstance(f, dict) and f.get("agent") == "db_agent":
            t = f.get("task", "")
            if t:
                template_used = t
                break

    return {
        "answer":        result.get("answer",      ""),
        "confidence":    result.get("confidence",   0.0),
        "elapsed_ms":    elapsed,
        "query_type":    result.get("query_type",   ""),
        "agents_used":   result.get("agents_used",  []),
        "template_used": template_used,
        "all_findings":  result.get("all_findings", []),
        "raw":           result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PASS / FAIL + FAILURE CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

def _check(answer: str,
           must_contain: list[str],
           must_not_contain: list[str]) -> tuple[bool, list[str], list[str]]:
    """Return (passed, missing_terms, found_bad_terms). Case-insensitive."""
    low       = answer.lower()
    missing   = [t for t in must_contain     if t.lower() not in low]
    found_bad = [t for t in must_not_contain if t.lower() in low]
    return (not missing and not found_bad), missing, found_bad


def _infer_failure_category(
    tc:        dict,
    answer:    str,
    missing:   list[str],
    found_bad: list[str],
    run:       dict,
) -> tuple[str, bool]:
    """
    Heuristically classify why a test case failed and whether it blocks a demo.

    Returns:
        (failure_category_str, is_demo_blocker_bool)

    Category hierarchy (first match wins):
        html_artifact         — </div> or similar HTML leaked into answer
        formatting_error      — split number artefacts (\n,\n)
        unsolicited_rec       — recommendation/suggest in answer when not expected
        wrong_dimension       — answer mentions wrong region, wrong category,
                                or "Supplier_Dispatch" where a named entity expected
        wrong_entity          — correct dimension but wrong supplier/category
        wrong_value           — right entity but wrong number
        routing_error         — template key doesn't match expected intent
        no_answer             — answer empty or "Data not available"
        content_mismatch      — generic: something missing, nothing specific
    """
    answer_low = answer.lower()

    # 1. HTML artifact
    if any("</div>" in b or "</" in b for b in found_bad):
        return "html_artifact", True

    # 2. Formatting artefact (split numbers)
    if any("\n,\n" in b for b in found_bad):
        return "formatting_error", True

    # 3. Unsolicited recommendation text
    if any(b.lower() in ("recommendation", "suggest") for b in found_bad):
        return "unsolicited_rec", False

    # 4. Wrong dimension (answer mentions an entity from a different dimension)
    wrong_dim_tokens = {"south", "supplier_dispatch", "north", "east", "west"}
    if any(b.lower() in wrong_dim_tokens for b in found_bad):
        return "wrong_dimension", True

    # 5. No answer at all
    if not answer.strip() or "data not available" in answer_low:
        return "no_answer", True

    # 6. Answer has content but required numbers/entities are missing
    if missing:
        # If missing items are numeric → value mismatch
        if all(_looks_numeric(m) for m in missing):
            return "wrong_value", True
        # If missing items are entity IDs (SUP*, PPE, North…) → entity mismatch
        if all(_looks_entity(m) for m in missing):
            return "wrong_entity", True
        # Generic content mismatch
        return "content_mismatch", True

    return "content_mismatch", False


def _looks_numeric(s: str) -> bool:
    """True if s is a number string (possibly with commas, %, $)."""
    import re
    return bool(re.match(r'^[\$]?[\d,\.]+%?$', s.strip()))


def _looks_entity(s: str) -> bool:
    """True if s looks like a supplier ID, region name, or product category."""
    import re
    return bool(re.match(r'^(SUP\d+|PPE|North|South|East|West)$', s.strip(), re.IGNORECASE))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _short_agents(agents: list) -> str:
    """Compress agent list to a short label for the one-liner output."""
    if not agents:
        return "—"
    labels = []
    for a in agents:
        al = str(a).lower()
        if "db"        in al:  labels.append("DB")
        elif "rag"     in al:  labels.append("RAG")
        elif "roi"     in al:  labels.append("ROI")
        elif "exec"    in al:  labels.append("Exec")
        elif "human"   in al:  labels.append("HumanLoop")
        elif "simul"   in al:  labels.append("Sim")
        else:                  labels.append(a[:8])
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            deduped.append(l)
    return "+".join(deduped)


def _snippet(text: str, width: int = 80) -> str:
    """Return a single-line snippet (no newlines) truncated to width."""
    s = text.replace("\n", " ").replace("\r", "").strip()
    return s[:width] + "…" if len(s) > width else s


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN DIAGNOSTIC LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(verbose: bool = False) -> tuple[int, int, list]:
    """
    Run all 20 diagnostic test cases and print results.

    Returns:
        (passed_count, total_count, results_list)
    """
    results:            list[dict] = []
    failure_categories: dict[str, int] = {}
    timing_ms:          list[int] = []

    width = 72
    print(f"\n{'='*width}")
    print(f"  SUPPLY COMMAND AI — DYNAMIC DIAGNOSTIC RUN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Queries: {len(TEST_CASES)}  |  Mode: diagnostic (no fixes)")
    print(f"{'='*width}\n")

    for tc in TEST_CASES:
        tid   = tc["id"]
        query = tc["query"]
        role  = tc["role"]

        # ── Run through pipeline ──────────────────────────────────────────────
        try:
            run = _run_one(query, role)
            answer       = run["answer"]
            confidence   = run["confidence"]
            elapsed_ms   = run["elapsed_ms"]
            query_type   = run["query_type"]
            agents_used  = run["agents_used"]
            template_key = run["template_used"]
        except Exception as exc:
            import traceback
            answer       = f"PIPELINE ERROR: {exc}"
            confidence   = 0.0
            elapsed_ms   = 0
            query_type   = "ERROR"
            agents_used  = []
            template_key = ""
            traceback.print_exc()

        timing_ms.append(elapsed_ms)

        # ── Pass / fail ───────────────────────────────────────────────────────
        passed, missing, found_bad = _check(
            answer,
            tc["must_contain"],
            tc["must_not_contain"],
        )

        failure_cat  = ""
        demo_blocker = False
        if not passed:
            failure_cat, demo_blocker = _infer_failure_category(
                tc, answer, missing, found_bad, run
            )
            failure_categories[failure_cat] = \
                failure_categories.get(failure_cat, 0) + 1

        # ── Build result record ───────────────────────────────────────────────
        rec = {
            "id":            tid,
            "query":         query,
            "role":          role,
            "passed":        passed,
            "query_type":    query_type,
            "agents_used":   agents_used,
            "template_used": template_key,
            "answer":        answer,
            "confidence":    confidence,
            "elapsed_ms":    elapsed_ms,
            "missing":       missing,
            "found_bad":     found_bad,
            "failure_cat":   failure_cat,
            "demo_blocker":  demo_blocker,
        }
        results.append(rec)

        # ── Print one-liner ───────────────────────────────────────────────────
        status   = "PASS" if passed else "FAIL"
        icon     = "✅" if passed else "❌"
        agents_s = _short_agents(agents_used)
        tmpl_s   = template_key[:30] if template_key else "—"
        conf_s   = f"{confidence:.2f}"

        if passed:
            print(
                f"[{tid}] {icon} {status} | "
                f"query_type={query_type or '—':<20} | "
                f"agents={agents_s:<16} | "
                f"template={tmpl_s:<32} | "
                f"confidence={conf_s}  ({elapsed_ms}ms)"
            )
        else:
            answer_snip = _snippet(answer, 60)
            print(
                f"[{tid}] {icon} {status} | "
                f"query_type={query_type or '—':<20} | "
                f"agents={agents_s:<16} | "
                f"template={tmpl_s:<32} | "
                f"failure={failure_cat}  demo_blocker={demo_blocker}"
            )

        # ── Verbose detail block ──────────────────────────────────────────────
        if verbose or not passed:
            print(f"       Query    : {query}")
            print(f"       Role     : {role}")
            if not passed:
                if missing:
                    print(f"       MISSING  : {missing}")
                if found_bad:
                    print(f"       BAD TEXT : {found_bad}")
            print(f"       Answer   : {_snippet(answer, 120)}")
            if verbose:
                print(f"       Full answer ({len(answer)} chars):")
                # Indent every line for readability
                for line in answer.split("\n"):
                    print(f"         {line}")
            print()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    total        = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    avg_ms       = int(sum(timing_ms) / len(timing_ms)) if timing_ms else 0
    p99_ms       = sorted(timing_ms)[int(len(timing_ms) * 0.99)] if len(timing_ms) > 1 else timing_ms[0] if timing_ms else 0

    pct   = round(100 * passed_count / total) if total else 0
    grade = "🟢" if pct == 100 else "🟡" if pct >= 70 else "🔴"

    print(f"\n{'='*width}")
    print(f"  {grade} RESULT: {passed_count}/{total} PASSED  ({pct}%)")
    print(f"  Timing: avg={avg_ms}ms  p99={p99_ms}ms")
    print(f"{'='*width}")

    # ── Failure category breakdown ─────────────────────────────────────────────
    if failure_categories:
        print(f"\n  FAILURE CATEGORIES:")
        print(f"  {'Category':<25} {'Count':>5}  {'Demo Blocker?'}")
        print(f"  {'-'*50}")
        # Determine if any failure of each category is a demo blocker
        cat_blocker: dict[str, bool] = {}
        for r in results:
            if not r["passed"] and r["failure_cat"]:
                cat_blocker[r["failure_cat"]] = \
                    cat_blocker.get(r["failure_cat"], False) or r["demo_blocker"]
        for cat in sorted(failure_categories, key=lambda c: -failure_categories[c]):
            cnt     = failure_categories[cat]
            blocker = "⛔ YES" if cat_blocker.get(cat) else "  no"
            print(f"  {'❌ ' + cat:<25} {cnt:>5}  {blocker}")
    else:
        print(f"\n  ✅ No failures — all categories clean")

    # ── Per-query result table ─────────────────────────────────────────────────
    print(f"\n  {'ID':<5} {'Status':<6} {'query_type':<22} "
          f"{'agents':<18} {'template':<32} {'conf':>5}  {'ms':>6}")
    print(f"  {'-'*100}")
    for r in results:
        status   = "PASS" if r["passed"] else "FAIL"
        icon     = "✅" if r["passed"] else "❌"
        agents_s = _short_agents(r["agents_used"])
        tmpl_s   = (r["template_used"] or "—")[:30]
        conf_s   = f"{r['confidence']:.2f}"
        ms_s     = str(r["elapsed_ms"])
        qt       = (r["query_type"] or "—")[:20]
        print(
            f"  {r['id']:<5} {icon} {status:<4} "
            f"{qt:<22} {agents_s:<18} {tmpl_s:<32} {conf_s:>5}  {ms_s:>6}ms"
        )

    # ── Failing queries with full diagnostics ─────────────────────────────────
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  FAILING QUERIES ({len(failures)} of {total}):")
        print(f"  {'='*70}")
        for r in failures:
            blocker_tag = " [DEMO BLOCKER]" if r["demo_blocker"] else ""
            print(f"\n  [{r['id']}] {r['query']}")
            print(f"    Role         : {r['role']}")
            print(f"    query_type   : {r['query_type'] or '—'}")
            print(f"    template     : {r['template_used'] or '—'}")
            print(f"    agents       : {r['agents_used']}")
            print(f"    failure      : {r['failure_cat']}{blocker_tag}")
            if r["missing"]:
                print(f"    missing      : {r['missing']}")
            if r["found_bad"]:
                print(f"    bad_text     : {r['found_bad']}")
            print(f"    answer snippet: {_snippet(r['answer'], 100)}")
    else:
        print(f"\n  ✅ All {total} queries passed — no failures to report")

    # ── Demo blocker summary ───────────────────────────────────────────────────
    blockers = [r for r in results if not r["passed"] and r["demo_blocker"]]
    print(f"\n  DEMO BLOCKER SUMMARY:")
    if blockers:
        print(f"  ⛔ {len(blockers)} demo-blocking failure(s):")
        for r in blockers:
            print(f"     [{r['id']}] {r['query'][:60]}  → {r['failure_cat']}")
    else:
        print(f"  ✅ No demo blockers — safe to proceed")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        os.path.dirname(__file__),
        f"dynamic_results_{ts}.json",
    )
    payload = {
        "run_at":             datetime.now().isoformat(),
        "total":              total,
        "passed":             passed_count,
        "failed":             failed_count,
        "pass_pct":           pct,
        "avg_ms":             avg_ms,
        "p99_ms":             p99_ms,
        "failure_categories": failure_categories,
        "results": [
            {k: v for k, v in r.items() if k != "all_findings"}
            for r in results
        ],
    }
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Full results → {output_file}")
    print(f"{'='*width}\n")

    return passed_count, total, results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply Command AI — dynamic diagnostic run (20 queries)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full answer text for every query (not just failures).",
    )
    args = parser.parse_args()

    passed, total, _ = run_diagnostics(verbose=args.verbose)
    sys.exit(0 if passed == total else 1)
