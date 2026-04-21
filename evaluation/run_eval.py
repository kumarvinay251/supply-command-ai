#!/usr/bin/env python3
# evaluation/run_eval.py
#
# Automated evaluation harness for Supply Command AI.
# Run with:   python3 evaluation/run_eval.py
#             python3 evaluation/run_eval.py --verbose
#             python3 evaluation/run_eval.py --category financial
#             python3 evaluation/run_eval.py --fast   (skips full pipeline, direct DB)
#
# Exit code 0  = all tests passed
# Exit code 1  = one or more tests failed
#
# WHY a standalone script instead of pytest?
#   pytest adds test-discovery complexity and can't easily emit the demo-ready
#   whitelist table. This script is self-contained: run it, read the output,
#   fix red categories, run again.

import sys
import os
import json
import time
import argparse
from datetime import datetime

# ── Path bootstrap — must run from repo root or evaluation/ ──────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.test_cases import TEST_CASES

# ── Pre-import the pipeline so services/logger.py bootstrap runs now ─────────
# Then immediately silence all loguru handlers.  The bootstrap in
# services/logger.py calls logger.add(sys.stdout, ...) at import time.
# We remove those handlers here so the eval output is clean.
from services.graph import run_pipeline as _run_pipeline_import  # noqa: F401
try:
    from loguru import logger as _log
    _log.remove()                              # drop ALL handlers (stdout + file)
    _log.add(sys.stderr, level="CRITICAL")     # keep only crash-level noise
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one(query: str, role: str) -> dict:
    """
    Run a single query through the full production pipeline.

    Returns a dict with at minimum:
        answer      str   — the final text shown to the user
        confidence  float — pipeline confidence (0.0 – 1.0)
        elapsed_ms  int   — wall-clock time
    """
    from services.graph import run_pipeline

    t0     = time.perf_counter()
    result = run_pipeline(query, role)
    elapsed = int((time.perf_counter() - t0) * 1000)

    return {
        "answer":     result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "elapsed_ms": elapsed,
        "raw":        result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PASS / FAIL LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _check(answer: str, must_contain: list, must_not_contain: list) -> tuple[bool, list, list]:
    """
    Return (passed, missing_terms, found_bad_terms).

    Case-insensitive. All strings in must_contain must appear in answer.
    No string in must_not_contain may appear in answer.
    """
    low = answer.lower()
    missing   = [e for e in must_contain      if e.lower() not in low]
    found_bad = [b for b in must_not_contain  if b.lower() in low]
    return (not missing and not found_bad), missing, found_bad


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    test_cases=None,
    verbose: bool = False,
    category_filter: str | None = None,
) -> tuple[int, int, list]:
    """
    Execute all test cases and print a structured report.

    Args:
        test_cases:      List of (query, role, contains, not_contains, category).
                         Defaults to evaluation.test_cases.TEST_CASES.
        verbose:         Print every test case, not just failures.
        category_filter: If set, only run cases in this category.

    Returns:
        (passed_count, total_count, results_list)
    """
    if test_cases is None:
        test_cases = TEST_CASES

    if category_filter:
        test_cases = [tc for tc in test_cases if tc[4] == category_filter]
        if not test_cases:
            print(f"No test cases found for category '{category_filter}'.")
            return 0, 0, []

    results          = []
    category_stats:  dict[str, dict] = {}
    timing_ms:       list[int]       = []

    print(f"\n{'='*72}")
    print(f"  SUPPLY COMMAND AI — AUTOMATED EVALUATION HARNESS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test cases: {len(test_cases)}"
          + (f"  |  Category: {category_filter}" if category_filter else ""))
    print(f"{'='*72}\n")

    for idx, (query, role, must_contain, must_not_contain, category) in \
            enumerate(test_cases, start=1):

        # ── Run pipeline ──────────────────────────────────────────────────────
        try:
            run = _run_one(query, role)
            answer     = run["answer"]
            confidence = run["confidence"]
            elapsed    = run["elapsed_ms"]
        except Exception as exc:
            import traceback
            answer     = f"PIPELINE ERROR: {exc}"
            confidence = 0.0
            elapsed    = 0
            traceback.print_exc()

        timing_ms.append(elapsed)

        # ── Check pass / fail ─────────────────────────────────────────────────
        passed, missing, found_bad = _check(answer, must_contain, must_not_contain)

        # ── Accumulate category stats ─────────────────────────────────────────
        if category not in category_stats:
            category_stats[category] = {"pass": 0, "fail": 0, "total": 0,
                                         "timing_ms": []}
        category_stats[category]["total"] += 1
        category_stats[category]["timing_ms"].append(elapsed)
        if passed:
            category_stats[category]["pass"] += 1
        else:
            category_stats[category]["fail"] += 1

        # ── Build result record ───────────────────────────────────────────────
        rec = {
            "idx":        idx,
            "query":      query,
            "role":       role,
            "category":   category,
            "passed":     passed,
            "answer":     answer[:300],
            "confidence": confidence,
            "elapsed_ms": elapsed,
            "missing":    missing,
            "found_bad":  found_bad,
        }
        results.append(rec)

        # ── Print per-test output ─────────────────────────────────────────────
        status = "✅ PASS" if passed else "❌ FAIL"

        if verbose or not passed:
            print(f"Q{idx:02d} [{category:<16}] {status}  ({elapsed}ms, conf={confidence:.2f})")
            print(f"     Query:  {query}")
            print(f"     Role:   {role}")
            if not passed:
                if missing:
                    print(f"     MISSING:   {missing}")
                if found_bad:
                    print(f"     BAD TEXT:  {found_bad}")
            print(f"     Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print()
        else:
            # Compact one-liner for passing tests
            print(f"Q{idx:02d} [{category:<16}] {status}  ({elapsed}ms)")

    # ── Print summary ─────────────────────────────────────────────────────────
    total  = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    avg_ms       = int(sum(timing_ms) / len(timing_ms)) if timing_ms else 0
    p100_ms      = sorted(timing_ms)[int(len(timing_ms) * 0.99)] if timing_ms else 0

    print(f"\n{'='*72}")
    pct = round(100 * passed_count / total) if total else 0
    grade = "🟢" if pct == 100 else "🟡" if pct >= 80 else "🔴"
    print(f"  {grade} RESULT: {passed_count}/{total} PASSED  ({pct}%)")
    print(f"  Timing: avg={avg_ms}ms  p99={p100_ms}ms")
    print(f"{'='*72}")

    print(f"\n  BY CATEGORY:")
    print(f"  {'Category':<22} {'Pass':>5} {'Total':>5} {'%':>5}  Bar")
    print(f"  {'-'*55}")
    for cat in sorted(category_stats):
        s   = category_stats[cat]
        pct = round(100 * s["pass"] / s["total"]) if s["total"] else 0
        bar = ("█" * (pct // 10)).ljust(10)
        ico = "✅" if pct == 100 else "⚠️ " if pct >= 70 else "❌"
        print(f"  {ico} {cat:<20} {s['pass']:>5}/{s['total']:<5} {pct:>4}%  {bar}")

    # ── Demo whitelist ────────────────────────────────────────────────────────
    perfect = [c for c, s in category_stats.items() if s["pass"] == s["total"]]
    broken  = [c for c, s in category_stats.items()
               if s["total"] > 0 and s["pass"] / s["total"] < 0.70]

    print(f"\n  ✅ DEMO WHITELIST  (100% pass rate):")
    if perfect:
        for c in sorted(perfect):
            print(f"     • {c}")
    else:
        print(f"     (none)")

    print(f"\n  ❌ DO NOT DEMO  (<70% pass rate):")
    if broken:
        for c in sorted(broken):
            s = category_stats[c]
            print(f"     • {c}  ({s['pass']}/{s['total']})")
    else:
        print(f"     (none — all categories above 70%)")

    # ── Failing query list ────────────────────────────────────────────────────
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  FAILING QUERIES ({len(failures)}):")
        for r in failures:
            print(f"     ❌ [{r['category']}] {r['query'][:70]}")
            if r["missing"]:
                print(f"          Missing:  {r['missing']}")
            if r["found_bad"]:
                print(f"          Bad text: {r['found_bad']}")

    # ── Save JSON results ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(os.path.dirname(__file__)), exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(__file__), f"eval_results_{ts}.json")
    payload     = {
        "run_at":      datetime.now().isoformat(),
        "total":       total,
        "passed":      passed_count,
        "failed":      failed_count,
        "pass_pct":    pct,
        "avg_ms":      avg_ms,
        "p99_ms":      p100_ms,
        "by_category": {
            c: {k: v for k, v in s.items() if k != "timing_ms"}
            for c, s in category_stats.items()
        },
        "results": results,
    }
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Full results → {output_file}")
    print(f"{'='*72}\n")

    return passed_count, total, results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supply Command AI — automated evaluation harness"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every test case, not just failures.",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Only run cases in this category (e.g. financial, rbac, whatif).",
    )
    args = parser.parse_args()

    passed, total, _ = run_evaluation(
        verbose=args.verbose,
        category_filter=args.category,
    )

    sys.exit(0 if passed == total else 1)
