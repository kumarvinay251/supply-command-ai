"""
Supply Command AI — Database Connection Layer
The single, secure doorway to supply_chain.db.

WHY this file exists:
    Every agent that needs data MUST go through here — never connect directly.
    This gives us one place to enforce security rules, log every query,
    track performance, and handle errors consistently across all agents.

LLM Usage:
    NONE — this entire file is pure Python + SQLite.
    Zero tokens consumed. Maximum trust, maximum speed.

Security model:
    validate_sql() blocks destructive statements BEFORE they reach the DB.
    If an agent is compromised or makes a mistake, this is the last line
    of defence that prevents data loss.
"""

import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from services.logger import get_logger

# ── Logger ────────────────────────────────────────────────────────────────────

log = get_logger("db_agent")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "database" / "supply_chain.db"

# ── SQL Guardrail — blocked keyword patterns ──────────────────────────────────
#
# WHY regex with word boundaries?
#   A naive `"DROP" in sql` would block a query that selects a column
#   called `drop_reason`. Word boundaries (\b) ensure we only match the
#   SQL keyword as a standalone word, not as part of a column/table name.
#
# WHY not just trust the agents?
#   Agents generate SQL from LLM output. LLMs can hallucinate, be
#   manipulated via prompt injection, or simply produce bad SQL under
#   edge cases. This guardrail is code — it never hallucinates.

BLOCKED_KEYWORDS: list[str] = [
    "DROP", "DELETE", "UPDATE", "INSERT",
    "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE",
]

# Pre-compile once at import time — faster than recompiling per query
_BLOCKED_PATTERN = re.compile(
    r"\b(" + "|".join(BLOCKED_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. GET CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    """
    Open and return a configured SQLite connection to supply_chain.db.

    WHY row_factory = sqlite3.Row?
        By default, sqlite3 returns rows as plain tuples — agents would
        need to know the column position of every field, which is fragile.
        sqlite3.Row lets rows behave like dicts (row["supplier_name"]),
        which is far safer and more readable. We then convert to true
        Python dicts before returning to callers so there are no SQLite
        objects anywhere in the application layer.

    WHY PRAGMA foreign_keys = ON?
        SQLite disables foreign-key enforcement by default for legacy
        compatibility. We enable it so referential integrity is always
        checked — protects against orphaned records if data is ever loaded
        incorrectly.

    Returns:
        sqlite3.Connection — caller is responsible for closing it, or
        use execute_query() which manages the connection lifecycle.

    Raises:
        FileNotFoundError — if the database file does not exist.
        sqlite3.Error     — on any connection-level failure.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run database/load_data.py first."
        )

    conn = sqlite3.connect(str(DB_PATH))

    # Rows behave like dicts: row["column_name"] instead of row[0]
    conn.row_factory = sqlite3.Row

    # Enforce referential integrity on every connection
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ═══════════════════════════════════════════════════════════════════════════════
#  2. VALIDATE SQL  —  pure Python guardrail, zero DB calls
# ═══════════════════════════════════════════════════════════════════════════════

def validate_sql(sql: str) -> tuple[bool, Optional[str]]:
    """
    Inspect a SQL string for dangerous keywords BEFORE it touches the DB.

    WHY this is code, not an LLM call?
        Speed: regex runs in microseconds, an LLM call takes 1-2 seconds.
        Reliability: a regex is deterministic — it always blocks DROP.
          An LLM might be convinced a DROP is "safe" via prompt injection.
        Cost: zero tokens consumed.

    Blocked keywords: DROP, DELETE, UPDATE, INSERT, ALTER, CREATE,
                      TRUNCATE, EXEC, EXECUTE

    Args:
        sql: The SQL string to inspect (may be multi-line).

    Returns:
        (True,  None)          — SQL is safe to execute
        (False, reason_string) — SQL is blocked; reason explains why
    """
    if not sql or not sql.strip():
        return False, "Empty SQL string rejected."

    match = _BLOCKED_PATTERN.search(sql)
    if match:
        keyword = match.group(0).upper()
        reason  = (
            f"Blocked keyword '{keyword}' detected in SQL. "
            f"Only SELECT statements are permitted."
        )
        log.warning(
            f"validate_sql | BLOCKED — keyword='{keyword}' "
            f"| query_preview='{sql.strip()[:80]}'"
        )
        return False, reason

    return True, None


# ═══════════════════════════════════════════════════════════════════════════════
#  3. EXECUTE QUERY
# ═══════════════════════════════════════════════════════════════════════════════

def execute_query(
    sql: str,
    params: Optional[tuple | list] = None,
) -> dict:
    """
    Validate, execute, time, and return results for a SQL SELECT query.

    WHY one function wraps everything?
        Every agent needs the same behaviour: validate → execute → time →
        convert → log. Putting it here means no agent can skip a step.
        If we add a new guardrail tomorrow, it applies everywhere instantly.

    WHY convert to plain dicts?
        sqlite3.Row objects look like dicts but they are not JSON-serialisable
        and they hold a reference to the connection cursor. Converting to
        plain Python dicts makes results safe to pass between agents, log,
        cache, and serialise.

    WHY track execution_time_ms?
        Slow queries (>500ms) indicate missing indexes or bad SQL generation.
        Logging this lets us spot performance regressions without profiling.

    Args:
        sql:    A SELECT statement. Parameterised queries are preferred.
        params: Optional tuple/list of values for ? placeholders — prevents
                SQL injection even from legitimate agent-generated queries.

    Returns:
        {
            "success":          bool,
            "data":             list[dict],   # [] on failure
            "row_count":        int,
            "execution_time_ms": int,
            "sql":              str,          # the query that was run
            "error":            str | None,   # None on success
        }
    """
    result_template = {
        "success":           False,
        "data":              [],
        "row_count":         0,
        "execution_time_ms": 0,
        "sql":               sql,
        "error":             None,
    }

    # ── Step 1: Guardrail — validate before touching the DB ───────────────────
    is_safe, block_reason = validate_sql(sql)
    if not is_safe:
        result_template["error"] = block_reason
        log.warning(f"execute_query | BLOCKED | reason='{block_reason}'")
        return result_template

    # ── Step 2: Execute with timing ───────────────────────────────────────────
    conn  = None
    start = time.perf_counter()

    try:
        conn   = get_connection()
        cursor = conn.cursor()

        # Use parameterised execution when params are provided.
        # WHY? Even though we trust agents to generate SQL, params values
        # might come from user input — parameterisation prevents injection.
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)

        # ── Step 3: Convert rows to plain Python dicts ────────────────────────
        # sqlite3.Row supports dict(row) directly — clean and efficient.
        raw_rows = cursor.fetchall()
        data     = [dict(row) for row in raw_rows]

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        log.success(
            f"execute_query | rows={len(data)} | "
            f"time={elapsed_ms}ms | "
            f"sql='{sql.strip()[:100]}'"
        )

        return {
            "success":           True,
            "data":              data,
            "row_count":         len(data),
            "execution_time_ms": elapsed_ms,
            "sql":               sql,
            "error":             None,
        }

    except sqlite3.Error as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        error_msg  = f"SQLite error: {exc}"

        # WHY log error but not raise?
        #   Raising would crash the agent and the UI. Returning a structured
        #   error dict lets the caller decide how to handle it — show the
        #   user a friendly message, retry, or escalate.
        log.error(
            f"execute_query | FAILED | time={elapsed_ms}ms | "
            f"error='{exc}' | sql='{sql.strip()[:100]}'"
        )
        result_template["error"]             = error_msg
        result_template["execution_time_ms"] = elapsed_ms
        return result_template

    finally:
        # Always close — SQLite connections are not thread-safe if shared.
        # Each query gets its own connection, opened and closed here.
        if conn:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. GET TABLE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

def get_table_schema() -> dict:
    """
    Return the schema of every user-facing table in the database.

    WHY the Planning Agent needs this:
        Before building a query plan, the agent must know what tables and
        columns exist. This avoids hallucinating column names or querying
        tables that don't exist.

    WHY exclude system tables?
        sqlite_sequence is an internal SQLite housekeeping table.
        rbac_audit_log and ai_decisions_log are write-only audit tables —
        agents should never query them for business data.

    Returns:
        {
            "tables": {
                "shipments": {
                    "columns": [
                        {"name": "shipment_id", "type": "TEXT", "pk": False},
                        ...
                    ],
                    "row_count": 100,
                },
                ...
            },
            "error": None | str,
        }
    """
    # Tables agents are NOT allowed to see or query
    EXCLUDED_TABLES = {"sqlite_sequence", "rbac_audit_log", "ai_decisions_log"}

    conn = None
    try:
        conn   = get_connection()
        cursor = conn.cursor()

        # ── Discover all tables ───────────────────────────────────────────────
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        all_tables = [row["name"] for row in cursor.fetchall()]
        user_tables = [t for t in all_tables if t not in EXCLUDED_TABLES]

        schema = {}
        for table in user_tables:

            # ── Column definitions via PRAGMA ─────────────────────────────────
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [
                {
                    "name": col["name"],
                    "type": col["type"] or "TEXT",
                    "pk":   bool(col["pk"]),
                }
                for col in cursor.fetchall()
            ]

            # ── Row count for context ─────────────────────────────────────────
            cursor.execute(f"SELECT COUNT(*) AS n FROM {table}")
            row_count = cursor.fetchone()["n"]

            schema[table] = {
                "columns":   cols,
                "row_count": row_count,
            }

        log.info(
            f"get_table_schema | {len(schema)} tables returned: "
            f"{list(schema.keys())}"
        )
        return {"tables": schema, "error": None}

    except sqlite3.Error as exc:
        log.error(f"get_table_schema | FAILED: {exc}")
        return {"tables": {}, "error": str(exc)}

    finally:
        if conn:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  5. LOG AGENT DECISION
# ═══════════════════════════════════════════════════════════════════════════════

def log_agent_decision(decision: dict) -> bool:
    """
    Persist a record of every agent response into ai_decisions_log.

    WHY log every decision?
        Auditability — every answer the system gives can be traced back
        to exactly which role asked, which agent answered, what SQL was
        run, and how confident the answer was. This is critical for a
        medical supply chain where decisions affect patient outcomes.

    WHY a separate function rather than inline inserts?
        Centralising audit writes means: one schema to maintain, one
        place to add fields, and one place to handle write failures.
        A failure here logs an error but never crashes the application —
        losing an audit record is bad, but crashing the UI is worse.

    Expected keys in `decision` dict:
        user_query       str   — the original question from the user
        role_used        str   — e.g. "Demand Planner"
        agent_used       str   — e.g. "db_agent"
        tables_accessed  str   — comma-separated table names
        sql_generated    str   — the SQL query that was executed
        result_summary   str   — short plain-English summary of result
        confidence_score float — 0.0–1.0 (agent-assigned)
        response_time_ms int   — total wall-clock time for the answer

    Args:
        decision: Dict with the keys listed above.
                  Missing keys default to None / 0 — never raises KeyError.

    Returns:
        True on successful write, False on failure.
    """
    # WHY not use execute_query() here?
    #   execute_query() blocks INSERT via validate_sql() — intentionally.
    #   This function has explicit, controlled INSERT authority for the
    #   single audit table only. Nothing else can INSERT via this path.

    conn = None
    try:
        conn = get_connection()

        conn.execute(
            """
            INSERT INTO ai_decisions_log (
                timestamp, user_query, role_used, agent_used,
                tables_accessed, sql_generated, result_summary,
                confidence_score, response_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                decision.get("user_query",       ""),
                decision.get("role_used",         ""),
                decision.get("agent_used",        ""),
                decision.get("tables_accessed",   ""),
                decision.get("sql_generated",     ""),
                decision.get("result_summary",    ""),
                decision.get("confidence_score",  0.0),
                decision.get("response_time_ms",  0),
            ),
        )
        conn.commit()

        log.info(
            f"log_agent_decision | saved | "
            f"agent={decision.get('agent_used')} | "
            f"confidence={decision.get('confidence_score')} | "
            f"time={decision.get('response_time_ms')}ms"
        )
        return True

    except sqlite3.Error as exc:
        # WHY not re-raise?
        #   A failed audit write must never block the user from getting
        #   their answer. Log the failure and move on.
        log.error(f"log_agent_decision | FAILED to write audit record: {exc}")
        return False

    finally:
        if conn:
            conn.close()
