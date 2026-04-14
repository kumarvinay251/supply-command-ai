"""
Supply Command AI — Centralised Logger
Powered by Loguru.

Usage:
    from services.logger import get_logger
    log = get_logger("db_agent")
    log.info("Query executed successfully")
    log.warning("Low confidence result")
    log.error("Database connection failed")

Every log line written to:
    • Console  — coloured, human-readable
    • File     — logs/control_tower.log (rotation: 10 MB, retention: 7 days)

Log format includes: timestamp | level | agent_name | message
"""

import sys
from pathlib import Path
from loguru import logger


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
LOG_DIR  = BASE_DIR / "logs"
LOG_FILE = LOG_DIR  / "control_tower.log"

LOG_DIR.mkdir(exist_ok=True)


# ── Format Strings ────────────────────────────────────────────────────────────

# Console — coloured, includes agent name injected via `extra`
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{extra[agent]:<20}</cyan> | "
    "<level>{message}</level>"
)

# File — plain text, same fields, machine-parseable
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level:<8} | "
    "{extra[agent]:<20} | "
    "{message}"
)


# ── Bootstrap — runs once at import ──────────────────────────────────────────

def _bootstrap_logger() -> None:
    """Remove default Loguru handler and attach console + file sinks."""
    logger.remove()                         # remove default stderr sink

    # Console sink
    logger.add(
        sys.stdout,
        format    = CONSOLE_FORMAT,
        level     = "DEBUG",
        colorize  = True,
        backtrace = True,
        diagnose  = True,
    )

    # File sink — rotates at 10 MB, keeps 7 days of history
    logger.add(
        str(LOG_FILE),
        format    = FILE_FORMAT,
        level     = "DEBUG",
        rotation  = "10 MB",
        retention = "7 days",
        compression = "zip",
        backtrace = True,
        diagnose  = True,
        encoding  = "utf-8",
    )


_bootstrap_logger()


# ── Public Factory ────────────────────────────────────────────────────────────

def get_logger(agent_name: str):
    """
    Return a Loguru logger bound to a specific agent name.

    The agent name is injected into every log line so you can filter
    the log file by agent (e.g. grep 'db_agent' logs/control_tower.log).

    Args:
        agent_name: Short label shown in the log line, e.g. 'db_agent',
                    'planning_agent', 'roi_agent', 'rag_agent'.

    Returns:
        A Loguru logger with `extra['agent']` pre-bound.

    Example:
        log = get_logger("db_agent")
        log.info("Running query: SELECT * FROM shipments")
        log.success("Query returned 42 rows in 12ms")
        log.warning("Result confidence below threshold: 0.61")
        log.error("SQLite connection failed — retrying")
    """
    return logger.bind(agent=agent_name)


# ── Convenience Loggers (pre-bound, importable directly) ─────────────────────

planning_log = get_logger("planning_agent")
db_log       = get_logger("db_agent")
roi_log      = get_logger("roi_agent")
rag_log      = get_logger("rag_agent")
app_log      = get_logger("app")


# ── Self-test (runs only when executed directly) ──────────────────────────────

if __name__ == "__main__":
    print(f"\nLog file → {LOG_FILE}\n")

    test_agents = ["planning_agent", "db_agent", "roi_agent", "rag_agent", "app"]
    levels      = ["debug", "info", "success", "warning", "error"]

    for agent in test_agents:
        log = get_logger(agent)
        log.debug(   f"[{agent}] DEBUG   — initialising agent")
        log.info(    f"[{agent}] INFO    — agent ready")
        log.success( f"[{agent}] SUCCESS — query completed in 18ms")
        log.warning( f"[{agent}] WARNING — confidence score 0.58, below threshold")
        log.error(   f"[{agent}] ERROR   — connection timeout, retrying (1/3)")

    print(f"\n✅ All test log lines written to {LOG_FILE}\n")
