"""
GlobalMedTech Supply Command AI
Database Loader — loads all CSVs into SQLite
Run: python database/load_data.py
"""
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "database" / "supply_chain.db"
DATA_DIR = BASE_DIR / "sample_data"
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}")
FILES = {
    "suppliers_master": "master/suppliers_master.csv",
    "shipments": "transactions/shipments.csv",
    "financial_impact": "transactions/financial_impact.csv",
}
def clean_columns(df):
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_"))
    return df
def create_audit_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_decisions_log (
            decision_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT,
            user_query     TEXT,
            role_used      TEXT,
            agent_used     TEXT,
            tables_accessed TEXT,
            sql_generated  TEXT,
            result_summary TEXT,
            confidence_score REAL,
            feedback_score REAL,
            response_time_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rbac_audit_log (
            audit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT,
            role_id        TEXT,
            query_attempted TEXT,
            access_granted TEXT,
            blocked_reason TEXT,
            tables_requested TEXT
        )
    """)
    conn.commit()
    logger.success("System tables created")
def load_file(conn, table_name, csv_path):
    full_path = DATA_DIR / csv_path
    if not full_path.exists():
        logger.warning(f"Skipping {table_name} — file not found")
        return 0
    df = pd.read_csv(full_path)
    df = clean_columns(df)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.success(f"{table_name}: {len(df)} rows loaded")
    return len(df)
def validate(conn):
    logger.info("Running validation...")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = cursor.fetchall()
    print("\n" + "="*45)
    print("  DATABASE VALIDATION")
    print("="*45)
    total = 0
    for (table,) in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        total += count
        print(f"  {table:<30} {count:>6} rows")
    print("="*45)
    print(f"  {'TOTAL':<30} {total:>6} rows")
    print("="*45)
def main():
    print("\n" + "="*45)
    print("  Supply Command AI — Data Loader")
    print("="*45 + "\n")

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    logger.info(f"Database: {DB_PATH}")

    create_audit_table(conn)

    for table_name, csv_path in FILES.items():
        load_file(conn, table_name, csv_path)

    conn.commit()
    validate(conn)
    conn.close()

    print(f"\n✅ Database ready: {DB_PATH}\n")
if __name__ == "__main__":
    main()
