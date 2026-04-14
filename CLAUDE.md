# CLAUDE.md — Supply Command AI
Instructions for Claude Code when working on this project.
Read this file before touching any code.

---

## Project Overview

**Product:**   Supply Command AI
**Company:**   GlobalMedTech Inc. (all data is synthetic — no real PII)
**Domain:**    Medical device supply chain decision intelligence
**Purpose:**   Multi-agent GenAI system that answers supply chain questions
               for three user roles using real SQLite data, FAISS RAG,
               and coded SQL templates — LLM used only for final formatting.

---

## Architecture Principles — Follow These Always

These are non-negotiable. Every file in this project was built around them.
Do not break them even if a simpler approach exists.

### 1. LLM for formatting only
The LLM (GPT-4o-mini) is called **once per user question** — at the very end,
to format the combined findings into a readable answer.
All routing, intent classification, SQL generation, and calculations are
**coded Python**. Never move logic into a prompt.

### 2. Human-in-the-loop above $50,000
Any recommendation or result with financial impact above $50,000 must be
flagged for human approval before being actioned.
This is enforced in `agents/guardrails.py → check_human_approval_needed()`.
Do not lower this threshold without stakeholder sign-off.

### 3. Guardrails before AND after every agent
- **Input guardrails** run before any agent touches the database
- **Output guardrails** run after agents return, before the UI renders
Both live in `agents/guardrails.py`. Never call a DB agent without first
running `check_query_access()` and `detect_prompt_injection()`.

### 4. Every answer must cite its source
Answers must always include:
- Which table(s) the data came from
- The SQL that was executed (or "RAG: knowledge_base/file.md")
- Confidence score (float 0.0–1.0)
Never return a plain text answer with no provenance.

### 5. Confidence score on every response
Confidence is calculated in `agents/db_agent.py → interpret_result()`:
- `row_count >= 10` → 0.90
- `row_count >= 5`  → 0.75
- `row_count >= 2`  → 0.60
- `row_count == 1`  → 0.40
- `row_count == 0`  → 0.00
If confidence < 0.60, a warning is shown. If < 0.60, human approval required.

### 6. RBAC enforced — role checked before data access
Three roles exist: `Demand Planner`, `Operations Manager`, `CFO`.
Role is validated in `agents/guardrails.py → ROLES` dict and
`validate_role()` before any query runs.
Never assume a role has access — always check `check_query_access()`.

### 7. All agent decisions logged to ai_decisions_log
Every agent response is written to the `ai_decisions_log` SQLite table via
`database/db_connection.py → log_agent_decision()`.
Security events (blocked queries, injections) are logged to `rbac_audit_log`.

### 8. SQL validated before execution — always
Every SQL string passes through `database/db_connection.py → validate_sql()`
before reaching SQLite. This blocks: `DROP`, `DELETE`, `UPDATE`, `INSERT`,
`ALTER`, `CREATE`, `TRUNCATE`, `EXEC`.
Never bypass this — not even in test code.

---

## File Structure

```
Control Tower/
│
├── CLAUDE.md                        ← You are here. Read before coding.
├── README.md                        ← Public-facing project documentation
├── requirements.txt                 ← Python dependencies (pinned versions)
├── .env                             ← API keys and config (never commit)
│
├── database/
│   ├── db_connection.py             ← Single secure doorway to SQLite.
│   │                                   get_connection, validate_sql,
│   │                                   execute_query, get_table_schema,
│   │                                   log_agent_decision
│   ├── load_data.py                 ← One-time loader: CSVs → SQLite tables.
│   │                                   Run once to initialise the database.
│   └── supply_chain.db              ← Live SQLite database (auto-generated)
│
├── agents/
│   ├── guardrails.py                ← Ethics, safety, RBAC layer.
│   │                                   ROLES config, input validation,
│   │                                   output masking, human-in-the-loop,
│   │                                   audit logging to rbac_audit_log
│   ├── planning_agent.py            ← Converts user question → structured plan.
│   │                                   Intent classification (coded keywords),
│   │                                   plan templates, step routing,
│   │                                   guardrail checks before execution
│   └── db_agent.py                  ← Executes SQL against supply_chain.db.
│                                       SQL template library (pre-written),
│                                       result interpretation, confidence scoring
│
├── services/
│   ├── logger.py                    ← Centralised Loguru logger.
│   │                                   Console + file sink, agent-tagged lines,
│   │                                   rotation at 10MB, 7-day retention
│   └── memory.py                    ← FAISS vector store service.
│                                       chunk_text, load_documents,
│                                       build_vector_store, search,
│                                       0.75 similarity guardrail
│
├── sample_data/
│   ├── master/
│   │   └── suppliers_master.csv     ← 3 suppliers: SUP001, SUP002, SUP003
│   ├── transactions/
│   │   ├── shipments.csv            ← 100 shipments (2023–2024)
│   │   └── financial_impact.csv     ← 36 monthly records (2022–2024)
│   └── documents/
│       └── Globalmedtech annual     ← PDF annual report — RAG source
│           report 2023 full.pdf
│
├── logs/
│   └── control_tower.log            ← Loguru output (auto-created, .gitignore)
│
└── vector_store/                    ← FAISS index (auto-created after build)
    ├── index.faiss                  ← Saved FAISS vectors
    └── chunks_meta.pkl              ← Parallel chunk metadata list
```

---

## Key Design Decisions

### Why LangGraph over plain LangChain?
LangGraph lets us define the agent flow as an explicit state machine — each
node (agent) is a discrete step with typed inputs and outputs. Plain LangChain
chains are opaque and hard to debug. LangGraph gives us:
- Explicit human-in-the-loop checkpoints (interrupt nodes)
- Clear state passing between agents (not hidden in memory)
- Visualisable graph for explainability
- Deterministic routing (not LLM-decided)

### Why SQLite for now?
GlobalMedTech demo uses 100 shipments and 36 financial records — SQLite is
fast, portable, and requires zero infrastructure. The data layer is abstracted
behind `db_connection.py` so migrating to PostgreSQL in production requires
changes in only one file. SQLite is not a limitation — it is a deliberate
choice for this phase.

### Why FAISS for vector store?
FAISS (Facebook AI Similarity Search) is:
- Local — no API calls, no cost, no latency to an external vector DB
- Fast — inner-product search on 1536-dim vectors in milliseconds
- Portable — ships as a Python package, no Docker needed
- Replaceable — `services/memory.py` abstracts the interface so Pinecone
  or Weaviate can replace it by changing only that file.
The 0.75 similarity guardrail ensures weak matches never reach the LLM.

### Why hybrid template + LLM fallback for SQL?
SQL templates in `agents/db_agent.py → SQL_TEMPLATES` handle 95% of the
expected question types for this domain. This gives us:
- **Zero hallucination** — pre-written SQL is always correct
- **Zero token cost** — no LLM needed for standard queries
- **Auditability** — exact SQL shown to user for every answer
- **Speed** — no round-trip to OpenAI for query generation
The LLM fallback (not yet built) handles edge cases the templates miss.

### Why semantic layer approach?
Raw column names like `sla_on_time_target_pct` are not user-friendly.
The planning agent's intent classifier maps natural language ("which supplier
is worst") to specific SQL templates ("supplier_sla_performance") using keyword
matching. This semantic layer means users never see raw SQL or column names,
but we still have full control over what SQL runs.

---

## What NOT to Do

```
NEVER bypass agents/guardrails.py
    ❌  db_connection.execute_query(sql)  — without validate_role() first
    ✅  guardrails.check_query_access(query, role) → then execute

NEVER call LLM before checking the SQL template library
    ❌  llm.invoke(f"Write SQL for: {query}")
    ✅  db_agent.get_sql_template(task, role)  — LLM only if no template matches

NEVER write SQL without validate_sql()
    ❌  conn.execute(user_supplied_sql)
    ✅  db_connection.execute_query(sql)  — validate_sql() is called inside

NEVER return an answer without a confidence score
    ❌  {"answer": "SupplierC has the highest delay rate"}
    ✅  {"answer": "...", "confidence": 0.9, "source": "shipments", "sql": "..."}

NEVER store PII in logs
    ✅  Log: role, intent, query_type, row_count, execution_time_ms
    ❌  Log: user_id, patient_id, personal names, email addresses
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini (final formatting only) |
| Embeddings | OpenAI text-embedding-ada-002 |
| Database | SQLite via Python sqlite3 |
| Vector store | FAISS (faiss-cpu) |
| Document parsing | pypdf |
| UI | Streamlit + Plotly |
| Logging | Loguru |
| Data processing | Pandas, NumPy |
| Environment | python-dotenv |

---

## Data

### CSV Files (sample_data/)

| File | Rows | Description |
|---|---|---|
| `suppliers_master.csv` | 3 | SUP001 (SupplierA), SUP002 (SupplierB), SUP003 (SupplierC) |
| `shipments.csv` | 100 | Shipments Jan 2023–Dec 2024 across 3 suppliers |
| `financial_impact.csv` | 36 | Monthly financials Jan 2022–Dec 2024 |

### SQLite Tables (supply_chain.db)

| Table | Rows | Purpose |
|---|---|---|
| `suppliers_master` | 3 | Supplier profiles, SLA targets, certifications |
| `shipments` | 100 | Individual shipment records with delay and risk data |
| `financial_impact` | 36 | Monthly cost, savings, ROI metrics |
| `ai_decisions_log` | 0 | Agent answer audit trail (grows at runtime) |
| `rbac_audit_log` | 0 | Guardrail event audit trail (grows at runtime) |

### Knowledge Base (RAG Sources)
- `sample_data/documents/` — GlobalMedTech Annual Report 2023 (PDF, 8 chunks)
- `knowledge_base/` — Markdown SOPs and decision frameworks (planned)
- All content is **synthetic** — GlobalMedTech Inc. does not exist.

---

## Environment Variables (.env)

```env
OPENAI_API_KEY=your_key_here        # Required for embeddings and final LLM call
OPENAI_MODEL=gpt-4o-mini            # Model for answer formatting step only
DB_PATH=database/supply_chain.db    # SQLite database path
DATA_DIR=sample_data                # CSV source directory
KNOWLEDGE_BASE_DIR=sample_data/documents
VECTOR_STORE_DIR=vector_store       # FAISS index output directory
APP_NAME=Supply Command AI
COMPANY_NAME=GlobalMedTech Inc.
LOG_LEVEL=INFO
LOG_FILE=logs/control_tower.log
CHUNK_SIZE=500                      # Words per RAG chunk
CHUNK_OVERLAP=50                    # Overlap between chunks
MAX_RETRIEVAL_DOCS=3                # Max RAG results per search
```

---

## Running the Project

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Load data into SQLite
python3 database/load_data.py

# 3. Test logger
python3 -m services.logger

# 4. Test memory/chunking (no LLM)
python3 -m services.memory

# 5. Build FAISS vector store (requires OPENAI_API_KEY)
python3 -c "from services.memory import *; build_vector_store(load_documents())"
```
