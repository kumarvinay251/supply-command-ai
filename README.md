# Supply Command AI
**GenAI-powered Supply Chain Decision Intelligence Platform for GlobalMedTech Inc.**

---

## What It Does

Supply Command AI transforms raw supply chain data into explainable, role-aware decisions.
Instead of dashboards that show data and leave interpretation to the user, it accepts
natural-language questions and returns structured answers grounded in real database
queries, RAG-retrieved documents, and financial calculations — every answer citing
its exact source and confidence level.

The system uses a multi-agent architecture orchestrated by LangGraph. A Planning Agent
classifies the user's intent and builds a step-by-step investigation plan before any
data is fetched. Specialist agents (DB, RAG, ROI) execute only their assigned steps.
Role-based access control is enforced at every layer — a Demand Planner cannot see
financial data even if they ask for it directly. Every decision above $50,000 financial
impact is flagged for human approval before being actioned.

All data is synthetic. GlobalMedTech Inc. is a fictional medical device company created
to demonstrate the platform with realistic supply chain scenarios across suppliers,
shipments, delays, SLA breaches, and AI-driven cost savings.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│              Streamlit App  |  Role Selector  |  Chat           │
└────────────────────────────┬────────────────────────────────────┘
                             │ question + role
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT GUARDRAILS                             │
│   validate_role() → check_query_access() →                      │
│   detect_prompt_injection()                                     │
│   [ agents/guardrails.py ]                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ approved query
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PLANNING AGENT                              │
│   classify_intent() → get_plan_template() → create_plan()      │
│   Pure Python — zero LLM tokens — returns structured JSON plan  │
│   [ agents/planning_agent.py ]                                  │
└──────────┬─────────────────┬──────────────────┬────────────────┘
           │                 │                  │
           ▼                 ▼                  ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│   DB AGENT   │   │    RAG AGENT     │   │   ROI AGENT  │
│              │   │                  │   │              │
│ SQL template │   │ FAISS vector     │   │ Financial    │
│ library →    │   │ search →         │   │ calculations │
│ SQLite query │   │ knowledge base   │   │ from         │
│ → findings   │   │ + annual report  │   │ financial_   │
│              │   │ → cited chunks   │   │ impact table │
│ [db_agent.py]│   │ [memory.py]      │   │ [db_agent.py]│
└──────────┬───┘   └───────┬──────────┘   └──────┬───────┘
           │               │                      │
           └───────────────┴──────────────────────┘
                           │ combined findings
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HUMAN CHECKPOINT                              │
│   check_human_approval_needed()                                 │
│   Triggers if: impact > $50K | confidence < 60% |              │
│   action = expedite/cancel/terminate | rows > 10               │
│   [ agents/guardrails.py ]                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ approved findings
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT GUARDRAILS                             │
│   validate_output() — mask [RESTRICTED] for Demand Planner     │
│   enforce row caps | add confidence warnings                    │
│   [ agents/guardrails.py ]                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ validated findings
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXECUTIVE AGENT       ← ONE LLM CALL HERE    │
│   GPT-4o-mini formats findings into a readable answer           │
│   Answer includes: source table | SQL used | confidence score   │
│   [ LangGraph final node ]                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  EXPLAINABLE ANSWER │
              │  + Source citation  │
              │  + SQL shown        │
              │  + Confidence score │
              │  + Human flag if    │
              │    approval needed  │
              └────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-agent orchestration** | LangGraph state machine routes each question through Planning → Specialist Agents → Output |
| **Role-based access control** | 3 roles (Demand Planner / Operations Manager / CFO) with distinct table access, keyword blocks, and row caps |
| **Hybrid SQL approach** | Pre-written SQL templates handle 95% of queries — zero LLM tokens for standard questions |
| **RAG on document corpus** | FAISS vector store over Annual Report PDF + knowledge base with 0.75 similarity guardrail |
| **Human-in-the-loop** | Automatic approval checkpoint for decisions above $50K, low confidence, or high-risk actions |
| **Full audit trail** | Every agent decision → `ai_decisions_log` | Every security event → `rbac_audit_log` |
| **Explainable AI** | Every answer shows: source table, SQL executed, confidence score, and reasoning steps |
| **Prompt injection protection** | 9 regex patterns + length+behavioral analysis block adversarial inputs before agents run |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent orchestration | LangGraph 0.3.34 | Stateful multi-agent workflow with human checkpoints |
| LLM | OpenAI GPT-4o-mini | Final answer formatting only — 1 call per question |
| Embeddings | OpenAI text-embedding-ada-002 | Document and query vectorisation for RAG |
| Database | SQLite (Python sqlite3) | Stores shipments, financials, suppliers, audit logs |
| Vector store | FAISS 1.9.0 (faiss-cpu) | Similarity search over knowledge base chunks |
| Document parsing | pypdf 3.17.1 | Extract text from Annual Report PDF |
| UI | Streamlit 1.40.0 | Role-selector chat interface |
| Charts | Plotly 5.18.0 | Interactive supply chain visualisations |
| Logging | Loguru 0.7.2 | Structured logs with agent tags, rotation, compression |
| Data processing | Pandas 2.2.3 + NumPy 2.1.3 | CSV loading, data manipulation |
| Environment | python-dotenv 1.0.0 | Secure config management |

---

## Project Structure

```
Control Tower/
│
├── CLAUDE.md                    Instructions for Claude Code
├── README.md                    This file
├── requirements.txt             Pinned Python dependencies
├── .env                         API keys and paths (not committed)
│
├── database/
│   ├── db_connection.py         Secure SQLite gateway — all agents use this
│   ├── load_data.py             One-time CSV → SQLite loader
│   └── supply_chain.db          Live database (3 tables + 2 audit tables)
│
├── agents/
│   ├── guardrails.py            RBAC config, input/output validation,
│   │                            prompt injection detection, human-in-the-loop
│   ├── planning_agent.py        Intent classification, plan templates,
│   │                            step routing — zero LLM
│   └── db_agent.py              SQL template library, query execution,
│                                result interpretation, confidence scoring
│
├── services/
│   ├── logger.py                Loguru wrapper — get_logger("agent_name")
│   └── memory.py                FAISS vector store — chunk, embed, search
│
├── sample_data/
│   ├── master/
│   │   └── suppliers_master.csv
│   ├── transactions/
│   │   ├── shipments.csv
│   │   └── financial_impact.csv
│   └── documents/
│       └── Globalmedtech annual report 2023 full .pdf
│
├── logs/
│   └── control_tower.log        Auto-created, rotates at 10MB
│
└── vector_store/
    ├── index.faiss              FAISS index (created by build_vector_store)
    └── chunks_meta.pkl          Chunk metadata parallel to index vectors
```

---

## Data Layer

### SQLite Tables — `database/supply_chain.db`

| Table | Rows | Key Columns |
|---|---|---|
| `suppliers_master` | 3 | supplier_id, supplier_name, sla_on_time_target_pct, risk_tier, annual_spend_usd_2023/2024 |
| `shipments` | 100 | shipment_id, status, delay_days, sla_breach, expedited_flag, expedited_cost_usd, risk_flag |
| `financial_impact` | 36 | period_label, total_sc_cost_usd, ai_savings_usd, cumulative_savings, roi_pct |
| `ai_decisions_log` | 0* | timestamp, role_used, agent_used, sql_generated, confidence_score |
| `rbac_audit_log` | 0* | timestamp, role_id, query_attempted, access_granted, blocked_reason |

*Grows at runtime — starts empty.

### Suppliers (synthetic)

| ID | Name | Delay Rate | SLA Target | Risk Tier |
|---|---|---|---|---|
| SUP001 | SupplierA | 19.4% | 92% | Medium |
| SUP002 | SupplierB | 5.9% | 95% | Medium |
| SUP003 | SupplierC | 20.0% | 90% | High |

### Key Verified Metrics (cross-validated against raw CSVs)
- **SLA breach rate:** 17.1% in 2023 → 2.1% in 2024 (13 breaches / 76 shipments)
- **Expedited costs:** $159,600 in 2023 → $41,795 in 2024 (−74%)
- **Peak cost month:** January 2024 at $528,000
- **AI ROI by Dec 2024:** 340% (cumulative savings $401,000)

---

## Agent Descriptions

### Planning Agent (`agents/planning_agent.py`)
The brain of the system. Receives the raw user question and produces a structured
JSON plan before any data is fetched. Intent is classified using pure keyword
matching across 7 intent categories (DELAY_ANALYSIS, SUPPLIER_RISK,
INVENTORY_RISK, FINANCIAL_IMPACT, BENCHMARK_COMPARISON, ROOT_CAUSE,
RECOMMENDATION). Each intent maps to a pre-built plan template with ordered steps
assigned to specific agents. Zero LLM tokens used. The plan is shown to the user
before execution so they see exactly what the system will do.

### DB Agent (`agents/db_agent.py`)
The data retrieval specialist. Receives a single step dict from the Planning Agent
and executes a pre-written SQL query from the template library — no LLM generates
SQL. Contains 8 named SQL templates covering delay analysis, supplier performance,
financial breakdowns, ROI progression, and risk identification. Results are
interpreted into structured findings with a confidence score based on row count.
Every query goes through `db_connection.execute_query()` which enforces the SQL
guardrail and tracks execution time.

### Guardrails (`agents/guardrails.py`)
Not a conversational agent — a pure Python safety layer. Runs before the Planning
Agent (input guardrails) and after all agents return (output guardrails). Enforces
the ROLES configuration for 3 user roles. Detects 9 categories of prompt injection.
Masks financial column values with `[RESTRICTED]` for roles without financial access.
Flags results for human approval when financial exposure exceeds $50,000, confidence
is below 60%, high-risk action words appear ("expedite", "cancel", "terminate"), or
more than 10 shipments are affected. All events written to `rbac_audit_log`.

### Memory Service (`services/memory.py`)
The RAG knowledge retrieval service. Chunks documents using pure Python word-count
splitting with configurable overlap (default: 500 words, 50 overlap). Preserves
`[SECTION: name]` tags as metadata. Loads PDFs via pypdf and markdown files as
plain text. Builds a FAISS inner-product index using OpenAI text-embedding-ada-002.
The 0.75 similarity guardrail is hard-coded: if the best matching chunk scores
below 0.75, the function returns `{"found": False}` and the LLM never sees it.

### Logger (`services/logger.py`)
Centralised Loguru wrapper. Every agent calls `get_logger("agent_name")` which
returns a logger pre-bound with the agent name. Every log line includes:
timestamp, level (DEBUG/INFO/SUCCESS/WARNING/ERROR), agent name (20-char padded),
and message. Writes to both console (coloured) and `logs/control_tower.log`
(plain text). Rotates at 10MB, retains 7 days, compresses to `.zip`.

---

## How to Run

### Prerequisites
- Python 3.11+ (3.13 recommended with miniconda)
- OpenAI API key
- ~500MB disk space for dependencies

### Setup

```bash
# 1. Clone / navigate to project
cd "Control Tower"

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Configure environment
# Edit .env and replace your_openai_key_here with your real key
# OPENAI_API_KEY=sk-...

# 4. Load data into SQLite
python3 database/load_data.py
# Expected: 3 suppliers, 100 shipments, 36 financial rows loaded

# 5. Verify logger works
python3 -m services.logger
# Expected: coloured log lines in console + written to logs/

# 6. Verify document chunking (no API key needed)
python3 -m services.memory
# Expected: 8 chunks from Annual Report PDF

# 7. Build FAISS vector store (requires OPENAI_API_KEY)
python3 -c "
from services.memory import load_documents, build_vector_store
build_vector_store(load_documents())
"
# Expected: index.faiss + chunks_meta.pkl in vector_store/
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | OpenAI key — used for embeddings and final LLM formatting |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |
| `DB_PATH` | No | Defaults to `database/supply_chain.db` |
| `DATA_DIR` | No | Defaults to `sample_data` |
| `KNOWLEDGE_BASE_DIR` | No | Defaults to `sample_data/documents` |
| `VECTOR_STORE_DIR` | No | Defaults to `vector_store` |
| `APP_NAME` | No | Display name in UI |
| `COMPANY_NAME` | No | `GlobalMedTech Inc.` |
| `LOG_LEVEL` | No | `INFO` (options: DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | No | `logs/control_tower.log` |
| `CHUNK_SIZE` | No | `500` — words per RAG chunk |
| `CHUNK_OVERLAP` | No | `50` — overlap words between chunks |
| `MAX_RETRIEVAL_DOCS` | No | `3` — max RAG results returned |

---

## Design Principles

| # | Principle | Why It Matters |
|---|---|---|
| 1 | **LLM for formatting only** | Eliminates hallucination from routing and SQL. Every data decision is auditable Python code. |
| 2 | **Human in the loop at $50K** | Mirrors GlobalMedTech procurement policy. AI advises — humans authorise. |
| 3 | **Guardrails before AND after** | Input blocks bad questions. Output masking handles unexpected data leakage. Both are needed. |
| 4 | **Every answer cites its source** | Explainability is a regulatory requirement in medical supply chains. Users must be able to audit answers. |
| 5 | **Confidence on every response** | Prevents false certainty. A 40% confidence answer looks different to a 90% answer — users calibrate accordingly. |
| 6 | **RBAC before data access** | Data governance failure in medical supply chains can cause compliance violations and competitive harm. |
| 7 | **Full audit trail** | Both agent decisions and security events are logged. Essential for post-incident forensics. |
| 8 | **SQL validation always** | An agent mistake or adversarial input must never corrupt or delete production data. |

---

## Roadmap

### Phase 2 — In Development
- [ ] `agents/rag_agent.py` — Dedicated RAG agent wrapping `services/memory.py`
- [ ] `agents/roi_agent.py` — Financial calculation specialist
- [ ] `agents/executive_agent.py` — LangGraph final node (the one LLM call)
- [ ] `app.py` — Streamlit UI with role selector and chat interface
- [ ] LangGraph workflow wiring all agents into a state machine
- [ ] Evaluation framework with golden test set (20 Q&A pairs)

### Phase 3 — Production Path
- [ ] Replace SQLite with PostgreSQL for concurrent users
- [ ] Replace FAISS with Pinecone for persistent, managed vector store
- [ ] Add streaming responses via LangGraph streaming API
- [ ] Add feedback mechanism — user rates answers → improves confidence calibration
- [ ] Knowledge base expansion — SOPs, troubleshooting guides, supplier contracts
- [ ] CI/CD pipeline with automated eval regression tests

### Scale Considerations
- Current: 1 user, SQLite, local FAISS — suitable for demo and pilot
- 10 users: Add connection pooling to `db_connection.py`, move to PostgreSQL
- 100 users: Containerise with Docker, add Redis for plan caching
- 1000 users: Kubernetes, Pinecone, async LangGraph execution

---

## Data Notice

All data in this project is **100% synthetic**.
GlobalMedTech Inc., SupplierA, SupplierB, and SupplierC are fictional entities.
No real patient data, no real financial records, no real supplier information
is used anywhere in this system. The data was generated to reflect realistic
supply chain patterns for demonstration purposes only.
