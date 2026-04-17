"""
Supply Command AI — LangGraph Orchestration
The state machine that wires every agent into a single, auditable pipeline.

This file contains the complete graph definition for the Supply Command AI
multi-agent system. Every user question follows the same fixed path:

    INPUT GUARDRAILS → PLANNER → [SPECIALIST AGENTS]* → EXECUTIVE → ANSWER

WHY LangGraph for orchestration?
    LangGraph gives us a stateful, reproducible execution graph where:
    • Every state transition is logged and inspectable.
    • Conditional routing (proceed vs block, loop vs finish) is explicit code
      — not an LLM deciding what to do next.
    • Human-in-the-loop checkpoints integrate naturally as conditional edges.
    • State accumulation (findings list) is handled by typed reducers, not
      manual list management scattered across agent files.

Graph topology:
    START
      └─► input_guardrails
            ├─ (blocked)  ──► END            ← RBAC / injection block
            └─ (proceed)  ──► planner
                                └─► execute_step  ◄─┐
                                      ├─ (more_steps) ─┘  ← step loop
                                      └─ (done) ──► executive
                                                        └─► END

LLM calls: EXACTLY ONE — inside executive_agent.run().
All other nodes are pure Python.

State flows forward only — no backward edges except the execute_step
self-loop, which terminates when current_step_index >= len(plan.steps).
"""

import operator
from typing import Annotated, Optional

from langgraph.graph import StateGraph, START, END

from services.logger   import get_logger
from agents.guardrails import (
    validate_role,
    check_query_access,
    detect_prompt_injection,
    pre_planning_rbac_check,
)
from agents.planning_agent import create_plan, explain_plan
from agents import db_agent, rag_agent, roi_agent, executive_agent

log = get_logger("graph")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — STATE SCHEMA
#  TypedDict defines every field that flows through the graph.
#
#  WHY TypedDict instead of a dataclass or plain dict?
#    LangGraph requires a TypedDict (or Pydantic model) for its state so it
#    can validate field types and apply reducers at runtime. TypedDict is
#    lighter than Pydantic and sufficient here since we control all inputs.
#
#  WHY Annotated[list[dict], operator.add] for findings?
#    Each execute_step node returns ONE new finding. Instead of manually
#    appending to a list, the operator.add reducer handles accumulation:
#        old_findings + [new_finding] → updated state
#    This means every node is a pure function — it returns only what changed,
#    and LangGraph merges it into the running state.
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(dict):
    """
    Typed state dictionary for the Supply Command AI graph.

    Fields
    ------
    query                : Original user question.
    role                 : Authenticated role ("Demand Planner" / "Operations Manager" / "CFO").
    conversation_history : Optional prior [{"role": ..., "content": ...}] for follow-ups.

    blocked              : True if input guardrails rejected this query.
    blocked_reason       : Human-readable reason for the block.

    plan                 : Full JSON plan from planning_agent.create_plan().
    plan_explanation     : Human-readable plan summary for the UI.
    current_step_index   : Which plan step to execute next (0-based).

    findings             : Accumulated list of agent finding dicts.
                           Uses operator.add reducer — append-only, never reset mid-run.

    final_answer         : Structured dict from executive_agent.run().
    """


# LangGraph requires the state type annotation separate from the class.
# We use a plain TypedDict-style annotation via the annotations dict below.
# This avoids requiring Python 3.12+ TypedDict class syntax while keeping
# compatibility with LangGraph 0.3.x.

from typing import TypedDict

class AgentState(TypedDict, total=False):  # type: ignore[no-redef]
    # ── Input ──────────────────────────────────────────────────────────────────
    query:                str
    role:                 str
    conversation_history: Optional[list[dict]]

    # ── Guardrail results ──────────────────────────────────────────────────────
    blocked:              bool
    blocked_reason:       str

    # ── Planning results ───────────────────────────────────────────────────────
    plan:                 Optional[dict]
    plan_explanation:     str
    current_step_index:   int

    # ── Specialist findings — accumulated across all execute_step calls ────────
    # operator.add is the reducer: old_list + new_list (both are lists of dicts)
    findings:             Annotated[list[dict], operator.add]

    # ── Alert-driven forced context (set by dashboard alert panel) ────────────
    forced_metric:        Optional[str]
    forced_entity:        Optional[str]

    # ── Final answer from executive agent ─────────────────────────────────────
    final_answer:         Optional[dict]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — GRAPH NODES
#  Each node is a pure function: (state) → partial state update dict.
#  Nodes never mutate state directly — they return only the keys they change.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Node 1: Input Guardrails ──────────────────────────────────────────────────

def input_guardrails_node(state: AgentState) -> dict:
    """
    Run all three input safety checks before any data is touched.

    Checks (in order):
        1. validate_role()          — Is this a recognised role?
        2. check_query_access()     — Does this role have permission for this query?
        3. detect_prompt_injection() — Does the query look adversarial?

    WHY this order?
        Role validation is cheapest (dict lookup). Query access is second
        (string matching). Injection detection is last (regex + length check).
        If the cheapest check fails, we never run the expensive ones.

    Returns:
        {"blocked": True,  "blocked_reason": "..."} if any check fails.
        {"blocked": False, "blocked_reason": ""}    if all checks pass.
    """
    query = state.get("query", "")
    role  = state.get("role",  "")

    log.info(
        f"input_guardrails_node | role='{role}' | "
        f"query='{query[:60]}{'...' if len(query) > 60 else ''}'"
    )

    # ── Check 1: Role validation ──────────────────────────────────────────────
    role_check = validate_role(role)
    if not role_check["valid"]:
        log.warning(
            f"input_guardrails_node | BLOCKED | invalid role: '{role}'"
        )
        return {
            "blocked":       True,
            "blocked_reason": f"Invalid role: {role_check['reason']}",
        }

    # ── Check 2: RBAC query access ────────────────────────────────────────────
    # WHY pre_planning_rbac_check() instead of check_query_access() directly?
    #   pre_planning_rbac_check() returns the clean reason string without any
    #   wrapper prefix. This ensures blocked_reason is EXACTLY the message
    #   intended for the user (e.g. the prescribed financial restriction message)
    #   with no "Access denied: " prefix added by this layer.
    rbac_check = pre_planning_rbac_check(query, role)
    if rbac_check["blocked"]:
        log.warning(
            f"input_guardrails_node | BLOCKED | RBAC denied | "
            f"reason='{rbac_check['reason']}'"
        )
        return {
            "blocked":        True,
            "blocked_reason": rbac_check["reason"],
        }

    # ── Check 3: Prompt injection detection ───────────────────────────────────
    injection_check = detect_prompt_injection(query)
    if not injection_check["safe"]:
        log.warning(
            f"input_guardrails_node | BLOCKED | injection detected | "
            f"reason='{injection_check['reason']}'"
        )
        return {
            "blocked":        True,
            "blocked_reason": f"Security check failed: {injection_check['reason']}",
        }

    log.info("input_guardrails_node | PASSED all three safety checks")
    return {
        "blocked":        False,
        "blocked_reason": "",
    }


# ── Node 2: Planner ───────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Classify the user's intent and build a structured investigation plan.

    WHY plan before fetching data?
        Without a plan, agents would fetch data ad-hoc. A pre-built plan:
        • Ensures only relevant agents are called (no redundant DB queries).
        • Makes the reasoning transparent — the UI shows the plan before
          results, so users understand what the system will do.
        • Enables full audit trail: plan steps are logged with their intent
          classification and confidence score.

    Returns:
        {
            "plan":               dict,  ← full JSON plan from create_plan()
            "plan_explanation":   str,   ← human-readable for UI display
            "current_step_index": 0,     ← always reset to 0 for a new plan
        }
    """
    query = state.get("query", "")
    role  = state.get("role",  "")

    log.info(
        f"planner_node | classifying intent | "
        f"query='{query[:60]}'"
    )

    plan        = create_plan(query, role)
    explanation = explain_plan(plan)

    # ── Apply forced context from alert panel (overrides ML detection) ─────────
    # WHY override after create_plan() rather than inside it?
    #   create_plan() takes only (query, role) — it has no visibility into
    #   session state. Overriding here keeps create_plan() pure and stateless.
    #   The alert knows exactly which metric fired, so we trust it over keyword
    #   matching which can be ambiguous ("expedited" → FINANCIAL_IMPACT vs
    #   the specific expedited_cost template).
    forced_metric = state.get("forced_metric")
    forced_entity = state.get("forced_entity")

    if forced_metric:
        plan["detected_metric"] = forced_metric
        plan["alert_driven"]    = True
        plan["no_reinterpret"]  = True
        log.info(f"planner_node | FORCED METRIC: '{forced_metric}'")

        # Also override the first DB step task to the correct SQL template.
        # WHY override here and not inside create_plan()?
        #   create_plan() uses keyword-based metric detection which can clash
        #   with the forced metric (e.g. "cost" matching before "expedited_cost").
        #   The forced override happens post-plan, so we must also fix the step.
        #
        # WHY delay_rate maps differently depending on forced_entity?
        #   When an alert fires on a specific supplier (e.g. SUP003), the exact
        #   supplier delay rate is needed — supplier_delay_rate uses a WHERE
        #   clause + parameterised ? for pinpoint accuracy.
        #   When no entity is set (fleet-wide query), delay_count_by_supplier
        #   returns all suppliers ranked, which is the correct multi-row view.
        _delay_template = (
            "supplier_delay_rate"       # entity-specific: WHERE supplier_id = ?
            if forced_entity
            else "delay_count_by_supplier"  # fleet-wide: all suppliers ranked
        )
        _FORCED_TASK_MAP = {
            "delay_rate":     _delay_template,
            "expedited_cost": "total_expedited_cost",
            "otd":            "fleet_otd_vs_benchmark",
            "sla_breaches":   "total_sla_breaches",
        }
        forced_task = _FORCED_TASK_MAP.get(forced_metric)
        if forced_task:
            for _s in plan.get("steps", []):
                if _s.get("agent") == "DB Agent":
                    _s["task"] = forced_task
                    # For entity-specific delay rate, inject supplier_id as param
                    if forced_task == "supplier_delay_rate" and forced_entity:
                        _s["params"] = (forced_entity,)
                        log.info(
                            f"planner_node | FORCED DB STEP → '{forced_task}' "
                            f"with params=({forced_entity!r},)"
                        )
                    else:
                        log.info(
                            f"planner_node | FORCED DB STEP → '{forced_task}'"
                        )
                    break

    if forced_entity:
        plan["detected_entity"]    = forced_entity
        plan["detected_dimension"] = "supplier"
        log.info(f"planner_node | FORCED ENTITY: '{forced_entity}'")

    log.info(
        f"planner_node | intent='{plan.get('intent')}' | "
        f"steps={plan.get('total_steps', 0)} | "
        f"confidence={plan.get('confidence', 0):.2f} | "
        f"keywords_matched={plan.get('keywords_matched', [])}"
    )

    return {
        "plan":               plan,
        "plan_explanation":   explanation,
        "current_step_index": 0,
    }


# ── Node 3: Execute Step ──────────────────────────────────────────────────────

def execute_step_node(state: AgentState) -> dict:
    """
    Run exactly one step from the plan using the appropriate specialist agent.

    This node is called in a loop (via conditional edge) until all steps
    in the plan are exhausted. Each call:
        1. Reads current_step_index from state.
        2. Fetches the step dict from plan.steps[current_step_index].
        3. Routes to the correct agent: db_agent / rag_agent / roi_agent.
        4. Returns the finding + increments current_step_index.

    WHY a loop instead of parallel execution?
        Some steps depend on prior results — specifically, roi_agent needs
        DB findings to calculate financial exposure. A sequential loop
        ensures findings_so_far is always populated when roi_agent runs.

    WHY not a subgraph per agent?
        The routing logic is trivial (agent name string match). A subgraph
        would add complexity without benefit. The current design keeps all
        routing in one place, which is easier to audit and modify.

    Returns:
        {
            "findings":           [new_finding_dict],  ← appended by operator.add
            "current_step_index": step_index + 1,
        }
    """
    plan            = state.get("plan", {})
    step_index      = state.get("current_step_index", 0)
    role            = state.get("role", "")
    steps           = plan.get("steps", [])
    findings_so_far = state.get("findings", [])

    # Guard: should not happen if conditional routing is correct
    if step_index >= len(steps):
        log.warning(
            f"execute_step_node | step_index={step_index} >= "
            f"len(steps)={len(steps)} — no more steps to execute"
        )
        return {"current_step_index": step_index}

    step       = steps[step_index]
    # Normalise agent name: planning_agent uses display names ("DB Agent"),
    # graph routes on snake_case ("db_agent"). Map both forms.
    _AGENT_NAME_MAP = {
        "db agent":          "db_agent",
        "rag agent":         "rag_agent",
        "roi agent":         "roi_agent",
        "executive agent":   "executive_agent",
        "human_checkpoint":  "human_checkpoint",
        "human checkpoint":  "human_checkpoint",
    }
    raw_agent_name = step.get("agent", "db_agent")
    agent_name     = _AGENT_NAME_MAP.get(raw_agent_name.lower(), raw_agent_name.lower().replace(" ", "_"))
    task_desc      = step.get("task", "")

    log.info(
        f"execute_step_node | step {step_index + 1}/{len(steps)} | "
        f"agent='{agent_name}' | task='{task_desc[:60]}'"
    )

    # ── Route to specialist agent ─────────────────────────────────────────────
    try:
        if agent_name == "db_agent":
            # DB Agent: SQL template lookup → SQLite query → structured finding
            finding = db_agent.run(step, role)

        elif agent_name == "rag_agent":
            # RAG Agent: FAISS search → similarity guardrail → cited chunks
            finding = rag_agent.run(step, role)

        elif agent_name == "roi_agent":
            # ROI Agent: financial calculations from prior DB findings
            # WHY pass findings_so_far?
            #   roi_agent extracts DB data (shipment rows, supplier rows) from
            #   prior findings to avoid a second DB query. This keeps the ROI
            #   calculations grounded in the same data the user just saw.
            finding = roi_agent.run(step, findings_so_far, role)

        elif agent_name == "executive_agent":
            # The executive_agent step in the plan is a placeholder —
            # actual LLM formatting is handled by executive_node after
            # all specialist steps complete. Skip it here gracefully.
            log.debug(
                f"execute_step_node | executive_agent step in plan — "
                f"handled by executive_node, skipping in step loop"
            )
            return {
                "findings":           [],           # nothing to append
                "current_step_index": step_index + 1,
            }

        elif agent_name == "human_checkpoint":
            # Human checkpoint is a planning marker — signals a pause point
            # for human review. In the automated pipeline it logs and passes
            # through. In the UI, app.py can surface this as a visible notice.
            # The actual approval enforcement happens in executive_agent via
            # check_human_approval_needed() — this is the plan-level marker only.
            log.info(
                f"execute_step_node | HUMAN CHECKPOINT | "
                f"step={step_index + 1} | task='{task_desc[:60]}' | "
                f"continuing automatically in pipeline mode"
            )
            return {
                "findings":           [],           # no data finding, just a marker
                "current_step_index": step_index + 1,
            }

        else:
            # Unknown agent — log and return a graceful skip
            log.warning(
                f"execute_step_node | UNKNOWN agent '{agent_name}' | "
                f"step skipped"
            )
            finding = {
                "agent":      agent_name,
                "task":       task_desc,
                "finding":    f"Agent '{agent_name}' not recognised — step skipped.",
                "confidence": 0.0,
                "source":     "graph.py routing",
            }

    except Exception as exc:
        # Agent threw an exception — log it, return a graceful error finding
        # We do NOT re-raise: one failed step should not abort the whole pipeline.
        log.error(
            f"execute_step_node | AGENT ERROR | agent='{agent_name}' | "
            f"step={step_index + 1} | error='{exc}'"
        )
        finding = {
            "agent":      agent_name,
            "task":       task_desc,
            "finding":    f"Step failed: {exc}. Remaining steps will still run.",
            "confidence": 0.0,
            "source":     "error",
        }

    log.info(
        f"execute_step_node | step {step_index + 1} complete | "
        f"confidence={finding.get('confidence', 0):.2f} | "
        f"finding='{str(finding.get('finding', ''))[:80]}'"
    )

    return {
        "findings":           [finding],       # operator.add appends this
        "current_step_index": step_index + 1,  # advance pointer
    }


# ── Node 4: Executive ─────────────────────────────────────────────────────────

def whatif_node(state: AgentState) -> dict:
    """
    Short-circuit node for What-If Simulation queries.

    Calls simulate_whatif() with parameters extracted by the planner, then
    formats the result using format_whatif_output(). Produces a final_answer
    dict directly — skipping execute_step and executive entirely.

    WHY a dedicated node instead of inline logic in executive_node?
        WHATIF queries require zero DB agent calls, zero RAG calls, zero LLM
        calls. A separate node makes the bypass explicit in the graph topology
        and keeps executive_node's logic clean.

    Returns:
        {"final_answer": dict}  ← same shape as executive_node output
    """
    import time as _time
    from agents.roi_agent      import simulate_whatif
    from agents.executive_agent import format_whatif_output

    plan   = state.get("plan", {})
    entity = plan.get("whatif_entity")
    metric = plan.get("whatif_metric", "delay_rate")
    target = plan.get("whatif_target")

    log.info(
        f"whatif_node | entity='{entity}' | metric='{metric}' | "
        f"target={target}"
    )

    t0         = _time.monotonic()
    sim_result = simulate_whatif(entity=entity, metric=metric, target_value=target)
    answer_text = format_whatif_output(sim_result)
    elapsed_ms  = int((_time.monotonic() - t0) * 1000)

    final_answer = {
        "answer":                  answer_text,
        "sources":                 ["What-If Simulation · DB-grounded"],
        "sql_shown":               [],
        "agents_used":             ["ROI Agent (simulation)"],
        "confidence":              0.85 if "error" not in sim_result else 0.0,
        "groundedness_score":      1.0,
        "human_approval_required": False,
        "approval_reason":         "",
        "impact_summary":          "",
        "tokens_used":             0,
        "cost_usd":                0.0,
        "execution_time_ms":       elapsed_ms,
        "warnings":                [],
    }

    log.success(
        f"whatif_node | COMPLETE | "
        f"entity='{entity}' | "
        f"time={elapsed_ms}ms | "
        f"error={'yes' if 'error' in sim_result else 'no'}"
    )
    return {"final_answer": final_answer}


def executive_node(state: AgentState) -> dict:
    """
    Format all verified findings into a role-appropriate answer.

    THIS IS THE ONLY NODE THAT CALLS AN LLM.

    Delegates entirely to executive_agent.run() which:
        • Applies output guardrails (mask [RESTRICTED] values)
        • Checks human approval threshold BEFORE calling LLM
        • Calls GPT-4o-mini ONCE with all findings in context
        • Validates groundedness of LLM response
        • Logs to ai_decisions_log for audit trail

    WHY is the LLM call here and not inside each specialist agent?
        Specialist agents produce structured Python dicts — they never need
        free-text generation. All interpretation is coded. The LLM's only
        job is to write a clear, role-appropriate sentence from verified facts.
        One call here is cheaper, faster, and easier to test than N calls
        spread across agents.

    Returns:
        {"final_answer": dict}  ← full executive_agent.run() return dict
    """
    findings             = state.get("findings", [])
    query                = state.get("query",    "")
    role                 = state.get("role",     "")
    conversation_history = state.get("conversation_history")

    # Pull query_type + metric_definition from the plan so the executive
    # agent can apply query-type-aware data rules in its system prompt.
    plan              = state.get("plan", {})
    query_type        = plan.get("query_type",        "METRIC_QUERY")
    metric_definition = plan.get("metric_definition", "")
    multi_row         = plan.get("multi_row",         False)
    alert_driven      = plan.get("alert_driven",      False)
    detected_metric   = plan.get("detected_metric",   "")

    log.info(
        f"executive_node | role='{role}' | "
        f"total_findings={len(findings)} | "
        f"query_type='{query_type}' | "
        f"query='{query[:60]}'"
    )

    answer = executive_agent.run(
        all_findings         = findings,
        query                = query,
        role                 = role,
        conversation_history = conversation_history,
        query_type           = query_type,
        metric_definition    = metric_definition,
        multi_row            = multi_row,
        alert_driven         = alert_driven,
        detected_metric      = detected_metric,
    )

    log.info(
        f"executive_node | COMPLETE | "
        f"tokens={answer.get('tokens_used', 0)} | "
        f"cost=${answer.get('cost_usd', 0):.6f} | "
        f"groundedness={answer.get('groundedness_score', 0):.2f} | "
        f"human_approval={answer.get('human_approval_required', False)} | "
        f"time={answer.get('execution_time_ms', 0)}ms"
    )

    return {"final_answer": answer}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CONDITIONAL ROUTING FUNCTIONS
#  These functions read state and return a string that LangGraph uses to
#  select the next edge. Pure Python — zero LLM tokens.
# ═══════════════════════════════════════════════════════════════════════════════

def _route_after_planner(state: AgentState) -> str:
    """
    Decide whether to run the normal step-loop or short-circuit to whatif_node.

    Returns:
        "whatif"  → WHATIF_QUERY detected — skip all agents, run simulation
        "execute" → normal plan — proceed to execute_step loop
    """
    plan = state.get("plan", {})
    if plan.get("query_type") == "WHATIF_QUERY":
        log.info("_route_after_planner | WHATIF_QUERY detected — routing to whatif_node")
        return "whatif"
    return "execute"


def _route_after_guardrails(state: AgentState) -> str:
    """
    Decide whether to proceed to the planner or terminate.

    Returns:
        "blocked"  → route to END (query rejected)
        "proceed"  → route to planner node
    """
    if state.get("blocked"):
        log.warning(
            f"_route_after_guardrails | BLOCKED | "
            f"reason='{state.get('blocked_reason', '')}'"
        )
        return "blocked"
    return "proceed"


def _route_after_step(state: AgentState) -> str:
    """
    Decide whether more plan steps remain or the pipeline is done.

    Returns:
        "more_steps" → loop back to execute_step
        "done"       → route to executive node

    WHY count against plan steps, not a flag?
        A flag can be set incorrectly. Counting remaining steps is
        deterministic: current_step_index >= len(plan.steps) is always True
        when there's nothing left to execute, regardless of any agent's
        internal state.
    """
    plan       = state.get("plan", {})
    steps      = plan.get("steps", [])
    step_index = state.get("current_step_index", 0)

    if step_index < len(steps):
        log.debug(
            f"_route_after_step | more_steps | "
            f"next_step={step_index + 1}/{len(steps)}"
        )
        return "more_steps"

    log.info(
        f"_route_after_step | done | "
        f"all {len(steps)} step(s) complete"
    )
    return "done"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GRAPH ASSEMBLY
#  Build and compile the LangGraph state machine.
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """
    Assemble and compile the LangGraph state machine.

    Graph structure:
        START
          │
          ▼
        input_guardrails ──(blocked)──► END
          │
          (proceed)
          │
          ▼
        planner
          │
          ▼
        execute_step ◄────────────────┐
          │                           │
          ├──(more_steps)─────────────┘
          │
          └──(done)
               │
               ▼
             executive
               │
               ▼
              END

    Returns:
        CompiledGraph — a LangGraph runnable ready for .invoke() calls.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("input_guardrails", input_guardrails_node)
    builder.add_node("planner",          planner_node)
    builder.add_node("whatif",           whatif_node)
    builder.add_node("execute_step",     execute_step_node)
    builder.add_node("executive",        executive_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "input_guardrails")

    # ── Guardrails → proceed or terminate ─────────────────────────────────────
    builder.add_conditional_edges(
        "input_guardrails",
        _route_after_guardrails,
        {
            "blocked": END,       # query rejected — no further processing
            "proceed": "planner", # safe to continue
        },
    )

    # ── Planner → WHATIF short-circuit OR normal execute_step loop ────────────
    # WHY conditional here instead of an unconditional edge?
    #   WHATIF_QUERY plans have steps=[] — there is nothing for execute_step
    #   to process. Routing them to whatif_node avoids the step loop entirely
    #   and keeps the graph topology honest about what actually runs.
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "whatif":  "whatif",       # simulation path — no agents, no LLM
            "execute": "execute_step", # normal path — run specialist agents
        },
    )

    # ── Whatif → end (no executive needed — answer already formatted) ─────────
    builder.add_edge("whatif", END)

    # ── Execute step → loop until all steps done, then executive ──────────────
    builder.add_conditional_edges(
        "execute_step",
        _route_after_step,
        {
            "more_steps": "execute_step",  # self-loop: run next step
            "done":       "executive",     # all steps done → format answer
        },
    )

    # ── Executive → end ───────────────────────────────────────────────────────
    builder.add_edge("executive", END)

    compiled = builder.compile()
    log.info("build_graph | graph compiled successfully")
    return compiled


# ── Compiled graph singleton ──────────────────────────────────────────────────
# WHY compile once at module load, not per request?
#   Compilation validates the graph topology and pre-computes routing tables.
#   It's a one-time cost (~50ms). Recompiling per request would waste time.
#   The compiled graph is stateless — it does not hold user data between calls.
graph = build_graph()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PUBLIC API
#  The single entry point used by app.py and any other caller.
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    query:                str,
    role:                 str,
    conversation_history: Optional[list[dict]] = None,
    forced_metric:        Optional[str] = None,
    forced_entity:        Optional[str] = None,
) -> dict:
    """
    Main entry point for the Supply Command AI pipeline.

    Takes a natural language question from an authenticated user, runs it
    through the full multi-agent pipeline, and returns a structured answer
    with sources, SQL, confidence, and cost metadata.

    Args:
        query:                Free-text question from the user.
        role:                 One of "Demand Planner" / "Operations Manager" / "CFO".
        conversation_history: Optional. Prior [{role, content}] messages for
                              follow-up question context.

    Returns:
        {
            "answer":                  str,    ← formatted natural language answer
            "plan_explanation":        str,    ← what the system planned to do
            "sources":                 list,   ← data sources cited in the answer
            "sql_shown":               list,   ← SQL queries that ran
            "agents_used":             list,   ← which specialist agents contributed
            "confidence":              float,  ← weighted confidence (0.0–1.0)
            "groundedness_score":      float,  ← LLM hallucination check score
            "human_approval_required": bool,   ← True if impact > $50K etc.
            "tokens_used":             int,    ← LLM tokens consumed this call
            "cost_usd":                float,  ← estimated API cost
            "execution_time_ms":       int,    ← total pipeline time
            "warnings":                list,   ← any guardrail or confidence warnings
            "blocked":                 bool,   ← True if query was rejected
            "blocked_reason":          str,    ← why it was blocked (if blocked)
        }

    Example:
        >>> result = run_pipeline(
        ...     query = "Which supplier has the highest delay rate?",
        ...     role  = "Operations Manager",
        ... )
        >>> print(result["answer"])
        >>> print(result["sql_shown"])
    """
    log.info(
        f"run_pipeline | role='{role}' | "
        f"query='{query[:80]}{'...' if len(query) > 80 else ''}'"
    )

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: AgentState = {
        "query":                query,
        "role":                 role,
        "conversation_history": conversation_history,
        "forced_metric":        forced_metric,
        "forced_entity":        forced_entity,
        "blocked":              False,
        "blocked_reason":       "",
        "plan":                 None,
        "plan_explanation":     "",
        "current_step_index":   0,
        "findings":             [],
        "final_answer":         None,
    }

    # ── Run the compiled LangGraph pipeline ───────────────────────────────────
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        log.error(f"run_pipeline | GRAPH EXECUTION FAILED | error='{exc}'")
        return {
            "answer":                  f"Pipeline error: {exc}. Please try again.",
            "plan_explanation":        "",
            "sources":                 [],
            "sql_shown":               [],
            "agents_used":             [],
            "confidence":              0.0,
            "groundedness_score":      0.0,
            "human_approval_required": False,
            "tokens_used":             0,
            "cost_usd":                0.0,
            "execution_time_ms":       0,
            "warnings":                [str(exc)],
            "blocked":                 False,
            "blocked_reason":          "",
        }

    # ── Handle blocked queries ────────────────────────────────────────────────
    if final_state.get("blocked"):
        reason = final_state.get("blocked_reason", "Query blocked.")
        log.warning(f"run_pipeline | BLOCKED | reason='{reason}'")
        return {
            "answer":                  reason,
            "plan_explanation":        "",
            "sources":                 [],
            "sql_shown":               [],
            "agents_used":             [],
            "confidence":              0.0,
            "groundedness_score":      0.0,
            "human_approval_required": False,
            "tokens_used":             0,
            "cost_usd":                0.0,
            "execution_time_ms":       0,
            "warnings":                [reason],
            "blocked":                 True,
            "blocked_reason":          reason,
        }

    # ── Extract and return final answer ───────────────────────────────────────
    final_answer = final_state.get("final_answer") or {}

    # ── Extract plan steps for UI trace panel ────────────────────────────────
    _plan      = final_state.get("plan") or {}
    _plan_dict = _plan if isinstance(_plan, dict) else {}
    _steps     = _plan_dict.get("steps", [])

    result = {
        "answer":                  final_answer.get("answer",                  "No answer generated."),
        "plan_explanation":        final_state.get("plan_explanation",         ""),
        "sources":                 final_answer.get("sources",                 []),
        "sql_shown":               final_answer.get("sql_shown",               []),
        "agents_used":             final_answer.get("agents_used",             []),
        "confidence":              final_answer.get("confidence",              0.0),
        "groundedness_score":      final_answer.get("groundedness_score",      0.0),
        "human_approval_required": final_answer.get("human_approval_required", False),
        "approval_reason":         final_answer.get("approval_reason",         ""),
        "impact_summary":          final_answer.get("impact_summary",          ""),
        "tokens_used":             final_answer.get("tokens_used",             0),
        "cost_usd":                final_answer.get("cost_usd",                0.0),
        "execution_time_ms":       final_answer.get("execution_time_ms",       0),
        "warnings":                final_answer.get("warnings",                []),
        "blocked":                 False,
        "blocked_reason":          "",
        # ── Internal state exposed for diagnostics and UI ─────────────────────
        # WHY expose these?
        #   query_type and metric_definition are set by the planner and used by
        #   the executive agent — surfaces them so the chat UI can display the
        #   disambiguation sentence and the agent trace can show query_type.
        #   all_findings lets callers inspect what data went into the answer.
        #   plan_steps drives the agent trace panel in the UI.
        "query_type":              final_state.get("plan", {}).get("query_type",        ""),
        "metric_definition":       final_state.get("plan", {}).get("metric_definition", ""),
        "all_findings":            final_state.get("findings",          []),
        "plan_steps":              [
            s.get("agent") if isinstance(s, dict) else getattr(s, "agent", "")
            for s in _steps
        ],
    }

    log.success(
        f"run_pipeline | COMPLETE | "
        f"role='{role}' | "
        f"confidence={result['confidence']:.2f} | "
        f"tokens={result['tokens_used']} | "
        f"cost=${result['cost_usd']:.6f} | "
        f"time={result['execution_time_ms']}ms | "
        f"human_approval={result['human_approval_required']}"
    )

    return result
