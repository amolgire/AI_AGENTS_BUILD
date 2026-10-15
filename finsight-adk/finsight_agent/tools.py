"""ADK function tools (Course Day 2). Heavy lifting (retrieval, joins, ranking, conflict/stale detection,
citation checks) is deterministic Python so numbers and evidence IDs are never hallucinated; the LLM agents
decide WHEN to call tools and write the narrative. State lives in session.state (Course Day 3)."""
import json, os, re, runpy, time
from datetime import datetime
try:
    from google.adk.tools.tool_context import ToolContext
except ImportError:                      # lets tests run without ADK installed
    ToolContext = object
from .datastore import DataStore, DATA_DIR
from .logic import planner_agent, analyst_agent, critic_agent, report_writer_agent, ADVICE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR, OUT_DIR = os.path.join(ROOT, "logs"), os.path.join(ROOT, "outputs")
_store = None
clean = lambda o: json.loads(json.dumps(o, default=str))
ID_RE = re.compile(r"\b[NER]\d{3}\b")

def get_store():
    global _store
    if _store is None:
        if not os.path.exists(os.path.join(DATA_DIR, "companies.csv")):
            runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_gen.py"))
        _store = DataStore()
    return _store

class _Mem:
    def __init__(self, ids): self.last_company_ids = ids

def _need(state, *keys):
    miss = [k for k in keys if not state.get(k)]
    return {"status": "error", "message": f"Missing {miss}. Run the earlier workflow steps first."} if miss else None

def _compact(e):
    return {k: e.get(k) for k in ("id", "source_type", "company", "date", "sentiment", "impact", "stale", "suspicious", "text")}

# ---------------- Planner ----------------
def create_research_plan(query: str, fresh_days: int, length: str, risk_focus: bool, tool_context: ToolContext) -> dict:
    """Interpret the user's research request and build a structured research plan.

    Args:
        query: The user's research question, e.g. 'Research Apple' or 'Summarize AI-in-banking developments'.
        fresh_days: Freshness threshold in days (use 30 unless the user says otherwise).
        length: One of 'short', 'standard', 'detailed' (use 'standard' unless asked).
        risk_focus: True if the user wants the report to emphasise risks.
    Returns:
        A plan with intent, scope, tasks and completion criteria, or status 'stop' with a message
        when the query is invalid or nothing matches the datasets.
    """
    st = tool_context.state
    plan = clean(planner_agent(query, get_store(), {"fresh_days": fresh_days, "length": length, "risk_focus": risk_focus},
                               _Mem(st.get("last_company_ids", []))))
    if plan.get("error") or plan.get("no_result"):
        return {"status": "stop", "message": plan.get("error") or plan["message"]}
    for k in ("news_evidence", "company_evidence", "evidence", "analysis", "critic_issues", "draft_report", "removed_ids"):
        st[k] = [] if k.endswith(("evidence", "issues", "ids")) else None
    st["plan"], st["review_log"] = plan, []
    if plan["intent"] in ("company", "comparison"): st["last_company_ids"] = plan["company_ids"]
    names = [get_store().name_of(c) for c in plan["company_ids"]]
    return {"status": "ok", "intent": plan["intent"], "scope": names, "topic": plan["topic"], "tasks": plan["tasks"],
            "completion_criteria": plan["completion"], "freshness_days": plan["fresh_days"]}

# ---------------- Research tools ----------------
def fetch_news(tool_context: ToolContext) -> dict:
    """Search the supplied financial_news dataset for the approved plan: filter, de-duplicate, rank by recency/impact.
    Takes no arguments; reads the approved plan from session state. Returns ranked news records with IDs."""
    st = tool_context.state
    if (e := _need(st, "plan")): return e
    p = st["plan"]
    ev = clean(get_store().search_news(p["company_ids"], p["topic"] if p["intent"] in ("market", "executive_brief") else None, p["fresh_days"]))
    st["news_evidence"] = ev
    return {"status": "ok", "count": len(ev), "records": [_compact(x) for x in ev]}

def fetch_company_data(tool_context: ToolContext) -> dict:
    """Retrieve company profiles, the latest earnings call and the latest analyst report for every company in the
    approved plan. Takes no arguments. Returns records with IDs (E... earnings, R... analyst)."""
    st = tool_context.state
    if (e := _need(st, "plan")): return e
    p, s = st["plan"], get_store()
    ev = clean(s.get_earnings(p["company_ids"]) + s.get_analyst(p["company_ids"], p["fresh_days"]))
    st["company_evidence"] = ev
    profiles = {s.name_of(c): {k: v for k, v in (s.profile(c) or {}).items() if k in ("sector", "country", "market_cap_bn")} for c in p["company_ids"]}
    return {"status": "ok", "count": len(ev), "records": [_compact(x) for x in ev], "profiles": clean(profiles)}

# ---------------- Gate 2: evidence review ----------------
def review_evidence(remove_ids: list[str], tool_context: ToolContext) -> dict:
    """Merge news + company evidence into the shared evidence store and apply the human reviewer's removals.
    Call with an empty list first to show the merged evidence; call again with IDs the reviewer wants removed.

    Args:
        remove_ids: Record IDs to remove, e.g. ['N004', 'R002']; empty list if none.
    """
    st = tool_context.state
    if (e := _need(st, "plan")): return e
    removed = set(st.get("removed_ids") or []) | {r.strip().upper() for r in remove_ids}
    merged = (st.get("news_evidence") or []) + (st.get("company_evidence") or [])
    known = {x["id"] for x in merged}
    ev = [x for x in merged if x["id"] not in removed]
    st["evidence"], st["removed_ids"] = ev, sorted(removed)
    unknown = sorted({r.strip().upper() for r in remove_ids} - known)
    if remove_ids:
        st["review_log"] = (st.get("review_log") or []) + [f"2 Evidence review: removed {', '.join(sorted(removed - set(unknown)))}"]
    return {"status": "ok", "kept": len(ev), "removed": sorted(removed - set(unknown)), "unknown_ids": unknown,
            "records": [_compact(x) for x in ev]}

# ---------------- Analyst + Critic ----------------
def run_signal_analysis(tool_context: ToolContext) -> dict:
    """Compute evidence-backed signals from the approved evidence: sentiment tallies, risks, opportunities, earnings and
    analyst views, conflicts, stale/missing data, and confidence explained by quality/coverage/recency/agreement.
    Takes no arguments. Every claim carries evidence IDs; claims are numbered for human review."""
    st = tool_context.state
    if (e := _need(st, "plan", "evidence")): return e
    a = clean(analyst_agent(st["plan"], st["evidence"], get_store()))
    issues = critic_agent(a, st["evidence"])
    st["analysis"], st["critic_issues"] = a, issues
    return {"status": "ok", "claims": [{"n": i, **c} for i, c in enumerate(a["claims"])], "conflicts": a["conflicts"],
            "limitations": a["limitations"], "critic_issues": issues}

def verify_citations(draft_text: str, tool_context: ToolContext) -> dict:
    """Critic check: every [ID] cited in the text must exist in the evidence store, and no advice-like wording is allowed.

    Args:
        draft_text: The analyst's draft text to verify.
    """
    ids = {x["id"] for x in tool_context.state.get("evidence") or []}
    cited = set(ID_RE.findall(draft_text))
    unknown = sorted(cited - ids)
    advice = bool(ADVICE.search(draft_text))
    return {"status": "ok", "cited_ids": sorted(cited), "unknown_ids": unknown, "advice_wording": advice,
            "uncited_text": not cited, "passed": not unknown and not advice and bool(cited)}

# ---------------- Gate 3: analysis challenge ----------------
def reject_claims(claim_numbers: list[int], reviewer_note: str, tool_context: ToolContext) -> dict:
    """Apply the human reviewer's challenge: mark numbered claims as rejected so they are excluded from the report.

    Args:
        claim_numbers: Claim numbers (as shown by run_signal_analysis) to reject; empty list if none.
        reviewer_note: Optional correction or comment from the reviewer ('' if none).
    """
    st = tool_context.state
    if (e := _need(st, "analysis")): return e
    a = st["analysis"]
    for n in claim_numbers:
        if 0 <= n < len(a["claims"]): a["claims"][n]["rejected"] = True
    st["analysis"] = a
    log = st.get("review_log") or []
    if claim_numbers: log = log + [f"3 Analysis challenge: rejected claims {sorted(claim_numbers)}"]
    if reviewer_note: log = log + [f"3 Analysis challenge: note - {reviewer_note}"]
    st["review_log"] = log
    return {"status": "ok", "rejected": claim_numbers, "remaining": sum(not c.get("rejected") for c in a["claims"])}

# ---------------- Writer ----------------
def render_report(executive_summary: str, tool_context: ToolContext) -> dict:
    """Assemble the full structured briefing (findings, conflicts, risks, stale data, confidence, sources, review log,
    disclaimer). The LLM supplies only the executive summary, which must cite evidence IDs already in the store.

    Args:
        executive_summary: 3-4 sentence summary using only approved claims, keeping [ID] citations.
    """
    st = tool_context.state
    if (e := _need(st, "plan", "evidence", "analysis")): return e
    a = dict(st["analysis"]); a["claims"] = [c for c in a["claims"] if not c.get("rejected")]
    ids = {x["id"] for x in st["evidence"]}
    ok = bool(executive_summary) and set(ID_RE.findall(executive_summary)) <= ids and not ADVICE.search(executive_summary)
    md = report_writer_agent(st["plan"], st["evidence"], a, st.get("critic_issues") or [], (st.get("review_log") or []) + ["4 Final approval: pending"],
                             get_store(), (lambda c, q: executive_summary) if ok else None)
    st["draft_report"] = md
    return {"status": "ok", "summary_accepted": ok, "report_markdown": md}

# ---------------- Gates 1 & 4 ----------------
def record_review_decision(gate: str, action: str, detail: str, tool_context: ToolContext) -> dict:
    """Record a human reviewer decision in the review log / audit trail (e.g. plan approval).

    Args:
        gate: e.g. '1 Plan approval'.
        action: e.g. 'approved', 'changed', 'rejected'.
        detail: Free text, '' if none.
    """
    st = tool_context.state
    st["review_log"] = (st.get("review_log") or []) + [f"{gate}: {action}" + (f" - {detail}" if detail else "")]
    return {"status": "ok"}

def finalize_report(decision: str, reviewer_note: str, tool_context: ToolContext) -> dict:
    """Final human gate. Only decision 'approve' publishes the report to the outputs folder.

    Args:
        decision: 'approve', 'reject' or 'another_cycle'.
        reviewer_note: Optional comment ('' if none).
    """
    st = tool_context.state
    if (e := _need(st, "draft_report")): return e
    st["review_log"] = (st.get("review_log") or []) + [f"4 Final approval: {decision}" + (f" - {reviewer_note}" if reviewer_note else "")]
    if decision != "approve": return {"status": "not_published", "decision": decision}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"finsight_report_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(st["draft_report"].replace("4 Final approval: pending", "4 Final approval: approved"))
    return {"status": "published", "path": path}

# ---------------- Observability callbacks (Course Day 4) ----------------
def _sid(tc):
    try: return tc._invocation_context.session.id
    except Exception: return "session"

def _log(tc, event, **kw):
    os.makedirs(LOG_DIR, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "agent": getattr(tc, "agent_name", "?"), "event": event, **kw}
    with open(os.path.join(LOG_DIR, f"{_sid(tc)}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")

def audit_before_tool(tool, args, tool_context):
    _log(tool_context, "tool_call", tool=getattr(tool, "name", str(tool)), args=args)
    return None

def audit_after_tool(tool, args, tool_context, tool_response):
    summary = {k: v for k, v in tool_response.items() if k != "report_markdown" and not isinstance(v, (list, dict))} if isinstance(tool_response, dict) else str(tool_response)[:200]
    ids = [r["id"] for r in tool_response.get("records", [])] if isinstance(tool_response, dict) else []
    _log(tool_context, "tool_result", tool=getattr(tool, "name", str(tool)), summary=summary, evidence_ids=ids)
    return None
