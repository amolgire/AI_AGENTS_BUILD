"""Specialist agents. Each takes/returns plain dicts so hand-offs are structured and loggable."""
import re
from collections import Counter

TOPICS = {"ai": "AI", "artificial intelligence": "AI", "generative": "AI", "earnings": "Earnings", "regulat": "Regulation", "risk": "Risk"}
ADVICE = re.compile(r"\b(you should (buy|sell|invest)|guaranteed|must buy|strong buy)\b", re.I)
DISCLAIMER = ("DISCLAIMER: This output is for educational use only, is generated from SYNTHETIC data, and is not financial advice. "
              "No recommendation or portfolio-impact statement is made.")

# ---------- Planner ----------
def planner_agent(query, store, filters, memory):
    q = (query or "").strip()
    if len(q) < 3: return {"error": "Please enter a research question (company, market, banking or technology topic)."}
    ids = store.resolve_entities(q) or list(filters.get("company_ids") or [])
    sector, sector_ids = store.sector_ids(q)
    topic = next((v for k, v in TOPICS.items() if re.search(r"\b" + k, q.lower())), None)
    followup = False
    if not ids and not sector_ids and not topic and memory.last_company_ids and re.search(r"\b(their|its|them|they|those)\b", q.lower()):
        ids, followup = memory.last_company_ids, True
    low = q.lower()
    intent = ("comparison" if len(ids) >= 2 and re.search(r"compar|vs|versus", low) else
              "executive_brief" if re.search(r"brief|executive|cio", low) else
              "company" if ids else "market")
    scope = ids or sector_ids
    if not scope and not topic:
        return {"error": None, "no_result": True, "query": q,
                "message": "No matching company, sector or topic in the supplied datasets. Try a company name (e.g. Apple) or a topic such as AI in banking."}
    scope = scope or [c for c in store.companies.company_id]
    tasks = [{"n": 1, "agent": "news_research", "do": f"Search financial_news.csv for {topic or 'all topics'} across {len(scope)} compan(ies); filter, dedupe, rank"},
             {"n": 2, "agent": "company_research", "do": "Attach company profiles, latest earnings call and latest analyst report"},
             {"n": 3, "agent": "financial_analyst", "do": "Identify trends, risks, opportunities, contradictions, missing info"},
             {"n": 4, "agent": "critic", "do": "Check citations, stale evidence, unsupported or advice-like claims"},
             {"n": 5, "agent": "report_writer", "do": "Produce audience-appropriate briefing after human approval"}]
    return {"error": None, "no_result": False, "query": q, "intent": intent, "company_ids": scope, "topic": topic, "sector": sector,
            "fresh_days": int(filters.get("fresh_days", 30)), "length": filters.get("length", "standard"), "risk_focus": bool(filters.get("risk_focus")),
            "followup": followup, "tasks": tasks, "datasets": ["companies.csv", "financial_news.csv", "earnings_calls.csv", "analyst_reports.csv"],
            "completion": "Every material claim cites >=1 record ID; conflicts, stale and missing data are visible; human approval obtained."}

# ---------- Research agents ----------
def news_research_agent(plan, store):
    return store.search_news(plan["company_ids"], plan["topic"] if plan["intent"] in ("market", "executive_brief") else None, plan["fresh_days"])

def company_research_agent(plan, store):
    ev = store.get_earnings(plan["company_ids"]) + store.get_analyst(plan["company_ids"], plan["fresh_days"])
    profiles = {c: store.profile(c) for c in plan["company_ids"] if store.profile(c)}
    return ev, profiles

# ---------- Financial analyst ----------
def analyst_agent(plan, evidence, store):
    ev = [e for e in evidence if not e["suspicious"]]
    claims, conflicts, limits = [], [], []
    for e in evidence:
        if e["suspicious"]: limits.append(f"{e['id']} withheld: contains instruction-like text (treated as untrusted data).")
    for cid in dict.fromkeys(e["company_id"] for e in ev):
        name, mine = store.name_of(cid), [e for e in ev if e["company_id"] == cid]
        news = [e for e in mine if e["source_type"] == "news"]
        fresh = [e for e in news if not e["stale"]]
        for e in news:
            if e["stale"]: limits.append(f"{e['id']} ({name}) is {e['age_days']} days old - older than the {plan['fresh_days']}-day freshness threshold; excluded from tallies.")
        cnt = Counter(e["sentiment"] for e in fresh)
        if fresh:
            claims.append({"kind": "fact", "company": name, "claim": f"{name}: {len(fresh)} fresh news records - {cnt['positive']} positive, {cnt['negative']} negative, {cnt['neutral']} neutral.", "evidence": [e["id"] for e in fresh]})
        neg_high = [e for e in fresh if e["sentiment"] == "negative" and e["impact"] == "High"]
        if neg_high:
            claims.append({"kind": "interpretation", "company": name, "tag": "risk", "claim": f"{name}: high-impact negative development(s) flagged: " + "; ".join(e["text"] for e in neg_high[:2]) + ".", "evidence": [e["id"] for e in neg_high]})
        pos_high = [e for e in fresh if e["sentiment"] == "positive" and e["impact"] == "High"]
        if pos_high:
            claims.append({"kind": "interpretation", "company": name, "tag": "opportunity", "claim": f"{name}: high-impact positive development(s): " + "; ".join(e["text"] for e in pos_high[:2]) + ".", "evidence": [e["id"] for e in pos_high]})
        er = next((e for e in mine if e["source_type"] == "earnings"), None)
        if er: claims.append({"kind": "fact", "company": name, "claim": f"{name} earnings ({er['text']}).", "evidence": [er["id"]]})
        an = next((e for e in mine if e["source_type"] == "analyst"), None)
        if an: claims.append({"kind": "fact", "company": name, "claim": f"{name} analyst view (one input, not unquestioned truth): {an['text']}.", "evidence": [an["id"]]})
        elif not mine: limits.append(f"{name}: no records available.")
        else: limits.append(f"{name}: analyst report not available." if not an else "")
        if pos_high or cnt["positive"] > cnt["negative"]:
            counter = [e["id"] for e in neg_high] + ([an["id"]] if an and an["risk"] == "High" else []) + ([er["id"]] if er and er["eps"] < 0 else [])
            if counter:
                conflicts.append({"company": name, "text": f"{name}: positive news momentum conflicts with negative/high-risk signals.", "evidence": [e["id"] for e in fresh if e["sentiment"] == "positive"][:3] + counter})
        # confidence (BR-05): explained from quality, coverage, recency, agreement - no unsupported percentage
        sev = max(cnt.values()) / sum(cnt.values()) if cnt else 0
        pts = (len({e["source_type"] for e in mine}) >= 3) + (len(mine) >= 4) + (bool(news) and len(fresh) / len(news) >= .6) + (sev >= .6)
        level = "High" if pts >= 4 else "Medium" if pts >= 2 else "Low"
        claims.append({"kind": "meta", "company": name, "tag": "confidence", "claim": f"{name} confidence: {level} - {len({e['source_type'] for e in mine})} source type(s), {len(mine)} record(s), {len(fresh)}/{len(news)} news fresh, sentiment agreement {sev:.0%}.", "evidence": [e["id"] for e in mine]})
    if plan["topic"]:
        tn = [e for e in ev if e["source_type"] == "news" and not e["stale"]]
        if tn:
            c = Counter(e["sentiment"] for e in tn)
            claims.insert(0, {"kind": "interpretation", "company": "ALL", "tag": "theme", "claim": f"Theme '{plan['topic']}': {len(tn)} fresh records across {len({e['company_id'] for e in tn})} companies ({c['positive']} positive / {c['negative']} negative / {c['neutral']} neutral).", "evidence": [e["id"] for e in tn]})
    return {"claims": claims, "conflicts": conflicts, "limitations": [l for l in limits if l]}

# ---------- Critic ----------
def critic_agent(analysis, evidence):
    byid, issues = {e["id"]: e for e in evidence}, []
    for i, c in enumerate(analysis["claims"]):
        if c["kind"] != "meta" and not c["evidence"]: issues.append(f"Claim {i}: no source record ID (BR-01).")
        if ADVICE.search(c["claim"]): issues.append(f"Claim {i}: advice-like wording (BR-08).")
        if any(byid.get(x, {}).get("stale") for x in c["evidence"]) and c["kind"] != "meta": issues.append(f"Claim {i}: relies on stale evidence (BR-03).")
        if c["kind"] == "interpretation" and len({byid[x]["source_type"] for x in c["evidence"] if x in byid}) < 2 and c.get("tag") in ("risk", "opportunity"):
            analysis["limitations"].append(f"Claim {i} rests on a single source type - treat as tentative (BR-02).")
    if analysis["conflicts"] and not analysis["conflicts"][0]["evidence"]: issues.append("Conflict listed without evidence (BR-04).")
    return issues

# ---------- Report writer ----------
def report_writer_agent(plan, evidence, analysis, issues, review_log, store, polish=None):
    n = 5 if plan["length"] == "short" else 12 if plan["length"] == "detailed" else 8
    cl = [c for c in analysis["claims"] if c["kind"] != "meta"]
    ids = lambda c: " [" + ", ".join(c["evidence"]) + "]"
    L = [f"# FinSight AI Briefing - {plan['query']}", "", "> **SYNTHETIC EDUCATIONAL DATA** - not real market information (BR-10).", ""]
    summ = polish([c["claim"] + ids(c) for c in cl[:n]], plan["query"]) if polish else None
    L += ["## Executive summary", summ or " ".join(c["claim"] + ids(c) for c in cl[:3]) or "Insufficient evidence available.", "", "## Key findings"]
    L += [f"- {c['claim']}{ids(c)}" for c in cl[:n]]
    L += ["", "## Conflicting signals (shown, not discarded)"] + ([f"- {c['text']} [{', '.join(c['evidence'])}]" for c in analysis["conflicts"]] or ["- None detected in the retrieved evidence."])
    risks = [c for c in analysis["claims"] if c.get("tag") == "risk"]; opps = [c for c in analysis["claims"] if c.get("tag") == "opportunity"]
    if plan["risk_focus"]: opps = opps[:2]
    L += ["", "## Risks"] + ([f"- {c['claim']}{ids(c)}" for c in risks] or ["- No high-impact negative developments retrieved."])
    L += ["", "## Opportunities"] + ([f"- {c['claim']}{ids(c)}" for c in opps] or ["- None identified."])
    L += ["", "## Confidence"] + [f"- {c['claim']}" for c in analysis["claims"] if c.get("tag") == "confidence"]
    L += ["", "## Stale, missing or withheld data"] + ([f"- {x}" for x in analysis["limitations"]] or ["- None."])
    L += ["", "## Questions requiring human judgment"] + [f"- Does the {c['company']} conflict change the relationship-manager narrative?" for c in analysis["conflicts"]]
    L += ["- Is the evidence base broad enough for this audience?"]
    if issues: L += ["", "## Critic notes"] + [f"- {i}" for i in issues]
    L += ["", "## Sources (evidence IDs)", "| ID | Type | Company | Date | Note |", "|---|---|---|---|---|"]
    used = {x for c in analysis["claims"] for x in c["evidence"]}
    L += [f"| {e['id']} | {e['source_type']} | {e['company']} | {e['date']} | {'STALE ' if e['stale'] else ''}{e['text'][:70]} |" for e in evidence if e["id"] in used]
    L += ["", "## Human review log"] + ([f"- {r}" for r in review_log] or ["- No edits; approved as generated."]) + ["", "---", DISCLAIMER]
    return "\n".join(L)
