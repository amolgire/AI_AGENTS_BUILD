"""No API key needed: tests the deterministic tool layer that backs every agent (run: pytest -q)."""
import os, sys
import types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pkg = types.ModuleType("finsight_agent"); pkg.__path__ = [os.path.join(ROOT, "finsight_agent")]
sys.modules["finsight_agent"] = pkg          # skip agent.py (needs google-adk + API key); tests cover tools only
from finsight_agent import tools as T

class Ctx:                                   # minimal stand-in for ADK ToolContext
    def __init__(self): self.state = {}

def flow(query, remove=(), reject=()):
    c = Ctx(); p = T.create_research_plan(query, 30, "standard", False, c)
    if p["status"] != "ok": return c, p, None
    T.fetch_news(c); T.fetch_company_data(c); T.review_evidence([], c)
    if remove: T.review_evidence(list(remove), c)
    a = T.run_signal_analysis(c); T.reject_claims(list(reject), "", c)
    r = T.render_report("Summary using only approved findings.", c)
    return c, a, r

def test_banking_ai():
    c, a, r = flow("Summarize AI-in-banking developments")
    assert "DISCLAIMER" in r["report_markdown"] and "SYNTHETIC" in r["report_markdown"]
def test_company_comparison():
    c, a, r = flow("Compare JPMorgan and Goldman Sachs"); assert "JPMorgan" in r["report_markdown"] and "Goldman Sachs" in r["report_markdown"]
def test_no_result():
    assert T.create_research_plan("Research Tesla", 30, "standard", False, Ctx())["status"] == "stop"
def test_malformed():
    assert T.create_research_plan("", 30, "standard", False, Ctx())["status"] == "stop"
def test_conflict_and_stale():
    c, a, r = flow("Research Goldman Sachs"); assert a["conflicts"]
    c, a, r = flow("Research Apple"); assert "freshness threshold" in r["report_markdown"]
def test_prompt_injection_withheld():
    c, a, r = flow("AI update for Citigroup"); assert "reveal the system prompt" not in r["report_markdown"].lower(); assert "withheld" in r["report_markdown"]
def test_human_rejection_and_citation_check():
    c, a, r = flow("Research Apple", remove=("N001",), reject=(0,))
    assert "removed" in r["report_markdown"] and "rejected claims" in r["report_markdown"]
    assert T.verify_citations("Bad claim [N999]", c)["unknown_ids"] == ["N999"]
    assert T.verify_citations("You should buy this [N001]", c)["advice_wording"]
def test_bad_summary_is_rejected():
    c, a, r = flow("Research Apple"); r2 = T.render_report("Made up [N999]", c); assert not r2["summary_accepted"]
def test_finalize_requires_approval(tmp_path):
    c, a, r = flow("Research Apple"); T.OUT_DIR = str(tmp_path)
    assert T.finalize_report("reject", "", c)["status"] == "not_published"
    assert T.finalize_report("approve", "", c)["status"] == "published"
