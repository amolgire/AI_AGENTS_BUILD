# FinSight AI on Google ADK (Kaggle x Google 5-Day AI Agents Intensive style)
**All data and outputs are SYNTHETIC and educational. Not financial advice.**

## Setup (Windows PowerShell; on Mac/Linux use `source .venv/bin/activate`)
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy finsight_agent\.env.example finsight_agent\.env     # then paste your Google AI Studio key (https://aistudio.google.com/apikey)
```
Synthetic CSVs are generated automatically on first run (finsight_agent/data/).

## Run
```
python -m pytest -q tests          # no API key needed - tests the tool layer
adk web --no-reload                # run from THIS folder (the parent of finsight_agent); pick "finsight_agent" in the UI
adk run finsight_agent             # terminal chat
python run_demo.py                 # scripted 5-turn demo through the ADK Runner
```
In the chat you will hit 4 human gates: plan approval -> evidence review -> analysis challenge -> final approval.
Reports land in outputs/, the audit trail in logs/<session>.jsonl, ADK traces in the adk web "Trace" tab.

## Course concept -> where it is used
| Course day | Concept | Here |
|---|---|---|
| 1 | Agents, Sequential/Parallel multi-agent | `research_team` (Parallel), `analysis_team` (Sequential), orchestrator root agent |
| 2 | Custom tools, agents as tools | `tools.py` function tools; `AgentTool` wraps each team |
| 3 | Sessions, state, memory | `tool_context.state` holds plan/evidence/analysis/review log; follow-ups reuse last companies |
| 4 | Quality, observability, guardrails | `audit_*` callbacks (BR-09), critic agent + `verify_citations`, injection-safe data layer |
| 5 | Prototype -> production | Run locally now; deploy later with `adk deploy` (Cloud Run / Agent Engine) |

## Known limits
- ADK wiring (agent.py) was written from the ADK docs but NOT executed by me (no network); if `adk web` errors, paste me the traceback.
- Model default `gemini-2.5-flash`; override with FINSIGHT_MODEL in .env.
- State is in-memory per `adk web` run; use `--session_service_uri sqlite:///sessions.db` for persistence.
- Semantic retrieval, debate agents, DOCX export are not included.
