"""Scripted end-to-end demo through the real ADK Runner (needs GOOGLE_API_KEY). Run: python run_demo.py"""
import asyncio, os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finsight_agent", ".env"))
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from finsight_agent.agent import root_agent

TURNS = ["Prepare a five-minute executive briefing on AI trends affecting global banks.",
         "Approve the plan.", "Keep all evidence, continue.", "Accept all claims, write the report.", "Approve and publish."]

async def main():
    svc = InMemorySessionService()
    await svc.create_session(app_name="finsight", user_id="amol", session_id="demo")
    runner = Runner(agent=root_agent, app_name="finsight", session_service=svc)
    for t in TURNS:
        print(f"\n=== USER: {t}")
        async for ev in runner.run_async(user_id="amol", session_id="demo", new_message=types.Content(role="user", parts=[types.Part(text=t)])):
            if ev.is_final_response() and ev.content and ev.content.parts:
                print(ev.content.parts[0].text)
asyncio.run(main())
