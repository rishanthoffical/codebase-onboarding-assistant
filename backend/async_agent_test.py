import asyncio
import time
from app.agent.loop import agent_loop

async def test():
    repo_id = "00edc45f-603b-434c-ae6e-aec6ed2e8a1d"

    print("STARTING AGENT LOOP")

    start = time.time()

    response = await agent_loop(
        question="What is llm_service.py?",
        repo_id=repo_id,
        session_id="async-agent-test-001",
    )

    print("AGENT FINISHED:", round(time.time() - start, 2), "s")
    print("TYPE:", type(response))
    print("ANSWER:", response.answer[:500])
    print("TOOLS:", response.tool_calls_made)
    print("CITATIONS:", response.citations)

asyncio.run(test())
