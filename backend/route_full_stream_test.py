import asyncio
import time
from app.api.routes import chat_stream, ChatRequest
from app.services.repo_service import RepoService
from app.services.status_service import StatusService

async def test():
    request = ChatRequest(
        message="What is llm_service.py?",
        repo_id="00edc45f-603b-434c-ae6e-aec6ed2e8a1d",
        session_id="route-full-stream-test-001",
    )

    print("CALLING ROUTE")
    response = await chat_stream(
        request=request,
        repo_service=RepoService(),
        status_service=StatusService(),
        _security=None,
    )

    print("RESPONSE TYPE:", type(response))
    print("STARTING STREAM")
    
    start = time.time()

    async for chunk in response.body_iterator:
        elapsed = round(time.time() - start, 2)
        print(f"[{elapsed}s] CHUNK:", repr(chunk))

    print("STREAM FINISHED")

asyncio.run(test())
