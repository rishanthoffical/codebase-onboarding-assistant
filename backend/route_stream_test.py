import asyncio
from app.api.routes import chat_stream, ChatRequest
from app.services.repo_service import RepoService
from app.services.status_service import StatusService

async def test():
    request = ChatRequest(
        message="What is llm_service.py?",
        repo_id="00edc45f-603b-434c-ae6e-aec6ed2e8a1d",
        session_id="route-direct-test-001",
    )

    print("CALLING ROUTE")
    response = await chat_stream(
        request=request,
        repo_service=RepoService(),
        status_service=StatusService(),
        _security=None,
    )

    print("RESPONSE TYPE:", type(response))
    print("ITERATOR:", type(response.body_iterator))
    print("GET FIRST CHUNK")

    chunk = await response.body_iterator.__anext__()

    print("FIRST CHUNK:", repr(chunk))

asyncio.run(test())
