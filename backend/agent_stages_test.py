import asyncio
import time

from app.agent.loop import (
    build_contextual_question,
    chat_response_cache,
    get_target_functions,
    find_function_sources,
    retrieve_information,
)

async def test():
    repo_id = "00edc45f-603b-434c-ae6e-aec6ed2e8a1d"
    question = "What is llm_service.py?"
    session_id = "stage-test-001"

    print("1. CONTEXTUAL QUESTION")
    t = time.time()
    contextual_question = build_contextual_question(
        question,
        session_id,
    )
    print("   OK:", contextual_question)
    print("   TIME:", round(time.time() - t, 2), "s")

    print("2. CACHE GET")
    t = time.time()
    cached_response = chat_response_cache.get(
        repo_id=repo_id,
        indexed_commit=None,
        question=contextual_question,
    )
    print("   OK:", cached_response is not None)
    print("   TIME:", round(time.time() - t, 2), "s")

    print("3. TARGET FUNCTIONS")
    t = time.time()
    target_functions = get_target_functions(
        contextual_question
    )
    print("   OK:", target_functions)
    print("   TIME:", round(time.time() - t, 2), "s")

    print("4. FIND FUNCTION SOURCES")
    t = time.time()
    function_sources = find_function_sources(
        target_functions,
        repo_id=repo_id,
    )
    print("   OK:", function_sources)
    print("   TIME:", round(time.time() - t, 2), "s")

    print("5. RETRIEVE INFORMATION")
    t = time.time()
    information = retrieve_information(
        contextual_question,
        repo_id,
    )
    print("   OK")
    print("   DOCUMENTS:", len(information.get("documents", [])))
    print("   TOOL CALLS:", information.get("tool_calls"))
    print("   TIME:", round(time.time() - t, 2), "s")

    print("ALL STAGES FINISHED")

asyncio.run(test())
