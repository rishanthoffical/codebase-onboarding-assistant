from fastapi import APIRouter, HTTPException, Depends

from app.api.deps import (
    get_repo_service,
    get_status_service,
)

from app.models.chat import (
    ChatRequest,
    ChatResponse,
)

from app.services.repo_service import RepoService
from app.services.status_service import StatusService

from app.guardrails.pipeline import scope_guardrail

from app.agent.loop import agent_loop


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    repo_service: RepoService = Depends(get_repo_service),
    status_service: StatusService = Depends(get_status_service),
):
    """
    Main Codebase Onboarding Assistant endpoint.
    """

    try:
        question = request.message.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty.",
            )

        if not scope_guardrail(question):
            return ChatResponse(
                answer=(
                    "This question is outside the scope "
                    "of the codebase assistant. Please ask "
                    "about the repository, code, files, "
                    "functions, classes, APIs, database, "
                    "tests, architecture, or bugs."
                ),
                citations=[],
                tool_calls_made=[
                    "scope_guardrail"
                ],
                confidence=0.95,
                next_steps_suggestion=(
                    "Ask a question about the project's "
                    "codebase."
                ),
            )

        repo_id = request.repo_id or "local"

        response = await agent_loop(
            question=question,
            repo_id=repo_id,
            session_id=request.session_id,
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(exc)}",
        )


@router.get("/repos")
async def list_repositories(
    repo_service: RepoService = Depends(
        get_repo_service
    ),
):
    """
    Return all repositories registered with the assistant.
    """

    repositories = repo_service.list_repos()

    return {
        "repositories": [
            repository.dict()
            for repository in repositories
        ]
    }


@router.get("/repos/{repo_id}")
async def get_repository(
    repo_id: str,
    repo_service: RepoService = Depends(
        get_repo_service
    ),
):
    """
    Return information about one repository.
    """

    repository = repo_service.get_repo_info(
        repo_id
    )

    return repository.dict()
