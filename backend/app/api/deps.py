from app.services.repo_service import RepoService
from app.services.status_service import StatusService


def get_repo_service() -> RepoService:
    return RepoService()


def get_status_service() -> StatusService:
    return StatusService()