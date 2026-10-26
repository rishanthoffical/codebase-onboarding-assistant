import asyncio
import logging
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    Request,
)
from fastapi.responses import StreamingResponse

from celery.result import AsyncResult

from app.services.chat_history import ChatHistoryService
from app.api.deps import (
    get_repo_service,
    get_status_service,
)
from app.core.security import (
    require_chat_security,
    require_api_security,
)

from app.models.chat import (
    ChatRequest,
    ChatResponse,
)

from app.config import settings
from app.services.repo_service import RepoService
from app.services.vector_store import VectorStore
from app.services.status_service import StatusService

from app.agent.loop import (
    agent_loop,
    agent_stream,
)

from app.tasks.celery_app import celery_app
from app.tasks.reindex import reindex_task


router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILES = settings.max_files
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024
MAX_SINGLE_FILE_BYTES = settings.max_single_file_mb * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_UPLOAD_BYTES
MAX_ARCHIVE_FILES = MAX_FILES
MAX_EXTRACTED_BYTES = MAX_UPLOAD_BYTES

EXCLUDED_UPLOAD_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "htmlcov",
    "chroma_db",
    "chroma_cache",
    "uploaded_repositories",
    ".vscode",
    ".idea",
}

EXCLUDED_UPLOAD_FILES = {
    ".coverage",
}

SENSITIVE_UPLOAD_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".der",
    ".keystore",
}


def _is_excluded_upload_path(path: PurePosixPath) -> bool:
    if path.name.lower().startswith(".env"):
        return True
    if path.name.lower() in EXCLUDED_UPLOAD_FILES:
        return True
    return any(
        part.lower() in EXCLUDED_UPLOAD_DIRECTORIES
        for part in path.parts[:-1]
    )


def _extract_project_archive(
    archive_bytes: bytes | bytearray,
    destination: Path,
) -> int:
    try:
        archive = zipfile.ZipFile(
            BytesIO(archive_bytes)
        )
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail="Upload a valid ZIP archive.",
        ) from exc

    with archive:
        members = archive.infolist()
        file_members = [
            member
            for member in members
            if not member.is_dir()
            and not _is_excluded_upload_path(
                PurePosixPath(
                    member.filename.replace("\\", "/")
                )
            )
        ]

        if len(file_members) > MAX_FILES:
            raise HTTPException(
                status_code=413,
                detail="Project contains too many files.",
            )

        total_size = sum(
            member.file_size
            for member in file_members
        )

        if total_size > MAX_EXTRACTED_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Project upload is larger than the allowed limit.",
            )

        extracted_size = 0
        excluded_files = 0
        seen_file_paths: set[str] = set()

        for member in members:
            windows_member_path = PureWindowsPath(
                member.filename
            )
            member_path = PurePosixPath(
                member.filename.replace("\\", "/")
            )

            if (
                member_path.is_absolute()
                or windows_member_path.is_absolute()
                or bool(windows_member_path.drive)
                or not member_path.parts
                or ".." in member_path.parts
                or member.flag_bits & 0x1
                or stat.S_ISLNK(
                    member.external_attr >> 16
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail="The ZIP archive contains an unsafe entry.",
                )

            target = destination.joinpath(
                *member_path.parts
            ).resolve()

            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="The ZIP archive contains an unsafe entry.",
                ) from exc

            if member.is_dir():
                if any(
                    part.lower() in EXCLUDED_UPLOAD_DIRECTORIES
                    for part in member_path.parts
                ):
                    continue

                target.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                continue

            if _is_excluded_upload_path(member_path):
                excluded_files += 1
                continue

            normalized_member_path = member_path.as_posix().casefold()
            if normalized_member_path in seen_file_paths:
                raise HTTPException(
                    status_code=400,
                    detail="The ZIP archive contains duplicate file paths.",
                )
            seen_file_paths.add(normalized_member_path)

            if member.file_size > MAX_SINGLE_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "A project file is larger than the "
                        "allowed single-file limit."
                    ),
                )

            if (
                member_path.suffix.lower()
                in SENSITIVE_UPLOAD_SUFFIXES
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The ZIP archive contains a sensitive "
                        "credential file. Remove it before uploading."
                    ),
                )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            member_size = 0
            with archive.open(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    member_size += len(chunk)
                    extracted_size += len(chunk)

                    if (
                        member_size > MAX_SINGLE_FILE_BYTES
                        or extracted_size > MAX_EXTRACTED_BYTES
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Project upload is larger than "
                                "the allowed limit."
                            ),
                        )

                    output.write(chunk)

    if not any(
        path.is_file()
        for path in destination.rglob("*")
    ):
        raise HTTPException(
            status_code=400,
            detail="The ZIP archive is empty.",
        )

    return excluded_files

def _register_uploaded_project(
    uploads: list[tuple[str, bytes | bytearray]],
    repo_service: RepoService,
) -> dict:
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Choose at least one project file.",
        )

    is_single_zip = (
        len(uploads) == 1
        and Path(uploads[0][0]).suffix.lower() == ".zip"
    )
    excluded_files = 0
    eligible_uploads: list[tuple[str, bytes | bytearray]] = []
    seen_paths: set[str] = set()
    for filename, content in uploads:
        relative_path = PurePosixPath(
            filename.replace("\\", "/")
        )
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or ".." in relative_path.parts
        ):
            raise HTTPException(
                status_code=400,
                detail="A selected file has an unsafe path.",
            )

        if _is_excluded_upload_path(relative_path):
            excluded_files += 1
            continue

        if relative_path.suffix.lower() in SENSITIVE_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A selected file has a sensitive credential extension. "
                    "Remove it before uploading."
                ),
            )

        if not is_single_zip and b"\x00" in content[:4096]:
            raise HTTPException(
                status_code=400,
                detail="A selected file is binary and cannot be uploaded as source.",
            )

        normalized_path = relative_path.as_posix()
        if normalized_path.casefold() in seen_paths:
            raise HTTPException(
                status_code=400,
                detail="The selection contains duplicate file paths.",
            )
        seen_paths.add(normalized_path.casefold())
        eligible_uploads.append((normalized_path, content))

    uploads = eligible_uploads
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="No readable project files were selected.",
        )

    if len(uploads) > MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail="Project contains too many files.",
        )

    if sum(len(content) for _, content in uploads) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Project upload is larger than the allowed limit.",
        )

    is_single_zip = (
        len(uploads) == 1
        and Path(uploads[0][0]).suffix.lower() == ".zip"
    )

    if not is_single_zip and any(
        len(content) > MAX_SINGLE_FILE_BYTES
        for _, content in uploads
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                "A project file is larger than the "
                "allowed single-file limit."
            ),
        )

    unsupported_archives = {
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
    }

    for filename, _ in uploads:
        suffix = Path(filename).suffix.lower()

        if suffix in unsupported_archives:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{suffix} archives are not supported. "
                    "Extract the archive and choose its folder, "
                    "or convert it to .zip and choose ZIP."
                ),
            )

    uploads_root = Path(
        os.getenv(
            "REPOSITORY_UPLOADS_PATH",
            str(
                Path(tempfile.gettempdir())
                / "codebase-inspector-uploads"
            ),
        )
    ).resolve()
    uploads_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    repo_id = str(uuid.uuid4())
    repo_path = uploads_root / repo_id
    repo_path.mkdir()
    registered = False
    try:
        if (
            len(uploads) == 1
            and Path(uploads[0][0]).suffix.lower() == ".zip"
        ):
            excluded_files += _extract_project_archive(
                uploads[0][1],
                repo_path,
            )
        else:
            written_files = 0

            for filename, content in uploads:
                relative_path = PurePosixPath(
                    filename.replace("\\", "/")
                )

                if (
                    relative_path.is_absolute()
                    or not relative_path.parts
                    or ".." in relative_path.parts
                    or relative_path.name.lower().startswith(".env")
                    or relative_path.suffix.lower()
                    in SENSITIVE_UPLOAD_SUFFIXES
                    or b"\x00" in content[:4096]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="A selected file has an unsafe path or is binary/sensitive.",
                    )

                if (
                    any(
                        part.lower() in EXCLUDED_UPLOAD_DIRECTORIES
                        for part in relative_path.parts[:-1]
                    )
                    or relative_path.name.lower()
                    in EXCLUDED_UPLOAD_FILES
                ):
                    excluded_files += 1
                    continue

                target = repo_path.joinpath(
                    *relative_path.parts
                ).resolve()

                try:
                    target.relative_to(repo_path.resolve())
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="A selected file has an unsafe path.",
                    ) from exc

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                target.write_bytes(bytes(content))
                written_files += 1

            if not written_files:
                raise HTTPException(
                    status_code=400,
                    detail="No readable project files were selected.",
                )

        first_parts = PurePosixPath(
            uploads[0][0].replace("\\", "/")
        ).parts
        common_root = first_parts[0] if len(first_parts) > 1 else ""
        if common_root and all(
            PurePosixPath(name.replace("\\", "/")).parts[:1]
            == (common_root,)
            for name, _ in uploads
        ):
            repo_name = common_root
        else:
            repo_name = Path(uploads[0][0]).stem or "Uploaded project"

        repository = repo_service.register_repo(
            repo_id=repo_id,
            name=repo_name,
            path=str(repo_path),
        )
        registered = True

        task = reindex_task.delay(
            repo_path=str(repo_path),
            repo_id=repo_id,
        )

    except HTTPException:
        if registered:
            repo_service.unregister_repo(repo_id)
        shutil.rmtree(
            repo_path,
            ignore_errors=True,
        )
        raise
    except Exception as exc:
        logger.exception("Failed to queue repository indexing task")
        if registered:
            repo_service.unregister_repo(repo_id)
        shutil.rmtree(
            repo_path,
            ignore_errors=True,
        )
        raise HTTPException(
            status_code=503,
            detail="The project could not be queued for indexing. Check that the indexing worker is running.",
        ) from exc

    return {
        "repository": repository.dict(),
        "job_id": task.id,
        "status": "indexing",
        "excluded_files": excluded_files,
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    repo_service: RepoService = Depends(get_repo_service),
    status_service: StatusService = Depends(get_status_service),
    _security: None = Depends(require_chat_security),
):
    """
    Main Codebase Onboarding Assistant endpoint.

    Also persists user/assistant messages when a session_id
    is provided.
    """

    try:
        question = request.message.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty.",
            )

        repo_id = request.repo_id or "local"

        history_service = ChatHistoryService()

        response = await agent_loop(
            question=question,
            repo_id=repo_id,
            session_id=request.session_id,
        )

        if request.session_id:
            history_service.add_exchange(
                session_id=request.session_id,
                repo_id=repo_id,
                user_message=question,
                assistant_message=response.answer,
            )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Chat processing failed")
        raise HTTPException(
            status_code=500,
            detail="Chat processing failed. Check the backend logs for details.",
        ) from exc


@router.get("/repos")
async def list_repositories(
    repo_service: RepoService = Depends(
        get_repo_service
    ),
    _security: None = Depends(require_api_security),
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


@router.post("/repos/upload", status_code=202)
async def upload_repository(
    request: Request,
    filename: str,
    repo_service: RepoService = Depends(
        get_repo_service
    ),
    _security: None = Depends(require_api_security),
):
    """Extract an uploaded project ZIP and enqueue its indexing."""

    filename = PurePosixPath(
        filename.replace("\\", "/")
    ).name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Choose a project ZIP or a text/code file.",
        )

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid Content-Length header.",
            ) from exc

        if declared_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Project upload is larger than the allowed limit.",
            )

    uploaded_chunks = bytearray()
    async for chunk in request.stream():
        uploaded_chunks.extend(chunk)

        if len(uploaded_chunks) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Project upload is larger than the allowed limit.",
            )

    return _register_uploaded_project(
        [(filename, uploaded_chunks)],
        repo_service,
    )


@router.get("/repos/{repo_id}")
async def get_repository(
    repo_id: str,
    repo_service: RepoService = Depends(
        get_repo_service
    ),
    _security: None = Depends(require_api_security),
):
    """
    Return information about one repository.
    """

    repository = repo_service.get_repo_info(
        repo_id
    )

    return repository.dict()


@router.get("/repos/{repo_id}/index-status")
async def get_repository_index_status(
    repo_id: str,
    repo_service: RepoService = Depends(get_repo_service),
    status_service: StatusService = Depends(get_status_service),
    _security: None = Depends(require_api_security),
):
    repository = repo_service.get_repo_info(repo_id)

    try:
        index_status = status_service.get_index_status(repo_id)
    except Exception as exc:
        logger.exception(
            "Unable to read vector index status for repo_id=%s",
            repo_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to read repository index status.",
        ) from exc

    return {
        **index_status,
        "repository_status": repository.status,
    }


@router.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    repo_id: str = "local",
    _security: None = Depends(require_api_security),
):
    history_service = ChatHistoryService()

    history = history_service.get_history(
        session_id=session_id,
        repo_id=repo_id,
    )

    return {
        "session_id": session_id,
        "repo_id": repo_id,
        "messages": history,
    }


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    repo_service: RepoService = Depends(get_repo_service),
    status_service: StatusService = Depends(get_status_service),
    _security: None = Depends(require_chat_security),
):
    """
    Stream assistant responses using Server-Sent Events.
    """

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    repo_id = request.repo_id or "local"

    try:
        repo_service.get_repo_info(repo_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise

        vector_count = VectorStore().count(repo_id=repo_id)

        if vector_count <= 0:
            raise HTTPException(
                status_code=404,
                detail=f"Repository '{repo_id}' not found.",
            )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Repository '{repo_id}' not found.",
        )

    def generate():
        yield from agent_stream(
            question=request.message,
            repo_id=repo_id,
            session_id=request.session_id,
        )
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    _security: None = Depends(require_api_security),
):
    """
    Return the current Celery indexing job status.

    The response includes the Celery state and, when
    available, the progress metadata written by the
    reindex task.
    """

    try:
        result = AsyncResult(
            job_id,
            app=celery_app,
        )

        response = {
            "job_id": job_id,
            "state": result.state,
        }

        if result.failed():
            response["status"] = "failed"
            response["error"] = (
                "Indexing failed. Check the backend logs for details."
            )

        elif result.info is not None:
            if isinstance(result.info, dict):
                response["meta"] = result.info
            else:
                response["result"] = result.info

        if result.successful():
            response["status"] = "completed"

        elif result.failed():
            response["status"] = "failed"

        elif result.state in {
            "PENDING",
            "RECEIVED",
            "STARTED",
            "RETRY",
            "REINDEXING",
        }:
            response["status"] = "running"

        else:
            response["status"] = result.state.lower()

        return response

    except Exception as exc:
        logger.exception(
            "Unable to read Celery status for job_id=%s",
            job_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to read job status.",
        ) from exc




