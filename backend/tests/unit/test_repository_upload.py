from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api import routes
from app.api.routes import _extract_project_archive
from app.api.routes import _register_uploaded_project


def make_request(content: bytes) -> Request:
    async def receive() -> dict:
        return {
            "type": "http.request",
            "body": content,
            "more_body": False,
        }

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/repos/upload",
            "raw_path": b"/repos/upload",
            "query_string": b"",
            "headers": [],
            "client": ("test", 123),
            "server": ("test", 80),
        },
        receive,
    )
from app.services.repo_service import RepoService


def make_archive(entries: list[tuple[str, str]]) -> bytes:
    buffer = BytesIO()

    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)

    return buffer.getvalue()


def test_extract_project_archive_writes_project_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "project"
    destination.mkdir()

    _extract_project_archive(
        make_archive(
            [
                ("sample-project/README.md", "A sample project"),
                ("sample-project/src/main.py", "print('hello')"),
            ]
        ),
        destination,
    )

    assert (
        destination / "sample-project" / "README.md"
    ).read_text() == "A sample project"
    assert (
        destination / "sample-project" / "src" / "main.py"
    ).read_text() == "print('hello')"


def test_extract_project_archive_excludes_runtime_and_secret_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "project"
    destination.mkdir()

    excluded = _extract_project_archive(
        make_archive(
            [
                ("sample/README.md", "# project"),
                ("sample/node_modules/pkg/index.js", "ignored"),
                ("sample/.git/config", "ignored"),
                ("sample/uploaded_repositories/old/main.py", "ignored"),
                ("sample/chroma_cache/model.json", "ignored"),
                ("sample/.env", "secret"),
            ]
        ),
        destination,
    )

    assert excluded == 5
    assert (destination / "sample" / "README.md").exists()
    assert not (destination / "sample" / "node_modules").exists()
    assert not (destination / "sample" / ".git").exists()
    assert not (destination / "sample" / "uploaded_repositories").exists()
    assert not (destination / "sample" / "chroma_cache").exists()
    assert not (destination / "sample" / ".env").exists()


def test_extract_project_archive_applies_limits_after_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "project"
    destination.mkdir()
    monkeypatch.setattr(routes, "MAX_FILES", 1)
    monkeypatch.setattr(routes, "MAX_EXTRACTED_BYTES", 5)
    monkeypatch.setattr(routes, "MAX_SINGLE_FILE_BYTES", 5)

    excluded = _extract_project_archive(
        make_archive(
            [
                ("sample/README.md", "ok\n"),
                ("sample/node_modules/pkg/large.js", "x" * 100),
            ]
        ),
        destination,
    )

    assert excluded == 1
    assert (destination / "sample" / "README.md").read_text() == "ok\n"


def test_extract_project_archive_enforces_single_file_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "project"
    destination.mkdir()
    monkeypatch.setattr(routes, "MAX_SINGLE_FILE_BYTES", 2)

    with pytest.raises(HTTPException) as error:
        _extract_project_archive(
            make_archive([("sample/main.py", "123")]),
            destination,
        )

    assert error.value.status_code == 413
    assert "single-file limit" in error.value.detail


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.py",
        "..\\..\\outside.py",
        "/outside.py",
        "C:\\outside.py",
        "C:outside.py",
        "\\\\server\\share\\outside.py",
    ],
)
def test_extract_project_archive_rejects_unsafe_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    destination = tmp_path / "project"
    destination.mkdir()

    with pytest.raises(HTTPException) as error:
        _extract_project_archive(
            make_archive([(unsafe_path, "unsafe")]),
            destination,
        )

    assert error.value.status_code == 400


def test_rar_archive_returns_conversion_instructions() -> None:
    with pytest.raises(HTTPException) as error:
        _register_uploaded_project(
            [("source.rar", b"RAR archive data")],
            repo_service=object(),
        )

    assert error.value.status_code == 415
    assert "convert it to .zip" in error.value.detail


def test_non_git_repository_stays_indexed_after_refresh(
    tmp_path: Path,
) -> None:
    class MemoryCache:
        def get(self, key: str) -> None:
            return None

        def set(self, key: str, value: object, expiration: int = 3600) -> None:
            return None

        def delete(self, key: str) -> None:
            return None

    repo_id = f"upload-test-{tmp_path.name}"
    service = RepoService(cache_service=MemoryCache())

    try:
        service.register_repo(
            repo_id=repo_id,
            name="Uploaded project",
            path=str(tmp_path),
        )

        indexed = service.mark_indexed(repo_id)
        refreshed = service.get_repo_info(repo_id)

        assert indexed.status == "indexed"
        assert refreshed.status == "indexed"
        assert refreshed.last_indexed_commit is None
    finally:
        service.unregister_repo(repo_id)


@pytest.mark.asyncio
async def test_upload_registers_project_and_queues_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRepoService:
        def register_repo(
            self,
            repo_id: str,
            name: str,
            path: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                dict=lambda: {
                    "repo_id": repo_id,
                    "name": name,
                    "path": path,
                }
            )

    queued: dict[str, str] = {}

    def enqueue_indexing(
        repo_path: str,
        repo_id: str,
    ) -> SimpleNamespace:
        queued.update(
            repo_path=repo_path,
            repo_id=repo_id,
        )
        return SimpleNamespace(id="job-123")

    monkeypatch.setenv(
        "REPOSITORY_UPLOADS_PATH",
        str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(
        routes.reindex_task,
        "delay",
        enqueue_indexing,
    )

    response = await routes.upload_repository(
        request=make_request(
            make_archive(
                [("sample/main.py", "print('ready')")]
            )
        ),
        filename="sample.zip",
        repo_service=FakeRepoService(),
        _security=None,
    )

    assert response["job_id"] == "job-123"
    assert response["status"] == "indexing"
    assert queued["repo_id"] == response["repository"]["repo_id"]
    assert (
        Path(queued["repo_path"]) / "sample" / "main.py"
    ).read_text() == "print('ready')"


@pytest.mark.asyncio
async def test_upload_registers_single_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRepoService:
        def register_repo(
            self,
            repo_id: str,
            name: str,
            path: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                dict=lambda: {
                    "repo_id": repo_id,
                    "name": name,
                    "path": path,
                }
            )

    monkeypatch.setenv(
        "REPOSITORY_UPLOADS_PATH",
        str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(
        routes.reindex_task,
        "delay",
        lambda **kwargs: SimpleNamespace(id="job-456"),
    )

    response = await routes.upload_repository(
        request=make_request(b"def run():\n    return True\n"),
        filename="main.py",
        repo_service=FakeRepoService(),
        _security=None,
    )

    uploaded_path = Path(
        response["repository"]["path"]
    ) / "main.py"

    assert response["job_id"] == "job-456"
    assert uploaded_path.read_text() == "def run():\n    return True\n"


@pytest.mark.asyncio
async def test_batch_upload_preserves_folder_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRepoService:
        def register_repo(
            self,
            repo_id: str,
            name: str,
            path: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                dict=lambda: {
                    "repo_id": repo_id,
                    "name": name,
                    "path": path,
                }
            )

    monkeypatch.setenv(
        "REPOSITORY_UPLOADS_PATH",
        str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(
        routes.reindex_task,
        "delay",
        lambda **kwargs: SimpleNamespace(id="job-789"),
    )

    response = routes._register_uploaded_project(
        uploads=[
            ("sample/README.md", b"# sample project\n"),
            ("sample/src/main.py", b"print('nested')\n"),
            ("sample/node_modules/pkg/index.js", b"ignored"),
        ],
        repo_service=FakeRepoService(),
    )

    project_path = Path(
        response["repository"]["path"]
    )

    assert response["job_id"] == "job-789"
    assert response["excluded_files"] == 1
    assert response["repository"]["name"] == "sample"
    assert (
        project_path / "sample" / "src" / "main.py"
    ).read_text() == "print('nested')\n"
    assert not (
        project_path / "sample" / "node_modules"
    ).exists()


def test_upload_rejects_file_count_over_configured_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "MAX_FILES", 1)

    with pytest.raises(HTTPException) as error:
        routes._register_uploaded_project(
            uploads=[
                ("first.py", b"pass\n"),
                ("second.py", b"pass\n"),
            ],
            repo_service=object(),
        )

    assert error.value.status_code == 413
    assert error.value.detail == "Project contains too many files."


def test_batch_upload_limits_ignore_excluded_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRepoService:
        def register_repo(
            self,
            repo_id: str,
            name: str,
            path: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                dict=lambda: {
                    "repo_id": repo_id,
                    "name": name,
                    "path": path,
                }
            )

    monkeypatch.setenv(
        "REPOSITORY_UPLOADS_PATH",
        str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(routes, "MAX_FILES", 1)
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(routes, "MAX_SINGLE_FILE_BYTES", 10)
    monkeypatch.setattr(
        routes.reindex_task,
        "delay",
        lambda **kwargs: SimpleNamespace(id="job-filtered"),
    )

    response = routes._register_uploaded_project(
        uploads=[
            ("sample/main.py", b"pass"),
            ("sample/node_modules/pkg/index.js", b"x" * 100),
            ("sample/.env.local", b"API_KEY=not-uploaded"),
        ],
        repo_service=FakeRepoService(),
    )

    project_path = Path(response["repository"]["path"])
    assert response["excluded_files"] == 2
    assert response["repository"]["name"] == "sample"
    assert (project_path / "sample" / "main.py").read_bytes() == b"pass"
    assert not (project_path / "sample" / "node_modules").exists()
    assert not (project_path / "sample" / ".env.local").exists()


def test_upload_rejects_total_size_over_configured_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 3)

    with pytest.raises(HTTPException) as error:
        routes._register_uploaded_project(
            uploads=[("main.py", b"pass\n")],
            repo_service=object(),
        )

    assert error.value.status_code == 413
    assert (
        error.value.detail
        == "Project upload is larger than the allowed limit."
    )


def test_upload_rejects_single_file_over_configured_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(routes, "MAX_SINGLE_FILE_BYTES", 3)

    with pytest.raises(HTTPException) as error:
        routes._register_uploaded_project(
            uploads=[("main.py", b"pass\n")],
            repo_service=object(),
        )

    assert error.value.status_code == 413
    assert "single-file limit" in error.value.detail
