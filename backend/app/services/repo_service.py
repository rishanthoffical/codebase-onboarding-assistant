from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.indexer.repo import RepoManager
from app.models.repo import Repo
from app.services.cache import Cache


class RepoService:
    """
    Repository registry and repository-information service.

    Redis is optional for local development. The service keeps an
    in-memory registry so repository management continues to work
    when Redis is unavailable.
    """

    _registry: Dict[str, Dict[str, Any]] = {}
    _initialized = False

    def __init__(
        self,
        cache_service: Optional[Cache] = None,
    ):
        if cache_service is not None:
            self.cache_service = cache_service
        else:
            try:
                self.cache_service = Cache(
                    os.getenv(
                        "REDIS_URL",
                        "redis://localhost:6379/0",
                    )
                )
            except Exception:
                self.cache_service = None

        self._initialize_registry()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_registry(self) -> None:
        if RepoService._initialized:
            return

        repositories = self._load_configured_repositories()

        if not repositories:
            repositories = [
                {
                    "repo_id": "local",
                    "name": "Local Repository",
                    "path": self._default_local_repository(),
                }
            ]

        for repository in repositories:
            try:
                self.register_repo(
                    repo_id=repository["repo_id"],
                    name=repository["name"],
                    path=repository["path"],
                    persist=True,
                )
            except Exception:
                continue

        RepoService._initialized = True

    def _default_local_repository(self) -> str:
        configured_path = os.getenv(
            "REPOSITORY_PATH"
        )

        if configured_path:
            return str(
                Path(configured_path)
                .expanduser()
                .resolve()
            )

        project_root = (
            Path(__file__).resolve().parents[3]
        )

        return str(project_root.resolve())

    def _load_configured_repositories(
        self,
    ) -> List[Dict[str, str]]:
        raw = os.getenv(
            "REPOSITORIES_JSON",
            "",
        ).strip()

        if not raw:
            return []

        try:
            parsed = json.loads(raw)
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return []

        if not isinstance(parsed, list):
            return []

        repositories: List[Dict[str, str]] = []

        for item in parsed:
            if not isinstance(item, dict):
                continue

            repo_id = str(
                item.get("repo_id", "")
            ).strip()

            name = str(
                item.get("name", "")
            ).strip()

            path = str(
                item.get("path", "")
            ).strip()

            if not repo_id or not path:
                continue

            repositories.append(
                {
                    "repo_id": repo_id,
                    "name": name or repo_id,
                    "path": path,
                }
            )

        return repositories

    # ------------------------------------------------------------------
    # Redis-safe helpers
    # ------------------------------------------------------------------

    def _cache_get(
        self,
        key: str,
    ) -> Any:
        if self.cache_service is None:
            return None

        try:
            return self.cache_service.get(key)
        except Exception:
            return None

    def _cache_set(
        self,
        key: str,
        value: Any,
    ) -> bool:
        if self.cache_service is None:
            return False

        try:
            self.cache_service.set(
                key,
                value,
                expiration=86400,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_repo(
        data: Dict[str, Any],
    ) -> Repo:
        return Repo(
            repo_id=str(
                data["repo_id"]
            ),
            name=str(
                data["name"]
            ),
            path=str(
                data["path"]
            ),
            current_commit=data.get(
                "current_commit"
            ),
            last_indexed_commit=data.get(
                "last_indexed_commit"
            ),
            status=str(
                data.get(
                    "status",
                    "registered",
                )
            ),
        )

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    def register_repo(
        self,
        repo_id: str,
        name: str,
        path: str,
        persist: bool = True,
    ) -> Repo:
        repo_id = repo_id.strip()
        name = name.strip()

        if not repo_id:
            raise ValueError(
                "repo_id cannot be empty"
            )

        if not path:
            raise ValueError(
                "Repository path cannot be empty"
            )

        resolved_path = (
            Path(path)
            .expanduser()
            .resolve()
        )

        current_commit = None

        if not resolved_path.exists():
            status = "path_not_found"

        elif not resolved_path.is_dir():
            status = "invalid_path"

        else:
            try:
                manager = RepoManager(
                    str(resolved_path)
                )

                current_commit = (
                    manager.get_current_commit()
                )

                status = "ready"

            except Exception:
                status = "not_git_repository"

        repository = Repo(
            repo_id=repo_id,
            name=name or repo_id,
            path=str(resolved_path),
            current_commit=current_commit,
            last_indexed_commit=None,
            status=status,
        )

        data = repository.dict()

        RepoService._registry[
            repo_id
        ] = data

        if persist:
            self._cache_set(
                f"repo:{repo_id}",
                data,
            )

            self._cache_set(
                f"repo:registry:{repo_id}",
                data,
            )

        return repository

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get_repo_info(
        self,
        repo_id: str,
    ) -> Repo:
        repo_id = repo_id.strip()

        cached = self._cache_get(
            f"repo:{repo_id}"
        )

        if isinstance(cached, dict):
            RepoService._registry[
                repo_id
            ] = cached

            repository = self._to_repo(
                cached
            )

            return self._refresh_repository(
                repository
            )

        stored = RepoService._registry.get(
            repo_id
        )

        if stored:
            repository = self._to_repo(
                stored
            )

            return self._refresh_repository(
                repository
            )

        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_id}' "
                "not found"
            ),
        )

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_repository(
        self,
        repository: Repo,
    ) -> Repo:
        try:
            manager = RepoManager(
                repository.path
            )

            current_commit = (
                manager.get_current_commit()
            )

            repository.current_commit = (
                current_commit
            )

            if repository.last_indexed_commit:
                if (
                    repository.last_indexed_commit
                    == current_commit
                ):
                    repository.status = "indexed"
                else:
                    repository.status = (
                        "changes_detected"
                    )
            else:
                repository.status = "ready"

        except Exception:
            path = Path(
                repository.path
            )

            if not path.exists():
                repository.status = (
                    "path_not_found"
                )
            else:
                repository.status = (
                    "not_git_repository"
                )

        data = repository.dict()

        RepoService._registry[
            repository.repo_id
        ] = data

        self._cache_set(
            f"repo:{repository.repo_id}",
            data,
        )

        return repository

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_repos(self) -> List[Repo]:
        repositories = dict(
            RepoService._registry
        )

        if self.cache_service is not None:
            try:
                keys = (
                    self.cache_service.redis_client.keys(
                        "repo:registry:*"
                    )
                )

                for key in keys:
                    if isinstance(
                        key,
                        bytes,
                    ):
                        key = key.decode(
                            "utf-8"
                        )

                    cached = self._cache_get(
                        key
                    )

                    if not isinstance(
                        cached,
                        dict,
                    ):
                        continue

                    repo_id = key.split(
                        "repo:registry:",
                        1,
                    )[-1]

                    repositories[
                        repo_id
                    ] = cached

            except Exception:
                pass

        result: List[Repo] = []

        for data in repositories.values():
            try:
                repository = self._to_repo(
                    data
                )

                repository = (
                    self._refresh_repository(
                        repository
                    )
                )

                result.append(
                    repository
                )

            except Exception:
                continue

        result.sort(
            key=lambda repo: (
                repo.name.lower(),
                repo.repo_id.lower(),
            )
        )

        return result

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_repo(
        self,
        repo_id: str,
        data: Dict[str, Any],
    ) -> Repo:
        repository = self.get_repo_info(
            repo_id
        )

        allowed_fields = {
            "name",
            "path",
            "last_indexed_commit",
            "status",
        }

        for key, value in data.items():
            if key in allowed_fields:
                setattr(
                    repository,
                    key,
                    value,
                )

        if "path" in data:
            repository.path = str(
                Path(
                    repository.path
                )
                .expanduser()
                .resolve()
            )

            try:
                manager = RepoManager(
                    repository.path
                )

                repository.current_commit = (
                    manager.get_current_commit()
                )

            except Exception:
                repository.current_commit = None

        stored = repository.dict()

        RepoService._registry[
            repo_id
        ] = stored

        self._cache_set(
            f"repo:{repo_id}",
            stored,
        )

        self._cache_set(
            f"repo:registry:{repo_id}",
            stored,
        )

        return repository

    # ------------------------------------------------------------------
    # Index state
    # ------------------------------------------------------------------

    def mark_indexed(
        self,
        repo_id: str,
        commit_sha: Optional[str] = None,
    ) -> Repo:
        repository = self.get_repo_info(
            repo_id
        )

        if commit_sha is None:
            try:
                manager = RepoManager(
                    repository.path
                )

                commit_sha = (
                    manager.get_current_commit()
                )

            except Exception:
                commit_sha = None

        repository.last_indexed_commit = (
            commit_sha
        )

        if (
            commit_sha
            and repository.current_commit
            == commit_sha
        ):
            repository.status = "indexed"
        else:
            repository.status = "ready"

        stored = repository.dict()

        RepoService._registry[
            repo_id
        ] = stored

        self._cache_set(
            f"repo:{repo_id}",
            stored,
        )

        self._cache_set(
            f"repo:registry:{repo_id}",
            stored,
        )

        return repository

    # ------------------------------------------------------------------
    # Walk repository
    # ------------------------------------------------------------------

    def walk_repository(
        self,
        repo_path: str,
    ) -> List[str]:
        return [
            str(path)
            for path in RepoManager(
                repo_path
            ).walk_repo()
        ]
