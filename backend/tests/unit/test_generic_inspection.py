from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.agent import loop as agent_loop_module
from app.agent.loop import (
    build_context,
    build_docker_backend_startup_answer,
    build_ollama_request_answer,
    context_shows_ollama_generation_call,
    find_ollama_request_sources,
    find_function_sources,
    has_unverified_function_implementation_claim,
    is_ollama_generation_request_question,
    is_docker_backend_startup_question,
    is_follow_up_question,
    is_project_overview_question,
    retrieve_information,
)
from app.agent.tool_registry import execute_tool
from app.indexer.chunker import chunk_file
from app.indexer.indexer import RepositoryIndexer
from app.agent.loop import verify_repository_sources
from app.api import routes
from app.guardrails.pipeline import scope_guardrail
from app.models.chat import ChatRequest, ChatResponse
from app.services import repo_service as repo_service_module
from app.services import vector_store as vector_store_module
from app.services.project_scanner import build_project_overview
from app.services.vector_store import VectorStore




def test_uploaded_python_class_ranges_use_absolute_chunk_lines() -> None:
    from app.agent.loop import find_python_symbol_line_range

    source = (
        "class OllamaProvider:\n"
        "    def generate(self):\n"
        "        return None\n"
        "\n"
        "class ClaudeProvider:\n"
        "    def generate(self):\n"
        "        return None\n"
    )

    assert find_python_symbol_line_range(
        source, "OllamaProvider", line_offset=109
    ) == (110, 112)

    assert find_python_symbol_line_range(
        source, "ClaudeProvider", line_offset=109
    ) == (114, 116)

    assert find_python_symbol_line_range(
        source, "MissingProvider", line_offset=109
    ) is None


def test_uploaded_python_class_citations_are_not_swapped() -> None:
    from app.agent.loop import correct_python_class_citations

    context = (
        "FILE: llm_service.py:110-112\n"
        "class OllamaProvider:\n"
        "    def generate(self):\n"
        "        return None\n"
        "\n"
        "FILE: llm_service.py:649-651\n"
        "class ClaudeProvider:\n"
        "    def generate(self):\n"
        "        return None\n"
    )
    answer = (
        "OllamaProvider [llm_service.py:649-651]. "
        "ClaudeProvider [llm_service.py:110-112]."
    )
    result = correct_python_class_citations(
        ["llm_service.py:649-651", "llm_service.py:110-112"],
        answer,
        context,
    )
    assert result == [
        "llm_service.py:110-112",
        "llm_service.py:649-651",
    ]


def test_chunk_file_accepts_unlisted_text_extensions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "settings.toml"
    source.write_text(
        "[application]\nname = 'sample'\n",
        encoding="utf-8",
    )

    chunks = chunk_file(str(source))

    assert chunks
    assert "name = 'sample'" in chunks[0]["content"]


def test_function_source_shortcuts_are_local_repository_only() -> None:
    assert find_function_sources(["agent_loop"], "local")
    assert find_function_sources(["agent_loop"], "uploaded-project") == []


def test_repository_summary_requests_use_the_deterministic_overview() -> None:
    assert is_project_overview_question(
        "Summarize this selected repository from the present files."
    )
    assert is_project_overview_question(
        "What does this codebase do?"
    )
    assert is_project_overview_question(
        "What does this repository do?"
    )
    assert not is_project_overview_question(
        "Summarize the webhook handler."
    )


def test_repository_overview_is_not_misclassified_as_follow_up() -> None:
    assert not is_follow_up_question(
        "What does this repository do?"
    )
    assert not is_follow_up_question(
        "Explain security.py."
    )
    assert is_follow_up_question(
        "What does it do?"
    )


def test_repository_tools_use_the_selected_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text(
        "def uploaded_entry():\n    return 'uploaded project'\n",
        encoding="utf-8",
    )

    contents = execute_tool(
        "read_file",
        {
            "path": "src/main.py",
            "line_start": 1,
        },
        repo_root=str(tmp_path),
    )
    matches = execute_tool(
        "grep_repo",
        {
            "pattern": "uploaded_entry",
            "path_glob": "**/*.py",
        },
        repo_root=str(tmp_path),
    )

    assert "uploaded project" in contents
    assert matches[0]["file_path"] == "src/main.py"


def test_citations_verify_against_uploaded_repository_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text(
        "def main():\n    return 'ready'\n\n\n# end\n",
        encoding="utf-8",
    )

    citations = verify_repository_sources(
        ["src/main.py"],
        metadatas=[
            {
                "file_path": "src/main.py",
                "line_start": 1,
                "line_end": 2,
            },
            {
                "file_path": "src/main.py",
                "line_start": 5,
                "line_end": 5,
            },
        ],
        repository_root=str(tmp_path),
    )

    assert citations == [
        "src/main.py:1-2",
        "src/main.py:5-5",
    ]


def test_function_answer_rejects_citation_to_unrelated_lines(
    tmp_path: Path,
) -> None:
    source = tmp_path / "agent" / "loop.py"
    source.parent.mkdir()
    source.write_text(
        "def helper():\n"
        "    return None\n"
        "\n"
        "def retrieve_information():\n"
        "    return []\n",
        encoding="utf-8",
    )

    incorrect_answer = (
        "The function `retrieve_information` is implemented in "
        "`agent/loop.py:1-2`."
    )
    correct_answer = (
        "The function `retrieve_information` is implemented in "
        "`agent/loop.py:4-5`."
    )

    assert has_unverified_function_implementation_claim(
        incorrect_answer,
        str(tmp_path),
    )
    assert not has_unverified_function_implementation_claim(
        correct_answer,
        str(tmp_path),
    )


def test_ollama_request_question_requires_direct_call_evidence() -> None:
    question = (
        "Which function sends a request to Ollama to generate an answer?"
    )

    assert is_ollama_generation_request_question(question)
    assert is_ollama_generation_request_question(
        "Which function sends the POST request to /api/generate?"
    )
    assert not is_ollama_generation_request_question(
        "What is Ollama?"
    )
    assert context_shows_ollama_generation_call(
        'requests.post(f"{self.base_url}/api/generate")'
    )
    assert not context_shows_ollama_generation_call(
        "def retrieve_information(): return vector_search()"
    )


def test_docker_startup_answer_traces_root_compose_to_fastapi_entry(
    tmp_path: Path,
) -> None:
    assert is_docker_backend_startup_question(
        "Which file does Docker use to start the FastAPI backend?"
    )
    assert not is_docker_backend_startup_question(
        "Trace the /chat endpoint from app/api/routes.py into the agent execution pipeline. "
        "Identify the actual functions called, explain the order in which they run, "
        "and show the relevant file paths and line numbers. Only report steps "
        "supported by source code. Do not infer execution order from module names."
    )

    (tmp_path / "backend" / "app").mkdir(parents=True)
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  backend:\n"
        "    build:\n"
        "      context: ./backend\n"
        "      dockerfile: Dockerfile\n"
        "    command: >\n"
        "      uvicorn app.main:app\n"
        "      --host 0.0.0.0\n"
        "  other:\n"
        "    image: unused\n",
        encoding="utf-8",
    )
    (tmp_path / "backend" / "Dockerfile").write_text(
        'CMD ["python", "app/backend/main.py"]\n',
        encoding="utf-8",
    )
    (tmp_path / "backend" / "app" / "main.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n",
        encoding="utf-8",
    )
    nested = tmp_path / "backend" / "app" / "backend"
    nested.mkdir()
    (nested / "docker-compose.yml").write_text(
        "services:\n  backend:\n    command: python wrong.py\n",
        encoding="utf-8",
    )

    answer = build_docker_backend_startup_answer(str(tmp_path))

    assert answer is not None
    text, citations = answer
    assert "`backend/app/main.py`" in text
    assert "`uvicorn app.main:app`" in text
    assert "`./backend/Dockerfile`" in text
    assert "deeper nested and backup copies were not used" in text
    assert "wrong.py" not in text
    assert "docker-compose.yml:6-8" in citations
    assert "backend/app/main.py:2-2" in citations
    assert "backend/Dockerfile:1-1" in citations


def test_docker_startup_answer_handles_one_folder_upload_wrapper(
    tmp_path: Path,
) -> None:
    project = tmp_path / "uploaded-project"
    (project / "backend" / "app").mkdir(parents=True)
    (project / "docker-compose.yml").write_text(
        "services:\n"
        "  backend:\n"
        "    build:\n"
        "      context: ./backend\n"
        "      dockerfile: Dockerfile\n"
        "    command: uvicorn app.main:app\n",
        encoding="utf-8",
    )
    (project / "backend" / "Dockerfile").write_text(
        "FROM python:3.14-slim\n", encoding="utf-8"
    )
    (project / "backend" / "app" / "main.py").write_text(
        "app = FastAPI()\n", encoding="utf-8"
    )
    duplicate = project / "backend" / "app" / "backend"
    duplicate.mkdir()
    (duplicate / "docker-compose.yml").write_text(
        "services:\n  backend:\n    command: python incorrect.py\n",
        encoding="utf-8",
    )

    answer = build_docker_backend_startup_answer(str(tmp_path))

    assert answer is not None
    text, citations = answer
    assert "`uploaded-project/docker-compose.yml`" in text
    assert "`uploaded-project/backend/app/main.py`" in text
    assert "`./backend/Dockerfile`" in text
    assert "incorrect.py" not in text
    assert "uploaded-project/docker-compose.yml:6-6" in citations


def test_docker_startup_answer_finds_compose_under_project_backend_folder(
    tmp_path: Path,
) -> None:
    project = tmp_path / "uploaded-project"
    backend = project / "backend"
    (backend / "app").mkdir(parents=True)
    (backend / "docker-compose.yml").write_text(
        "services:\n"
        "  backend:\n"
        "    build:\n"
        "      context: ./backend\n"
        "      dockerfile: Dockerfile\n"
        "    command: >\n"
        "      uvicorn app.main:app\n",
        encoding="utf-8",
    )
    (backend / "Dockerfile").write_text(
        "FROM python:3.14-slim\n", encoding="utf-8"
    )
    (backend / "app" / "main.py").write_text(
        "app = FastAPI()\n", encoding="utf-8"
    )
    duplicate = backend / "app" / "backend"
    duplicate.mkdir()
    (duplicate / "docker-compose.yml").write_text(
        "services:\n  backend:\n    command: python wrong.py\n",
        encoding="utf-8",
    )

    answer = build_docker_backend_startup_answer(str(tmp_path))

    assert answer is not None
    text, citations = answer
    assert "uploaded-project/backend/docker-compose.yml" in text
    assert "uploaded-project/backend/app/main.py" in text
    assert "wrong.py" not in text
    assert "shallowest Compose file present in the upload" in text


def test_ollama_request_answer_distinguishes_generate_and_chat_endpoints(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "backend" / "app" / "services" / "llm_service.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "import requests\n"
        "class OllamaProvider:\n"
        "    def generate(self, question, context):\n"
        "        response = requests.post(\"http://ollama/api/generate\")\n"
        "        return response.text\n"
        "    def generate_with_tools(self, question, context):\n"
        "        response = requests.post(\"http://ollama/api/chat\")\n"
        "        return response.text\n",
        encoding="utf-8",
    )

    answer = build_ollama_request_answer(str(tmp_path))

    assert answer is not None
    text, citations = answer
    assert "`OllamaProvider.generate()`" in text
    assert "`/api/generate`" in text
    assert "`OllamaProvider.generate_with_tools()`" in text
    assert "`/api/chat`" in text
    assert "backend/app/services/llm_service.py:3-3" in citations
    assert "backend/app/services/llm_service.py:4-4" in citations
    assert "backend/app/services/llm_service.py:6-6" in citations
    assert "backend/app/services/llm_service.py:7-7" in citations


def test_ollama_request_answer_when_both_methods_are_requested(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "backend" / "app" / "services" / "llm_service.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "import requests\n"
        "class OllamaProvider:\n"
        "    def generate(self, question, context):\n"
        "        response = requests.post(\"http://ollama/api/generate\")\n"
        "        return response.text\n"
        "    def generate_with_tools(self, question, context):\n"
        "        response = requests.post(\"http://ollama/api/chat\")\n"
        "        return response.text\n",
        encoding="utf-8",
    )

    question = (
        "Explain both OllamaProvider.generate and "
        "OllamaProvider.generate_with_tools in services/llm_service.py, "
        "including their endpoints and exact method citations."
    )
    answer = build_ollama_request_answer(str(tmp_path), question)

    assert answer is not None
    text, citations = answer
    assert "`OllamaProvider.generate()`" in text
    assert "`/api/generate`" in text
    assert "`OllamaProvider.generate_with_tools()`" in text
    assert "`/api/chat`" in text
    assert citations == [
        "app/services/llm_service.py:3-5",
        "app/services/llm_service.py:6-8",
    ]


def test_ollama_request_answer_excludes_unrequested_method(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "backend" / "app" / "services" / "llm_service.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "import requests\n"
        "class OllamaProvider:\n"
        "    def generate(self, question, context):\n"
        "        response = requests.post(\"http://ollama/api/generate\")\n"
        "        return response.text\n"
        "    def generate_with_tools(self, question, context):\n"
        "        response = requests.post(\"http://ollama/api/chat\")\n"
        "        return response.text\n",
        encoding="utf-8",
    )

    question = (
        "Find the exact definition of OllamaProvider.generate in "
        "services/llm_service.py. Do not use generate_with_tools."
    )
    answer = build_ollama_request_answer(str(tmp_path), question)

    assert answer is not None
    text, citations = answer
    assert "`OllamaProvider.generate()`" in text
    assert "`/api/generate`" in text
    assert "generate_with_tools()" not in text
    assert "`/api/chat`" not in text
    assert citations == ["app/services/llm_service.py:3-5"]


def test_rag_context_redacts_detected_secrets() -> None:
    context = build_context(
        {
            "documents": ['API_KEY = "0123456789abcdef"'],
            "metadatas": [
                {
                    "file_path": "settings.py",
                    "line_start": 1,
                    "line_end": 1,
                }
            ],
        }
    )

    assert "0123456789abcdef" not in context
    assert "[REDACTED]" in context


def test_vector_search_always_filters_by_selected_repository() -> None:
    store = VectorStore.__new__(VectorStore)
    store.collection = MagicMock()

    store.search("payment flow", repo_id="project-b")

    store.collection.query.assert_called_once_with(
        query_texts=["payment flow"],
        n_results=5,
        where={"repo_id": "project-b"},
    )


def test_vector_search_retries_once_for_transient_chroma_id_error() -> None:
    store = VectorStore.__new__(VectorStore)
    store.collection = MagicMock()
    expected = {
        "ids": [["chunk-1"]],
        "documents": [["selected source"]],
        "metadatas": [[{"repo_id": "project-b", "file_path": "main.py"}]],
        "distances": [[0.1]],
    }
    store.collection.query.side_effect = [
        RuntimeError("InternalError: Error finding id"),
        expected,
    ]

    result = store.search("where is the app initialized", repo_id="project-b")

    assert result == expected
    assert store.collection.query.call_count == 2


def test_vector_search_falls_back_unfiltered_and_keeps_repo_isolation() -> None:
    store = VectorStore.__new__(VectorStore)
    store.collection = MagicMock()
    store.collection.count.return_value = 3

    def query(**kwargs):
        if "where" in kwargs:
            raise RuntimeError("InternalError: Error finding id")
        return {
            "ids": [["target-id", "other-id", "second-target-id"]],
            "documents": [["target source", "other source", "second target"]],
            "metadatas": [[
                {"repo_id": "project-b", "file_path": "main.py"},
                {"repo_id": "project-a", "file_path": "main.py"},
                {"repo_id": "project-b", "file_path": "api.py"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }

    store.collection.query.side_effect = query

    result = store.search(
        "find the selected repository",
        n_results=2,
        repo_id="project-b",
    )

    assert result["ids"] == [["target-id", "second-target-id"]]
    assert result["documents"] == [["target source", "second target"]]
    assert all(
        metadata["repo_id"] == "project-b"
        for metadata in result["metadatas"][0]
    )
    assert store.collection.query.call_count == 3
    assert store.collection.query.call_args.kwargs == {
        "query_texts": ["find the selected repository"],
        "n_results": 3,
    }


def test_vector_search_can_filter_by_repository_and_file() -> None:
    store = VectorStore.__new__(VectorStore)
    store.collection = MagicMock()
    store.collection.get.return_value = {
        "ids": ["auth-id"],
        "metadatas": [
            {"repo_id": "project-b", "file_path": "src/auth/login.py"},
        ],
    }
    store.collection.query.return_value = {
        "ids": [["auth-id", "other-repo-auth-id"]],
        "documents": [["auth document", "other repo auth document"]],
        "metadatas": [[
            {"repo_id": "project-b", "file_path": "src/auth/login.py"},
            {"repo_id": "project-a", "file_path": "src/auth/login.py"},
        ]],
        "distances": [[0.1, 0.2]],
    }

    result = store.search(
        "Explain src/auth/login.py",
        repo_id="project-b",
        file_path="src/auth/login.py",
    )

    expected_where = {
        "$and": [
            {"repo_id": "project-b"},
            {"file_path": "src/auth/login.py"},
        ]
    }
    store.collection.get.assert_called_once_with(
        where=expected_where,
        include=["metadatas"],
    )
    store.collection.query.assert_called_once_with(
        query_texts=["Explain src/auth/login.py"],
        n_results=1,
        where=expected_where,
    )
    assert result["ids"] == [["auth-id"]]
    assert result["documents"] == [["auth document"]]
    assert result["metadatas"] == [[
        {"repo_id": "project-b", "file_path": "src/auth/login.py"}
    ]]
    assert result["distances"] == [[0.1]]


@pytest.mark.asyncio
async def test_two_indexed_repositories_do_not_retrieve_each_others_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    (project_a / "README.md").write_text(
        "This project contains a unique greeting function.",
        encoding="utf-8",
    )
    (project_a / "hello.py").write_text(
        'def hello_from_project_a():\n'
        '    return "PROJECT_A_UNIQUE"\n',
        encoding="utf-8",
    )
    (project_b / "README.md").write_text(
        "This project contains a unique payment class.",
        encoding="utf-8",
    )
    (project_b / "payment.java").write_text(
        "class ProjectBUniquePayment {}\n",
        encoding="utf-8",
    )
    (project_a / "shared.py").write_text(
        'PROJECT_A_SHARED = "A_ONLY"\n',
        encoding="utf-8",
    )
    (project_b / "shared.py").write_text(
        'PROJECT_B_SHARED = "B_ONLY"\n',
        encoding="utf-8",
    )
    repo_id_a = f"repo-{uuid4().hex}"
    repo_id_b = f"repo-{uuid4().hex}"
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    class EmptyCache:
        def get(self, **kwargs):
            return None

        def set(self, **kwargs):
            return None

    class FakeRepoService:
        def get_repo_info(self, repo_id: str) -> SimpleNamespace:
            paths = {
                repo_id_a: str(project_a),
                repo_id_b: str(project_b),
            }
            return SimpleNamespace(
                path=paths[repo_id],
                last_indexed_commit=None,
            )

    def echo_retrieved_context(question: str, context: str) -> str:
        return context

    try:
        RepositoryIndexer(
            str(project_a),
            repo_id=repo_id_a,
            vector_store=store,
        ).index()
        RepositoryIndexer(
            str(project_b),
            repo_id=repo_id_b,
            vector_store=store,
        ).index()

        metadata_a = store.get_by_repo(repo_id_a)["metadatas"]
        metadata_b = store.get_by_repo(repo_id_b)["metadatas"]
        assert metadata_a and all(
            metadata.get("repo_id") == repo_id_a
            for metadata in metadata_a
        )
        assert metadata_b and all(
            metadata.get("repo_id") == repo_id_b
            for metadata in metadata_b
        )

        monkeypatch.setattr(
            agent_loop_module,
            "vector_store",
            store,
        )
        explicit_a = retrieve_information(
            "Explain shared.py",
            repo_id=repo_id_a,
        )
        explicit_b = retrieve_information(
            "Explain shared.py",
            repo_id=repo_id_b,
        )
        assert any(
            "PROJECT_A_SHARED" in document
            for document in explicit_a["documents"]
        )
        assert all(
            metadata.get("repo_id") == repo_id_a
            for metadata in explicit_a["metadatas"]
        )
        assert all(
            "PROJECT_B_SHARED" not in document
            for document in explicit_a["documents"]
        )
        assert any(
            "PROJECT_B_SHARED" in document
            for document in explicit_b["documents"]
        )
        assert all(
            metadata.get("repo_id") == repo_id_b
            for metadata in explicit_b["metadatas"]
        )
        assert all(
            "PROJECT_A_SHARED" not in document
            for document in explicit_b["documents"]
        )

        monkeypatch.setattr(
            agent_loop_module,
            "chat_response_cache",
            EmptyCache(),
        )
        monkeypatch.setattr(
            agent_loop_module,
            "llm_service",
            SimpleNamespace(generate=echo_retrieved_context),
        )
        monkeypatch.setattr(
            repo_service_module,
            "RepoService",
            FakeRepoService,
        )

        question = "What unique code exists in this project?"
        answer_a = await agent_loop_module.agent_loop(
            question,
            repo_id=repo_id_a,
        )
        answer_b = await agent_loop_module.agent_loop(
            question,
            repo_id=repo_id_b,
        )

        assert "PROJECT_A_UNIQUE" in answer_a.answer
        assert "hello.py" in answer_a.answer
        assert "ProjectBUniquePayment" not in answer_a.answer
        assert "payment.java" not in answer_a.answer
        assert "ProjectBUniquePayment" in answer_b.answer
        assert "payment.java" in answer_b.answer
        assert "PROJECT_A_UNIQUE" not in answer_b.answer
        assert "hello.py" not in answer_b.answer
    finally:
        store.client.close()


@pytest.mark.asyncio
async def test_webhook_questions_use_repository_evidence_and_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "app" / "api" / "webhook.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def handle_webhook(payload):\n"
        '    """A webhook is an HTTP callback for event notifications."""\n'
        "    verify_signature(payload)\n"
        "    enqueue_event(payload)\n"
        "    return {'accepted': True}\n",
        encoding="utf-8",
    )
    repo_id = f"repo-{uuid4().hex}"
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    class EmptyCache:
        def get(self, **kwargs):
            return None

        def set(self, **kwargs):
            return None

    class FakeRepoService:
        def get_repo_info(self, requested_repo_id: str) -> SimpleNamespace:
            assert requested_repo_id == repo_id
            return SimpleNamespace(
                path=str(repository),
                last_indexed_commit=None,
            )

    def echo_retrieved_context(question: str, context: str) -> str:
        return context

    try:
        RepositoryIndexer(
            str(repository),
            repo_id=repo_id,
            vector_store=store,
        ).index()
        monkeypatch.setattr(
            agent_loop_module,
            "vector_store",
            store,
        )
        monkeypatch.setattr(
            agent_loop_module,
            "chat_response_cache",
            EmptyCache(),
        )
        monkeypatch.setattr(
            agent_loop_module,
            "llm_service",
            SimpleNamespace(generate=echo_retrieved_context),
        )
        monkeypatch.setattr(
            repo_service_module,
            "RepoService",
            FakeRepoService,
        )

        for question in (
            "What is a webhook?",
            "Explain webhook.py.",
        ):
            assert scope_guardrail(question) is True
            response = await agent_loop_module.agent_loop(
                question,
                repo_id=repo_id,
            )
            assert "off-topic" not in response.answer.lower()
            assert "FILE: app/api/webhook.py:" in response.answer
            assert any(
                citation.startswith("app/api/webhook.py:")
                for citation in response.citations
            )

        assert all(
            1 <= int(citation.split(":")[1].split("-")[0])
            <= len(source.read_text(encoding="utf-8").splitlines())
            for citation in response.citations
        )
    finally:
        store.client.close()


@pytest.mark.asyncio
async def test_unrelated_question_is_rejected_only_without_repository_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyCache:
        def get(self, **kwargs):
            return None

    class FakeRepoService:
        def get_repo_info(self, repo_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                path="",
                last_indexed_commit=None,
            )

    monkeypatch.setattr(
        agent_loop_module,
        "chat_response_cache",
        EmptyCache(),
    )
    monkeypatch.setattr(
        agent_loop_module,
        "retrieve_information",
        lambda question, repo_id, **kwargs: {
            "documents": [],
            "metadatas": [],
            "tool_calls": ["vector_search"],
        },
    )
    monkeypatch.setattr(
        repo_service_module,
        "RepoService",
        FakeRepoService,
    )

    response = await agent_loop_module.agent_loop(
        "Tell me a joke.",
        repo_id=f"repo-{uuid4().hex}",
    )

    assert response.answer == (
        "This question is off-topic for the Codebase Inspector."
    )
    assert response.citations == []
    assert response.confidence == 0.0


def test_vector_store_uses_configured_persistence_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    monkeypatch.setattr(
        vector_store_module.settings,
        "chroma_persist_directory",
        "configured-vector-store",
    )
    monkeypatch.setattr(
        vector_store_module.settings,
        "chroma_host",
        "",
    )
    monkeypatch.setattr(
        vector_store_module.chromadb,
        "PersistentClient",
        lambda path: client,
    )

    store = VectorStore()

    assert store.persist_directory == "configured-vector-store"
    assert store.client is client
    client.get_or_create_collection.assert_called_once_with(
        name="codebase"
    )


def test_vector_store_uses_shared_chroma_http_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    monkeypatch.setattr(
        vector_store_module.settings,
        "chroma_host",
        "chroma",
    )
    monkeypatch.setattr(
        vector_store_module.settings,
        "chroma_persist_directory",
        str(tmp_path / "fallback"),
    )
    monkeypatch.setattr(
        vector_store_module.settings,
        "chroma_port",
        8000,
    )
    http_client = MagicMock(return_value=client)
    monkeypatch.setattr(
        vector_store_module.chromadb,
        "HttpClient",
        http_client,
    )

    store = VectorStore()

    assert store.client is client
    http_client.assert_called_once_with(
        host="chroma",
        port=8000,
    )
    client.get_or_create_collection.assert_called_once_with(
        name="codebase"
    )


@pytest.mark.parametrize(
    "requested_class,other_class",
    [
        ("OllamaProvider", "ClaudeProvider"),
        ("ClaudeProvider", "LLMService"),
        ("LLMService", "ClaudeProvider"),
    ],
)
def test_explicit_file_retrieval_prioritizes_requested_class_declaration(
    monkeypatch: pytest.MonkeyPatch,
    requested_class: str,
    other_class: str,
) -> None:
    class SearchStore:
        def get_file_paths(self, repo_id):
            assert repo_id == "project-a"
            return ["backend/app/services/llm_service.py"]

        def search(self, query, n_results, repo_id, file_path=None):
            assert repo_id == "project-a"
            if file_path is None:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            assert file_path == "backend/app/services/llm_service.py"
            assert f"class {requested_class}" in query
            docs = [
                f"class {other_class}:\n    pass",
                f"class {requested_class}:\n    pass",
                "supporting implementation chunk",
            ]
            metadata = [
                {"repo_id": repo_id, "file_path": file_path, "line_start": i * 10}
                for i in range(len(docs))
            ]
            return {"documents": [docs], "metadatas": [metadata], "distances": [[0.1, 0.2, 0.3]]}

    monkeypatch.setattr(agent_loop_module, "vector_store", SearchStore())

    result = retrieve_information(
        f"Show the implementation of {requested_class} in llm_service.py",
        repo_id="project-a",
    )

    assert result["documents"][0].startswith(f"class {requested_class}:")
    assert result["metadatas"][0]["file_path"] == "backend/app/services/llm_service.py"
    assert all(metadata["repo_id"] == "project-a" for metadata in result["metadatas"])


def test_retrieval_drops_distant_semantic_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SearchStore:
        def search(self, query, n_results, repo_id):
            assert repo_id == "project-a"
            return {
                "documents": [["webhook handler", "unrelated source"]],
                "metadatas": [
                    [
                        {"repo_id": "project-a", "file_path": "src/webhook.py"},
                        {"repo_id": "project-a", "file_path": "README.md"},
                    ]
                ],
                "distances": [[0.9, 1.9]],
            }

    monkeypatch.setattr(
        agent_loop_module,
        "vector_store",
        SearchStore(),
    )

    result = retrieve_information(
        "what is a webhook",
        repo_id="project-a",
    )

    assert result["documents"] == ["webhook handler"]
    assert result["metadatas"] == [
        {"repo_id": "project-a", "file_path": "src/webhook.py"}
    ]
    assert result["tool_calls"] == ["vector_search"]


def test_fastapi_initialization_question_scans_selected_repo_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SearchStore:
        def search(self, query, n_results, repo_id):
            return {
                "documents": [["from .main import app"]],
                "metadatas": [[
                    {"repo_id": repo_id, "file_path": "app/__init__.py"}
                ]],
                "distances": [[0.1]],
            }

        def get_by_repo(self, repo_id):
            return {
                "documents": [
                    "app = FastAPI(title=\"Codebase Onboarding Assistant\")",
                    "app = FastAPI(title=\"another repository\")",
                ],
                "metadatas": [
                    {"repo_id": repo_id, "file_path": "backend/app/main.py", "line_start": 13},
                    {"repo_id": "different-repo", "file_path": "backend/app/main.py", "line_start": 13},
                ],
            }

    monkeypatch.setattr(agent_loop_module, "vector_store", SearchStore())

    result = retrieve_information(
        "Where is the FastAPI app initialized?",
        repo_id="selected-repo",
    )

    assert result["documents"][0].startswith("app = FastAPI(")
    assert result["metadatas"][0]["file_path"] == "backend/app/main.py"
    assert result["metadatas"][0]["repo_id"] == "selected-repo"
    assert "get_by_repo" in result["tool_calls"]


def test_fastapi_initialization_prefers_runtime_path_over_duplicate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "backend" / "app" / "main.py"
    duplicate = tmp_path / "backend" / "app" / "backend" / "app" / "main.py"
    package_init = tmp_path / "app" / "__init__.py"
    for source in (canonical, duplicate, package_init):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "from fastapi import FastAPI\n"
            "\n"
            "app = FastAPI(title='sample')\n",
            encoding="utf-8",
        )

    class SearchStore:
        def search(self, query, n_results, repo_id):
            return {
                "documents": [["from .main import app"]],
                "metadatas": [[
                    {"repo_id": repo_id, "file_path": "app/__init__.py"}
                ]],
                "distances": [[0.1]],
            }

    monkeypatch.setattr(agent_loop_module, "vector_store", SearchStore())

    result = retrieve_information(
        "Where is the FastAPI app initialized?",
        repo_id="selected-repo",
        repository_root=str(tmp_path),
    )

    assert result["metadatas"][0]["file_path"] == "backend/app/main.py"
    assert result["metadatas"][0]["line_start"] == 1
    assert "app = FastAPI" in result["documents"][0]
    assert result["metadatas"] == [
        {
            "repo_id": "selected-repo",
            "file_path": "backend/app/main.py",
            "line_start": 1,
            "line_end": 6,
        }
    ]


@pytest.mark.parametrize(
    ("question", "requested_path", "source", "marker"),
    [
        (
            "Explain webhook.py",
            "webhook.py",
            '# Unique spectroscopy calibration dataset\n'
            'SPECTRUM = ["ORION_NEUTRAL_LINE", "QUASAR_BLUE_SHIFT"]\n'
            "CALIBRATION_VALUE = 7391\n",
            "ORION_NEUTRAL_LINE",
        ),
        (
            "Explain hello.py",
            "hello.py",
            '# Greeting template\ndef greet():\n'
            '    return "HELLO_COPPER_KITE"\n',
            "HELLO_COPPER_KITE",
        ),
        (
            "Explain routes.py",
            "routes.py",
            '# Orchard navigation\nROUTE_TABLE = '
            '{"north": "OBSERVATORY_DOME", "south": "ARCHIVE"}\n',
            "OBSERVATORY_DOME",
        ),
        (
            "Explain src/auth/login.py",
            "src/auth/login.py",
            "def authenticate(credentials):\n"
            '    """Validate a login and establish an authenticated session."""\n'
            '    return "AUTHENTICATION_SENTINEL"\n',
            "AUTHENTICATION_SENTINEL",
        ),
    ],
)
def test_explicit_file_reference_retrieves_scoped_chroma_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    requested_path: str,
    source: str,
    marker: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.md").write_text(
        "# Orchard Observatory\n"
        "A catalog of distant stars and spectral observations.\n",
        encoding="utf-8",
    )
    target = repository / requested_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    repo_id = f"repo-{uuid4().hex}"
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    try:
        RepositoryIndexer(
            str(repository),
            repo_id=repo_id,
            vector_store=store,
        ).index()
        raw = store.search(
            question,
            n_results=10,
            repo_id=repo_id,
        )
        raw_matches = [
            (metadata, distance)
            for metadata, distance in zip(
                raw["metadatas"][0],
                raw["distances"][0],
            )
            if metadata.get("file_path") == requested_path
        ]
        assert raw_matches

        if requested_path == "webhook.py":
            assert all(
                distance > agent_loop_module.MAX_RETRIEVAL_DISTANCE
                for _, distance in raw_matches
            )

        monkeypatch.setattr(
            agent_loop_module,
            "vector_store",
            store,
        )
        result = retrieve_information(
            question,
            repo_id=repo_id,
        )

        assert result["resolved_file_references"] == [
            requested_path
        ]
        assert result["metadatas"][0]["file_path"] == requested_path
        assert result["metadatas"][0]["repo_id"] == repo_id
        assert any(
            marker in document
            for document in result["documents"]
        )
    finally:
        store.client.close()


def test_conceptual_question_keeps_semantic_repository_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "src" / "auth" / "login.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def authenticate(credentials):\n"
        '    """Authentication verifies credentials before login."""\n'
        '    return "SESSION_CREATED"\n',
        encoding="utf-8",
    )
    repo_id = f"repo-{uuid4().hex}"
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    try:
        RepositoryIndexer(
            str(repository),
            repo_id=repo_id,
            vector_store=store,
        ).index()
        monkeypatch.setattr(
            agent_loop_module,
            "vector_store",
            store,
        )

        result = retrieve_information(
            "What is authentication?",
            repo_id=repo_id,
        )

        assert result["requested_file_references"] == []
        assert any(
            metadata.get("file_path") == "src/auth/login.py"
            for metadata in result["metadatas"]
        )
        assert any(
            "Authentication verifies credentials" in document
            for document in result["documents"]
        )
    finally:
        store.client.close()


@pytest.mark.asyncio
async def test_missing_explicit_file_is_not_invented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.md").write_text(
        "A small example repository.",
        encoding="utf-8",
    )
    repo_id = f"repo-{uuid4().hex}"
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    class EmptyCache:
        def get(self, **kwargs):
            return None

    class FakeRepoService:
        def get_repo_info(self, requested_repo_id: str) -> SimpleNamespace:
            assert requested_repo_id == repo_id
            return SimpleNamespace(
                path=str(repository),
                last_indexed_commit=None,
            )

    try:
        RepositoryIndexer(
            str(repository),
            repo_id=repo_id,
            vector_store=store,
        ).index()
        monkeypatch.setattr(
            agent_loop_module,
            "vector_store",
            store,
        )
        monkeypatch.setattr(
            agent_loop_module,
            "chat_response_cache",
            EmptyCache(),
        )
        monkeypatch.setattr(
            agent_loop_module,
            "llm_service",
            SimpleNamespace(
                generate=lambda **kwargs: pytest.fail(
                    "Missing-file questions must not reach answer generation."
                )
            ),
        )
        monkeypatch.setattr(
            repo_service_module,
            "RepoService",
            FakeRepoService,
        )

        response = await agent_loop_module.agent_loop(
            "Explain missing_file.py",
            repo_id=repo_id,
        )

        assert response.answer == (
            "I could not find the requested file `missing_file.py` "
            "in the current repository."
        )
        assert response.citations == []
        assert response.confidence == 0.0
    finally:
        store.client.close()


def test_overview_retrieval_uses_project_evidence_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SearchStore:
        query = ""
        limit = 0

        def search(self, query, n_results, repo_id):
            self.query = query
            self.limit = n_results
            return {
                "documents": [["Project setup information"]],
                "metadatas": [[
                    {
                        "repo_id": repo_id,
                        "file_path": "README.md",
                        "language": "Markdown",
                    }
                ]],
                "distances": [[1.8]],
            }

    store = SearchStore()
    monkeypatch.setattr(agent_loop_module, "vector_store", store)

    result = retrieve_information(
        "Give me a beginner-friendly overview of this codebase.",
        repo_id="project-a",
    )

    assert "README" in store.query
    assert store.limit == 10
    assert result["documents"] == ["Project setup information"]


def test_follow_up_context_does_not_force_overview_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SearchStore:
        query = ""
        limit = 0

        def search(self, query, n_results, repo_id):
            self.query = query
            self.limit = n_results
            return {
                "documents": [["Provider selection implementation"]],
                "metadatas": [[
                    {
                        "repo_id": repo_id,
                        "file_path": "llm_service.py",
                        "language": "Python",
                    }
                ]],
                "distances": [[0.2]],
            }

    store = SearchStore()
    monkeypatch.setattr(agent_loop_module, "vector_store", store)
    contextual_question = (
        "Previous assistant answer: the project overview describes the "
        "important files and main execution flow.\n\n"
        "Current developer follow-up: How does this file choose between "
        "Ollama and Claude?"
    )

    result = retrieve_information(
        contextual_question,
        repo_id="project-a",
        overview_request=False,
    )

    assert store.query == contextual_question
    assert store.limit == 5
    assert result["documents"] == ["Provider selection implementation"]


@pytest.mark.asyncio
async def test_follow_up_after_overview_gets_a_normal_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_id = "follow-up-after-overview"
    session_id = f"session-{uuid4().hex}"
    overview_calls: list[bool | None] = []

    class EmptyCache:
        def get(self, **kwargs):
            return None

        def set(self, **kwargs):
            return True

    class FakeRepoService:
        def get_repo_info(self, requested_repo_id: str) -> SimpleNamespace:
            assert requested_repo_id == repo_id
            return SimpleNamespace(
                path=str(tmp_path),
                last_indexed_commit=None,
            )

    def retrieve(question, requested_repo_id, **kwargs):
        overview_calls.append(kwargs.get("overview_request"))
        return {
            "documents": ["FILE: llm_service.py:1-4\nprovider selection code"],
            "metadatas": [
                {
                    "repo_id": requested_repo_id,
                    "file_path": "llm_service.py",
                    "line_start": 1,
                    "line_end": 4,
                }
            ],
            "tool_calls": ["vector_search"],
            "requested_file_references": [],
            "resolved_file_references": [],
            "file_reference_lookup_complete": True,
        }

    monkeypatch.setattr(repo_service_module, "RepoService", FakeRepoService)
    monkeypatch.setattr(agent_loop_module, "chat_response_cache", EmptyCache())
    monkeypatch.setattr(agent_loop_module, "retrieve_information", retrieve)
    monkeypatch.setattr(agent_loop_module, "find_function_sources", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        agent_loop_module,
        "llm_service",
        SimpleNamespace(
            generate=lambda **kwargs: "Ollama and Claude are selected by provider configuration."
        ),
    )
    monkeypatch.setattr(
        agent_loop_module,
        "build_project_overview",
        lambda **kwargs: pytest.fail("A follow-up must not return a project overview."),
    )

    agent_loop_module._save_session(
        session_id=session_id,
        question="Give me an overview of this codebase.",
        answer="The overview described important files and execution flow.",
        citations=[],
    )
    try:
        response = await agent_loop_module.agent_loop(
            "How does this file choose between Ollama and Claude?",
            repo_id=repo_id,
            session_id=session_id,
        )
    finally:
        agent_loop_module.SESSION_MEMORY.pop(session_id, None)

    assert overview_calls == [False]
    assert "provider configuration" in response.answer
    assert "repository_scan" not in response.tool_calls_made


@pytest.mark.asyncio
async def test_index_status_reports_only_the_selected_repository() -> None:
    class FakeRepoService:
        def get_repo_info(self, repo_id: str) -> SimpleNamespace:
            return SimpleNamespace(status="indexed")

    class FakeStatusService:
        def get_index_status(self, repo_id: str) -> dict:
            return {
                "repo_id": repo_id,
                "vector_count": 3,
                "index_freshness": "indexed",
            }

    response = await routes.get_repository_index_status(
        repo_id="project-a",
        repo_service=FakeRepoService(),
        status_service=FakeStatusService(),
        _security=None,
    )

    assert response == {
        "repo_id": "project-a",
        "vector_count": 3,
        "index_freshness": "indexed",
        "repository_status": "indexed",
    }


@pytest.mark.asyncio
async def test_webhook_question_reaches_repository_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = ChatResponse(
        answer="Webhook handling is described in the selected repository.",
        citations=["src/webhook.py:1-4"],
        tool_calls_made=["vector_search"],
        confidence=0.85,
    )
    agent_call = AsyncMock(return_value=answer)
    monkeypatch.setattr(routes, "agent_loop", agent_call)

    response = await routes.chat(
        request=ChatRequest(
            message="what is a webhook",
            repo_id="project-a",
        ),
        repo_service=object(),
        status_service=object(),
        _security=None,
    )

    assert response.answer == answer.answer
    agent_call.assert_awaited_once_with(
        question="what is a webhook",
        repo_id="project-a",
        session_id=None,
    )


@pytest.mark.asyncio
async def test_chat_errors_do_not_expose_internal_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_chat(**kwargs):
        raise RuntimeError("API_KEY=0123456789abcdef")

    monkeypatch.setattr(routes, "agent_loop", fail_chat)

    with pytest.raises(HTTPException) as error:
        await routes.chat(
            request=ChatRequest(
                message="Explain the API",
                repo_id="project-a",
            ),
            repo_service=object(),
            status_service=object(),
            _security=None,
        )

    assert error.value.status_code == 500
    assert "0123456789abcdef" not in error.value.detail
    assert "Check the backend logs" in error.value.detail


@pytest.mark.asyncio
async def test_streaming_chat_rejects_empty_messages() -> None:
    with pytest.raises(HTTPException) as error:
        await routes.chat_stream(
            request=ChatRequest(message="   ", repo_id="project-a"),
            repo_service=object(),
            status_service=object(),
            _security=None,
        )

    assert error.value.status_code == 400
    assert error.value.detail == "Message cannot be empty."


@pytest.mark.asyncio
async def test_general_technical_question_without_repository_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyCache:
        def get(self, **kwargs):
            return None

    class FakeRepoService:
        def get_repo_info(self, repo_id):
            return SimpleNamespace(
                path="",
                last_indexed_commit=None,
            )

    def general_answer(question: str, context: str) -> str:
        assert question == "What is an API?"
        assert "No relevant repository evidence" in context
        return (
            "An API is a defined interface that software uses to "
            "communicate with another system. This is a general "
            "explanation; no implementation evidence was found in "
            "the uploaded repository."
        )

    monkeypatch.setattr(
        agent_loop_module,
        "chat_response_cache",
        EmptyCache(),
    )
    monkeypatch.setattr(
        agent_loop_module,
        "retrieve_information",
        lambda question, repo_id, **kwargs: {
            "documents": [],
            "metadatas": [],
            "tool_calls": ["vector_search"],
        },
    )
    monkeypatch.setattr(
        repo_service_module,
        "RepoService",
        FakeRepoService,
    )
    monkeypatch.setattr(
        agent_loop_module,
        "llm_service",
        SimpleNamespace(generate=general_answer),
    )

    response = await agent_loop_module.agent_loop(
        "What is an API?",
        repo_id="empty-project",
    )

    assert "general explanation" in response.answer
    assert "no implementation evidence" in response.answer
    assert response.confidence == 0.45
    assert response.citations == []
    assert response.tool_calls_made == [
        "vector_search",
        "answer_generation",
    ]


@pytest.mark.asyncio
async def test_unrelated_question_without_repository_evidence_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyCache:
        def get(self, **kwargs):
            return None

    class FakeRepoService:
        def get_repo_info(self, repo_id):
            return SimpleNamespace(
                path="",
                last_indexed_commit=None,
            )

    monkeypatch.setattr(
        agent_loop_module,
        "chat_response_cache",
        EmptyCache(),
    )
    monkeypatch.setattr(
        agent_loop_module,
        "retrieve_information",
        lambda question, repo_id, **kwargs: {
            "documents": [],
            "metadatas": [],
            "tool_calls": ["vector_search"],
        },
    )
    monkeypatch.setattr(
        repo_service_module,
        "RepoService",
        FakeRepoService,
    )
    monkeypatch.setattr(
        agent_loop_module,
        "llm_service",
        SimpleNamespace(
            generate=lambda question, context: "OUT_OF_SCOPE"
        ),
    )

    response = await agent_loop_module.agent_loop(
        "What is the weather tomorrow?",
        repo_id="empty-project",
    )

    assert response.answer == (
        "This question is off-topic for the Codebase Inspector."
    )
    assert response.confidence == 0.0
    assert response.citations == []


def test_project_overview_scans_only_existing_repository_evidence(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    source = repository / "src" / "main.py"
    source.parent.mkdir(parents=True)
    (repository / "README.md").write_text(
        "# Billing service\n\nProcesses invoices for customers.\n",
        encoding="utf-8",
    )
    (repository / "requirements.txt").write_text(
        "fastapi==1.0\nsqlalchemy==2.0\n",
        encoding="utf-8",
    )
    source.write_text(
        "from fastapi import FastAPI\n"
        "from sqlalchemy import create_engine\n"
        "app = FastAPI()\n"
        "@app.post('/invoices')\n"
        "def create_invoice():\n"
        "    return {}\n",
        encoding="utf-8",
    )

    overview = build_project_overview(
        repo_path=str(repository),
        repo_id="project-a",
        vector_store=MagicMock(),
    )

    assert "Processes invoices for customers." in overview.answer
    assert "FastAPI" in overview.answer
    assert "SQLAlchemy" in overview.answer
    assert "/invoices" in overview.answer
    assert "Framework could not be determined" not in overview.answer
    assert "app/services" not in overview.answer
    assert overview.confidence >= 0.85
    assert all(
        (repository / citation.rsplit(":", 1)[0]).is_file()
        for citation in overview.citations
    )


def test_project_overview_detects_manifest_metadata_and_web_frameworks(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "web-project"
    source = repository / "src"
    source.mkdir(parents=True)
    manifest = repository / "package.json"
    manifest.write_text(
        '{\n'
        '  "description": "Tracks inventory changes for a small shop.",\n'
        '  "dependencies": {\n'
        '    "@angular/core": "17.0.0",\n'
        '    "next": "14.0.0",\n'
        '    "vue": "3.0.0"\n'
        '  }\n'
        '}\n',
        encoding="utf-8",
    )
    (source / "App.vue").write_text(
        "<template><main>Inventory</main></template>\n",
        encoding="utf-8",
    )
    (source / "main.ts").write_text(
        "import { createApp } from 'vue'\n"
        "createApp({}).mount('#app')\n",
        encoding="utf-8",
    )

    overview = build_project_overview(
        repo_path=str(repository),
        repo_id="web-project",
        vector_store=MagicMock(),
    )

    assert "Tracks inventory changes for a small shop." in overview.answer
    assert "Angular" in overview.answer
    assert "Next.js" in overview.answer
    assert "Vue" in overview.answer
    assert "Vue (" in overview.answer
    assert "package.json" in overview.answer
    assert all(
        (repository / citation.rsplit(":", 1)[0]).is_file()
        for citation in overview.citations
    )


def test_project_overview_ignores_tests_backups_and_generated_data(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "chroma_cache").mkdir()
    (repository / "uploaded_repositories" / "old").mkdir(parents=True)
    (repository / "README.md").write_text(
        "# Sample project\n\nA small service.\n",
        encoding="utf-8",
    )
    (repository / "src" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n",
        encoding="utf-8",
    )
    (repository / "src" / "framework_map.ts").write_text(
        'const languages = { "Vue": ".vue" };\n',
        encoding="utf-8",
    )
    (repository / "src" / "patterns.py").write_text(
        'FRAMEWORK_PATTERNS = {"Stripe": re.compile(r"stripe")}\n',
        encoding="utf-8",
    )
    (repository / "src" / "payments_backup.py").write_text(
        "import stripe\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_fixtures.ts").write_text(
        "import { Component } from '@angular/core';\n"
        "@app.post('/test-route')\n",
        encoding="utf-8",
    )
    (repository / "chroma_cache" / "tokenizer.json").write_text(
        '{"tokens": ["@nestjs/core"]}\n',
        encoding="utf-8",
    )
    (repository / "uploaded_repositories" / "old" / "app.py").write_text(
        "from django import test_application\n",
        encoding="utf-8",
    )

    overview = build_project_overview(
        repo_path=str(repository),
        repo_id="project-a",
        vector_store=MagicMock(),
    )

    assert "FastAPI" in overview.answer
    assert "Vue" not in overview.answer
    assert "NestJS" not in overview.answer
    assert "Angular" not in overview.answer
    assert "Django" not in overview.answer
    assert "Stripe" not in overview.answer
    assert "/test-route" not in overview.answer
    assert "uploaded_repositories" not in overview.answer
    assert "chroma_cache" not in overview.answer


def test_project_overview_does_not_cite_empty_source_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "empty-project"
    source = repository / "main.py"
    repository.mkdir()
    source.write_text("", encoding="utf-8")

    overview = build_project_overview(
        repo_path=str(repository),
        repo_id="empty-project",
        vector_store=MagicMock(),
    )

    assert overview.confidence == 0.1
    assert overview.citations == []
    assert (
        "I could not determine this confidently from the uploaded files."
        in overview.answer
    )
    assert "no non-empty, readable" in overview.answer
