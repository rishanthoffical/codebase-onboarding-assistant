from pathlib import Path
from unittest.mock import MagicMock, patch

from app.indexer.chunker import language_for_path
from app.indexer.indexer import RepositoryIndexer
from app.indexer.diff import FileChange


def make_indexer(tmp_path: Path) -> RepositoryIndexer:
    repository = tmp_path / "repository"
    repository.mkdir()

    indexer = RepositoryIndexer(
        str(repository),
        repo_id="test-repo",
    )

    indexer.vector_store = MagicMock()

    return indexer


def test_get_files_skips_runtime_and_unsupported_files(tmp_path):
    indexer = make_indexer(tmp_path)
    repository = indexer.repo_path

    (repository / "README.md").write_text(
        "# Project\n",
        encoding="utf-8",
    )
    (repository / "settings.xml").write_text(
        "<settings />\n",
        encoding="utf-8",
    )
    (repository / "agent").mkdir()
    (repository / "agent" / "loop.py").write_text(
        "print('source')\n",
        encoding="utf-8",
    )
    (repository / "agent" / "loop.py.backup").write_text(
        "print('backup')\n",
        encoding="utf-8",
    )
    (repository / "agent" / "extract_function.tmp").write_text(
        "temporary output\n",
        encoding="utf-8",
    )
    (repository / "celerybeat-schedule").write_text(
        "runtime schedule data\n",
        encoding="utf-8",
    )
    (repository / ".env.example").write_text(
        "API_KEY=example\n",
        encoding="utf-8",
    )

    assert {
        file.relative_to(repository).as_posix()
        for file in indexer.get_files()
    } == {
        "README.md",
        "settings.xml",
        "agent/loop.py",
    }


def test_index_redacts_detected_secrets_before_storing_vectors(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "secret-project"
    repository.mkdir()
    (repository / "settings.py").write_text(
        'API_KEY = "0123456789abcdef"\n',
        encoding="utf-8",
    )
    vector_store = MagicMock()
    indexer = RepositoryIndexer(
        str(repository),
        repo_id="secret-project",
        vector_store=vector_store,
    )

    with patch.object(
        indexer,
        "get_current_commit",
        return_value=None,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value=None,
    ):
        result = indexer.index()

    stored_documents = [
        document
        for call in vector_store.add_documents.call_args_list
        for document in call.kwargs["documents"]
    ]
    assert result["vectors_created"] == 1
    assert stored_documents
    assert all("0123456789abcdef" not in text for text in stored_documents)
    assert all("[REDACTED]" in text for text in stored_documents)


def test_full_index_writes_vectors_in_bounded_batches(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "large-project"
    repository.mkdir()
    for index in range(101):
        (repository / f"module_{index}.py").write_text(
            "pass\n",
            encoding="utf-8",
        )

    vector_store = MagicMock()
    indexer = RepositoryIndexer(
        str(repository),
        repo_id="large-project",
        vector_store=vector_store,
    )
    chunk = {
        "content": "pass",
        "start_line": 1,
        "end_line": 1,
    }

    with patch(
        "app.indexer.indexer.chunk_file",
        return_value=[chunk],
    ), patch.object(
        indexer,
        "get_current_commit",
        return_value=None,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value=None,
    ):
        result = indexer.index()

    assert result["files_indexed"] == 101
    assert result["chunks_created"] == 101
    assert vector_store.add_documents.call_count == 2
    assert [
        len(call.kwargs["documents"])
        for call in vector_store.add_documents.call_args_list
    ] == [100, 1]


def test_generic_languages_are_indexed_with_repository_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "generic-project"
    repository.mkdir()
    (repository / "Payment.cs").write_text(
        "public class Payment {}\n",
        encoding="utf-8",
    )
    (repository / "payment.java").write_text(
        "class Payment {}\n",
        encoding="utf-8",
    )
    (repository / "runtime.tmp").write_text(
        "not source\n",
        encoding="utf-8",
    )

    vector_store = MagicMock()
    indexer = RepositoryIndexer(
        str(repository),
        repo_id="project-b",
        vector_store=vector_store,
    )
    chunk = {
        "content": "class Payment {}",
        "start_line": 1,
        "end_line": 1,
    }

    with patch(
        "app.indexer.indexer.chunk_file",
        return_value=[chunk],
    ), patch.object(
        indexer,
        "get_current_commit",
        return_value=None,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value=None,
    ):
        result = indexer.index()

    assert result["files_discovered"] == 3
    assert result["files_indexed"] == 2
    assert result["files_skipped"] == 1
    assert result["chunks_created"] == 2
    assert result["vectors_created"] == 2

    stored_metadata = vector_store.add_documents.call_args.kwargs[
        "metadatas"
    ]
    assert {
        (metadata["repo_id"], metadata["language"])
        for metadata in stored_metadata
    } == {
        ("project-b", "C#"),
        ("project-b", "Java"),
    }
    assert all("chunk_index" in metadata for metadata in stored_metadata)


def test_vue_single_file_components_are_supported() -> None:
    assert language_for_path(Path("src/App.vue")) == "Vue"


def test_requested_project_languages_have_extension_mappings() -> None:
    expected = {
        "app.py": "Python",
        "Main.java": "Java",
        "app.js": "JavaScript",
        "app.ts": "TypeScript",
        "index.html": "HTML",
        "site.css": "CSS",
        "main.c": "C",
        "main.cpp": "C++",
        "Main.cs": "C#",
        "main.go": "Go",
        "main.rs": "Rust",
        "index.php": "PHP",
        "app.rb": "Ruby",
        "Main.kt": "Kotlin",
        "App.swift": "Swift",
        "schema.sql": "SQL",
        "config.xml": "XML",
        "config.yaml": "YAML",
        "config.json": "JSON",
        "README.md": "Markdown",
    }

    assert {
        name: language_for_path(Path(name))
        for name in expected
    } == expected


def mock_chunk_file(*args, **kwargs):
    return [
        {
            "content": "def hello():\n    return 'hello'",
            "start_line": 1,
            "end_line": 2,
        }
    ]


def test_incremental_new_file(tmp_path):
    indexer = make_indexer(tmp_path)

    new_file = indexer.repo_path / "new_file.py"
    new_file.write_text(
        "def hello():\n    return 'hello'\n",
        encoding="utf-8",
    )

    changes = [
        FileChange(
            file_path="new_file.py",
            status="NEW",
        )
    ]

    with patch(
        "app.indexer.indexer.get_file_changes",
        return_value=changes,
    ), patch(
        "app.indexer.indexer.chunk_file",
        side_effect=mock_chunk_file,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value="commit-new",
    ):
        result = indexer.incremental_index(
            base_commit="base",
            head_commit="head",
        )

    assert result["new_files"] == 1
    assert result["modified_files"] == 0
    assert result["deleted_files"] == 0
    assert result["renamed_files"] == 0
    assert result["copied_files"] == 0
    assert result["files_processed"] == 1
    assert result["chunks_added"] == 1

    indexer.vector_store.add_documents.assert_called_once()


def test_incremental_modified_file(tmp_path):
    indexer = make_indexer(tmp_path)

    modified_file = indexer.repo_path / "existing.py"
    modified_file.write_text(
        "def updated():\n    return 'updated'\n",
        encoding="utf-8",
    )

    changes = [
        FileChange(
            file_path="existing.py",
            status="MODIFIED",
        )
    ]

    with patch(
        "app.indexer.indexer.get_file_changes",
        return_value=changes,
    ), patch(
        "app.indexer.indexer.chunk_file",
        side_effect=mock_chunk_file,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value="commit-modified",
    ):
        result = indexer.incremental_index(
            base_commit="base",
            head_commit="head",
        )

    assert result["new_files"] == 0
    assert result["modified_files"] == 1
    assert result["deleted_files"] == 0
    assert result["renamed_files"] == 0
    assert result["copied_files"] == 0
    assert result["files_processed"] == 1
    assert result["chunks_added"] == 1

    indexer.vector_store.delete_by_file.assert_called_once_with(
        file_path="existing.py",
        repo_id="test-repo",
    )

    indexer.vector_store.add_documents.assert_called_once()


def test_incremental_deleted_file(tmp_path):
    indexer = make_indexer(tmp_path)

    changes = [
        FileChange(
            file_path="deleted.py",
            status="DELETED",
        )
    ]

    with patch(
        "app.indexer.indexer.get_file_changes",
        return_value=changes,
    ):
        result = indexer.incremental_index(
            base_commit="base",
            head_commit="head",
        )

    assert result["new_files"] == 0
    assert result["modified_files"] == 0
    assert result["deleted_files"] == 1
    assert result["renamed_files"] == 0
    assert result["copied_files"] == 0
    assert result["files_processed"] == 0

    indexer.vector_store.delete_by_file.assert_called_once_with(
        file_path="deleted.py",
        repo_id="test-repo",
    )

    indexer.vector_store.add_documents.assert_not_called()


def test_incremental_renamed_file(tmp_path):
    indexer = make_indexer(tmp_path)

    renamed_file = indexer.repo_path / "new_name.py"
    renamed_file.write_text(
        "def renamed():\n    return 'renamed'\n",
        encoding="utf-8",
    )

    changes = [
        FileChange(
            file_path="new_name.py",
            status="RENAMED",
            old_path="old_name.py",
        )
    ]

    with patch(
        "app.indexer.indexer.get_file_changes",
        return_value=changes,
    ), patch(
        "app.indexer.indexer.chunk_file",
        side_effect=mock_chunk_file,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value="commit-renamed",
    ):
        result = indexer.incremental_index(
            base_commit="base",
            head_commit="head",
        )

    assert result["new_files"] == 0
    assert result["modified_files"] == 0
    assert result["deleted_files"] == 0
    assert result["renamed_files"] == 1
    assert result["copied_files"] == 0
    assert result["files_processed"] == 1
    assert result["chunks_added"] == 1

    delete_calls = (
        indexer.vector_store
        .delete_by_file
        .call_args_list
    )

    assert delete_calls[0].kwargs == {
        "file_path": "old_name.py",
        "repo_id": "test-repo",
    }

    assert delete_calls[1].kwargs == {
        "file_path": "new_name.py",
        "repo_id": "test-repo",
    }

    indexer.vector_store.add_documents.assert_called_once()


def test_incremental_copied_file(tmp_path):
    indexer = make_indexer(tmp_path)

    copied_file = indexer.repo_path / "copy.py"
    copied_file.write_text(
        "def copied():\n    return 'copied'\n",
        encoding="utf-8",
    )

    changes = [
        FileChange(
            file_path="copy.py",
            status="COPIED",
            old_path="original.py",
        )
    ]

    with patch(
        "app.indexer.indexer.get_file_changes",
        return_value=changes,
    ), patch(
        "app.indexer.indexer.chunk_file",
        side_effect=mock_chunk_file,
    ), patch.object(
        indexer,
        "get_last_modified_commit",
        return_value="commit-copied",
    ):
        result = indexer.incremental_index(
            base_commit="base",
            head_commit="head",
        )

    assert result["new_files"] == 0
    assert result["modified_files"] == 0
    assert result["deleted_files"] == 0
    assert result["renamed_files"] == 0
    assert result["copied_files"] == 1
    assert result["files_processed"] == 1
    assert result["chunks_added"] == 1

    # COPIED must not delete the original file's vectors.
    indexer.vector_store.delete_by_file.assert_not_called()

    indexer.vector_store.add_documents.assert_called_once()
