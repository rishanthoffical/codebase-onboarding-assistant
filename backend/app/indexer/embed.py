from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """
    Local embedding provider.

    Default model:
        sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name
        )

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        return [
            embedding.tolist()
            for embedding in embeddings
        ]


class Embedder(EmbeddingProvider):
    """
    Backward-compatible alias for the existing Embedder API.
    """

    pass


def embed_chunks(
    chunks: list[dict[str, Any]],
    provider: EmbeddingProvider | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for indexer chunks.

    Supports the current chunker output:
        chunk["content"]

    and the older format:
        chunk["text"]
    """

    if not chunks:
        return []

    provider = provider or EmbeddingProvider()

    texts: list[str] = []

    for chunk in chunks:
        text = chunk.get(
            "content",
            chunk.get("text", ""),
        )

        texts.append(
            str(text)
        )

    return provider.embed(texts)


def embed_and_upsert(
    chunks: list[dict[str, Any]],
    vector_store: Any,
    provider: EmbeddingProvider | None = None,
) -> None:
    """
    Backward-compatible embedding helper.

    The active RepositoryIndexer does not need this function because
    Chroma handles document embedding through the configured collection.

    This helper is retained so older callers do not fail.
    """

    if not chunks:
        return

    provider = provider or EmbeddingProvider()

    embeddings = embed_chunks(
        chunks,
        provider=provider,
    )

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        chunk_id = chunk.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        metadata = {
            "file_path": chunk.get(
                "file_path",
                chunk.get("file", ""),
            ),
            "line_start": int(
                chunk.get(
                    "line_start",
                    chunk.get("start_line", 1),
                )
            ),
            "line_end": int(
                chunk.get(
                    "line_end",
                    chunk.get("end_line", 1),
                )
            ),
            "last_modified_commit": chunk.get(
                "last_modified_commit",
                "",
            ),
            "doc_type": chunk.get(
                "doc_type",
                "code",
            ),
            "repo_id": chunk.get(
                "repo_id",
                "local",
            ),
            "chunk_id": chunk_id,
        }

        # Support a custom vector store that exposes upsert().
        upsert = getattr(
            vector_store,
            "upsert",
            None,
        )

        if upsert is None:
            raise AttributeError(
                "The supplied vector store does not "
                "provide an upsert() method."
            )

        try:
            upsert(
                chunk_id=chunk_id,
                vector=embedding,
                metadata=metadata,
            )
        except TypeError:
            # Compatibility with the previous in-memory
            # store implementation.
            upsert(
                chunk_id,
                metadata,
            )