"""Semantic retrieval over normalized research evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models import Evidence
from app.rag.chunker import EvidenceChunker
from app.rag.embeddings import GeminiEmbeddingModel
from app.rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A semantically retrieved chunk with provenance."""

    chunk_id: str
    evidence_id: str
    content: str
    similarity_score: float
    source: str
    source_type: str
    title: str
    url: str | None
    metadata: dict[str, object]


class RAGRetriever:
    """Index Evidence and retrieve semantic chunks."""

    def __init__(
        self,
        *,
        embedding_model: GeminiEmbeddingModel | None = None,
        vector_store: ChromaVectorStore | None = None,
        chunker: EvidenceChunker | None = None,
        persist_directory: str | Path = ".data/chroma",
        collection_name: str = "openintel_evidence",
    ) -> None:
        self.embedding_model = (
            embedding_model
            or GeminiEmbeddingModel()
        )

        self.vector_store = (
            vector_store
            or ChromaVectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        )

        self.chunker = (
            chunker
            or EvidenceChunker()
        )

    def index_evidence(
        self,
        evidence: list[Evidence],
    ) -> int:
        """Chunk, embed and persist evidence."""

        chunks = [
            chunk
            for item in evidence
            for chunk in self.chunker.chunk(item)
        ]

        if not chunks:
            return 0

        embeddings = (
            self.embedding_model.embed_documents(
                [
                    chunk.content
                    for chunk in chunks
                ]
            )
        )

        self.vector_store.add(
            ids=[
                chunk.chunk_id
                for chunk in chunks
            ],
            documents=[
                chunk.content
                for chunk in chunks
            ],
            embeddings=embeddings,
            metadatas=[
                self._metadata(chunk)
                for chunk in chunks
            ],
        )

        return len(chunks)

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most semantically relevant chunks."""

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        rows = self.vector_store.query(
            embedding=query_embedding,
            n_results=k,
        )

        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                evidence_id=str(
                    row["metadata"].get(
                        "evidence_id",
                        "",
                    )
                ),
                content=row["content"],
                similarity_score=row[
                    "similarity"
                ],
                source=str(
                    row["metadata"].get(
                        "source",
                        "",
                    )
                ),
                source_type=str(
                    row["metadata"].get(
                        "source_type",
                        "",
                    )
                ),
                title=str(
                    row["metadata"].get(
                        "title",
                        "",
                    )
                ),
                url=(
                    row["metadata"].get(
                        "url"
                    )
                    or None
                ),
                metadata=self._restore_metadata(
                    row["metadata"]
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _metadata(
        chunk: Any,
    ) -> dict[str, Any]:
        """Convert chunk provenance to Chroma-compatible metadata."""

        metadata: dict[str, Any] = {
            "evidence_id": chunk.evidence_id,
            "source": chunk.source,
            "source_type": chunk.source_type,
            "title": chunk.title,
            "url": chunk.url or "",
            "chunk_id": chunk.chunk_id,
        }

        metadata[
            "source_metadata_json"
        ] = json.dumps(
            chunk.metadata,
            sort_keys=True,
            default=str,
        )

        return metadata

    @staticmethod
    def _restore_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, object]:
        """Restore original Evidence metadata."""

        raw = metadata.get(
            "source_metadata_json",
            "{}",
        )

        try:
            restored = json.loads(raw)
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            restored = {}

        if not isinstance(
            restored,
            dict,
        ):
            return {}

        return restored