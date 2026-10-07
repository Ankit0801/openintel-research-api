"""ChromaDB abstraction for local vector persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class ChromaVectorStore:
    """Application-level abstraction over a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path = ".data/chroma",
        collection_name: str = "openintel_evidence_gemini_v2",
    ) -> None:
        self.persist_directory = Path(
            persist_directory
        )

        self.collection_name = collection_name

        self._client: (
            chromadb.PersistentClient | None
        ) = None

        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        """Lazily initialize the Chroma collection."""

        if self._collection is None:
            self.persist_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._client = chromadb.PersistentClient(
                path=str(
                    self.persist_directory
                )
            )

            self._collection = (
                self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine"
                    },
                )
            )

        return self._collection

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or replace chunks by deterministic ID."""

        if not (
            ids
            and documents
            and embeddings
            and metadatas
        ):
            raise ValueError(
                "ids, documents, embeddings, "
                "and metadatas are required."
            )

        lengths = {
            len(ids),
            len(documents),
            len(embeddings),
            len(metadatas),
        }

        if len(lengths) != 1:
            raise ValueError(
                "All vector-store inputs "
                "must have the same length."
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        *,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Return chunks ordered by cosine distance."""

        if n_results <= 0:
            raise ValueError(
                "n_results must be greater than zero."
            )

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result.get(
            "documents",
            [[]],
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]],
        )[0]

        distances = result.get(
            "distances",
            [[]],
        )[0]

        ids = result.get(
            "ids",
            [[]],
        )[0]

        rows: list[dict[str, Any]] = []

        for (
            chunk_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "content": document,
                    "metadata": metadata or {},
                    "distance": float(distance),
                    "similarity": max(
                        0.0,
                        min(
                            1.0,
                            1.0 - float(distance),
                        ),
                    ),
                }
            )

        return rows

    def count(self) -> int:
        """Return number of stored chunks."""

        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the collection."""

        if self._client is None:
            return

        self._client.delete_collection(
            self.collection_name
        )

        self._collection = None