"""Local sentence-transformers embedding backend."""

from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class LocalEmbeddingModel:
    """Lazy, reusable local embedding model.

    The model is loaded only on the first embedding request
    and then reused.

    No external LLM/API call is involved.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> SentenceTransformer:
        """Return the cached model, loading it lazily."""

        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name
            )

        return self._model

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        if not texts:
            return []

        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a query."""

        if not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vector.tolist()