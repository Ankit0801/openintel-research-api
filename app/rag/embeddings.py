"""Gemini embedding backend."""

from __future__ import annotations

from collections.abc import Sequence

from google import genai
from google.genai import types

from app.config import get_settings


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class GeminiEmbeddingModel:
    """Gemini API embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        client=None,
    ) -> None:
        settings = get_settings()

        self.model_name = model_name
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        if not texts:
            return []

        response = self._client.models.embed_content(
            model=self.model_name,
            contents=list(texts),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )

        return [
            list(embedding.values)
            for embedding in response.embeddings
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a search query."""

        if not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        response = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )

        return list(response.embeddings[0].values)