"""OpenAlex API client for OpenIntel."""

from typing import Any

import httpx

from app.models import Evidence


class OpenAlexClient:
    """Client for retrieving scholarly works from OpenAlex."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        """Initialize the OpenAlex client."""

        self.timeout = timeout
        self.api_key = api_key

        self.headers = {
            "User-Agent": "OpenIntel/0.1",
        }

    def search_works(
        self,
        query: str,
        per_page: int = 5,
    ) -> list[Evidence]:
        """Search OpenAlex works and normalize them into Evidence."""

        if not query.strip():
            raise ValueError(
                "OpenAlex search query cannot be empty."
            )

        if not 1 <= per_page <= 25:
            raise ValueError(
                "per_page must be between 1 and 25."
            )

        params: dict[str, Any] = {
            "search": query.strip(),
            "per-page": per_page,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = httpx.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
        except httpx.RequestError as error:
            raise RuntimeError(
                f"OpenAlex request failed: {error}"
            ) from error

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected OpenAlex response format."
            )

        results = data.get("results", [])

        if not isinstance(results, list):
            raise RuntimeError(
                "OpenAlex results field has an unexpected format."
            )

        return [
            self._work_to_evidence(work)
            for work in results
            if isinstance(work, dict)
        ]

    @staticmethod
    def _work_to_evidence(
        work: dict[str, Any],
    ) -> Evidence:
        """Convert an OpenAlex work into normalized Evidence."""

        title = (
            work.get("display_name")
            or work.get("title")
            or "Untitled work"
        )

        url = work.get("doi")

        if not url:
            url = work.get("id")

        publication_date = work.get(
            "publication_date",
            "Unknown",
        )

        cited_by_count = work.get(
            "cited_by_count",
            0,
        )

        publication_year = work.get(
            "publication_year",
            "Unknown",
        )

        authorships = work.get(
            "authorships",
            [],
        )

        authors: list[str] = []

        if isinstance(authorships, list):
            for authorship in authorships:
                if not isinstance(authorship, dict):
                    continue

                author = authorship.get("author")

                if not isinstance(author, dict):
                    continue

                display_name = author.get("display_name")

                if display_name:
                    authors.append(display_name)

        concepts = work.get(
            "concepts",
            [],
        )

        concept_names: list[str] = []

        if isinstance(concepts, list):
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue

                name = concept.get("display_name")

                if name:
                    concept_names.append(name)

        abstract = OpenAlexClient._reconstruct_abstract(
            work.get("abstract_inverted_index")
        )

        content = (
            f"Title: {title}\n"
            f"Authors: {', '.join(authors) or 'Unknown'}\n"
            f"Publication date: {publication_date}\n"
            f"Publication year: {publication_year}\n"
            f"Citations: {cited_by_count}\n"
            f"Concepts: "
            f"{', '.join(concept_names) or 'Unknown'}\n"
            f"Abstract: "
            f"{abstract or 'No abstract available.'}"
        )

        return Evidence(
            source="openalex",
            source_type="academic_work",
            title=title,
            url=url,
            content=content,
            metadata={
                "authors": authors,
                "publication_date": publication_date,
                "publication_year": publication_year,
                "cited_by_count": cited_by_count,
                "concepts": concept_names,
            },
        )

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: Any,
    ) -> str:
        """Reconstruct an abstract from OpenAlex inverted-index data."""

        if not isinstance(inverted_index, dict):
            return ""

        words: list[tuple[int, str]] = []

        for word, positions in inverted_index.items():
            if not isinstance(word, str):
                continue

            if not isinstance(positions, list):
                continue

            for position in positions:
                if isinstance(position, int):
                    words.append((position, word))

        words.sort(key=lambda item: item[0])

        return " ".join(
            word
            for _, word in words
        )