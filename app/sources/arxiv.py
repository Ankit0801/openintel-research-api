"""arXiv API client for OpenIntel."""

from xml.etree import ElementTree

import httpx

from app.models import Evidence


class ArxivClient:
    """Client for retrieving public arXiv paper metadata."""

    BASE_URL = "https://export.arxiv.org/api/query"

    ATOM_NAMESPACE = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def __init__(
        self,
        timeout: float = 20.0,
    ) -> None:
        """Initialize the arXiv client."""

        self.timeout = timeout

        self.headers = {
            "User-Agent": "OpenIntel/0.1",
        }

    def search_papers(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Evidence]:
        """Search arXiv title and abstract fields."""

        if not query.strip():
            raise ValueError(
                "arXiv search query cannot be empty."
            )

        if not 1 <= max_results <= 20:
            raise ValueError(
                "max_results must be between 1 and 20."
            )

        terms = [
            term.strip().strip('"')
            for term in query.split(" OR ")
            if term.strip()
        ]

        search_parts = []

        for term in terms:
            escaped = term.replace('"', '\\"')

            search_parts.append(
                f'ti:"{escaped}" OR abs:"{escaped}"'
            )

        search_query = " OR ".join(
            f"({part})"
            for part in search_parts
        )

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = httpx.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
        except httpx.RequestError as error:
            raise RuntimeError(
                f"arXiv request failed: {error}"
            ) from error

        response.raise_for_status()

        return self._parse_response(
            response.text
        )

    def _parse_response(
        self,
        xml_text: str,
    ) -> list[Evidence]:
        """Parse an arXiv Atom response into Evidence objects."""

        try:
            root = ElementTree.fromstring(
                xml_text
            )
        except ElementTree.ParseError as error:
            raise RuntimeError(
                "arXiv returned invalid XML."
            ) from error

        evidence: list[Evidence] = []

        for entry in root.findall(
            "atom:entry",
            self.ATOM_NAMESPACE,
        ):
            parsed = self._entry_to_evidence(
                entry
            )

            if parsed is not None:
                evidence.append(parsed)

        return evidence

    def _entry_to_evidence(
        self,
        entry: ElementTree.Element,
    ) -> Evidence | None:
        """Convert one arXiv Atom entry into Evidence."""

        title = self._text(
            entry,
            "atom:title",
        )

        summary = self._text(
            entry,
            "atom:summary",
        )

        published = self._text(
            entry,
            "atom:published",
        )

        updated = self._text(
            entry,
            "atom:updated",
        )

        paper_id = self._text(
            entry,
            "atom:id",
        )

        if not title or not paper_id:
            return None

        authors: list[str] = []

        for author in entry.findall(
            "atom:author",
            self.ATOM_NAMESPACE,
        ):
            name = self._text(
                author,
                "atom:name",
            )

            if name:
                authors.append(name)

        categories: list[str] = []

        for category in entry.findall(
            "atom:category",
            self.ATOM_NAMESPACE,
        ):
            term = category.attrib.get(
                "term"
            )

            if term:
                categories.append(term)

        content = (
            f"Title: {title}\n"
            f"Authors: "
            f"{', '.join(authors) or 'Unknown'}\n"
            f"Published: "
            f"{published or 'Unknown'}\n"
            f"Updated: "
            f"{updated or 'Unknown'}\n"
            f"Categories: "
            f"{', '.join(categories) or 'Unknown'}\n"
            f"Abstract: "
            f"{summary or 'No abstract provided.'}"
        )

        return Evidence(
            source="arxiv",
            source_type="paper",
            title=title,
            url=paper_id,
            content=content,
            metadata={
                "authors": authors,
                "published": published,
                "updated": updated,
                "categories": categories,
                "has_abstract": bool(summary),
            },
        )

    @staticmethod
    def _text(
        element: ElementTree.Element,
        path: str,
    ) -> str:
        """Extract and normalize text from an XML element."""

        child = element.find(
            path,
            ArxivClient.ATOM_NAMESPACE,
        )

        if child is None or child.text is None:
            return ""

        return " ".join(
            child.text.split()
        )