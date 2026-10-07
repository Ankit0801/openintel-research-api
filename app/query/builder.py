"""Deterministic source-specific query construction for OpenIntel."""

from dataclasses import dataclass

from app.models import ResearchRequest


@dataclass(frozen=True)
class SourceQueries:
    """Queries generated for each supported research source."""

    github: str
    arxiv: str
    openalex: str
    nvd: str


class QueryBuilder:
    """Build deterministic queries from a research request."""

    def build(self, request: ResearchRequest) -> SourceQueries:
        """Build source-specific queries."""

        question = request.question.strip()

        return SourceQueries(
            github=self._github_query(question),
            arxiv=self._arxiv_query(question),
            openalex=self._openalex_query(question),
            nvd=self._nvd_query(question),
        )

    @staticmethod
    def _github_query(question: str) -> str:
        """Build a GitHub repository search query."""

        return question

    @staticmethod
    def _arxiv_query(question: str) -> str:
        """Build an arXiv research-oriented query."""

        normalized = question.lower()

        if "open source" in normalized and "ai agent" in normalized:
            return (
                '"AI agents" OR '
                '"agentic AI" OR '
                '"autonomous agents"'
            )

        if "ai agent" in normalized:
            return (
                '"AI agents" OR '
                '"agentic AI" OR '
                '"autonomous agents"'
            )

        return question

    @staticmethod
    def _openalex_query(question: str) -> str:
        """Build an OpenAlex academic search query."""

        normalized = question.lower()

        if "open source" in normalized and "ai agent" in normalized:
            return (
                "agentic artificial intelligence"
            )

        if "ai agent" in normalized:
            return (
                "agentic artificial intelligence"
            )

        return question

    @staticmethod
    def _nvd_query(question: str) -> str:
        """Build a security-oriented query."""

        normalized = question.lower()

        if "ai agent" in normalized:
            return "AI agent"

        return question