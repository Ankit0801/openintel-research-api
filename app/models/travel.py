"""Structured data contracts for OpenIntel research workflows."""

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """A user's research question and optional investigation constraints."""

    question: str = Field(
        min_length=10,
        description="The question OpenIntel should investigate.",
    )
    preferred_sources: list[str] = Field(
        default_factory=list,
        description="Optional sources the user wants the system to prioritize.",
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of source categories to investigate.",
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Reject blank questions after removing incidental whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("Research question cannot be blank.")

        return normalized

    @field_validator("preferred_sources")
    @classmethod
    def normalize_sources(cls, values: list[str]) -> list[str]:
        """Normalize source names and remove duplicates."""

        normalized = [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

        return list(dict.fromkeys(normalized))


class ResearchPlan(BaseModel):
    """Structured investigation plan produced by the planning agent."""

    research_objective: str = Field(
        description="What the investigation is trying to establish.",
    )
    key_questions: list[str] = Field(
        description="Specific questions that should be answered.",
    )
    recommended_sources: list[str] = Field(
        description="Source categories that should be consulted.",
    )
    search_strategy: list[str] = Field(
        description="High-level steps for gathering evidence.",
    )


class Evidence(BaseModel):
    """A normalized and scored piece of evidence."""

    source: str = Field(
        description="Name of the source that provided the evidence.",
    )

    source_type: str = Field(
        default="unknown",
        description="Category of evidence, such as repository, paper, or vulnerability.",
    )

    title: str = Field(
        description="Title or identifying label of the source item.",
    )

    url: str | None = Field(
        default=None,
        description="URL where the evidence can be inspected.",
    )

    content: str = Field(
        description="Relevant evidence extracted from the source.",
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Source-specific metadata.",
    )

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic relevance score for the research question.",
    )

    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic quality score for the evidence.",
    )

    recency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic recency score for the evidence.",
    )

    combined_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Combined deterministic score used for final evidence ranking.",
    )

    evidence_id: str | None = Field(
        default=None,
        description="Stable identifier for deduplication and provenance.",
    )
class ResearchReport(BaseModel):
    """Final evidence-backed research result."""

    answer: str = Field(
        description="Synthesized answer to the original research question.",
    )
    key_findings: list[str] = Field(
        description="Most important findings from the investigation.",
    )
    limitations: list[str] = Field(
        description="Important uncertainty, missing data, or limitations.",
    )
    evidence: list[Evidence] = Field(
        description="Evidence supporting the report.",
    )