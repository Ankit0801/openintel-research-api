"""Tests for deterministic evidence intelligence."""

from app.evidence.processor import EvidenceProcessor
from app.models import Evidence


def make_evidence(
    *,
    source: str = "github",
    title: str = "Example AI Agent",
    content: str = "An open-source AI agent project.",
    metadata: dict[str, object] | None = None,
) -> Evidence:
    """Create deterministic test evidence."""

    return Evidence(
        source=source,
        source_type="repository",
        title=title,
        url=f"https://example.com/{title.replace(' ', '-').lower()}",
        content=content,
        metadata=metadata or {},
    )


def test_ai_agent_context_beats_generic_agent() -> None:
    """AI-agent evidence should outrank unrelated generic agents."""

    processor = EvidenceProcessor()

    relevant = make_evidence(
        title="Open Source AI Agent Framework",
        content=(
            "An open-source framework for building "
            "autonomous AI agents."
        ),
    )

    unrelated = make_evidence(
        title="SNMP Agent Security Tool",
        content=(
            "A network management agent used for "
            "SNMP infrastructure."
        ),
    )

    results = processor.process(
        evidence=[unrelated, relevant],
        question="How is the open-source AI agent ecosystem evolving?",
        max_results=2,
    )

    assert results[0].title == "Open Source AI Agent Framework"
    assert results[0].relevance_score > results[1].relevance_score


def test_nvd_generic_agent_is_heavily_penalized() -> None:
    """NVD entries about generic software agents should rank very low."""

    processor = EvidenceProcessor()

    evidence = make_evidence(
        source="nvd",
        title="CVE-2001-0352",
        content=(
            "SNMP agents in a network access point allow "
            "remote attackers to obtain encryption keys."
        ),
        metadata={
            "cvss_score": None,
            "references": [
                "https://example.com/reference"
            ],
            "published": "2001-07-21T04:00:00.000",
        },
    )

    results = processor.process(
        evidence=[evidence],
        question="How is the open-source AI agent ecosystem evolving?",
    )

    assert len(results) == 1
    assert results[0].relevance_score < 0.10


def test_gemini_cli_is_relevant_to_ai_agent_question() -> None:
    """A repository explicitly described as an AI agent should match."""

    processor = EvidenceProcessor()

    evidence = make_evidence(
        title="google-gemini/gemini-cli",
        content=(
            "An open-source AI agent that brings "
            "the power of Gemini directly into your terminal."
        ),
        metadata={
            "stars": 100000,
            "forks": 10000,
            "updated_at": "2026-10-05T20:22:57Z",
        },
    )

    results = processor.process(
        evidence=[evidence],
        question="How is the open-source AI agent ecosystem evolving?",
    )

    assert results[0].relevance_score > 0.50
    assert results[0].combined_score > 0.50


def test_ragflow_is_relevant_even_without_exact_ai_agent_phrase() -> None:
    """Agentic/RAG projects should receive meaningful relevance."""

    processor = EvidenceProcessor()

    evidence = make_evidence(
        title="infiniflow/ragflow",
        content=(
            "RAGFlow is an open-source Retrieval-Augmented "
            "Generation engine that fuses RAG with Agent "
            "capabilities to create a context layer for LLMs."
        ),
        metadata={
            "stars": 90000,
            "forks": 10000,
            "updated_at": "2026-10-05T20:12:08Z",
        },
    )

    results = processor.process(
        evidence=[evidence],
        question="How is the open-source AI agent ecosystem evolving?",
    )

    assert results[0].relevance_score > 0.30


def test_agentic_artificial_intelligence_is_recognized() -> None:
    """Academic evidence using the expanded agentic terminology should match."""

    processor = EvidenceProcessor()

    evidence = make_evidence(
        source="openalex",
        title="Agentic Artificial Intelligence",
        content=(
            "Agentic artificial intelligence enables "
            "autonomous systems capable of reasoning and acting."
        ),
        metadata={
            "publication_date": "2025-04-28",
            "publication_year": 2025,
            "cited_by_count": 46,
        },
    )

    results = processor.process(
        evidence=[evidence],
        question="How is the open-source AI agent ecosystem evolving?",
    )

    assert results[0].relevance_score > 0.20


def test_combined_score_is_deterministic() -> None:
    """Combined score should use the documented weighted formula."""

    processor = EvidenceProcessor()

    evidence = make_evidence(
        content="Open-source AI agent framework.",
        metadata={
            "updated_at": "2026-10-05T20:00:00Z",
        },
    )

    results = processor.process(
        evidence=[evidence],
        question="Open source AI agents",
    )

    item = results[0]

    expected = round(
        (
            0.60 * item.relevance_score
            + 0.30 * item.quality_score
            + 0.10 * item.recency_score
        ),
        4,
    )

    assert item.combined_score == expected


def test_evidence_id_is_stable() -> None:
    """The same evidence should always receive the same identifier."""

    processor = EvidenceProcessor()

    evidence = make_evidence(
        title="Stable AI Agent",
        content="An open-source AI agent.",
    )

    first = processor.process(
        evidence=[evidence],
        question="Open source AI agents",
    )[0]

    second = processor.process(
        evidence=[evidence],
        question="Open source AI agents",
    )[0]

    assert first.evidence_id == second.evidence_id


def test_duplicate_evidence_is_removed() -> None:
    """Duplicate evidence should be reduced to one item."""

    processor = EvidenceProcessor()

    first = make_evidence(
        title="Duplicate AI Agent",
        content="Open-source AI agent.",
    )

    duplicate = make_evidence(
        title="Duplicate AI Agent",
        content="Open-source AI agent.",
    )

    results = processor.process(
        evidence=[first, duplicate],
        question="Open source AI agents",
        max_results=10,
    )

    assert len(results) == 1


def test_results_are_ranked_by_combined_score() -> None:
    """Evidence should be returned in descending combined-score order."""

    processor = EvidenceProcessor()

    highly_relevant = make_evidence(
        title="Open Source AI Agent Framework",
        content=(
            "An open-source AI agent framework "
            "for autonomous agent systems."
        ),
        metadata={
            "updated_at": "2026-10-05T20:00:00Z",
            "stars": 10000,
            "forks": 1000,
        },
    )

    weakly_relevant = make_evidence(
        title="Database Utility",
        content=(
            "A utility for managing database "
            "connections and migrations."
        ),
        metadata={
            "updated_at": "2026-10-05T20:00:00Z",
            "stars": 10000,
            "forks": 1000,
        },
    )

    results = processor.process(
        evidence=[
            weakly_relevant,
            highly_relevant,
        ],
        question="How is the open-source AI agent ecosystem evolving?",
        max_results=2,
    )

    assert (
        results[0].combined_score
        >= results[1].combined_score
    )

    assert (
        results[0].title
        == "Open Source AI Agent Framework"
    )