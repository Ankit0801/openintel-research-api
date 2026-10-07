from types import SimpleNamespace

import pytest

from app.report import generate_research_report


def make_chunk(
    source: str = "GitHub",
    source_type: str = "repository",
    title: str = "AI Research",
    url: str = "https://example.com/evidence",
    similarity_score: float = 0.91,
    content: str = "AI can assist research workflows.",
):
    return SimpleNamespace(
        source=source,
        source_type=source_type,
        title=title,
        url=url,
        similarity_score=similarity_score,
        content=content,
    )


def make_validation_result():
    return SimpleNamespace(
        status="corroborated",
        source_count=3,
        evidence_count=4,
        claim_count=2,
        corroborated_claim_count=1,
        single_source_claim_count=1,
        corroboration_ratio=0.5,
        supporting_sources=(
            "arxiv",
            "github",
            "openalex",
        ),
        claim_validations=(
            SimpleNamespace(
                claim="ai can assist research workflows.",
                status="corroborated",
                supporting_sources=(
                    "arxiv",
                    "github",
                ),
            ),
            SimpleNamespace(
                claim="ai can improve model evaluation.",
                status="single_source",
                supporting_sources=(
                    "openalex",
                ),
            ),
        ),
    )


class FakeStructuredLLM:
    def __init__(self, captured_prompt):
        self.captured_prompt = captured_prompt

    def invoke(self, prompt):
        self.captured_prompt["value"] = prompt

        from app.models import ResearchReport

        return ResearchReport(
            answer="AI can assist research workflows.",
            key_findings=[
                "The claim is corroborated."
            ],
            limitations=[
                "One claim has single-source support."
            ],
            evidence=[],
        )


class FakeLLM:
    def __init__(self, captured_prompt):
        self.captured_prompt = captured_prompt

    def with_structured_output(self, model):
        return FakeStructuredLLM(
            self.captured_prompt
        )


def test_generate_research_report_returns_structured_output(
    monkeypatch,
):
    captured_prompt = {}

    monkeypatch.setattr(
        "app.report.get_llm",
        lambda settings: FakeLLM(captured_prompt),
    )

    retrieved_chunks = [
        make_chunk(),
    ]

    settings = SimpleNamespace()

    result = generate_research_report(
        question="How can AI assist research workflows?",
        retrieved_chunks=retrieved_chunks,
        settings=settings,
    )

    assert result.answer == (
        "AI can assist research workflows."
    )

    assert result.key_findings == [
        "The claim is corroborated."
    ]

    assert result.limitations == [
        "One claim has single-source support."
    ]

    assert len(result.evidence) == 1


def test_generate_research_report_includes_validation_context(
    monkeypatch,
):
    captured_prompt = {}

    monkeypatch.setattr(
        "app.report.get_llm",
        lambda settings: FakeLLM(captured_prompt),
    )

    retrieved_chunks = [
        make_chunk(),
    ]

    settings = SimpleNamespace()

    validation_result = make_validation_result()

    generate_research_report(
        question="How can AI assist research workflows?",
        retrieved_chunks=retrieved_chunks,
        settings=settings,
        validation_result=validation_result,
    )

    prompt = captured_prompt["value"]

    assert "Overall status: corroborated" in prompt
    assert "Source count: 3" in prompt
    assert "Evidence count: 4" in prompt
    assert "Claim count: 2" in prompt
    assert "Corroborated claims: 1" in prompt
    assert "Single-source claims: 1" in prompt
    assert "Corroboration ratio: 0.5000" in prompt

    assert (
        "ai can assist research workflows."
        in prompt
    )

    assert "Status: corroborated" in prompt
    assert "arxiv, github" in prompt

    assert (
        "ai can improve model evaluation."
        in prompt
    )

    assert "Status: single_source" in prompt
    assert "openalex" in prompt


def test_generate_research_report_uses_retrieved_evidence_as_source_of_truth(
    monkeypatch,
):
    captured_prompt = {}

    monkeypatch.setattr(
        "app.report.get_llm",
        lambda settings: FakeLLM(captured_prompt),
    )

    retrieved_chunks = [
        make_chunk(
            source="GitHub",
            source_type="repository",
            title="AI Research",
            url="https://github.com/example/research",
            similarity_score=0.91,
            content="AI can assist research workflows.",
        ),
        make_chunk(
            source="arXiv",
            source_type="paper",
            title="AI Evaluation",
            url="https://arxiv.org/abs/1234.5678",
            similarity_score=0.84,
            content="AI can improve model evaluation.",
        ),
    ]

    settings = SimpleNamespace()

    result = generate_research_report(
        question="How can AI assist research workflows?",
        retrieved_chunks=retrieved_chunks,
        settings=settings,
    )

    assert len(result.evidence) == 2

    first = result.evidence[0]

    assert first.source == "GitHub"
    assert first.source_type == "repository"
    assert first.title == "AI Research"
    assert first.url == (
        "https://github.com/example/research"
    )
    assert first.content == (
        "AI can assist research workflows."
    )
    assert first.relevance_score == pytest.approx(0.91)
    assert first.combined_score == pytest.approx(0.91)

    second = result.evidence[1]

    assert second.source == "arXiv"
    assert second.source_type == "paper"
    assert second.title == "AI Evaluation"
    assert second.url == (
        "https://arxiv.org/abs/1234.5678"
    )
    assert second.content == (
        "AI can improve model evaluation."
    )
    assert second.relevance_score == pytest.approx(0.84)
    assert second.combined_score == pytest.approx(0.84)


def test_generate_research_report_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="Research question cannot be empty.",
    ):
        generate_research_report(
            question="   ",
            retrieved_chunks=[
                make_chunk(),
            ],
            settings=SimpleNamespace(),
        )


def test_generate_research_report_rejects_empty_evidence():
    with pytest.raises(
        ValueError,
        match="At least one retrieved chunk is required.",
    ):
        generate_research_report(
            question="How can AI assist research workflows?",
            retrieved_chunks=[],
            settings=SimpleNamespace(),
        )