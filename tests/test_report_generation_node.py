from app.graph import report_generation_node
from app.models import Evidence, ResearchReport


def test_report_generation_node_returns_report(
    monkeypatch,
):
    expected_report = ResearchReport(
        answer="AI is a field of computer science.",
        key_findings=[
            "AI can assist research workflows.",
        ],
        limitations=[
            "The retrieved evidence is limited.",
        ],
        evidence=[
            Evidence(
                source="test",
                source_type="article",
                title="Test Evidence",
                content="AI can assist research workflows.",
            )
        ],
    )

    class FakeSettings:
        pass

    class FakeValidationResult:
        status = "corroborated"
        source_count = 2
        supporting_sources = (
            "arxiv",
            "github",
        )

    fake_validation_result = FakeValidationResult()

    def fake_get_settings():
        return FakeSettings()

    def fake_generate_research_report(
        question,
        retrieved_chunks,
        settings,
        validation_result,
    ):
        assert question == "What is AI?"
        assert retrieved_chunks == [
            "chunk-1",
            "chunk-2",
        ]
        assert isinstance(settings, FakeSettings)
        assert validation_result is fake_validation_result

        return expected_report

    monkeypatch.setattr(
        "app.graph.get_settings",
        fake_get_settings,
    )

    monkeypatch.setattr(
        "app.graph.generate_research_report",
        fake_generate_research_report,
    )

    state = {
        "request": type(
            "Request",
            (),
            {"question": "What is AI?"},
        )(),
        "retrieved_chunks": [
            "chunk-1",
            "chunk-2",
        ],
        "validation_result": fake_validation_result,
    }

    result = report_generation_node(state)

    assert result["report"] == expected_report
    assert "errors" not in result