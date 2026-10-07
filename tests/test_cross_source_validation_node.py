from app.graph import cross_source_validation_node


def test_cross_source_validation_node_returns_validation_result(
    monkeypatch,
):
    expected_result = "validated"

    class FakeValidator:
        def validate(self, evidence):
            assert evidence == [
                "evidence-1",
                "evidence-2",
            ]

            return expected_result

    monkeypatch.setattr(
        "app.graph.EvidenceValidator",
        FakeValidator,
    )

    state = {
        "processed_evidence": [
            "evidence-1",
            "evidence-2",
        ],
    }

    result = cross_source_validation_node(state)

    assert result["validation_result"] == expected_result
    assert "errors" not in result