from app.graph import rag_retrieval_node


def test_rag_retrieval_node_returns_retrieved_chunks(
    monkeypatch,
):
    class FakeRetriever:
        def index_evidence(self, evidence):
            assert evidence == ["evidence"]
            return 1

        def retrieve(self, query, *, k):
            assert query == "What is AI?"
            assert k == 5
            return ["chunk-1", "chunk-2"]

    monkeypatch.setattr(
        "app.graph.RAGRetriever",
        FakeRetriever,
    )

    state = {
        "request": type(
            "Request",
            (),
            {"question": "What is AI?"},
        )(),
        "processed_evidence": ["evidence"],
    }

    result = rag_retrieval_node(state)

    assert result["retrieved_chunks"] == [
        "chunk-1",
        "chunk-2",
    ]
    assert "errors" not in result