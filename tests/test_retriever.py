from app.models import Evidence
from app.rag.chunker import EvidenceChunker
from app.rag.retriever import (
    RAGRetriever,
)


class FakeEmbeddingModel:
    def embed_documents(
        self,
        texts,
    ):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        text,
    ):
        return [
            1.0,
            0.0,
            0.0,
        ]


def test_retriever_returns_provenance(
    tmp_path,
):
    retriever = RAGRetriever(
        embedding_model=(
            FakeEmbeddingModel()
        ),
        persist_directory=(
            tmp_path / "chroma"
        ),
        collection_name=(
            "retriever_test"
        ),
        chunker=EvidenceChunker(
            chunk_size=500,
            chunk_overlap=50,
        ),
    )

    evidence = Evidence(
        source="arxiv",
        source_type="paper",
        title="AI Agents Research",
        url=(
            "https://arxiv.org/abs/"
            "1234.5678"
        ),
        content=(
            "AI agents can plan, "
            "use tools, and retrieve "
            "information."
        ),
        metadata={
            "authors": [
                "A. Researcher"
            ],
            "published": (
                "2026-01-01"
            ),
        },
        evidence_id="paper-1",
    )

    indexed = retriever.index_evidence(
        [evidence]
    )

    assert indexed == 1

    results = retriever.retrieve(
        "tool-using AI agents",
        k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result.evidence_id
        == "paper-1"
    )

    assert (
        result.source
        == "arxiv"
    )

    assert (
        result.source_type
        == "paper"
    )

    assert (
        result.title
        == "AI Agents Research"
    )

    assert (
        result.url
        == "https://arxiv.org/abs/"
        "1234.5678"
    )

    assert result.chunk_id

    assert (
        result.similarity_score
        == 1.0
    )

    assert (
        result.metadata["authors"]
        == ["A. Researcher"]
    )