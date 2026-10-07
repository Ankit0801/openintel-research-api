from app.models import Evidence
from app.rag.chunker import EvidenceChunker


def make_evidence(
    content: str = " ".join(
        ["AI agents"] * 500
    ),
) -> Evidence:
    return Evidence(
        source="github",
        source_type="repository",
        title="Example",
        url="https://example.com/repo",
        content=content,
        metadata={"stars": 123},
        evidence_id="evidence-123",
    )


def test_chunker_preserves_provenance():
    chunks = EvidenceChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).chunk(
        make_evidence()
    )

    assert len(chunks) > 1

    assert all(
        chunk.evidence_id
        == "evidence-123"
        for chunk in chunks
    )

    assert all(
        chunk.source == "github"
        for chunk in chunks
    )

    assert all(
        chunk.title == "Example"
        for chunk in chunks
    )

    assert all(
        chunk.url
        == "https://example.com/repo"
        for chunk in chunks
    )

    assert all(
        chunk.metadata["stars"] == 123
        for chunk in chunks
    )


def test_chunk_ids_are_deterministic():
    chunker = EvidenceChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    first = chunker.chunk(
        make_evidence()
    )

    second = chunker.chunk(
        make_evidence()
    )

    assert [
        chunk.chunk_id
        for chunk in first
    ] == [
        chunk.chunk_id
        for chunk in second
    ]


def test_empty_content_returns_no_chunks():
    evidence = make_evidence(
        content="   "
    )

    assert (
        EvidenceChunker().chunk(
            evidence
        )
        == []
    )