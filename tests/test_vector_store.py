from app.rag.vector_store import (
    ChromaVectorStore,
)


def test_chroma_vector_store_round_trip(
    tmp_path,
):
    store = ChromaVectorStore(
        persist_directory=(
            tmp_path / "chroma"
        ),
        collection_name=(
            "test_collection"
        ),
    )

    store.add(
        ids=["chunk-1"],
        documents=[
            "AI agents use tools."
        ],
        embeddings=[
            [1.0, 0.0, 0.0]
        ],
        metadatas=[
            {
                "evidence_id": "e1",
                "source": "github",
                "source_type": "repository",
                "title": "Example",
                "url": "https://example.com",
                "chunk_id": "chunk-1",
            }
        ],
    )

    assert store.count() == 1

    rows = store.query(
        embedding=[
            1.0,
            0.0,
            0.0,
        ],
        n_results=1,
    )

    assert len(rows) == 1

    assert (
        rows[0]["chunk_id"]
        == "chunk-1"
    )

    assert (
        rows[0]["content"]
        == "AI agents use tools."
    )

    assert (
        rows[0]["similarity"]
        == 1.0
    )