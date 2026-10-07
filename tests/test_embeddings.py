from types import SimpleNamespace

from app.rag.embeddings import (
    LocalEmbeddingModel,
)


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = 0

    def encode(
        self,
        values,
        **kwargs,
    ):
        self.calls += 1

        if isinstance(values, str):
            return SimpleNamespace(
                tolist=lambda: [
                    1.0,
                    0.0,
                    0.0,
                ]
            )

        return SimpleNamespace(
            tolist=lambda: [
                [
                    1.0,
                    0.0,
                    0.0,
                ]
                for _ in values
            ]
        )


def test_embedding_model_is_lazy_and_reused():
    fake = FakeSentenceTransformer()

    embeddings = LocalEmbeddingModel(
        model=fake
    )

    assert fake.calls == 0

    assert (
        embeddings.embed_query(
            "AI agents"
        )
        == [1.0, 0.0, 0.0]
    )

    assert fake.calls == 1

    embeddings.embed_documents(
        [
            "one",
            "two",
        ]
    )

    assert fake.calls == 2

    embeddings.embed_query(
        "another"
    )

    assert fake.calls == 3