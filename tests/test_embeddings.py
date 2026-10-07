from types import SimpleNamespace

from app.rag.embeddings import GeminiEmbeddingModel


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_content(self, **kwargs):
        self.calls += 1

        contents = kwargs["contents"]

        if isinstance(contents, str):
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(
                        values=[
                            1.0,
                            0.0,
                            0.0,
                        ]
                    )
                ]
            )

        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[
                        1.0,
                        0.0,
                        0.0,
                    ]
                )
                for _ in contents
            ]
        )


def test_embedding_model_uses_gemini():
    fake = FakeEmbeddings()

    embeddings = GeminiEmbeddingModel(
        client=SimpleNamespace(
            models=fake
        )
    )

    assert (
        embeddings.embed_query("AI agents")
        == [1.0, 0.0, 0.0]
    )

    assert fake.calls == 1

    assert (
        embeddings.embed_documents(
            ["one", "two"]
        )
        == [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert fake.calls == 2