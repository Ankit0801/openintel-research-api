from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "openintel-research-api",
    }


def test_research(monkeypatch):
    class FakeGraph:
        def invoke(self, state):
            assert state["request"].question == "How can AI assist research workflows?"

            return {
                "report": {
                    "answer": "AI can assist research workflows.",
                    "key_findings": [
                        "AI can automate research tasks."
                    ],
                    "limitations": [
                        "Human validation is still required."
                    ],
                    "evidence": [],
                }
            }

    monkeypatch.setattr(
        "app.api.main.build_research_graph",
        lambda: FakeGraph(),
    )

    response = client.post(
        "/research",
        json={
            "question": "How can AI assist research workflows?",
            "preferred_sources": ["github", "arxiv"],
            "max_sources": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "AI can assist research workflows."
    assert len(body["key_findings"]) == 1
    assert len(body["limitations"]) == 1
