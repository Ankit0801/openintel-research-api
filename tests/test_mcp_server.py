import pytest

from mcp import Client

from app.mcp.server import mcp


@pytest.mark.anyio
async def test_mcp_server_exposes_health_check():
    async with Client(mcp) as client:
        result = await client.list_tools()

    tool_names = [
        tool.name
        for tool in result.tools
    ]

    assert "health_check" in tool_names


@pytest.mark.anyio
async def test_mcp_health_check_tool():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "health_check",
            {},
        )

    assert result.is_error is False

    assert result.structured_content == {
        "result": "OpenIntel MCP server is healthy.",
    }


@pytest.mark.anyio
async def test_mcp_server_exposes_research_tool():
    async with Client(mcp) as client:
        result = await client.list_tools()

    tool_names = [
        tool.name
        for tool in result.tools
    ]

    assert "research" in tool_names


@pytest.mark.anyio
async def test_mcp_research_tool_calls_research_workflow(
    monkeypatch,
):
    from app.mcp import server

    class FakeReport:
        def model_dump(self, mode="json"):
            return {
                "answer": (
                    "AI can assist research workflows."
                ),
                "key_findings": [
                    "AI can assist research workflows."
                ],
                "limitations": [
                    "Evidence was limited."
                ],
                "evidence": [],
            }

    class FakeGraph:
        def invoke(self, state):
            assert state["request"].question == (
                "How can AI assist research workflows?"
            )

            assert state["request"].preferred_sources == [
                "github",
                "arxiv",
            ]

            assert state["request"].max_sources == 2

            return {
                "report": FakeReport(),
                "errors": [],
            }

    monkeypatch.setattr(
        server,
        "build_research_graph",
        lambda: FakeGraph(),
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "research",
            {
                "question": (
                    "How can AI assist research workflows?"
                ),
                "preferred_sources": [
                    "github",
                    "arxiv",
                ],
                "max_sources": 2,
            },
        )

    assert result.is_error is False

    assert result.structured_content == {
    "answer": (
        "AI can assist research workflows."
    ),
    "key_findings": [
        "AI can assist research workflows."
    ],
    "limitations": [
        "Evidence was limited."
    ],
    "evidence": [],
}