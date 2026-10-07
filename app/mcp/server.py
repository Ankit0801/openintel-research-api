"""MCP server for the OpenIntel research system."""

from typing import Any

from mcp.server import MCPServer

from app.graph import build_research_graph
from app.models import ResearchRequest


mcp = MCPServer(
    "OpenIntel Research Server",
)


@mcp.tool(
    name="health_check",
    description="Check whether the OpenIntel MCP server is running.",
)
def health_check() -> str:
    """Check whether the OpenIntel MCP server is running."""

    return "OpenIntel MCP server is healthy."


@mcp.tool(
    name="research",
    description=(
        "Run the OpenIntel multi-source research workflow for a "
        "research question and return the structured research report."
    ),
    structured_output=True,
)
def research(
    question: str,
    preferred_sources: list[str] | None = None,
    max_sources: int = 5,
) -> dict[str, Any]:
    """Run the complete OpenIntel research workflow."""

    request = ResearchRequest(
        question=question,
        preferred_sources=preferred_sources or [],
        max_sources=max_sources,
    )

    initial_state = {
        "request": request,
        "research_plan": None,
        "selected_sources": [],
        "source_queries": {},
        "raw_evidence": [],
        "processed_evidence": [],
        "validation_result": None,
        "retrieved_chunks": [],
        "report": None,
        "errors": [],
    }

    graph = build_research_graph()

    final_state = graph.invoke(initial_state)

    report = final_state.get("report")

    if report is None:
        errors = final_state.get("errors", [])

        raise RuntimeError(
            "Research workflow failed."
            + (
                f" Errors: {'; '.join(errors)}"
                if errors
                else ""
            )
        )

    return report.model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()