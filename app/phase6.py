"""Phase 6: deterministic multi-source OpenIntel research."""

from app.graph import ResearchState, build_research_graph
from app.models import ResearchRequest


def sample_request() -> ResearchRequest:
    """Create a sample OpenIntel research request."""

    return ResearchRequest(
        question="Open source AI agents",
        preferred_sources=[
            "github",
            "arxiv",
            "openalex",
            "nvd",
        ],
        max_sources=5,
    )


def main() -> None:
    """Run the deterministic multi-source research graph."""

    request = sample_request()

    initial_state: ResearchState = {
        "request": request,
        "research_plan": None,
        "selected_sources": [],
        "source_queries": {},
        "raw_evidence": [],
        "processed_evidence": [],
        "errors": [],
    }

    graph = build_research_graph()

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 80)
    print("OPENINTEL DETERMINISTIC RESEARCH")
    print("=" * 80)

    print("\nSelected sources:")
    for source in final_state["selected_sources"]:
        print(f"- {source}")

    print("\nSource queries:")
    for source, query in final_state["source_queries"].items():
        if source in final_state["selected_sources"]:
            print(f"- {source}: {query}")

    print(
        f"\nRaw evidence retrieved: "
        f"{len(final_state['raw_evidence'])}"
    )

    print(
        f"Processed evidence: "
        f"{len(final_state['processed_evidence'])}"
    )

    print("\nRanked evidence:")

    for rank, item in enumerate(
        final_state["processed_evidence"],
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"Source: {item.source}")
        print(f"Source type: {item.source_type}")
        print(f"Title: {item.title}")
        print(f"URL: {item.url}")
        print(f"Relevance: {item.relevance_score}")
        print(f"Quality: {item.quality_score}")
        print(f"Recency: {item.recency_score}")
        print(f"Combined: {item.combined_score}")
        print(f"Evidence ID: {item.evidence_id}")
        print("\nContent:")
        print(item.content)

    if final_state["errors"]:
        print("\n" + "=" * 80)
        print("Errors:")
        for error in final_state["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()