"""Phase 6B: test live GitHub retrieval."""

from app.sources.github import GitHubClient


def main() -> None:
    """Search GitHub for AI agent repositories."""

    client = GitHubClient()

    evidence = client.search_repositories(
        query="AI agents",
        per_page=5,
    )

    print(f"Retrieved {len(evidence)} GitHub repositories.\n")

    for item in evidence:
        print("=" * 80)
        print(f"Source: {item.source}")
        print(f"Title: {item.title}")
        print(f"URL: {item.url}")
        print(item.content)


if __name__ == "__main__":
    main()