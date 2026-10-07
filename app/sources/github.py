"""GitHub REST API client for OpenIntel."""

from typing import Any

import httpx

from app.config import get_settings
from app.models import Evidence


class GitHubClient:
    """Client for retrieving public GitHub repository information."""

    BASE_URL = "https://api.github.com"
    API_VERSION = "2026-03-10"

    def __init__(
        self,
        token: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        """Initialize the GitHub client."""

        self.timeout = timeout

        # If a token is not explicitly supplied, use the application config.
        if token is None:
            token = get_settings().github_token

        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.API_VERSION,
            "User-Agent": "OpenIntel/0.1",
        }

        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a GitHub API request and return decoded JSON."""

        url = f"{self.BASE_URL}{path}"

        try:
            response = httpx.request(
                method=method,
                url=url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
        except httpx.RequestError as error:
            raise RuntimeError(
                f"GitHub request failed: {error}"
            ) from error

        self._raise_for_status(response)

        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Unexpected GitHub response format from {path}."
            )

        return data

    def search_repositories(
        self,
        query: str,
        per_page: int = 5,
    ) -> list[Evidence]:
        """Search GitHub repositories and normalize results into Evidence."""

        if not query.strip():
            raise ValueError("GitHub search query cannot be empty.")

        if not 1 <= per_page <= 30:
            raise ValueError("per_page must be between 1 and 30.")

        data = self._request(
            "GET",
            "/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
            },
        )

        repositories = data.get("items", [])

        return [
            self._repository_to_evidence(repository)
            for repository in repositories
            if isinstance(repository, dict)
        ]

    def get_repository(
        self,
        owner: str,
        repository: str,
    ) -> Evidence:
        """Retrieve one GitHub repository."""

        if not owner.strip():
            raise ValueError("GitHub repository owner cannot be empty.")

        if not repository.strip():
            raise ValueError("GitHub repository name cannot be empty.")

        data = self._request(
            "GET",
            f"/repos/{owner}/{repository}",
        )

        return self._repository_to_evidence(data)

    @staticmethod
    def _repository_to_evidence(
        repository: dict[str, Any],
    ) -> Evidence:
        """Convert GitHub repository JSON into OpenIntel Evidence."""

        name = repository.get(
            "full_name",
            "Unknown repository",
        )

        description = (
            repository.get("description")
            or "No description provided."
        )

        stars = repository.get(
            "stargazers_count",
            0,
        )

        forks = repository.get(
            "forks_count",
            0,
        )

        language = (
            repository.get("language")
            or "Unknown"
        )

        open_issues = repository.get(
            "open_issues_count",
            0,
        )

        updated_at = repository.get(
            "updated_at",
            "Unknown",
        )

        content = (
            f"Repository: {name}\n"
            f"Description: {description}\n"
            f"Stars: {stars}\n"
            f"Forks: {forks}\n"
            f"Primary language: {language}\n"
            f"Open issues: {open_issues}\n"
            f"Last updated: {updated_at}"
        )

        return Evidence(
            source="github",
            source_type="repository",
            title=name,
            url=repository.get("html_url"),
            content=content,
            metadata={
                "stars": stars,
                "forks": forks,
                "language": language,
                "open_issues": open_issues,
                "updated_at": updated_at,
                "owner": repository.get(
                    "owner",
                    {},
                ).get("login")
                if isinstance(
                    repository.get("owner"),
                    dict,
                )
                else None,
            },
        )
    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise useful errors for GitHub API failures."""

        if response.status_code in {403, 429}:
            remaining = response.headers.get("x-ratelimit-remaining")
            reset = response.headers.get("x-ratelimit-reset")
            retry_after = response.headers.get("retry-after")

            raise RuntimeError(
                "GitHub rate limit reached. "
                f"remaining={remaining}, "
                f"reset={reset}, "
                f"retry_after={retry_after}"
            )

        if response.status_code == 404:
            raise RuntimeError(
                "GitHub resource was not found: "
                f"{response.request.url}"
            )

        response.raise_for_status()