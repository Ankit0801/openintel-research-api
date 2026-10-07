"""NVD CVE API client for OpenIntel."""

from typing import Any

import httpx

from app.models import Evidence


class NVDClient:
    """Client for retrieving CVE information from the NVD."""

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        """Initialize the NVD client."""

        self.timeout = timeout
        self.api_key = api_key

        self.headers = {
            "User-Agent": "OpenIntel/0.1",
        }

        if api_key:
            self.headers["apiKey"] = api_key

    def search_cves(
        self,
        keyword: str,
        results_per_page: int = 5,
    ) -> list[Evidence]:
        """Search CVEs using the NVD keyword search."""

        if not keyword.strip():
            raise ValueError(
                "NVD keyword cannot be empty."
            )

        if not 1 <= results_per_page <= 20:
            raise ValueError(
                "results_per_page must be between 1 and 20."
            )

        params: dict[str, Any] = {
            "keywordSearch": keyword.strip(),
            "resultsPerPage": results_per_page,
        }

        try:
            response = httpx.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
        except httpx.RequestError as error:
            raise RuntimeError(
                f"NVD request failed: {error}"
            ) from error

        if response.status_code == 429:
            raise RuntimeError(
                "NVD rate limit reached. "
                "Please wait before retrying."
            )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected NVD response format."
            )

        vulnerabilities = data.get(
            "vulnerabilities",
            [],
        )

        if not isinstance(vulnerabilities, list):
            raise RuntimeError(
                "NVD vulnerabilities field has "
                "an unexpected format."
            )

        evidence: list[Evidence] = []

        for item in vulnerabilities:
            if not isinstance(item, dict):
                continue

            cve = item.get("cve")

            if not isinstance(cve, dict):
                continue

            evidence_item = self._cve_to_evidence(cve)

            if evidence_item:
                evidence.append(evidence_item)

        return evidence

    @staticmethod
    def _cve_to_evidence(
        cve: dict[str, Any],
    ) -> Evidence | None:
        """Convert one NVD CVE object into Evidence."""

        cve_id = cve.get("id")

        if not cve_id:
            return None

        descriptions = cve.get(
            "descriptions",
            [],
        )

        description = ""

        if isinstance(descriptions, list):
            for item in descriptions:
                if not isinstance(item, dict):
                    continue

                if item.get("lang") == "en":
                    description = item.get(
                        "value",
                        "",
                    )
                    break

        published = cve.get(
            "published",
            "Unknown",
        )

        last_modified = cve.get(
            "lastModified",
            "Unknown",
        )

        references = cve.get(
            "references",
            [],
        )

        reference_urls: list[str] = []

        if isinstance(references, list):
            for reference in references:
                if not isinstance(reference, dict):
                    continue

                url = reference.get("url")

                if url:
                    reference_urls.append(url)

        metrics = cve.get(
            "metrics",
            {}
        )

        cvss_score = None

        if isinstance(metrics, dict):
            cvss_v31 = metrics.get(
                "cvssMetricV31",
                [],
            )

            if isinstance(cvss_v31, list) and cvss_v31:
                first_metric = cvss_v31[0]

                if isinstance(first_metric, dict):
                    cvss_data = first_metric.get(
                        "cvssData"
                    )

                    if isinstance(cvss_data, dict):
                        cvss_score = cvss_data.get(
                            "baseScore"
                        )

        content = (
            f"CVE: {cve_id}\n"
            f"Published: {published}\n"
            f"Last modified: {last_modified}\n"
            f"CVSS score: "
            f"{cvss_score if cvss_score is not None else 'Unknown'}\n"
            f"Description: "
            f"{description or 'No description available.'}\n"
            f"References: "
            f"{', '.join(reference_urls) or 'None'}"
        )

        return Evidence(
            source="nvd",
            source_type="vulnerability",
            title=cve_id,
            url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
            content=content,
            metadata={
                "cve_id": cve_id,
                "published": published,
                "last_modified": last_modified,
                "cvss_score": cvss_score,
                "references": reference_urls,
            },
        )