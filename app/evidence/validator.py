"""Cross-source validation for normalized research evidence."""

from __future__ import annotations

from dataclasses import dataclass

from app.models import Evidence


@dataclass(frozen=True)
class ClaimValidation:
    """Validation result for a single normalized claim."""

    claim: str
    status: str
    supporting_sources: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    """Result of deterministic cross-source evidence validation."""

    status: str
    source_count: int
    supporting_sources: tuple[str, ...]
    evidence_count: int
    claim_validations: tuple[ClaimValidation, ...]
    claim_count: int
    corroborated_claim_count: int
    single_source_claim_count: int
    corroboration_ratio: float


class EvidenceValidator:
    """Validate evidence coverage across independent sources."""

    def validate(
        self,
        evidence: list[Evidence],
    ) -> ValidationResult:
        """Classify evidence based on independent sources and claims."""

        if not evidence:
            return ValidationResult(
                status="no_evidence",
                source_count=0,
                supporting_sources=(),
                evidence_count=0,
                claim_validations=(),
                claim_count=0,
                corroborated_claim_count=0,
                single_source_claim_count=0,
                corroboration_ratio=0.0,
            )

        sources = sorted(
            {
                item.source.strip().lower()
                for item in evidence
                if item.source.strip()
            }
        )

        source_count = len(sources)

        if source_count == 0:
            return ValidationResult(
                status="no_source",
                source_count=0,
                supporting_sources=(),
                evidence_count=len(evidence),
                claim_validations=(),
                claim_count=0,
                corroborated_claim_count=0,
                single_source_claim_count=0,
                corroboration_ratio=0.0,
            )

        if source_count == 1:
            status = "single_source"
        else:
            status = "corroborated"

        claim_validations = self._validate_claims(evidence)

        claim_count = len(claim_validations)

        corroborated_claim_count = sum(
            1
            for claim in claim_validations
            if claim.status == "corroborated"
        )

        single_source_claim_count = sum(
            1
            for claim in claim_validations
            if claim.status == "single_source"
        )

        corroboration_ratio = (
            corroborated_claim_count / claim_count
            if claim_count > 0
            else 0.0
        )

        return ValidationResult(
            status=status,
            source_count=source_count,
            supporting_sources=tuple(sources),
            evidence_count=len(evidence),
            claim_validations=claim_validations,
            claim_count=claim_count,
            corroborated_claim_count=corroborated_claim_count,
            single_source_claim_count=single_source_claim_count,
            corroboration_ratio=corroboration_ratio,
        )

    def _validate_claims(
        self,
        evidence: list[Evidence],
    ) -> tuple[ClaimValidation, ...]:
        """Group identical normalized claims by independent source."""

        claims: dict[str, set[str]] = {}

        for item in evidence:
            source = item.source.strip().lower()
            claim = self._normalize_claim(item.content)

            if not source or not claim:
                continue

            claims.setdefault(claim, set()).add(source)

        validations = []

        for claim, sources in claims.items():
            supporting_sources = tuple(sorted(sources))

            if len(supporting_sources) == 1:
                status = "single_source"
            else:
                status = "corroborated"

            validations.append(
                ClaimValidation(
                    claim=claim,
                    status=status,
                    supporting_sources=supporting_sources,
                )
            )

        return tuple(
            sorted(
                validations,
                key=lambda item: item.claim,
            )
        )

    @staticmethod
    def _normalize_claim(content: str) -> str:
        """Normalize evidence content for deterministic claim matching."""

        return " ".join(content.strip().lower().split())