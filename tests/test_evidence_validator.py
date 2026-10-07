from app.evidence.validator import (
    ClaimValidation,
    EvidenceValidator,
)
from app.models import Evidence


def make_evidence(
    source: str,
    title: str = "Test Evidence",
    content: str = "AI can assist research workflows.",
) -> Evidence:
    return Evidence(
        source=source,
        source_type="article",
        title=title,
        content=content,
    )


def test_validator_returns_no_evidence_for_empty_input():
    validator = EvidenceValidator()

    result = validator.validate([])

    assert result.status == "no_evidence"
    assert result.source_count == 0
    assert result.evidence_count == 0
    assert result.supporting_sources == ()
    assert result.claim_validations == ()
    assert result.claim_count == 0
    assert result.corroborated_claim_count == 0
    assert result.single_source_claim_count == 0
    assert result.corroboration_ratio == 0.0


def test_validator_returns_no_source_when_source_names_are_blank():
    validator = EvidenceValidator()

    evidence = [
        make_evidence("   "),
    ]

    result = validator.validate(evidence)

    assert result.status == "no_source"
    assert result.source_count == 0
    assert result.evidence_count == 1
    assert result.supporting_sources == ()
    assert result.claim_validations == ()
    assert result.claim_count == 0
    assert result.corroborated_claim_count == 0
    assert result.single_source_claim_count == 0
    assert result.corroboration_ratio == 0.0


def test_validator_identifies_single_source():
    validator = EvidenceValidator()

    evidence = [
        make_evidence("GitHub"),
        make_evidence("github", "Second Result"),
    ]

    result = validator.validate(evidence)

    assert result.status == "single_source"
    assert result.source_count == 1
    assert result.evidence_count == 2
    assert result.supporting_sources == (
        "github",
    )
    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="single_source",
            supporting_sources=(
                "github",
            ),
        ),
    )
    assert result.claim_count == 1
    assert result.corroborated_claim_count == 0
    assert result.single_source_claim_count == 1
    assert result.corroboration_ratio == 0.0


def test_validator_identifies_cross_source_corroboration():
    validator = EvidenceValidator()

    evidence = [
        make_evidence("GitHub"),
        make_evidence("arXiv"),
        make_evidence("OpenAlex"),
    ]

    result = validator.validate(evidence)

    assert result.status == "corroborated"
    assert result.source_count == 3
    assert result.evidence_count == 3
    assert result.supporting_sources == (
        "arxiv",
        "github",
        "openalex",
    )
    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="corroborated",
            supporting_sources=(
                "arxiv",
                "github",
                "openalex",
            ),
        ),
    )
    assert result.claim_count == 1
    assert result.corroborated_claim_count == 1
    assert result.single_source_claim_count == 0
    assert result.corroboration_ratio == 1.0


def test_validator_deduplicates_source_names():
    validator = EvidenceValidator()

    evidence = [
        make_evidence("GitHub"),
        make_evidence("github"),
        make_evidence("GITHUB"),
        make_evidence("arXiv"),
        make_evidence("arxiv"),
    ]

    result = validator.validate(evidence)

    assert result.status == "corroborated"
    assert result.source_count == 2
    assert result.evidence_count == 5
    assert result.supporting_sources == (
        "arxiv",
        "github",
    )
    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="corroborated",
            supporting_sources=(
                "arxiv",
                "github",
            ),
        ),
    )
    assert result.claim_count == 1
    assert result.corroborated_claim_count == 1
    assert result.single_source_claim_count == 0
    assert result.corroboration_ratio == 1.0


def test_validator_groups_identical_claims_across_sources():
    validator = EvidenceValidator()

    evidence = [
        make_evidence("GitHub"),
        make_evidence("arXiv"),
    ]

    result = validator.validate(evidence)

    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="corroborated",
            supporting_sources=(
                "arxiv",
                "github",
            ),
        ),
    )
    assert result.claim_count == 1
    assert result.corroborated_claim_count == 1
    assert result.single_source_claim_count == 0
    assert result.corroboration_ratio == 1.0


def test_validator_deduplicates_same_claim_from_same_source():
    validator = EvidenceValidator()

    evidence = [
        make_evidence("GitHub"),
        make_evidence("github", "Second Result"),
        make_evidence("arXiv"),
    ]

    result = validator.validate(evidence)

    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="corroborated",
            supporting_sources=(
                "arxiv",
                "github",
            ),
        ),
    )
    assert result.claim_count == 1
    assert result.corroborated_claim_count == 1
    assert result.single_source_claim_count == 0
    assert result.corroboration_ratio == 1.0


def test_validator_keeps_different_claims_separate():
    validator = EvidenceValidator()

    evidence = [
        make_evidence(
            source="GitHub",
            title="AI Research",
            content="AI can assist research workflows.",
        ),
        make_evidence(
            source="arXiv",
            title="Model Evaluation",
            content="AI can improve model evaluation.",
        ),
    ]

    result = validator.validate(evidence)

    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="single_source",
            supporting_sources=(
                "github",
            ),
        ),
        ClaimValidation(
            claim="ai can improve model evaluation.",
            status="single_source",
            supporting_sources=(
                "arxiv",
            ),
        ),
    )
    assert result.claim_count == 2
    assert result.corroborated_claim_count == 0
    assert result.single_source_claim_count == 2
    assert result.corroboration_ratio == 0.0


def test_validator_counts_mixed_claim_validation_statuses():
    validator = EvidenceValidator()

    evidence = [
        make_evidence(
            source="GitHub",
            content="AI can assist research workflows.",
        ),
        make_evidence(
            source="arXiv",
            content="AI can assist research workflows.",
        ),
        make_evidence(
            source="OpenAlex",
            content="AI can improve model evaluation.",
        ),
    ]

    result = validator.validate(evidence)

    assert result.claim_validations == (
        ClaimValidation(
            claim="ai can assist research workflows.",
            status="corroborated",
            supporting_sources=(
                "arxiv",
                "github",
            ),
        ),
        ClaimValidation(
            claim="ai can improve model evaluation.",
            status="single_source",
            supporting_sources=(
                "openalex",
            ),
        ),
    )
    assert result.claim_count == 2
    assert result.corroborated_claim_count == 1
    assert result.single_source_claim_count == 1
    assert result.corroboration_ratio == 0.5