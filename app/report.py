"""Generate structured research reports from retrieved evidence."""

from __future__ import annotations

from app.config import Settings
from app.llm import get_llm
from app.models import Evidence, ResearchReport
from app.rag.retriever import RetrievedChunk


def _build_evidence_context(
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    """Build the evidence context supplied to the language model."""

    return "\n\n".join(
        (
            f"[Evidence {index}]\n"
            f"Source: {chunk.source}\n"
            f"Source type: {chunk.source_type}\n"
            f"Title: {chunk.title}\n"
            f"URL: {chunk.url or 'N/A'}\n"
            f"Similarity: {chunk.similarity_score:.4f}\n"
            f"Content:\n{chunk.content}"
        )
        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        )
    )


def _build_validation_context(
    validation_result,
) -> str:
    """Build validation context for the language model."""

    if validation_result is None:
        return ""

    claim_validation_context = "\n".join(
        (
            f"- Claim: {claim.claim}\n"
            f"  Status: {claim.status}\n"
            f"  Supporting sources: "
            f"{', '.join(claim.supporting_sources)}"
        )
        for claim in validation_result.claim_validations
    )

    if not claim_validation_context:
        claim_validation_context = (
            "- No normalized claims were available for validation."
        )

    return f"""
Cross-source validation:

Overall status: {validation_result.status}
Source count: {validation_result.source_count}
Evidence count: {validation_result.evidence_count}
Claim count: {validation_result.claim_count}
Corroborated claims: {validation_result.corroborated_claim_count}
Single-source claims: {validation_result.single_source_claim_count}
Corroboration ratio: {validation_result.corroboration_ratio:.4f}
Supporting sources: {", ".join(validation_result.supporting_sources)}

Claim-level validation:
{claim_validation_context}
"""


def _convert_retrieved_evidence(
    retrieved_chunks: list[RetrievedChunk],
) -> list[Evidence]:
    """Convert retrieved chunks into traceable report evidence."""

    return [
        Evidence(
            source=chunk.source,
            source_type=chunk.source_type,
            title=chunk.title,
            url=chunk.url,
            content=chunk.content,
            relevance_score=chunk.similarity_score,
            combined_score=chunk.similarity_score,
            metadata={
                "retrieval_similarity": chunk.similarity_score,
            },
        )
        for chunk in retrieved_chunks
    ]


def generate_research_report(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    settings: Settings,
    validation_result=None,
) -> ResearchReport:
    """Generate an evidence-backed structured research report."""

    if not question.strip():
        raise ValueError("Research question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError(
            "At least one retrieved chunk is required."
        )

    evidence_context = _build_evidence_context(
        retrieved_chunks
    )

    validation_context = _build_validation_context(
        validation_result
    )

    prompt = f"""
You are an evidence-based research assistant.

Answer the user's research question using ONLY the supplied
retrieved evidence.

Research question:
{question}

Retrieved evidence:
{evidence_context}

{validation_context}

Requirements:
- Synthesize the evidence accurately.
- Do not invent facts that are not supported by the evidence.
- Identify important findings clearly.
- Consider the cross-source validation when assessing confidence.
- Distinguish corroborated claims from claims supported by only one source.
- Do not describe a claim as independently corroborated unless the
  validation data explicitly identifies it as corroborated.
- Mention meaningful limitations or uncertainty.
- Do not invent URLs, source names, titles, or evidence records.
- The final evidence list will be populated from the retrieved
  evidence by the application, so focus on producing the answer,
  key findings, and limitations.
"""

    llm = get_llm(settings)

    structured_llm = llm.with_structured_output(
        ResearchReport
    )

    generated_report = structured_llm.invoke(prompt)

    traceable_evidence = _convert_retrieved_evidence(
        retrieved_chunks
    )

    return generated_report.model_copy(
        update={
            "evidence": traceable_evidence,
        }
    )