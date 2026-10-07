"""Deterministic chunking for normalized research evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.models import Evidence


@dataclass(frozen=True)
class EvidenceChunk:
    """A chunk of Evidence with complete provenance."""

    chunk_id: str
    evidence_id: str
    source: str
    source_type: str
    title: str
    url: str | None
    content: str
    metadata: dict[str, object]


class EvidenceChunker:
    """Split evidence into deterministic overlapping character chunks."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        evidence: Evidence,
    ) -> list[EvidenceChunk]:
        """Create deterministic chunks while preserving provenance."""

        content = " ".join(
            evidence.content.split()
        ).strip()

        if not content:
            return []

        evidence_id = (
            evidence.evidence_id
            or self._fallback_evidence_id(evidence)
        )

        chunks: list[EvidenceChunk] = []

        start = 0
        chunk_index = 0

        step = (
            self.chunk_size
            - self.chunk_overlap
        )

        while start < len(content):
            end = min(
                start + self.chunk_size,
                len(content),
            )

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunk_id = self._chunk_id(
                    evidence_id=evidence_id,
                    chunk_index=chunk_index,
                    content=chunk_text,
                )

                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        evidence_id=evidence_id,
                        source=evidence.source,
                        source_type=evidence.source_type,
                        title=evidence.title,
                        url=evidence.url,
                        content=chunk_text,
                        metadata=dict(evidence.metadata),
                    )
                )

            if end >= len(content):
                break

            start += step
            chunk_index += 1

        return chunks

    @staticmethod
    def _chunk_id(
        evidence_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        """Generate a deterministic ID for a chunk."""

        raw = (
            f"{evidence_id}|"
            f"{chunk_index}|"
            f"{content}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:16]

        return (
            f"{evidence_id}:"
            f"{chunk_index}:"
            f"{digest}"
        )

    @staticmethod
    def _fallback_evidence_id(
        evidence: Evidence,
    ) -> str:
        """Generate a stable ID if Evidence has no evidence_id."""

        raw = "|".join(
            [
                evidence.source,
                evidence.source_type,
                evidence.title,
                evidence.url or "",
                evidence.content,
            ]
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:16]