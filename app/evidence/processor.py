"""Deterministic evidence intelligence for OpenIntel."""

import hashlib
import re
from datetime import datetime, timezone

from app.models import Evidence


class EvidenceProcessor:
    """Score, deduplicate, and rank research evidence."""

    STOPWORDS = {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "this",
        "that",
        "with",
        "from",
        "into",
        "about",
        "what",
        "when",
        "where",
        "which",
        "how",
        "why",
        "does",
        "can",
        "could",
        "would",
        "should",
        "have",
        "has",
        "had",
        "its",
        "their",
        "there",
        "they",
        "them",
        "these",
        "those",
        "using",
        "use",
        "used",
        "based",
        "such",
        "evolving",
    }

    CONCEPT_GROUPS = {
        "ai_agent": {
            "ai agent",
            "ai agents",
            "agentic ai",
            "agentic artificial intelligence",
            "artificial intelligence agent",
            "artificial intelligence agents",
            "ai agent system",
            "ai agent systems",
            "llm agent",
            "llm agents",
            "autonomous agent",
            "autonomous agents",
            "autonomous ai agent",
            "autonomous ai agents",
            "intelligent agent",
            "intelligent agents",
        },
        "open_source": {
            "open source",
            "open-source",
            "open sourced",
            "open-sourced",
        },
        "agentic": {
            "agentic",
            "agentic ai",
            "agentic system",
            "agentic systems",
            "agent framework",
            "agent frameworks",
            "agent runtime",
            "agent runtimes",
        },
        "rag": {
            "retrieval augmented generation",
            "retrieval-augmented generation",
            "rag",
        },
        "llm": {
            "large language model",
            "large language models",
            "llm",
            "llms",
        },
    }

    SOURCE_RELEVANCE_MULTIPLIERS = {
        "github": 1.0,
        "arxiv": 1.0,
        "openalex": 1.0,
        "nvd": 0.55,
    }

    def process(
        self,
        evidence: list[Evidence],
        question: str,
        max_results: int = 15,
    ) -> list[Evidence]:
        """Process raw evidence into ranked evidence."""

        if max_results < 1:
            raise ValueError(
                "max_results must be at least 1."
            )

        if not question.strip():
            raise ValueError(
                "Research question cannot be empty."
            )

        scored = [
            self._score_evidence(
                item=item,
                question=question,
            )
            for item in evidence
        ]

        deduplicated = self._deduplicate(
            scored
        )

        ranked = sorted(
            deduplicated,
            key=lambda item: (
                item.combined_score,
                item.relevance_score,
                item.quality_score,
            ),
            reverse=True,
        )

        return ranked[:max_results]

    def _score_evidence(
        self,
        item: Evidence,
        question: str,
    ) -> Evidence:
        """Calculate all deterministic evidence scores."""

        relevance = self._relevance_score(
            question=question,
            evidence=item,
        )

        quality = self._quality_score(
            item
        )

        recency = self._recency_score(
            item
        )

        combined = self._combined_score(
            relevance=relevance,
            quality=quality,
            recency=recency,
        )

        evidence_id = self._generate_evidence_id(
            item
        )

        return item.model_copy(
            update={
                "relevance_score": relevance,
                "quality_score": quality,
                "recency_score": recency,
                "combined_score": combined,
                "evidence_id": evidence_id,
            }
        )

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> set[str]:
        """Convert text into meaningful normalized tokens."""

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) > 2
            and token not in cls.STOPWORDS
        }

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """Normalize text for phrase matching."""

        return re.sub(
            r"\s+",
            " ",
            re.sub(
                r"[^a-zA-Z0-9]+",
                " ",
                text.lower(),
            ),
        ).strip()

    @classmethod
    def _contains_phrase(
        cls,
        text: str,
        phrase: str,
    ) -> bool:
        """Check whether a normalized phrase exists in text."""

        normalized_text = cls._normalize_text(
            text
        )

        normalized_phrase = cls._normalize_text(
            phrase
        )

        return normalized_phrase in normalized_text

    @classmethod
    def _concept_matches(
        cls,
        text: str,
        question: str,
    ) -> set[str]:
        """
        Identify domain concepts shared by the question and evidence.

        Concepts are intentionally deterministic. This avoids using an
        LLM for retrieval ranking and keeps evidence selection reproducible.
        """

        question_lower = question.lower()
        text_lower = text.lower()

        matched: set[str] = set()

        for concept, phrases in cls.CONCEPT_GROUPS.items():
            question_has_concept = any(
                phrase in question_lower
                for phrase in phrases
            )

            if not question_has_concept:
                continue

            evidence_has_concept = any(
                phrase in text_lower
                for phrase in phrases
            )

            if evidence_has_concept:
                matched.add(concept)

        return matched

    @classmethod
    def _ai_agent_context_match(
        cls,
        text: str,
    ) -> bool:
        """
        Detect AI-agent context even when the exact phrase is absent.

        Example:
            "open-source AI agent"

        and:

            "LLMs and AI agents"

        should both qualify.

        A generic phrase such as "SNMP agents" should not.
        """

        normalized = cls._normalize_text(
            text
        )

        tokens = normalized.split()

        ai_terms = {
            "ai",
            "llm",
            "llms",
            "genai",
            "generative",
        }

        agent_terms = {
            "agent",
            "agents",
            "agentic",
        }

        ai_positions = {
            index
            for index, token in enumerate(tokens)
            if token in ai_terms
        }

        agent_positions = {
            index
            for index, token in enumerate(tokens)
            if token in agent_terms
        }

        for ai_index in ai_positions:
            for agent_index in agent_positions:
                if abs(ai_index - agent_index) <= 5:
                    return True

        return False

    def _relevance_score(
        self,
        question: str,
        evidence: Evidence,
    ) -> float:
            """
            Calculate source-aware concept relevance.

            Ranking signals:

            1. Exact domain concepts
            2. AI-agent contextual relationship
            3. Meaningful token overlap
            4. Title overlap
            5. Important phrase overlap
            6. Source-specific penalties

            Concept matching is based on shared concepts rather than requiring
            every concept mentioned in the question to appear in the evidence.
            This prevents a missing secondary concept such as "open source"
            from incorrectly making a directly relevant academic paper score low.
            """

            question_tokens = self._tokenize(
                question
            )

            if not question_tokens:
                return 0.0

            title = evidence.title
            content = evidence.content
            metadata = str(
                evidence.metadata
            )

            combined_text = " ".join(
                [
                    title,
                    content,
                    metadata,
                ]
            )

            evidence_tokens = self._tokenize(
                combined_text
            )

            overlap = (
                question_tokens
                & evidence_tokens
            )

            token_score = (
                len(overlap)
                / len(question_tokens)
            )

            title_tokens = self._tokenize(
                title
            )

            title_overlap = (
                question_tokens
                & title_tokens
            )

            title_score = (
                len(title_overlap)
                / len(question_tokens)
            )

            concept_matches = self._concept_matches(
                combined_text,
                question,
            )

            question_concept_count = (
                self._question_concept_count(
                    question
                )
            )

            concept_score = (
                len(concept_matches)
                / max(
                    question_concept_count,
                    1,
                )
            )

            ai_agent_match = (
                self._ai_agent_context_match(
                    combined_text
                )
                and self._question_mentions_ai_agent(
                    question
                )
            )

            # AI-agent terminology is intentionally treated as a strong
            # semantic concept match. For example:
            #
            # Question:
            #     "open-source AI agent ecosystem"
            #
            # Evidence:
            #     "Agentic Artificial Intelligence"
            #
            # The evidence is directly relevant to the AI-agent concept even
            # though it does not contain the "open source" concept.
            if "ai_agent" in concept_matches:
                concept_score = max(
                    concept_score,
                    0.90,
                )

            if ai_agent_match:
                concept_score = max(
                    concept_score,
                    0.90,
                )

            phrase_score = self._phrase_match_score(
                question=question,
                evidence_text=combined_text,
            )

            relevance = (
                (0.35 * token_score)
                + (0.25 * title_score)
                + (0.25 * concept_score)
                + (0.15 * phrase_score)
            )

            source_multiplier = (
                self.SOURCE_RELEVANCE_MULTIPLIERS.get(
                    evidence.source.lower(),
                    1.0,
                )
            )

            relevance *= source_multiplier

            relevance = self._apply_source_penalties(
                relevance=relevance,
                question=question,
                evidence=evidence,
                concept_matches=concept_matches,
                ai_agent_match=ai_agent_match,
            )

            return round(
                min(max(relevance, 0.0), 1.0),
                4,
            )

    @classmethod
    def _question_mentions_ai_agent(
        cls,
        question: str,
    ) -> bool:
        """Determine whether the question concerns AI agents."""

        normalized = cls._normalize_text(
            question
        )

        if any(
            phrase in normalized
            for phrase in [
                "ai agent",
                "ai agents",
                "agentic ai",
                "agentic artificial intelligence",
                "artificial intelligence agent",
                "artificial intelligence agents",
                "llm agent",
                "llm agents",
                "autonomous agent",
                "autonomous agents",
            ]
        ):
            return True

        return False

    @classmethod
    def _question_concept_count(
        cls,
        question: str,
    ) -> int:
        """Count domain concepts explicitly requested by the question."""

        normalized = question.lower()

        count = 0

        for phrases in cls.CONCEPT_GROUPS.values():
            if any(
                phrase in normalized
                for phrase in phrases
            ):
                count += 1

        return count

    @classmethod
    def _phrase_match_score(
        cls,
        question: str,
        evidence_text: str,
    ) -> float:
        """Score important multi-word phrase matches."""

        normalized_question = cls._normalize_text(
            question
        )

        score = 0.0

        important_phrases = [
            "ai agent",
            "ai agents",
            "agentic ai",
            "agentic artificial intelligence",
            "llm agent",
            "llm agents",
            "autonomous agent",
            "autonomous agents",
            "open source",
            "open-source",
        ]

        matched = 0

        for phrase in important_phrases:
            normalized_phrase = cls._normalize_text(
                phrase
            )

            if (
                normalized_phrase in normalized_question
                and cls._contains_phrase(
                    evidence_text,
                    phrase,
                )
            ):
                matched += 1

        if matched:
            score = min(
                1.0,
                matched / 2,
            )

        return score

    @classmethod
    def _apply_source_penalties(
        cls,
        relevance: float,
        question: str,
        evidence: Evidence,
        concept_matches: set[str],
        ai_agent_match: bool,
    ) -> float:
        """Apply source-specific relevance safeguards."""

        source = evidence.source.lower()

        if source == "nvd":
            question_is_ai_agent = (
                cls._question_mentions_ai_agent(
                    question
                )
            )

            if question_is_ai_agent:
                if not ai_agent_match:
                    relevance *= 0.15

                if (
                    "ai_agent"
                    not in concept_matches
                    and "agentic"
                    not in concept_matches
                ):
                    relevance *= 0.50

        return relevance

    @staticmethod
    def _quality_score(
        evidence: Evidence,
    ) -> float:
        """Calculate source-aware evidence quality."""

        source = evidence.source.lower()

        if source == "github":
            return EvidenceProcessor._github_quality(
                evidence
            )

        if source == "arxiv":
            return EvidenceProcessor._arxiv_quality(
                evidence
            )

        if source == "openalex":
            return EvidenceProcessor._openalex_quality(
                evidence
            )

        if source == "nvd":
            return EvidenceProcessor._nvd_quality(
                evidence
            )

        return 0.5

    @staticmethod
    def _github_quality(
        evidence: Evidence,
    ) -> float:
        """Score GitHub repository quality."""

        metadata = evidence.metadata

        stars = metadata.get(
            "stars",
            0,
        )

        forks = metadata.get(
            "forks",
            0,
        )

        score = 0.2

        if isinstance(stars, int):
            if stars >= 10000:
                score += 0.5
            elif stars >= 1000:
                score += 0.35
            elif stars >= 100:
                score += 0.2

        if (
            isinstance(forks, int)
            and forks >= 100
        ):
            score += 0.1

        if evidence.url:
            score += 0.1

        return round(
            min(score, 1.0),
            4,
        )

    @staticmethod
    def _arxiv_quality(
        evidence: Evidence,
    ) -> float:
        """Score arXiv paper quality."""

        score = 0.4

        if evidence.url:
            score += 0.2

        if len(evidence.content) >= 500:
            score += 0.2

        if "Abstract:" in evidence.content:
            score += 0.2

        return round(
            min(score, 1.0),
            4,
        )

    @staticmethod
    def _openalex_quality(
        evidence: Evidence,
    ) -> float:
        """Score OpenAlex work quality."""

        metadata = evidence.metadata

        citations = metadata.get(
            "cited_by_count",
            0,
        )

        score = 0.4

        if isinstance(citations, int):
            if citations >= 100:
                score += 0.4
            elif citations >= 20:
                score += 0.25
            elif citations >= 5:
                score += 0.15

        if evidence.url:
            score += 0.2

        return round(
            min(score, 1.0),
            4,
        )

    @staticmethod
    def _nvd_quality(
        evidence: Evidence,
    ) -> float:
        """Score NVD vulnerability quality."""

        metadata = evidence.metadata

        score = 0.5

        cvss = metadata.get(
            "cvss_score"
        )

        if isinstance(
            cvss,
            (int, float),
        ):
            score += 0.2

        references = metadata.get(
            "references",
            [],
        )

        if (
            isinstance(references, list)
            and references
        ):
            score += 0.2

        if evidence.url:
            score += 0.1

        return round(
            min(score, 1.0),
            4,
        )

    @staticmethod
    def _recency_score(
        evidence: Evidence,
    ) -> float:
        """Calculate deterministic recency."""

        metadata = evidence.metadata

        date_value = (
            metadata.get("updated_at")
            or metadata.get("updated")
            or metadata.get("published")
            or metadata.get("publication_date")
        )

        if not isinstance(
            date_value,
            str,
        ):
            return 0.5

        try:
            normalized = date_value.replace(
                "Z",
                "+00:00",
            )

            published_date = datetime.fromisoformat(
                normalized
            )

            if published_date.tzinfo is None:
                published_date = published_date.replace(
                    tzinfo=timezone.utc
                )

            now = datetime.now(
                timezone.utc
            )

            age_days = (
                now - published_date
            ).days

            if age_days < 0:
                return 1.0

            if age_days <= 30:
                return 1.0

            if age_days <= 180:
                return 0.9

            if age_days <= 365:
                return 0.8

            if age_days <= 730:
                return 0.6

            if age_days <= 1825:
                return 0.4

            return 0.2

        except ValueError:
            return 0.5

    @staticmethod
    def _combined_score(
        relevance: float,
        quality: float,
        recency: float,
    ) -> float:
        """Combine deterministic evidence signals."""

        score = (
            (0.60 * relevance)
            + (0.30 * quality)
            + (0.10 * recency)
        )

        return round(
            min(score, 1.0),
            4,
        )

    @staticmethod
    def _generate_evidence_id(
        evidence: Evidence,
    ) -> str:
        """Generate a stable identifier for evidence."""

        identity = (
            evidence.source
            + "|"
            + (evidence.url or "")
            + "|"
            + evidence.title.strip().lower()
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _deduplicate(
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """Remove duplicate evidence by stable identity."""

        unique: dict[str, Evidence] = {}

        for item in evidence:
            evidence_id = (
                item.evidence_id
                or EvidenceProcessor._generate_evidence_id(
                    item
                )
            )

            existing = unique.get(
                evidence_id
            )

            if existing is None:
                unique[evidence_id] = item
                continue

            if (
                item.combined_score
                > existing.combined_score
            ):
                unique[evidence_id] = item

        return list(
            unique.values()
        )