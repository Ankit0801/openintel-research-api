"""LangGraph workflow for OpenIntel research."""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.evidence.processor import EvidenceProcessor
from app.evidence.validator import EvidenceValidator
from app.models import Evidence, ResearchPlan, ResearchRequest
from app.query.builder import QueryBuilder
from app.rag.retriever import RAGRetriever
from app.report import generate_research_report
from app.sources.arxiv import ArxivClient
from app.sources.github import GitHubClient
from app.sources.nvd import NVDClient
from app.sources.openalex import OpenAlexClient


class ResearchState(TypedDict):
    """State carried through the OpenIntel research workflow."""

    request: ResearchRequest
    research_plan: ResearchPlan | None
    selected_sources: list[str]
    source_queries: dict[str, str]

    raw_evidence: Annotated[
        list[Evidence],
        add,
    ]

    processed_evidence: list[Evidence]

    validation_result: object | None

    retrieved_chunks: list

    report: object | None

    errors: Annotated[
        list[str],
        add,
    ]


# def source_router_node(
#     state: ResearchState,
# ) -> dict:
#     """Select supported sources and build source-specific queries."""

#     requested_sources = {
#         source.strip().lower()
#         for source in state["request"].preferred_sources
#         if source.strip()
#     }

#     supported_sources = {
#         "github",
#         "arxiv",
#         "openalex",
#         "nvd",
#         "web",
#     }

#     selected_sources = [
#         source
#         for source in requested_sources
#         if source in supported_sources
#     ]

#     queries = QueryBuilder().build(
#         state["request"]
#     )

#     source_queries = {
#         "github": queries.github,
#         "arxiv": queries.arxiv,
#         "openalex": queries.openalex,
#         "nvd": queries.nvd,
#     }

#     return {
#         "selected_sources": selected_sources,
#         "source_queries": source_queries,
#     }

def source_router_node(
    state: ResearchState,
) -> dict:
    """Select supported sources and build source-specific queries."""

    requested_sources = []

    for source in state["request"].preferred_sources:
        normalized = source.strip().lower()

        if normalized and normalized not in requested_sources:
            requested_sources.append(normalized)

    supported_sources = {
        "github",
        "arxiv",
        "openalex",
        "nvd",
    }

    selected_sources = [
        source
        for source in requested_sources
        if source in supported_sources
    ][: state["request"].max_sources]

    queries = QueryBuilder().build(state["request"])

    source_queries = {
        "github": queries.github,
        "arxiv": queries.arxiv,
        "openalex": queries.openalex,
        "nvd": queries.nvd,
    }

    return {
        "selected_sources": selected_sources,
        "source_queries": source_queries,
    }

def github_retrieval_node(
    state: ResearchState,
) -> dict:
    """Retrieve GitHub evidence using the generated GitHub query."""

    if "github" not in state["selected_sources"]:
        return {
            "raw_evidence": [],
        }

    try:
        client = GitHubClient()

        query = state["source_queries"]["github"]

        evidence = client.search_repositories(
            query=query,
            per_page=5,
        )

        return {
            "raw_evidence": evidence,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "raw_evidence": [],
            "errors": [
                f"GitHub retrieval failed: {error}"
            ],
        }


def arxiv_retrieval_node(
    state: ResearchState,
) -> dict:
    """Retrieve arXiv evidence using the generated arXiv query."""

    if "arxiv" not in state["selected_sources"]:
        return {
            "raw_evidence": [],
        }

    try:
        client = ArxivClient()

        query = state["source_queries"]["arxiv"]

        evidence = client.search_papers(
            query=query,
            max_results=5,
        )

        return {
            "raw_evidence": evidence,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "raw_evidence": [],
            "errors": [
                f"arXiv retrieval failed: {error}"
            ],
        }


def openalex_retrieval_node(
    state: ResearchState,
) -> dict:
    """Retrieve OpenAlex evidence using the generated query."""

    if "openalex" not in state["selected_sources"]:
        return {
            "raw_evidence": [],
        }

    try:
        client = OpenAlexClient()

        query = state["source_queries"]["openalex"]

        evidence = client.search_works(
            query=query,
            per_page=5,
        )

        return {
            "raw_evidence": evidence,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "raw_evidence": [],
            "errors": [
                f"OpenAlex retrieval failed: {error}"
            ],
        }


def nvd_retrieval_node(
    state: ResearchState,
) -> dict:
    """Retrieve NVD security evidence using the generated query."""

    if "nvd" not in state["selected_sources"]:
        return {
            "raw_evidence": [],
        }

    try:
        client = NVDClient()

        query = state["source_queries"]["nvd"]

        evidence = client.search_cves(
            keyword=query,
            results_per_page=5,
        )

        return {
            "raw_evidence": evidence,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "raw_evidence": [],
            "errors": [
                f"NVD retrieval failed: {error}"
            ],
        }


def evidence_processing_node(
    state: ResearchState,
) -> dict:
    """Score, deduplicate, and rank retrieved evidence."""

    try:
        processor = EvidenceProcessor()

        processed = processor.process(
            evidence=state["raw_evidence"],
            question=state["request"].question,
            max_results=15,
        )

        return {
            "processed_evidence": processed,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "processed_evidence": [],
            "errors": [
                f"Evidence processing failed: {error}"
            ],
        }


def rag_retrieval_node(
    state: ResearchState,
) -> dict:
    """Index processed evidence and retrieve the most relevant chunks."""

    try:
        retriever = RAGRetriever()

        retriever.index_evidence(
            state["processed_evidence"]
        )

        retrieved_chunks = retriever.retrieve(
            state["request"].question,
            k=5,
        )

        return {
            "retrieved_chunks": retrieved_chunks,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "retrieved_chunks": [],
            "errors": [
                f"RAG retrieval failed: {error}"
            ],
        }

def cross_source_validation_node(
    state: ResearchState,
) -> dict:
    """Validate processed evidence across independent sources."""

    try:
        validator = EvidenceValidator()

        validation_result = validator.validate(
            state["processed_evidence"]
        )

        return {
            "validation_result": validation_result,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "validation_result": None,
            "errors": [
                f"Cross-source validation failed: {error}"
            ],
        }
    
def report_generation_node(
    state: ResearchState,
) -> dict:
    """Generate a structured research report from retrieved chunks."""

    try:
        settings = get_settings()

        report = generate_research_report(
            question=state["request"].question,
            retrieved_chunks=state["retrieved_chunks"],
            settings=settings,
            validation_result=state["validation_result"],
        )

        return {
            "report": report,
        }

    except (RuntimeError, ValueError) as error:
        return {
            "report": None,
            "errors": [
                f"Report generation failed: {error}"
            ],
        }


def build_research_graph():
    """Build the deterministic multi-source OpenIntel graph."""

    graph = StateGraph(ResearchState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    graph.add_node(
        "source_router",
        source_router_node,
    )

    graph.add_node(
        "github_retrieval",
        github_retrieval_node,
    )

    graph.add_node(
        "arxiv_retrieval",
        arxiv_retrieval_node,
    )

    graph.add_node(
        "openalex_retrieval",
        openalex_retrieval_node,
    )

    graph.add_node(
        "nvd_retrieval",
        nvd_retrieval_node,
    )

    graph.add_node(
        "evidence_processing",
        evidence_processing_node,
    )

    graph.add_node(
        "cross_source_validation",
        cross_source_validation_node,
    )
    graph.add_node(
        "rag_retrieval",
        rag_retrieval_node,
    )

    graph.add_node(
        "report_generation",
        report_generation_node,
    )

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    graph.add_edge(
        START,
        "source_router",
    )

    # ------------------------------------------------------------------
    # Parallel retrieval
    # ------------------------------------------------------------------

    graph.add_edge(
        "source_router",
        "github_retrieval",
    )

    graph.add_edge(
        "source_router",
        "arxiv_retrieval",
    )

    graph.add_edge(
        "source_router",
        "openalex_retrieval",
    )

    graph.add_edge(
        "source_router",
        "nvd_retrieval",
    )

    # ------------------------------------------------------------------
    # Retrieval -> Evidence Intelligence
    #
    # All four retrieval branches converge here.
    # raw_evidence uses the `add` reducer, so all evidence is accumulated
    # before evidence_processing runs.
    # ------------------------------------------------------------------

    graph.add_edge(
        "github_retrieval",
        "evidence_processing",
    )

    graph.add_edge(
        "arxiv_retrieval",
        "evidence_processing",
    )

    graph.add_edge(
        "openalex_retrieval",
        "evidence_processing",
    )

    graph.add_edge(
        "nvd_retrieval",
        "evidence_processing",
    )

    # ------------------------------------------------------------------
    # Evidence -> RAG -> Report
    # ------------------------------------------------------------------

    graph.add_edge(
        "evidence_processing",
        "cross_source_validation",
    )

    graph.add_edge(
        "cross_source_validation",
        "rag_retrieval",
    )

    graph.add_edge(
        "rag_retrieval",
        "report_generation",
    )

    # ------------------------------------------------------------------
    # End
    # ------------------------------------------------------------------

    graph.add_edge(
        "report_generation",
        END,
    )

    return graph.compile()