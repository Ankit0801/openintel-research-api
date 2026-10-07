import logging

from fastapi import FastAPI, HTTPException

from app.graph import build_research_graph
from app.models import ResearchRequest, ResearchReport


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("openintel.api")


app = FastAPI(
    title="OpenIntel Research API",
    description="Evidence-backed multi-agent research API",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "openintel-research-api",
    }


@app.post(
    "/research",
    response_model=ResearchReport,
)
def research(request: ResearchRequest) -> ResearchReport:
    logger.info(
        "Research request received: question=%r sources=%s max_sources=%s",
        request.question,
        request.preferred_sources,
        request.max_sources,
    )

    try:
        graph = build_research_graph()

        result = graph.invoke(
            {
                "request": request,
                "research_plan": None,
                "selected_sources": [],
                "source_queries": {},
                "raw_evidence": [],
                "processed_evidence": [],
                "validation_result": None,
                "retrieved_chunks": [],
                "report": None,
                "errors": [],
            }
        )

        report = result.get("report")

        if report is None:
            errors = result.get("errors", [])
            detail = "; ".join(errors) if errors else "Research failed"

            logger.error("Research graph failed: %s", detail)

            raise HTTPException(
                status_code=500,
                detail=detail,
            )

        logger.info("Research completed successfully")

        return report

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected research API failure")

        raise HTTPException(
            status_code=500,
            detail="Internal research service error",
        )
