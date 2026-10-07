"""Phase 4 entry point: validate a trip request and create a typed planning brief."""

from datetime import date
from decimal import Decimal
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from langchain_google_genai.chat_models import GoogleAPIError

from app.config import get_settings
from app.llm import get_llm
from app.models import TravelPlanningBrief, TravelRequest


def sample_request() -> TravelRequest:
    """Represent one user's trip request as validated application data."""
    return TravelRequest(
        origin="Kolkata",
        destination="Japan",
        start_date=date(2027, 4, 5),
        end_date=date(2027, 4, 12),
        travelers=2,
        budget=Decimal("200000"),
        currency="INR",
        preferences=["food", "culture", "moderate daily pace"],
    )


def main() -> None:
    """Create a structured LLM planning brief from a validated trip request."""
    settings = get_settings()
    llm = get_llm(settings)
    request = sample_request()
    planner = llm.with_structured_output(
        TravelPlanningBrief,
        method="json_schema",
        include_raw=True,
    )
    messages = [
        SystemMessage(
            content=(
                "You extract travel requirements into a planning brief. "
                "Do not invent live prices, schedules, or availability. "
                "Ask questions only when missing information would materially change the plan."
            )
        ),
        HumanMessage(
            content="Create a planning brief for this validated request:\n"
            + json.dumps(request.model_dump(mode="json"), indent=2)
        ),
    ]

    try:
        result = planner.invoke(messages)
    except (GoogleAPIError, OutputParserException) as error:
        print(f"Gemini request failed: {error}")
        print("Retry later if Gemini reports a temporary 503, 504, or malformed response.")
        return

    response = result["parsed"]
    if response is None:
        print("Gemini returned an unreadable structured response. Please retry the request.")
        return

    print(f"Model: {settings.gemini_model}")
    print(f"Validated request: {request.model_dump_json(indent=2)}")
    print(f"Planning brief: {response.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
