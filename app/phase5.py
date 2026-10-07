"""Phase 5 entry point: run the travel planning LangGraph."""

from datetime import date
from decimal import Decimal

from app.graph import TravelState, build_travel_graph
from app.models import TravelRequest


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
    """Run the travel planning graph."""

    request = sample_request()

    initial_state: TravelState = {
        "request": request,
        "planning_brief": None,
        "error": None,
    }

    graph = build_travel_graph()
    final_state = graph.invoke(initial_state)

    if final_state["error"]:
        print(f"Gemini request failed: {final_state['error']}")
        return

    print("Planning brief:")
    print(final_state["planning_brief"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()