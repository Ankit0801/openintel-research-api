"""Provider-specific LLM construction behind one reusable function."""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings


def get_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    """Create the chat model used by the travel-planning application."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
        temperature=0.2,
        max_tokens=2048,
        request_timeout=120,
        retries=2,
    )
