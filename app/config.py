"""Centralized environment-based application configuration."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class Settings:
    """Configuration needed by external services."""

    gemini_api_key: str
    gemini_model: str
    github_token: str | None


def get_settings() -> Settings:
    """Load settings from .env and validate required configuration."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to the local .env file before running the app."
        )

    github_token = os.getenv("GITHUB_TOKEN", "").strip() or None

    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        github_token=github_token,
    )