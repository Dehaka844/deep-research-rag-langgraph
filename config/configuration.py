"""Configuration shared by the report graph and its nodes."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Configuration:
    """Runtime settings supplied in ``RunnableConfig.configurable``.

    ``report_structure`` and ``number_of_queries`` are intentionally required:
    the exercise refers to them but does not prescribe values.
    """

    report_structure: str
    number_of_queries: int
    search_api: str = "exa"
    search_api_config: dict[str, Any] = field(default_factory=dict)
    planner_provider: str = "openai"
    planner_model: str = "gpt-4.1"
    writer_provider: str = "openai"
    writer_model: str = "gpt-4.1"
    max_search_depth: int = 3

    @classmethod
    def from_runnable_config(cls, config: dict[str, Any] | None) -> "Configuration":
        """Build settings while ignoring LangGraph keys such as ``thread_id``."""

        configurable = (config or {}).get("configurable", {})
        accepted_fields = {item.name for item in fields(cls)}
        values = {
            key: value
            for key, value in configurable.items()
            if key in accepted_fields
        }
        return cls(**values)
