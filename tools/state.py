"""Pydantic models and LangGraph state schemas for the research workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A planned or completed section of the final report."""

    name: str
    description: str
    research: bool
    content: str = ""


class Sections(BaseModel):
    """Structured output containing the report plan."""

    sections: list[Section]


class SearchQuery(BaseModel):
    search_query: str


class Queries(BaseModel):
    queries: list[SearchQuery]


class Feedback(BaseModel):
    grade: Literal["pass", "fail"]
    follow_up_queries: list[SearchQuery] = Field(default_factory=list)


class ReportStateInput(TypedDict):
    topic: str


class ReportStateOutput(TypedDict):
    final_report: str


class ReportState(TypedDict, total=False):
    topic: str
    feedback_on_report_plan: str
    sections: list[Section]
    completed_sections: Annotated[list[Section], operator.add]
    report_sections_from_research: str
    final_report: str


class SectionState(TypedDict, total=False):
    topic: str
    section: Section
    search_iterations: int
    search_queries: list[SearchQuery]
    source_str: str
    report_sections_from_research: str
    completed_sections: list[Section]


class SectionOutputState(TypedDict):
    completed_sections: list[Section]
