"""Node implementations for the outer report graph and per-section subgraph."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from exa_py import Exa
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, Send, interrupt

from config.configuration import Configuration, PROJECT_ROOT
from prompts.prompts import (
    final_section_prompt,
    query_planning_prompt,
    report_planning_prompt,
    research_section_prompt,
    section_queries_prompt,
    section_review_prompt,
)
from rag.get_data import retrieve_from_db
from tools.state import Feedback, Queries, Section, Sections


def _settings(config: RunnableConfig | None) -> Configuration:
    return Configuration.from_runnable_config(config)


def _model(settings: Configuration, role: str):
    if role == "planner":
        provider, model_name = settings.planner_provider, settings.planner_model
    else:
        provider, model_name = settings.writer_provider, settings.writer_model
    return init_chat_model(model_name, model_provider=provider)


def _exa_client() -> Exa:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("Falta EXA_API_KEY en el entorno o en el archivo .env.")
    return Exa(api_key=api_key)


async def _search_exa(query: str, settings: Configuration) -> list[dict[str, Any]]:
    """Call the synchronous Exa SDK off the event loop using its async node API."""

    options = dict(settings.search_api_config)
    options.setdefault("num_results", 10)
    options.setdefault("text", True)
    response = await asyncio.to_thread(_exa_client().search_and_contents, query, **options)
    results: list[dict[str, Any]] = []
    for result in response.results:
        results.append(
            {
                "title": result.title or "Sin título",
                "url": result.url or "URL desconocida",
                "author": result.author or "Autor desconocido",
                "published_date": result.published_date or "Fecha desconocida",
                "text": result.text or "",
            }
        )
    return results


def _format_web_sources(
    query_results: list[tuple[str, list[dict[str, Any]]]],
    source_offset: int = 0,
) -> str:
    formatted: list[str] = []
    source_number = source_offset + 1
    for query, results in query_results:
        for result in results:
            formatted.append(
                f"[{source_number}] Consulta: {query}\n"
                f"Título: {result['title']}\nURL: {result['url']}\n"
                f"Autor: {result['author']}\nFecha: {result['published_date']}\n"
                f"Contenido: {result['text']}"
            )
            source_number += 1
    return "\n\n".join(formatted) or "No se obtuvieron resultados web."


def _clean_section_content(content: str, section_name: str) -> str:
    """Remove model-added code fences and a repeated section heading."""

    lines = content.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]

    expected_title = re.sub(r"\s+", " ", section_name).strip().casefold()
    while lines:
        if lines[0].strip():
            heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", lines[0].strip())
            if heading:
                actual_title = re.sub(r"\s+", " ", heading.group(1)).strip().casefold()
                if actual_title == expected_title:
                    lines.pop(0)
                    continue
            break
        lines.pop(0)
    return "\n".join(lines).strip()


async def _run_queries(
    queries: list[str], settings: Configuration
) -> list[tuple[str, list[dict[str, Any]]]]:
    if settings.search_api.lower() != "exa":
        raise NotImplementedError(
            f"El proveedor de búsqueda {settings.search_api!r} no está implementado; "
            "el enunciado y la configuración de ejecución especifican Exa."
        )
    responses = await asyncio.gather(
        *(_search_exa(query, settings) for query in queries)
    )
    return list(zip(queries, responses, strict=True))


async def generate_report_plan(
    state: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    """Create planning queries, research the topic, and produce report sections."""

    settings = _settings(config)
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan")

    query_model = _model(settings, "writer").with_structured_output(Queries)
    query_result = await query_model.ainvoke(
        query_planning_prompt(topic, settings.number_of_queries, feedback)
    )
    queries = [item.search_query for item in query_result.queries][
        : settings.number_of_queries
    ]
    if not queries:
        raise ValueError("El modelo no generó consultas para planificar el informe.")
    search_results = await _run_queries(queries, settings)

    planner = _model(settings, "planner").with_structured_output(Sections)
    plan = await planner.ainvoke(
        report_planning_prompt(
            topic=topic,
            report_structure=settings.report_structure,
            search_context=_format_web_sources(search_results),
            feedback=feedback,
        )
    )
    return {"sections": plan.sections, "feedback_on_report_plan": ""}


def human_feedback(state: dict[str, Any]) -> Command:
    """Pause for plan review, then dispatch researched sections in parallel."""

    plan_text = "\n".join(
        f"{index}. {section.name}: {section.description} "
        f"({'investigación web' if section.research else 'redacción final'})"
        for index, section in enumerate(state.get("sections", []), start=1)
    )
    response = interrupt(
        "Revisa el plan del informe. Responde True para aprobarlo o escribe comentarios "
        "para regenerarlo.\n\n" + plan_text
    )
    if response is True:
        sends = [
            Send(
                "build_section_with_web_research",
                {
                    "topic": state["topic"],
                    "section": section,
                    "search_iterations": 0,
                    "search_queries": [],
                    "source_str": "",
                },
            )
            for section in state.get("sections", [])
            if section.research
        ]
        return Command(goto=sends or "get_rag_data")
    if isinstance(response, str):
        return Command(
            update={"feedback_on_report_plan": response},
            goto="generate_report_plan",
        )
    raise ValueError("La revisión debe ser True para aprobar o texto para pedir cambios.")


async def generate_queries(
    state: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    settings = _settings(config)
    section = state["section"]
    query_model = _model(settings, "writer").with_structured_output(Queries)
    result = await query_model.ainvoke(
        section_queries_prompt(state["topic"], section.name, section.description)
    )
    queries = result.queries[:1]
    if not queries:
        raise ValueError(f"No se generó una consulta para la sección {section.name!r}.")
    return {"search_queries": queries}


async def search_web(
    state: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    settings = _settings(config)
    queries = [item.search_query for item in state.get("search_queries", [])]
    results = await _run_queries(queries, settings)
    previous_sources = state.get("source_str", "")
    previous_source_count = sum(
        1 for line in previous_sources.splitlines() if line.startswith("[") and "] Consulta:" in line
    )
    new_sources = _format_web_sources(results, source_offset=previous_source_count)
    combined_sources = "\n\n".join(
        source for source in (previous_sources, new_sources) if source
    )
    return {
        "source_str": combined_sources,
        "search_iterations": state.get("search_iterations", 0) + 1,
    }


async def write_section(
    state: dict[str, Any], config: RunnableConfig
) -> Command:
    settings = _settings(config)
    section = state["section"]
    sources = state.get("source_str", "")
    writer = _model(settings, "writer")
    draft = await writer.ainvoke(
        research_section_prompt(
            topic=state["topic"],
            section_name=section.name,
            description=section.description,
            sources=sources,
        )
    )
    updated_section = section.model_copy(
        update={"content": _clean_section_content(draft.content, section.name)}
    )

    reviewer = _model(settings, "planner").with_structured_output(Feedback)
    review = await reviewer.ainvoke(
        section_review_prompt(
            topic=state["topic"],
            section_name=section.name,
            description=section.description,
            content=updated_section.content,
            sources=sources,
        )
    )
    at_search_limit = state.get("search_iterations", 0) >= settings.max_search_depth
    follow_up_queries = review.follow_up_queries[:1]
    if review.grade == "pass" or at_search_limit or not follow_up_queries:
        return Command(
            update={"section": updated_section, "completed_sections": [updated_section]},
            goto=END,
        )
    return Command(
        update={
            "section": updated_section,
            "search_queries": follow_up_queries,
        },
        goto="search_web",
    )


def get_rag_data(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve internal PDF evidence and add it as a non-research section."""

    persist_dir = PROJECT_ROOT / "rag" / "chroma_db"
    content = retrieve_from_db(state["topic"], str(persist_dir), k=10)
    rag_section = Section(
        name="Datos desde archivos",
        description="PDFs",
        research=False,
        content=content,
    )
    return {"completed_sections": [rag_section]}


def gather_completed_sections(state: dict[str, Any]) -> dict[str, Any]:
    completed = state.get("completed_sections", [])
    report_sections = "\n\n".join(
        f"## {section.name}\n{section.content}" for section in completed
    )
    return {"report_sections_from_research": report_sections}


async def write_final_sections(
    state: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    settings = _settings(config)
    section = state["section"]
    response = await _model(settings, "writer").ainvoke(
        final_section_prompt(
            topic=state["topic"],
            section_name=section.name,
            description=section.description,
            report_context=state.get("report_sections_from_research", ""),
        )
    )
    completed_section = section.model_copy(
        update={
            "content": _clean_section_content(response.content, section.name)
        }
    )
    return {"completed_sections": [completed_section]}


def compile_final_report(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble planned sections in order, then append additional sections such as RAG."""

    completed_by_name = {
        section.name: section.content for section in state.get("completed_sections", [])
    }
    ordered_names: set[str] = set()
    ordered_sections: list[str] = []
    for planned_section in state.get("sections", []):
        content = _clean_section_content(
            completed_by_name.get(planned_section.name, planned_section.content),
            planned_section.name,
        )
        ordered_sections.append(f"## {planned_section.name}\n{content}")
        ordered_names.add(planned_section.name)
    additional_sections: list[str] = []
    for completed_section in state.get("completed_sections", []):
        if completed_section.name not in ordered_names:
            additional_sections.append(
                f"## {completed_section.name}\n{completed_section.content}"
            )
            ordered_names.add(completed_section.name)
    conclusion_index = next(
        (
            index
            for index, planned_section in enumerate(state.get("sections", []))
            if "conclu" in planned_section.name.casefold()
        ),
        len(ordered_sections),
    )
    ordered_sections[conclusion_index:conclusion_index] = additional_sections
    return {"final_report": "\n\n".join(ordered_sections)}


def initiate_final_section_writing(state: dict[str, Any]) -> list[Send]:
    """Fan out work for plan sections that do not require web research."""

    return [
        Send(
            "write_final_sections",
            {
                "topic": state["topic"],
                "section": section,
                "report_sections_from_research": state.get(
                    "report_sections_from_research", ""
                ),
            },
        )
        for section in state.get("sections", [])
        if not section.research
    ]
