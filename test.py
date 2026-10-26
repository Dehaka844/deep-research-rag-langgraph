"""Run the required end-to-end report workflow with human approval automated."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from graph.graph import build_report_graph


TOPIC = "What are the gold & silver market predictions and analysis for 2025?"

REPORT_STRUCTURE = """Crea exactamente estas cinco secciones en este orden:
1. Introducción (research=False): presenta el alcance del informe y el análisis de oro y plata.
2. Previsiones del oro para 2025 (research=True): analiza las previsiones y los argumentos citados.
3. Previsiones de la plata para 2025 (research=True): analiza las previsiones y los argumentos citados.
4. Factores y escenarios compartidos (research=True): compara los factores económicos, la demanda,
   los riesgos y los escenarios relevantes para ambos metales.
5. Conclusiones (research=False): sintetiza los hallazgos sin añadir datos nuevos.
No incluyas una sección sobre los PDFs; el grafo la añadirá mediante RAG."""

EXPECTED_RESEARCH_FLAGS = [False, True, True, True, False]


def _validate_plan(sections) -> None:
    actual_flags = [section.research for section in sections]
    if actual_flags != EXPECTED_RESEARCH_FLAGS:
        raise RuntimeError(
            "El plan no respeta la estructura de prueba esperada. "
            "Se detiene antes de aprobarlo para evitar búsquedas adicionales. "
            f"research flags recibidos: {actual_flags}"
        )


async def run_report() -> str:
    project_root = Path(__file__).resolve().parent
    pdf_dir = project_root / "rag" / "pdfs"
    persist_dir = project_root / "rag" / "chroma_db"
    if not any(pdf_dir.rglob("*.pdf")):
        raise FileNotFoundError(
            f"No hay PDFs en {pdf_dir}. Añade los documentos internos antes de ejecutar."
        )
    if not persist_dir.is_dir() or not any(persist_dir.iterdir()):
        raise FileNotFoundError(
            "No existe una base Chroma inicializada. Créala primero con "
            "`python -m rag.create_db`. El test se detiene antes de usar Exa."
        )

    checkpoint_serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[("tools.state", "Section")]
    )
    workflow = build_report_graph(
        checkpointer=MemorySaver(serde=checkpoint_serializer)
    )
    run_config = {
        "configurable": {
            "thread_id": str(uuid4()),
            "search_api": "exa",
            "search_api_config": {
                "num_results": 15,
                "include_domains": ["eleconomista.es", "expansion.com"],
                "subpages": 5,
            },
            "planner_provider": "openai",
            "planner_model": "gpt-4.1",
            "writer_provider": "openai",
            "writer_model": "gpt-4.1",
            "max_search_depth": 2,
            "report_structure": REPORT_STRUCTURE,
            "number_of_queries": 2,
        }
    }

    print("Iniciando el flujo; el plan se aprobará automáticamente al interrumpirse.")
    async for update in workflow.astream(
        {"topic": TOPIC}, config=run_config, stream_mode="updates"
    ):
        print(f"Nodos ejecutados: {', '.join(update.keys())}")
        if "__interrupt__" in update:
            current_state = workflow.get_state(run_config).values
            _validate_plan(current_state.get("sections", []))
            print("Plan aprobado automáticamente. Reanudando la investigación.")
            async for resumed_update in workflow.astream(
                Command(resume=True), config=run_config, stream_mode="updates"
            ):
                print(f"Nodos ejecutados: {', '.join(resumed_update.keys())}")
            break

    final_state = workflow.get_state(run_config).values
    final_report = final_state.get("final_report")
    if not final_report:
        raise RuntimeError("El flujo terminó sin producir final_report.")
    print("\nInforme final:\n")
    print(final_report)
    return final_report


if __name__ == "__main__":
    asyncio.run(run_report())
