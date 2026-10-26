"""Retrieve relevant chunks from an existing persistent Chroma database."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def retrieve_from_db(topic: str, persist_dir: str, k: int = 10) -> str:
    """Return the top-k matching chunks with source, page, and author metadata."""

    database_path = Path(persist_dir).expanduser().resolve()
    if not database_path.is_dir() or not any(database_path.iterdir()):
        raise FileNotFoundError(
            f"No se encontró una base Chroma persistida en {database_path}. "
            "Créala primero con rag.create_db."
        )
    if k < 1:
        raise ValueError("k debe ser un entero mayor que cero.")

    vector_store = Chroma(
        persist_directory=str(database_path),
        embedding_function=OpenAIEmbeddings(),
        collection_name="deep_research_documents",
    )
    documents = vector_store.similarity_search(topic, k=k)
    if not documents:
        return "No se encontraron documentos relevantes para el tema indicado."

    formatted_documents: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source") or "Fuente desconocida"
        page = metadata.get("page")
        if isinstance(page, int):
            page = page + 1  # PyPDFDirectoryLoader representa las páginas desde cero.
        page_text = str(page) if page is not None else "desconocida"
        author = metadata.get("author") or metadata.get("Author") or "Desconocido"
        formatted_documents.append(
            f"[{index}] Fuente: {source}, página: {page_text}, autor: {author}. "
            f"Contenido: {document.page_content.strip()}"
        )

    return "\n\n".join(formatted_documents)
