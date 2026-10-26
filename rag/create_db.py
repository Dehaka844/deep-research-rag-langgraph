"""Load PDFs, split them, embed the chunks, and persist them in Chroma."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings

from config.configuration import PROJECT_ROOT


def create_vector_db(
    pdf_dir: str | Path,
    persist_dir: str | Path,
) -> dict[str, int]:
    """Create or update a persistent Chroma collection from a PDF directory."""

    pdf_path = Path(pdf_dir).expanduser().resolve()
    database_path = Path(persist_dir).expanduser().resolve()
    if not pdf_path.is_dir():
        raise FileNotFoundError(f"No existe el directorio de PDFs: {pdf_path}")
    if not any(pdf_path.rglob("*.pdf")):
        raise FileNotFoundError(f"No se encontraron archivos PDF en: {pdf_path}")

    loader = PyPDFDirectoryLoader(str(pdf_path), recursive=True)
    documents = loader.load_and_split()
    if not documents:
        raise ValueError(f"No se pudieron extraer documentos desde: {pdf_path}")

    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(database_path),
        collection_name="deep_research_documents",
    )
    return {"status": 200}


def main() -> None:
    parser = argparse.ArgumentParser(description="Crear la base vectorial Chroma desde PDFs.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PROJECT_ROOT / "rag" / "pdfs",
        help="Directorio que contiene los PDFs de entrada.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=PROJECT_ROOT / "rag" / "chroma_db",
        help="Directorio donde se persistirá Chroma.",
    )
    args = parser.parse_args()
    result: dict[str, Any] = create_vector_db(args.pdf_dir, args.persist_dir)
    print(result)


if __name__ == "__main__":
    main()
