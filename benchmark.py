"""Measure RepoSage indexing, retrieval, and Gemini response timings.

Examples:
    python benchmark.py --pdf AI_Engineering_Building_Applications_-_Chip_Huyen.pdf
    python benchmark.py --pdf book.pdf --question "What is RAG?"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


COLLECTION_NAME = "reposage_benchmark"


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(content, list):
        return "\n\n".join(
            block.get("text", "").strip()
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block.get("text", "").strip()
        )
    return ""


def build_index(pdf_path: Path, index_dir: Path) -> tuple[Chroma, dict]:
    started = time.perf_counter()
    text = read_pdf(pdf_path)
    extraction_seconds = time.perf_counter() - started

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(
        [Document(page_content=text, metadata={"source": pdf_path.name})]
    )

    embedding_started = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(index_dir),
        collection_name=COLLECTION_NAME,
    )
    total_seconds = time.perf_counter() - started

    return store, {
        "file": pdf_path.name,
        "file_size_mb": round(pdf_path.stat().st_size / 1_048_576, 2),
        "pages": len(PdfReader(str(pdf_path)).pages),
        "characters_extracted": len(text),
        "chunks": len(chunks),
        "extraction_seconds": round(extraction_seconds, 3),
        "embedding_and_index_seconds": round(
            time.perf_counter() - embedding_started, 3
        ),
        "cold_index_seconds": round(total_seconds, 3),
    }


def warm_load(index_dir: Path, embeddings) -> tuple[Chroma, float, int]:
    started = time.perf_counter()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(index_dir),
        embedding_function=embeddings,
    )
    seconds = time.perf_counter() - started
    return store, seconds, store._collection.count()


def benchmark_question(store: Chroma, question: str, model_name: str) -> dict:
    started = time.perf_counter()
    sources = store.similarity_search(question, k=3)
    retrieval_seconds = time.perf_counter() - started
    context = "\n\n---\n\n".join(doc.page_content for doc in sources)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from the supplied context. If unavailable, say so.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    first_token = None
    output_parts: list[str] = []
    for chunk in (prompt | llm).stream({"context": context, "question": question}):
        text = clean_text(chunk.content)
        if text:
            first_token = first_token or time.perf_counter()
            output_parts.append(text)

    finished = time.perf_counter()
    return {
        "question": question,
        "retrieved_chunks": len(sources),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "time_to_first_token_seconds": round(
            (first_token or finished) - started, 3
        ),
        "total_answer_seconds": round(finished - started, 3),
        "answer_characters": len("".join(output_parts)),
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--question")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    parser.add_argument("--index-dir", type=Path, default=Path(".reposage_benchmark"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_metrics.json"))
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if args.index_dir.exists():
        shutil.rmtree(args.index_dir)

    store, metrics = build_index(args.pdf, args.index_dir)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    _, warm_seconds, warm_chunks = warm_load(args.index_dir, embeddings)
    metrics["warm_index_load_seconds"] = round(warm_seconds, 3)
    metrics["warm_index_chunks"] = warm_chunks

    if args.question:
        if not os.getenv("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY is required when --question is used")
        metrics["question"] = benchmark_question(store, args.question, args.model)

    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
