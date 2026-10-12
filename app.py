"""RepoSage: a minimal document Q&A RAG app."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

st.set_page_config(page_title="RepoSage", page_icon="📚", layout="wide")

PROJECT_DIR = Path(__file__).parent
INDEX_DIR = PROJECT_DIR / ".reposage_chroma"
INDEX_MANIFEST = PROJECT_DIR / ".reposage_index.json"
COLLECTION_NAME = "reposage_documents"


def read_uploaded_file(uploaded_file) -> str:
    """Extract text from a PDF, Markdown, or plain-text upload."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.getvalue().decode("utf-8", errors="replace")


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def file_signature(uploaded_files) -> list[dict[str, Any]]:
    """Create a lightweight signature so unchanged uploads reuse the local index."""
    return [
        {"name": file.name, "size": getattr(file, "size", None)}
        for file in uploaded_files
    ]


def load_cached_store(uploaded_files) -> tuple[Chroma | None, int, bool]:
    if not INDEX_DIR.exists() or not INDEX_MANIFEST.exists():
        return None, 0, False

    try:
        manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("files") != file_signature(uploaded_files):
            return None, 0, False

        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(INDEX_DIR),
            embedding_function=get_embeddings(),
        )
        return store, int(manifest.get("chunk_count", 0)), True
    except Exception:
        return None, 0, False


def build_vector_store(uploaded_files) -> tuple[Chroma, int, bool]:
    cached_store, cached_count, reused = load_cached_store(uploaded_files)
    if reused and cached_store is not None:
        return cached_store, cached_count, True

    documents: list[Document] = []
    for uploaded_file in uploaded_files:
        text = read_uploaded_file(uploaded_file).strip()
        if text:
            documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

    if not documents:
        raise ValueError("The uploaded files did not contain readable text.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(INDEX_DIR),
        collection_name=COLLECTION_NAME,
    )
    INDEX_MANIFEST.write_text(
        json.dumps({"files": file_signature(uploaded_files), "chunk_count": len(chunks)}, indent=2),
        encoding="utf-8",
    )
    return store, len(chunks), False


def clean_model_response(content: Any) -> str:
    """Keep only readable model text; discard type, signature, and metadata."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if isinstance(text, str) else ""

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts)

    return ""


def build_question_chain(store: Chroma, question: str):
    results = store.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in results
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using only the supplied document context. "
                "If the answer is not present, say you could not find it in the uploaded documents. "
                "Be concise and explain your reasoning briefly.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )
    return prompt | llm, {"context": context, "question": question}, results


def stream_answer(store: Chroma, question: str) -> tuple[Iterator[str], list[Document]]:
    """Yield model text as it arrives and return the retrieved source documents."""
    chain, inputs, results = build_question_chain(store, question)

    def token_stream() -> Iterator[str]:
        for chunk in chain.stream(inputs):
            text = clean_model_response(chunk.content)
            if text:
                yield text

    return token_stream(), results


st.title("📚 RepoSage")
st.caption("Ask questions about your PDFs, text files, and Markdown notes.")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Add GOOGLE_API_KEY to a local .env file before asking questions.")

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", type="primary", disabled=not uploaded_files):
        try:
            st.session_state.store, chunk_count, reused = build_vector_store(uploaded_files)
            st.session_state.indexed_names = [file.name for file in uploaded_files]
            message = f"Reused cached index ({chunk_count} chunks)." if reused else f"Indexed {chunk_count} chunks."
            st.success(message)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    if st.session_state.get("indexed_names"):
        st.caption("Indexed: " + ", ".join(st.session_state.indexed_names))

st.header("2. Ask a question")
question = st.chat_input("What would you like to know?")

if question:
    if not st.session_state.get("store"):
        st.info("Upload and index at least one document first.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is missing. Add it to .env and restart Streamlit.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    answer_stream, sources = stream_answer(st.session_state.store, question)
                    st.write_stream(answer_stream)
                    with st.expander("Source snippets"):
                        for index, source in enumerate(sources, start=1):
                            st.markdown(f"**{index}. {source.metadata.get('source', 'unknown')}**")
                            st.caption(source.page_content[:700] + ("…" if len(source.page_content) > 700 else ""))
                except Exception as exc:
                    st.error(f"Question failed: {exc}")
