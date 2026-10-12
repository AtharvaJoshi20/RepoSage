# RepoSage

RepoSage is a small Streamlit app for testing a basic Retrieval-Augmented Generation (RAG) workflow.

Upload a PDF, text file, or Markdown file, index it locally, and ask questions about the uploaded content. The app retrieves relevant document chunks, sends them to Gemini, and displays the answer with source snippets.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Gemini](https://img.shields.io/badge/Gemini-API-8E75B2)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Embeddings-yellow?logo=huggingface)
![RAG](https://img.shields.io/badge/Project-RAG%20MVP-success)
![PDF](https://img.shields.io/badge/Input-PDF%20%7C%20TXT%20%7C%20MD-informational)

## What it demonstrates

- PDF, TXT, and Markdown upload
- PDF text extraction with `pypdf`
- Local Hugging Face embeddings
- ChromaDB vector storage
- Similarity-based document retrieval
- Gemini-powered answer generation
- Streaming answers in the Streamlit UI
- Source snippets for basic traceability
- A small benchmark script for indexing and response timings

## Tech stack

- Python
- Streamlit
- LangChain
- Gemini API
- ChromaDB
- Hugging Face `all-MiniLM-L6-v2` embeddings

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Add your Gemini API key to `.env`:

```env
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Never commit `.env` or expose the API key publicly.

## Run the app

```powershell
streamlit run app.py
```

Then open the local Streamlit URL, upload a document, select **Index documents**, and ask a question.

The first indexing run may take longer because the local embedding model needs to load and the document must be embedded. The app stores the Chroma index locally and reuses it when the same file is uploaded again.

## Measure performance

The included `benchmark.py` script measures:

- Document extraction time
- Number of generated chunks
- Cold indexing time
- Warm index loading time
- Retrieval time
- Time to first token
- Total answer time

Run the benchmark with any supported document:

```powershell
python benchmark.py `
  --pdf "path\to\your\document.pdf" `
  --output benchmark_metrics.json
```

To measure a complete RAG question:

```powershell
python benchmark.py `
  --pdf "path\to\your\document.pdf" `
  --question "Ask a question about the document" `
  --output benchmark_metrics_with_qa.json
```

The benchmark output is written as JSON. Results depend on hardware, model-cache state, and network/API latency.

## Performance

Measured using a local PDF, results vary depending on document size, hardware, model-cache state, and API latency.

| Metric | Result |
|---|---:|
| PDF size | 29.84 MB |
| Pages | 991 |
| Extracted characters | 1,081,665 |
| Generated chunks | 966 |
| Cold indexing time | 57.7–60.7 seconds |
| PDF extraction time | 13.9–14.4 seconds |
| Embedding and vector indexing | 44.1–46.4 seconds |
| Warm index load time | 0.009 seconds |
| Retrieval time | 0.016 seconds |
| Time to first token | 11.2 seconds |
| Total answer time | 11.5 seconds |

The persisted index reduces repeat loading from approximately one minute to under 0.01 seconds. Query latency is mainly caused by the Gemini API response time.

## Project files

```text
app.py             Streamlit RAG application
benchmark.py       Local timing and performance benchmark
requirements.txt   Python dependencies
.env.example       Environment variable template
.gitignore         Local secrets and generated files to ignore
```

## Limitations

- Answers are restricted to the uploaded document context.
- Scanned or image-only PDFs may not produce useful text without OCR.
- The local vector index is intended for this small experiment.
- There is no authentication, multi-user storage, evaluation dataset, or production deployment configuration.
- Gemini API usage may incur quota limits or costs depending on the selected model and account.

## Why this project exists

RepoSage is intentionally small. Its purpose is to make the main RAG flow easy to inspect:

```text
Upload → Extract → Chunk → Embed → Store → Retrieve → Generate → Show snippets
```
