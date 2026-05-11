# Pramana Backend

The backend folder contains the compatibility FastAPI app and database models. The production AI pipeline lives in the service package:

```text
service/src/powermind_rag
```

The removed backend RAG v2/LangGraph/Pinecone path is no longer part of the active project.

## Run Backend

```powershell
cd D:\PowerMind\backend
uv sync
uv run fastapi dev app/main.py
```

Health endpoints:

```text
GET /
GET /health
```

## Mounted Routers

The backend currently mounts compatibility routers:

```text
/chat
/history
/ingest
/rag
```

The production Pramana CLI/service path is preferred for ingest and QA:

```powershell
cd D:\PowerMind
python -m powermind_rag.cli ingest "D:\PowerMind\service\data\AEL_Earnings_Presentation_Q2-FY26_copy.pdf"
python -m powermind_rag.cli ask "Your question" --show-timings
```

## Integrating The Service In FastAPI

Use a long-lived pipeline instance:

```python
from powermind_rag.config import RAGConfig
from powermind_rag.pipeline import MultimodalRAGPipeline

config = RAGConfig.from_env()
pipeline = MultimodalRAGPipeline(config)
pipeline.load_from_storage()


def answer_question(question: str) -> dict:
    result = pipeline.answer(question)
    return {
        "answer": result.text,
        "citations": [chunk.citation for chunk in result.retrieved],
        "sources": [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "citation": chunk.citation,
                "score": chunk.score,
            }
            for chunk in result.retrieved
        ],
        "is_fallback": result.is_fallback,
        "timings": result.timings,
    }
```

## Response Contract

Recommended API response:

```json
{
  "answer": "Grounded answer with citations.",
  "citations": ["[p3:c12]"],
  "sources": [
    {
      "document_id": "AEL_Earnings_Presentation_Q2-FY26_copy",
      "page_number": 3,
      "citation": "[p3:c12]",
      "score": 0.92
    }
  ],
  "is_fallback": false,
  "timings": {
    "total": 4.2,
    "dense_retrieval": 0.5,
    "relevance_grading": 1.2,
    "generation": 2.1
  }
}
```

## Operational Notes

- Load `.env` from the repo root.
- Keep one pipeline object warm per process.
- Re-run ingestion when source PDFs change.
- Stored embedding dimension upgrades are handled automatically on first query after model changes.
- CRAG relevance fails visibly when Gemini keys fail, unless `POWERMIND_ALLOW_LEXICAL_CRAG_FALLBACK=1`.

