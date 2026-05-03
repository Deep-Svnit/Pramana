# Backend Integration Guide

This project exposes the PowerMind multimodal RAG pipeline as a Python service layer. The backend should create one long-lived `MultimodalRAGPipeline` instance, load the stored records once, and reuse that instance for incoming questions.

## Recommended Flow

1. Initialize the pipeline during backend startup.
2. Call `load_from_storage()` after the first ingestion run has populated `storage/`.
3. For each user query, call `answer(query)`.
4. Return the answer text, citations, fallback flag, verifier report, and timing breakdown to the client if needed.

## Minimal Example

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
        "is_fallback": result.is_fallback,
        "verifier_report": result.verifier_report,
        "timings": result.timings,
    }
```

## Timing Fields

`result.timings` contains a response-time breakdown for the main stages of the architecture:

- `storage_load`
- `query_expansion`
- `bm25_retrieval`
- `dense_retrieval`
- `keyword_retrieval`
- `text_rrf`
- `visual_retrieval`
- `final_rrf`
- `relevance_grading`
- `generation`
- `verification`
- `total`

Use these values to log latency, build dashboards, or return observability metadata in a backend API response.

## Released Timing Surface

The normal CLI answer path can now print timings directly:

```cmd
python -m powermind_rag.cli ask "What is the consolidated total income in H1-26?" --show-timings
```

This is the recommended way to release the timing breakdown to backend logs, staging checks, or operator-facing tools without using the standalone profiling script.

## Environment Variables

At minimum, set the same environment variables used by the CLI:

- `POWERMIND_DEVICE`
- `POWERMIND_LOCAL_ONLY`
- `POWERMIND_STORAGE_DIR`
- `QWEN_MODEL_PATH`
- `MISTRAL_API_KEY` when OCR is enabled
- `GROQ_API_KEY` when relevance grading is enabled

The `.env` file is loaded automatically by `RAGConfig.from_env()`.

## Profiling Script

Use `scripts/profile_response_timings.py` to inspect the component-wise timing split for a single query:

```cmd
python .\scripts\profile_response_timings.py "What operational metrics are mentioned for airport performance in H1-26?"
```

Pass `--json-output` if the backend team wants a machine-readable latency report.