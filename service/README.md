# Pramana Service

This is the production multimodal RAG package used by Pramana. It is runnable as:

```powershell
python -m powermind_rag.cli ...
```

from the repo root.

See also:

- [Frontend Docs](../docs/frontend.md)
- [Backend Docs](../docs/backend.md)
- [AI Workflow Docs](../docs/ai_workflow.md)

## Default Pipeline

The default service flow is API-first:

- Embeddings: NVIDIA `nvidia/nv-embed-v1`
- OCR: Mistral OCR
- Ingestion page VLM: Gemini first, NVIDIA Phi fallback
- Retrieval: BM25 + keyword + dense FAISS + RRF
- CRAG relevance: Gemini JSON-mode scoring
- Generation: NVIDIA hosted model
- Last visual check when not found: NVIDIA first, Gemini fallback

No local model is required by default.

## Optional Components

These remain available but are off by default:

- ColPali/byaldi visual index:
  `POWERMIND_ENABLE_COLPALI_VISUAL_INDEX=1`
- Local Qwen/Qwen-VL classes from earlier experiments.
- Lettuce verification code from earlier experiments.

Keep `POWERMIND_ENABLE_COLPALI_VISUAL_INDEX=0` for the API-first Pramana pipeline.

## Important Environment

```env
POWERMIND_STORAGE_DIR="D:\PowerMind\service\storage"
POWERMIND_DEVICE="api"

NVIDIA_KEY="..."
NVIDIA_EMBEDDING_MODEL="nvidia/nv-embed-v1"
NVIDIA_VLM_MODEL="microsoft/phi-4-multimodal-instruct"
NVIDIA_GENERATION_MODEL="microsoft/phi-4-multimodal-instruct"

MISTRAL_API_KEY="..."

GEMINI_API_KEY_1="..."
GEMINI_API_KEY_2="..."
GEMINI_API_KEY_3="..."
GEMINI_API_KEY_4="..."
GEMINI_API_KEY_5="..."
GEMINI_API_KEY_6="..."

POWERMIND_ENABLE_VISUAL_UNDERSTANDING="true"
POWERMIND_VISUAL_UNDERSTANDING_PROVIDER="gemini_nvidia"
POWERMIND_IMAGE_PROVIDER="nvidia_gemini"
POWERMIND_ENABLE_QUERY_VLM_FALLBACK="1"

POWERMIND_RELEVANCE_PROVIDER="gemini"
POWERMIND_ALLOW_LEXICAL_CRAG_FALLBACK="0"
POWERMIND_ENABLE_COLPALI_VISUAL_INDEX="0"
```

## Install

```powershell
cd D:\PowerMind\service
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Ingest

```powershell
cd D:\PowerMind
python -m powermind_rag.cli ingest "D:\PowerMind\service\data\AEL_Earnings_Presentation_Q2-FY26_copy.pdf"
```

The command prints a visual analysis summary. If both hosted VLMs fail on any page, those pages are also saved to:

```text
service/storage/<document_id>/visual_page_failures.json
```

## Ask

```powershell
cd D:\PowerMind
python -m powermind_rag.cli ask "What is the consolidated total income in H1-26?" --show-timings
```

If stored embeddings were created with an older model, the first query auto-re-embeds them with the current NVIDIA embedder and saves the upgraded records.

## Test Set

Questions live in:

```text
service/tests.json
```

Run the built-in batch:

```powershell
cd D:\PowerMind
python -m powermind_rag.cli ask-batch --output service\outputs\qa_results.md --json-output service\outputs\qa_results.json
```
