# Pramana AI Workflow

This document describes the production AI path in `service/src/powermind_rag`.

## Goals

Pramana answers questions from documents using only retrieved evidence. It is built for visually dense PDFs where facts may appear in selectable text, tables, charts, diagrams, or page layout.

## Ingestion

Input:

```text
PDF document
```

Steps:

1. Render PDF pages to images.
2. Extract normal PDF text.
3. Run Mistral OCR for table and layout-heavy content.
4. Run visual page understanding when enabled:
   - Primary: Gemini (`POWERMIND_VISUAL_UNDERSTANDING_PROVIDER=gemini_nvidia`)
   - Fallback: NVIDIA `microsoft/phi-4-multimodal-instruct`
5. Combine page text, OCR tables, and visual analysis.
6. Split content into factual propositions.
7. Embed propositions with NVIDIA `nvidia/nv-embed-v1`.
8. Save records under `service/storage/<document_id>/`.

Generated files:

```text
text_records.json              # chunks, metadata, embeddings
visual_page_analysis.json      # VLM page summaries
visual_page_failures.json      # pages where both hosted VLMs failed
pages/page_0001.png            # rendered page images
```

## Retrieval

At query time Pramana loads `text_records.json` and builds in-memory indexes:

- BM25 lexical index
- keyword search
- FAISS dense index over NVIDIA embeddings

The query is embedded with the same NVIDIA model. If stored records were created with an older embedding dimension, Pramana re-embeds them once and saves the upgraded records.

Results are fused with Reciprocal Rank Fusion.

## CRAG Relevance

Before answer generation, candidates are graded by Gemini as a CRAG relevance grader:

```env
POWERMIND_RELEVANCE_PROVIDER=gemini
POWERMIND_ALLOW_LEXICAL_CRAG_FALLBACK=0
```

The grader uses JSON-mode generation and parser repair for narrow Gemini JSON issues. If all Gemini keys fail, the run fails visibly instead of silently using lexical relevance, unless emergency fallback is explicitly enabled.

## Generation

The text answer is generated from retrieved evidence only. The prompt requires:

- citations for every factual claim
- exact numbers and units
- no outside knowledge
- `Not found in the document.` when evidence is insufficient

## Final VLM Fallback

If the normal pipeline cannot answer, Pramana performs one final page-image check:

```env
POWERMIND_ENABLE_QUERY_VLM_FALLBACK=1
POWERMIND_IMAGE_PROVIDER=nvidia_gemini
```

Query fallback order:

1. NVIDIA Phi
2. Gemini key pool

This is intentionally the opposite of ingestion, where Gemini is preferred first.

## Optional Local Components

The following remain in code for earlier experiments, but are off by default:

- ColPali/byaldi visual index:
  `POWERMIND_ENABLE_COLPALI_VISUAL_INDEX=1`
- local Qwen/Qwen-VL classes
- Lettuce verification code

Default production deployment does not require local model weights.

