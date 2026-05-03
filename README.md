# PowerMind Multimodal RAG

Strict implementation of the requested architecture:

- Dual ingestion:
  - Visual pages rendered to images, embedded with ColPali through `byaldi`, and stored in FAISS.
  - Text plus Mistral OCR table Markdown converted into LLM-generated atomic propositions.
- Dual indexing:
  - Visual FAISS page index.
  - Hybrid text index with BM25 plus dense FAISS.
- Retrieval:
  - Visual retrieval.
  - BM25 retrieval.
  - Dense retrieval.
  - Reciprocal Rank Fusion using `1 / (60 + rank)` with no normalization or weighted averaging.
- CRAG:
  - Relevance grader before generation.
  - LettuceDetect after generation.
- Generation:
  - Local Qwen only.
  - Citations like `[p3:c12]`.
  - Unsupported answers return `Not found in the document.`

## Runtime Requirements

Use the GPU conda environment created for the RTX 5050 setup:

```cmd
conda activate powermind_rtx5050
```

This project defaults to GPU execution:

```cmd
set POWERMIND_DEVICE=cuda
set POWERMIND_CUDA_ARCH=sm_120
```

For CPU execution:

```cmd
set POWERMIND_DEVICE=cpu
```

Your RTX 5050 laptop GPU uses compute capability `sm_120`, so the PyTorch build must be new enough to support that architecture. If CUDA is not available and `POWERMIND_DEVICE` is still `cuda`, the pipeline fails closed instead of silently falling back to CPU.

Install dependencies inside `powermind_rtx5050`:

```cmd
conda create -n powermind_rtx5050 python=3.11 pip -y
conda activate powermind_rtx5050
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If `torchvision` fails with `RuntimeError: operator torchvision::nms does not exist`, reinstall a matching PyTorch and torchvision pair. For example, if `pip show torchvision` says `Requires-Dist: torch (==2.10.0)`, run:

```cmd
python -m pip install --force-reinstall torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --force-reinstall "transformers>=5.7.0,<6.0.0" "sentence-transformers>=5.4.1,<6.0.0"
```

For local-only runs that do not send document pages or chunks to external OCR/grading services:

```cmd
set POWERMIND_LOCAL_ONLY=1
set PYTHONDONTWRITEBYTECODE=1
```

Set Mistral OCR key:

```cmd
set MISTRAL_API_KEY=...
```

Set Groq CRAG relevance grading key and model:

```cmd
set GROQ_API_KEY=...
set GROQ_RELEVANCE_MODEL=llama-3.1-8b-instant
```

The `.env` file is loaded automatically.

Make sure `QWEN_MODEL_PATH` points to a complete local Qwen model.

Qwen 2.5 7B Instruct is suitable for final grounded text generation. It is text-only, so it cannot read chart pixels directly; chart/image facts must first be converted into textual evidence by the ingestion path. If only a retrieved image page is available and no supported text evidence exists, the answer is `Fallback: Not found in the document.`

## Usage

### One-Time Ingestion

Run this once for all PDFs in `data`. It creates stored records under `storage/`.

```cmd
conda activate powermind_rtx5050
set POWERMIND_LOCAL_ONLY=1
set PYTHONDONTWRITEBYTECODE=1
python -m powermind_rag.cli ingest-dir .\data --doc-type "AEL disclosure pack" --section "Q2 FY26 and H1-26 results" --context "business segments, consolidated income, EBITDA drivers, and airport performance"
```

### Ask Queries After Ingestion

This reuses the stored embeddings and records. Do not ingest again unless the PDFs change.

```cmd
conda activate powermind_rtx5050
set POWERMIND_LOCAL_ONLY=1
set PYTHONDONTWRITEBYTECODE=1
python -m powermind_rag.cli ask "What is the consolidated total income in H1-26?"
```

### Run The Provided Question Set

```cmd
conda activate powermind_rtx5050
set POWERMIND_LOCAL_ONLY=1
set PYTHONDONTWRITEBYTECODE=1
python -m powermind_rag.cli ask-batch --output .\outputs\qa_results.md --json-output .\outputs\qa_results.json
```

The batch command writes:

- `outputs/qa_results.md` for readable answers
- `outputs/qa_results.json` with answers, citations, retrieved chunks, fallback flags, and verifier report
