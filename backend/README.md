# Pramana Backend

The backend folder contains the compatibility FastAPI application. The active AI pipeline lives in the service package at `service/src/powermind_rag`.

Full backend notes live in [docs/backend.md](../docs/backend.md).

## Run

```powershell
cd D:\PowerMind\backend
uv sync
uv run fastapi dev app/main.py
```

Health checks:

```text
GET /
GET /health
```

## Active Routers

```text
/chat
/history
/ingest
/rag
```

The old backend RAG v2 path has been removed. Keep new AI work pointed at the Pramana service package.

## Preferred Service Commands

From the repo root:

```powershell
python -m powermind_rag.cli ingest "D:\PowerMind\service\data\AEL_Earnings_Presentation_Q2-FY26_copy.pdf"
python -m powermind_rag.cli ask "Your question" --show-timings
```
