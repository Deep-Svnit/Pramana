# Pramana Frontend

The frontend is a Next.js app in `frontend/`. It provides the document upload and chat experience for Pramana.

## Run Frontend

```powershell
cd D:\PowerMind\frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Structure

```text
frontend/
|-- app/
|   |-- page.tsx
|   |-- layout.tsx
|   |-- globals.css
|   |-- api/
|       |-- chat/route.ts
|       |-- upload/route.ts
|-- components/
|-- lib/
```

## User Flow

1. User uploads a PDF.
2. Frontend sends the file to an upload API route.
3. Backend/service ingests the document.
4. User asks a question.
5. Frontend sends the question to a chat API route.
6. API returns answer, citations, source metadata, fallback flag, and optional timings.
7. Frontend renders answer and source evidence.

## Chat API Contract

Recommended request:

```json
{
  "message": "What is the Adani Family's equity stake in AEL?",
  "sessionId": "optional-session-id"
}
```

Recommended response:

```json
{
  "answer": "The Adani Family's equity stake in AEL is ... [p3:c12].",
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
    "total": 4.2
  }
}
```

## Upload API Contract

Recommended multipart field:

```text
file
```

Recommended response:

```json
{
  "status": "completed",
  "document_id": "AEL_Earnings_Presentation_Q2-FY26_copy",
  "text_records": 320,
  "visual_pages_succeeded": 39,
  "visual_pages_failed": 2
}
```

## UI Guidance

- Show citations near the answer, not hidden in logs.
- Surface `is_fallback` with a subtle status label.
- If `visual_pages_failed > 0`, show a non-blocking warning in document details.
- Keep timing data available in developer/debug mode.
- Do not ask users to interpret raw `text_records.json`; use source panels/page citations instead.

