# Pramana Frontend

The frontend is the Next.js user interface for Pramana. It gives users a document upload surface, a chat interface, and citation/source panels for answers returned by the backend or service API.

Full integration notes live in [docs/frontend.md](../docs/frontend.md).

## Run

```powershell
cd D:\PowerMind\frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Responsibilities

- Upload PDFs to the ingestion API.
- Send user questions to the chat API.
- Render grounded answers with citations.
- Show source pages and confidence/timing metadata when available.
- Surface visual-analysis failures as document-level warnings.

## Expected API Shape

Chat requests send a user message and optional session id. Chat responses should include:

```json
{
  "answer": "Grounded answer with citations.",
  "citations": ["[p3:c12]"],
  "sources": [],
  "is_fallback": false,
  "timings": {}
}
```

Upload responses should include the document id and ingestion status, including any pages where both hosted VLMs failed.

## Build

```powershell
npm run build
npm start
```
