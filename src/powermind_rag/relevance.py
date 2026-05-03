from __future__ import annotations

import json

from powermind_rag.schema import RetrievedChunk


class RelevanceGrader:
    def __init__(self, api_key: str | None, model_name: str):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for CRAG relevance grading.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install groq to use the mandatory Groq CRAG relevance grader.") from exc
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def grade(self, query: str, chunks: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
        if not chunks:
            return []
        return [(chunk, self._score(query, chunk)) for chunk in chunks]

    def _score(self, query: str, chunk: RetrievedChunk) -> float:
        prompt = f"""
Return a relevance score for the chunk with respect to the query.
Use 1.0 only when the chunk directly supports answering the query.
Use 0.0 when unrelated.
Return only JSON like {{"score": 0.73}}.

QUERY:
{query}

CHUNK:
{chunk.text}
""".strip()
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            max_tokens=32,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict CRAG relevance grader. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            score = float(json.loads(content)["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Groq relevance grader returned invalid JSON: {content}") from exc
        return max(0.0, min(1.0, score))
