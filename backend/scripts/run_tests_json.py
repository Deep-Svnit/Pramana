from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag_v2.config import RagV2Config
from app.rag_v2.graph import build_graph


def progress(message: str) -> None:
    print(f"[tests-json] {message}", flush=True)


def main() -> int:
    tests_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BACKEND_ROOT / "tests.json"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else BACKEND_ROOT / "tests_results.json"
    md_output_path = output_path.with_suffix(".md")

    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    questions = tests.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {tests_path}")

    cfg = RagV2Config.from_env()
    progress(
        f"building graph index={cfg.pinecone_index_name} namespace={cfg.pinecone_namespace} "
        f"embedding_model={cfg.embedding_model}"
    )
    graph = build_graph(cfg)

    results = []
    suite_start = perf_counter()
    for index, question in enumerate(questions, start=1):
        start = perf_counter()
        progress(f"question {index}/{len(questions)} started: {question}")
        try:
            result = graph.invoke({
                "query": question,
                "history": [],
                "chunks": [],
                "answer": "",
                "sources": [],
                "crag_report": {},
                "is_fallback": False,
            })
            elapsed = perf_counter() - start
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            progress(
                f"question {index}/{len(questions)} done in {elapsed:.1f}s "
                f"sources={len(sources)} fallback={result.get('is_fallback', False)}"
            )
            results.append({
                "index": index,
                "question": question,
                "answer": answer,
                "is_fallback": result.get("is_fallback", False),
                "sources": sources,
                "chunks_found": len(result.get("chunks", [])),
                "crag_report": result.get("crag_report", {}),
                "elapsed_seconds": elapsed,
            })
        except Exception as exc:  # noqa: BLE001
            elapsed = perf_counter() - start
            progress(f"question {index}/{len(questions)} failed in {elapsed:.1f}s error={exc}")
            results.append({
                "index": index,
                "question": question,
                "answer": "",
                "is_fallback": True,
                "sources": [],
                "chunks_found": 0,
                "crag_report": {},
                "elapsed_seconds": elapsed,
                "error": str(exc),
            })

    payload = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "tests_path": str(tests_path),
        "pinecone_index": cfg.pinecone_index_name,
        "pinecone_namespace": cfg.pinecone_namespace,
        "total_questions": len(questions),
        "total_elapsed_seconds": perf_counter() - suite_start,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# tests.json results",
        "",
        f"- Index: `{cfg.pinecone_index_name}`",
        f"- Namespace: `{cfg.pinecone_namespace}`",
        f"- Questions: `{len(questions)}`",
        "",
    ]
    for item in results:
        lines.extend([
            f"## {item['index']}. {item['question']}",
            "",
            item.get("answer") or f"ERROR: {item.get('error', '')}",
            "",
            f"- Fallback: `{item.get('is_fallback')}`",
            f"- Chunks found: `{item.get('chunks_found')}`",
            f"- Elapsed: `{item.get('elapsed_seconds'):.1f}s`",
            "",
        ])
    md_output_path.write_text("\n".join(lines), encoding="utf-8")

    progress(f"wrote JSON results to {output_path}")
    progress(f"wrote Markdown results to {md_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
