from __future__ import annotations

import argparse
import json
from pathlib import Path

from powermind_rag.config import RAGConfig
from powermind_rag.pipeline import MultimodalRAGPipeline


DEFAULT_QUESTIONS = [
    ("Q1", "What are the major business segments discussed in the document?"),
    ("Q2", "What is the consolidated total income in H1-26?"),
    ("Q3", "What drivers are mentioned for EBITDA changes in H1-26?"),
    ("Q4", "What is the CEO's email address?"),
    ("Q5 Part 1", "Summarize airport performance in H1-26."),
    ("Q5 Part 2", "Break airport performance in H1-26 down into passenger and cargo changes."),
    ("Q6", "What operational metrics are mentioned for airport performance in H1-26?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PowerMind multimodal RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--doc-type", required=True)
    ingest.add_argument("--section", required=True)
    ingest.add_argument("--context", required=True)

    ingest_dir = sub.add_parser("ingest-dir")
    ingest_dir.add_argument("folder", type=Path)
    ingest_dir.add_argument("--doc-type", default="corporate disclosure documents")
    ingest_dir.add_argument("--section", default="financial and operational performance")
    ingest_dir.add_argument("--context", default="AEL Q2 FY26 and H1-26 performance")

    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument("--show-timings", action="store_true")

    ask_batch = sub.add_parser("ask-batch")
    ask_batch.add_argument("--output", type=Path, default=Path("outputs/qa_results.md"))
    ask_batch.add_argument("--json-output", type=Path, default=Path("outputs/qa_results.json"))

    args = parser.parse_args()
    if args.command == "ingest":
        pipeline = MultimodalRAGPipeline(RAGConfig.from_env())
        pipeline.ingest_pdf(args.pdf, args.doc_type, args.section, args.context)
        print("Ingestion complete.")
    elif args.command == "ingest-dir":
        pipeline = MultimodalRAGPipeline(RAGConfig.from_env())
        pdfs = sorted(args.folder.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError(f"No PDF files found in {args.folder}.")
        for pdf in pdfs:
            print(f"Ingesting {pdf.name}...")
            pipeline.ingest_pdf(pdf, args.doc_type, args.section, args.context)
        print(f"Ingestion complete for {len(pdfs)} PDF(s).")
    elif args.command == "ask":
        pipeline = MultimodalRAGPipeline(RAGConfig.from_env())
        pipeline.load_from_storage()
        answer = pipeline.answer(args.query)
        print(answer.text)
        if args.show_timings:
            print(_format_timing_summary(answer.timings))
    elif args.command == "ask-batch":
        print("Loading RAG pipeline for batch questions...", flush=True)
        pipeline = MultimodalRAGPipeline(RAGConfig.from_env())
        print("Loading stored records...", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("# PowerMind RAG Question Results\n\nStarting batch run...\n", encoding="utf-8")
        args.json_output.write_text("[]", encoding="utf-8")
        results: list[dict] = []
        try:
            pipeline.load_from_storage()
        except Exception as exc:
            error_text = f"Batch run failed before questions could be asked: {exc}"
            args.output.write_text(
                f"# PowerMind RAG Question Results\n\n{error_text}\n",
                encoding="utf-8",
            )
            args.json_output.write_text(json.dumps({"error": error_text}, indent=2), encoding="utf-8")
            print(error_text, flush=True)
            return
        for label, question in DEFAULT_QUESTIONS:
            print(f"Asking {label}...", flush=True)
            answer = pipeline.answer(question)
            results.append(
                {
                    "id": label,
                    "question": question,
                    "answer": answer.text,
                    "is_fallback": answer.is_fallback,
                    "citations": [chunk.citation for chunk in answer.retrieved],
                    "retrieved": [
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "page_number": chunk.page_number,
                            "modality": chunk.modality,
                            "score": chunk.score,
                            "citation": chunk.citation,
                        }
                        for chunk in answer.retrieved
                    ],
                    "verifier_report": answer.verifier_report,
                    "timings": answer.timings,
                }
            )
        args.output.write_text(_format_markdown(results), encoding="utf-8")
        args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}", flush=True)
        print(f"Wrote {args.json_output}", flush=True)


def _format_markdown(results: list[dict]) -> str:
    lines = ["# PowerMind RAG Question Results", ""]
    for item in results:
        lines.extend(
            [
                f"## {item['id']}",
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Answer:** {item['answer']}",
                "",
                f"**Fallback:** {item['is_fallback']}",
                "",
                f"**Citations:** {', '.join(item['citations']) if item['citations'] else 'None'}",
                "",
                f"**Timing:** {_format_timing_summary(item.get('timings', {}))}",
                "",
            ]
        )
    return "\n".join(lines)


def _format_timing_summary(timings: dict[str, float]) -> str:
    total = timings.get("total", 0.0)
    if total <= 0:
        return "Unavailable"
    ordered = [
        ("storage_load", "storage load"),
        ("query_expansion", "query expansion"),
        ("bm25_retrieval", "bm25 retrieval"),
        ("dense_retrieval", "dense retrieval"),
        ("keyword_retrieval", "keyword retrieval"),
        ("text_rrf", "text fusion"),
        ("visual_retrieval", "visual retrieval"),
        ("final_rrf", "final fusion"),
        ("relevance_grading", "relevance grading"),
        ("generation", "generation"),
        ("verification", "verification"),
    ]
    parts = [f"total {total * 1000:.1f} ms"]
    for key, label in ordered:
        value = timings.get(key, 0.0)
        if value <= 0:
            continue
        parts.append(f"{label} {value * 1000:.1f} ms")
    return "; ".join(parts)


main()
