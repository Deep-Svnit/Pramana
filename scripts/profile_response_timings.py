from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from powermind_rag.config import RAGConfig
from powermind_rag.pipeline import MultimodalRAGPipeline


DEFAULT_QUERY = "What operational metrics are mentioned for airport performance in H1-26?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile component-wise PowerMind response timings")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    pipeline = MultimodalRAGPipeline(RAGConfig.from_env())
    pipeline.load_from_storage()
    answer = pipeline.answer(args.query)

    total_ms = answer.timings.get("total", 0.0) * 1000
    print(f"Query: {args.query}")
    print(f"Answer: {answer.text}")
    print(f"Total response time: {total_ms:.1f} ms")
    print("Component breakdown:")
    for name, value in _ordered_timing_items(answer.timings):
        if name == "total" or value <= 0:
            continue
        share = (value / answer.timings["total"] * 100) if answer.timings.get("total", 0.0) > 0 else 0.0
        label = name.replace("_", " ")
        print(f"  - {label}: {value * 1000:.1f} ms ({share:.1f}%)")

    if args.json_output:
        args.json_output.write_text(
            json.dumps(
                {
                    "query": args.query,
                    "answer": answer.text,
                    "timings": answer.timings,
                    "retrieved": [chunk.citation for chunk in answer.retrieved],
                    "is_fallback": answer.is_fallback,
                    "verifier_report": answer.verifier_report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _ordered_timing_items(timings: dict[str, float]) -> list[tuple[str, float]]:
    order = [
        "storage_load",
        "query_expansion",
        "bm25_retrieval",
        "dense_retrieval",
        "keyword_retrieval",
        "text_rrf",
        "visual_retrieval",
        "final_rrf",
        "relevance_grading",
        "generation",
        "verification",
        "total",
    ]
    return [(name, timings.get(name, 0.0)) for name in order]


if __name__ == "__main__":
    main()