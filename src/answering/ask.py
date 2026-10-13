"""CLI for manual testing of the grounded answer layer.

    python -m src.answering.ask "What treats type 2 diabetes?"

Educational retrieval prototype. Not clinical decision support.
"""

import argparse

from src.answering.answer_builder import (
    DEFAULT_MAX_CITATIONS,
    build_answer,
    print_answer,
)
from src.retrieval.hybrid_retriever import build_default_retriever


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer a question using only retrieved, cited guideline text."
    )

    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-citations", type=int, default=DEFAULT_MAX_CITATIONS)
    parser.add_argument("--no-graph", action="store_true", help="Vector retrieval only.")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    retriever = build_default_retriever(graph_enabled=not args.no_graph)
    retrieval = retriever.retrieve(args.question, top_k=args.top_k)

    answer = build_answer(
        args.question,
        retrieval,
        max_citations=args.max_citations,
    )

    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print_answer(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
