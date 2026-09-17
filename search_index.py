#!/usr/bin/env python3
"""Search the generated FAISS index from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a query and print the nearest chunks from a saved FAISS index.",
        epilog=(
            "Example: ./search_index.py 'Who trained Kael Ardyn?' "
            "--index-dir vector_index --top-k 3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Natural-language query to search for.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("vector_index"),
        help="Directory containing faiss.index, chunks.jsonl, and manifest.json.",
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        help="Optional model cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model only from the local cache.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest chunks to print (default: 3).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    return records


def main() -> int:
    args = parse_args()
    print("FAISS search started")
    print(f"Index directory: {args.index_dir.resolve()}")
    print(f"Query: {args.query}")
    if args.top_k < 1:
        print("Invalid arguments: --top-k must be positive")
        return 2

    index_path = args.index_dir / "faiss.index"
    chunks_path = args.index_dir / "chunks.jsonl"
    manifest_path = args.index_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = load_jsonl(chunks_path)
        index = faiss.read_index(str(index_path))
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"Cannot load index files from {args.index_dir}: {error}")
        return 1

    if index.ntotal != len(chunks):
        print(
            f"Index and metadata disagree: {index.ntotal} vectors, "
            f"{len(chunks)} chunk records"
        )
        return 1

    model_name = str(manifest["model"])
    print(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=str(args.cache_folder) if args.cache_folder else None,
            local_files_only=args.local_files_only,
        )
        query_vector = model.encode(
            [f"query: {args.query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as error:  # External model loaders raise several exception types.
        print(f"Query embedding failed for model {model_name}: {error}")
        return 1

    scores, indices = index.search(
        np.ascontiguousarray(query_vector, dtype=np.float32),
        min(args.top_k, index.ntotal),
    )
    for rank, (score, item_index) in enumerate(zip(scores[0], indices[0]), start=1):
        chunk = chunks[int(item_index)]
        preview = str(chunk["text"])[:500]
        print(f"Result {rank}: score={float(score):.4f}")
        print(f"Title: {chunk['title']}")
        print(f"Source: {chunk['source']} ({chunk['chunk_id']})")
        print(f"Text: {preview}")
    print("FAISS search finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
