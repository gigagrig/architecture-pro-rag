#!/usr/bin/env python3
"""Build a normalized FAISS index from anonymized Markdown documents."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_bot.cli import ScriptParser


DEFAULT_MODEL = "intfloat/multilingual-e5-base"
MODEL_URL = "https://huggingface.co/intfloat/multilingual-e5-base"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    start_word: int
    end_word: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = ScriptParser(
        description=(
            "Split entity Markdown files into word-bounded chunks, encode them "
            "with multilingual-e5-base, and save an IndexFlatIP FAISS index."
        ),
        epilog=(
            "Example: ./build_index.py --knowledge-dir knowledge_base "
            "--output-dir vector_index"
            "\nOptional: --extra-document fixtures/prompt_injection.txt"
            "\nOutputs: faiss.index, chunks.jsonl, manifest.json. Exit: 0 success, 1 build error, 2 invalid input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extra-document", type=Path, action="append", default=[],
        help="Additional UTF-8 text document; may be short. Repeat for multiple files.",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path("knowledge_base"),
        help="Directory containing entity_*.md files (default: knowledge_base).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vector_index"),
        help="Directory for index and metadata (default: vector_index).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"Sentence Transformers model (default: {DEFAULT_MODEL}).",
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
        "--min-words",
        type=int,
        default=100,
        help="Minimum target size of a chunk in words (default: 100).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=250,
        help="Maximum size of a chunk in words (default: 250).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=40,
        help="Word overlap between adjacent chunks (default: 40).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size (default: 16).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_words < 1:
        raise ValueError("--min-words must be positive")
    if args.max_words < args.min_words:
        raise ValueError("--max-words must be greater than or equal to --min-words")
    if args.overlap_words < 0 or args.overlap_words >= args.min_words:
        raise ValueError("--overlap-words must be non-negative and below --min-words")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")


def read_document(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8").strip()
    lines = content.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("first line must be a level-one Markdown heading")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).strip()
    if not body:
        raise ValueError("document body is empty")
    return title, body


def split_words(text: str, min_words: int, max_words: int, overlap: int) -> list[tuple[int, int, str]]:
    words = re.findall(r"\S+", text)
    if len(words) < min_words:
        raise ValueError(f"document has {len(words)} words; minimum is {min_words}")
    if len(words) <= max_words:
        return [(0, len(words), " ".join(words))]

    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        if len(words) - start < min_words:
            start = max(0, len(words) - min_words)
            end = len(words)
        chunks.append((start, end, " ".join(words[start:end])))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def load_chunks(
    knowledge_dir: Path,
    min_words: int,
    max_words: int,
    overlap: int,
) -> tuple[list[Chunk], int]:
    paths = sorted(knowledge_dir.glob("entity_*.md"))
    if not paths:
        raise ValueError(f"no entity_*.md files found in {knowledge_dir}")

    chunks: list[Chunk] = []
    for path in paths:
        print(f"Reading document: {path}")
        title, body = read_document(path)
        pieces = split_words(body, min_words, max_words, overlap)
        for sequence, (start, end, text) in enumerate(pieces, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}-chunk-{sequence:03d}",
                    source=path.as_posix(),
                    title=title,
                    start_word=start,
                    end_word=end,
                    text=text,
                )
            )
        print(f"Created chunks for {path}: {len(pieces)}")
    return chunks, len(paths)


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    print("FAISS index build started")
    print(f"Knowledge directory: {args.knowledge_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Embedding model: {args.model_name}")
    try:
        validate_args(args)
        chunks, document_count = load_chunks(
            args.knowledge_dir,
            args.min_words,
            args.max_words,
            args.overlap_words,
        )
        for path in args.extra_document:
            print(f"Reading additional document: {path}")
            body = path.read_text(encoding="utf-8").strip()
            if not body:
                raise ValueError(f"Empty additional document: {path}")
            pieces = split_words(body, 1, args.max_words, 0)
            for sequence, (start, end, text) in enumerate(pieces, start=1):
                chunks.append(Chunk(f"{path.stem}-chunk-{sequence:03d}",
                                    path.as_posix(), path.stem, start, end, text))
            document_count += 1
            print(f"Created chunks for {path}: {len(pieces)}")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs in input documents")
    except (OSError, ValueError) as error:
        print(f"Cannot prepare chunks: {error}")
        return 2

    started = time.monotonic()
    print(f"Loading embedding model: {args.model_name}")
    try:
        model = SentenceTransformer(
            args.model_name,
            cache_folder=str(args.cache_folder) if args.cache_folder else None,
            local_files_only=args.local_files_only,
        )
        passages = [f"passage: {chunk.text}" for chunk in chunks]
        print(f"Generating embeddings for chunks: {len(passages)}")
        vectors = model.encode(
            passages,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    except Exception as error:  # External model loaders raise several exception types.
        print(f"Embedding generation failed for model {args.model_name}: {error}")
        return 1

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        print(f"Unexpected embedding matrix shape: {vectors.shape}")
        return 1

    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created or reused directory: {args.output_dir}")
    index_path = args.output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"Created FAISS index: {index_path}")

    chunks_path = args.output_dir / "chunks.jsonl"
    write_chunks(chunks_path, chunks)
    print(f"Created chunk metadata: {chunks_path}")

    elapsed = time.monotonic() - started
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name,
        "model_url": MODEL_URL if args.model_name == DEFAULT_MODEL else None,
        "dimension": dimension,
        "index_type": "IndexFlatIP",
        "similarity": "cosine (normalized vectors and inner product)",
        "document_count": document_count,
        "chunk_count": len(chunks),
        "chunk_min_words": args.min_words,
        "chunk_max_words": args.max_words,
        "chunk_overlap_words": args.overlap_words,
        "extra_documents": [path.as_posix() for path in args.extra_document],
        "build_seconds": round(elapsed, 2),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created index manifest: {manifest_path}")
    print(f"Documents indexed: {document_count}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print("FAISS index build finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
