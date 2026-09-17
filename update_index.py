#!/usr/bin/env python3
"""Publish consistent FAISS snapshots and optionally run a daily scheduler."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from build_index import Chunk, load_chunks, split_words, write_chunks
from rag_bot.cli import ScriptParser


def emit(event: str, **fields: object) -> None:
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=event,
                          **fields), ensure_ascii=False), flush=True)


def active_directory(root: Path) -> Path:
    return (root / "current").resolve(strict=True) if (root / "current").is_symlink() else root


def scan(knowledge: Path, extras: list[Path]) -> dict[str, str]:
    paths = sorted(knowledge.glob("entity_*.md"))
    if not paths:
        raise ValueError(f"No entity_*.md documents in {knowledge}")
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + extras}


def update(args: argparse.Namespace) -> int:
    started = time.monotonic()
    emit("start", knowledge_dir=str(args.knowledge_dir.resolve()),
         index_dir=str(args.index_dir.resolve()))
    try:
        # An existing index supplies the encoder and chunking contract.
        root = args.index_dir
        with (root / "update.lock").open("a") as lock:
            emit("file_opened", path=str(root / "update.lock"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                emit("finish", status="locked", errors=0, new_chunks=0, index_size=None)
                return 0
            old_dir = active_directory(root)
            manifest = json.loads((old_dir / "manifest.json").read_text())
            extras = [Path(p) for p in manifest.get("extra_documents", [])]
            hashes = scan(args.knowledge_dir, extras)
            old_hashes = manifest.get("source_hashes", {})
            if hashes == old_hashes:
                emit("finish", status="unchanged", new_chunks=0,
                     index_size=manifest["chunk_count"], errors=0,
                     elapsed_seconds=round(time.monotonic() - started, 2))
                return 0
            chunks, count = load_chunks(args.knowledge_dir, manifest["chunk_min_words"],
                                        manifest["chunk_max_words"], manifest["chunk_overlap_words"])
            for path in extras:
                body = path.read_text(encoding="utf-8").strip()
                if not body:
                    raise ValueError(f"Empty extra document: {path}")
                for seq, (start, end, text) in enumerate(split_words(body, 1, manifest["chunk_max_words"], 0), 1):
                    chunks.append(Chunk(f"{path.stem}-chunk-{seq:03d}", path.as_posix(),
                                        path.stem, start, end, text))
            if len({c.chunk_id for c in chunks}) != len(chunks):
                raise ValueError("Duplicate chunk IDs in source documents")
            old_chunks = [json.loads(line) for line in (old_dir / "chunks.jsonl").read_text().splitlines()]
            old_index = faiss.read_index(str(old_dir / "faiss.index"))
            if old_index.ntotal != len(old_chunks) or old_index.d != manifest["dimension"]:
                raise ValueError(f"Inconsistent existing index: {old_dir}")
            if old_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError(f"Unexpected index metric: {old_dir}")
            # Embeddings depend only on passage text and the unchanged encoder.
            previous = {c["text"]: pos for pos, c in enumerate(old_chunks)}
            vectors = np.empty((len(chunks), old_index.d), dtype=np.float32)
            pending = []
            for pos, chunk in enumerate(chunks):
                if chunk.text in previous:
                    vectors[pos] = old_index.reconstruct(previous[chunk.text])
                else:
                    pending.append(pos)
            emit("embedding", new_chunks=len(pending), reused_chunks=len(chunks) - len(pending))
            if pending:
                model = SentenceTransformer(manifest["model"], cache_folder=str(args.cache_folder) if args.cache_folder else None,
                                            local_files_only=args.local_files_only)
                vectors[pending] = model.encode([f"passage: {chunks[p].text}" for p in pending],
                                                normalize_embeddings=True, convert_to_numpy=True,
                                                batch_size=args.batch_size, show_progress_bar=False)
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                raise ValueError("Embeddings are not finite normalized vectors")
            if scan(args.knowledge_dir, extras) != hashes:
                raise ValueError("Sources changed during update; retry required")
            index = faiss.IndexFlatIP(old_index.d)
            index.add(vectors)
            revision = root / "revisions" / uuid.uuid4().hex
            revision.mkdir(parents=True)
            emit("directory_created", path=str(revision))
            faiss.write_index(index, str(revision / "faiss.index"))
            write_chunks(revision / "chunks.jsonl", chunks)
            manifest.update(generated_at=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
                            document_count=count + len(extras), chunk_count=len(chunks))
            (revision / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            for name in ("faiss.index", "chunks.jsonl", "manifest.json"):
                emit("file_created", path=str(revision / name))
            pointer = root / f".current-{revision.name}"
            pointer.symlink_to(Path("revisions") / revision.name)
            pointer.replace(root / "current")
            emit("published", path=str(root / "current"), revision=revision.name)
            emit("finish", status="updated", new_chunks=len(pending), index_size=index.ntotal,
                 added_files=len(hashes.keys() - old_hashes.keys()),
                 changed_files=sum(hashes[p] != old_hashes[p] for p in hashes.keys() & old_hashes.keys()),
                 deleted_files=len(old_hashes.keys() - hashes.keys()), errors=0,
                 elapsed_seconds=round(time.monotonic() - started, 2))
            return 0
    except Exception as error:
        emit("finish", status="error", errors=1, new_chunks=0, index_size=None,
             knowledge_dir=str(args.knowledge_dir), index_dir=str(args.index_dir),
             error=str(error), elapsed_seconds=round(time.monotonic() - started, 2))
        return 1


def parse_args() -> argparse.Namespace:
    parser = ScriptParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                          epilog="Example: python update_index.py --local-files-only --cache-folder .cache/embeddings\n"
                          "Scheduler: python update_index.py --schedule --hour-utc 3\n"
                          "Outputs: INDEX/revisions/{id}/{faiss.index,chunks.jsonl,manifest.json}, current symlink; stdout log.\n"
                          "Exit codes: 0 success/no changes/locked; 1 update error; 2 invalid arguments.")
    parser.add_argument("--knowledge-dir", type=Path, default=Path("knowledge_base"), help="Source directory (default: knowledge_base)")
    parser.add_argument("--index-dir", type=Path, default=Path("vector_index"), help="Existing index directory (default: vector_index)")
    parser.add_argument("--cache-folder", type=Path, help="Embedding cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Disable model downloads")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size (default: 16)")
    parser.add_argument("--schedule", action="store_true", help="Update on startup, then daily; retry failed jobs")
    parser.add_argument("--hour-utc", type=int, default=3, help="Daily hour in UTC (default: 3 = 06:00 Moscow)")
    parser.add_argument("--retry-seconds", type=int, default=300, help="Retry delay after errors (default: 300 seconds)")
    args = parser.parse_args()
    if not 0 <= args.hour_utc <= 23 or args.batch_size < 1 or args.retry_seconds < 1:
        parser.error("hour must be 0..23; batch size and retry seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    while True:
        result = update(args)
        if not args.schedule:
            return result
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=args.hour_utc, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        if result:
            next_run = now + timedelta(seconds=args.retry_seconds)
        emit("scheduled", next_run=next_run.isoformat())
        while datetime.now(timezone.utc) < next_run:
            time.sleep(min(30, max(0, (next_run - datetime.now(timezone.utc)).total_seconds())))


if __name__ == "__main__":
    sys.exit(main())
