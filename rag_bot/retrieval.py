"""Read the existing FAISS index; use exactly its encoder and E5 prefix."""

import json
import logging
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_bot.config import Settings


class Retriever:
    def __init__(self, settings: Settings):
        self.root = settings.index_dir
        self.lock = Lock()
        self.path = self._active_path()
        self.manifest, self.chunks, self.index = self._read_snapshot(self.path)
        self.model = SentenceTransformer(
            self.manifest["model"], cache_folder=settings.cache_folder,
            local_files_only=settings.local_files_only,
        )
        if self.model.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError("Embedding model and FAISS dimensions disagree")

    def _active_path(self) -> Path:
        pointer = self.root / "current"
        return pointer.resolve(strict=True) if pointer.is_symlink() else self.root

    @staticmethod
    def _read_snapshot(path: Path) -> tuple:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        chunks = [json.loads(line) for line in
                       (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        index = faiss.read_index(str(path / "faiss.index"))
        if index.ntotal != len(chunks) or not chunks:
            raise ValueError("FAISS index and chunks.jsonl disagree or are empty")
        if index.d != manifest["dimension"]:
            raise ValueError("FAISS dimension and manifest disagree")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected normalized inner-product FAISS index")
        if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs")
        return manifest, chunks, index

    def _reload(self) -> None:
        try:
            path = self._active_path()
            if path == self.path:
                return
            manifest, chunks, index = self._read_snapshot(path)
            if manifest["model"] != self.manifest["model"] or index.d != self.index.d:
                raise ValueError("Hot reload requires the same embedding model and dimension")
            self.manifest, self.chunks, self.index, self.path = manifest, chunks, index, path
        except Exception:
            logging.getLogger(__name__).exception("Cannot reload index; keeping previous snapshot")

    def search(self, question: str, top_k: int) -> list[dict]:
        with self.lock:
            self._reload()
            vector = self.model.encode([f"query: {question}"],
                                       normalize_embeddings=True, convert_to_numpy=True)
            scores, positions = self.index.search(
                np.ascontiguousarray(vector, dtype=np.float32),
                min(top_k, self.index.ntotal),
            )
            return [dict(self.chunks[int(position)], score=float(score))
                    for score, position in zip(scores[0], positions[0]) if position >= 0]
