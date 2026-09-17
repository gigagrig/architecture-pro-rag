"""Read the existing FAISS index; use exactly its encoder and E5 prefix."""

import json
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_bot.config import Settings


class Retriever:
    def __init__(self, settings: Settings):
        path: Path = settings.index_dir
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = [json.loads(line) for line in
                       (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        self.index = faiss.read_index(str(path / "faiss.index"))
        if self.index.ntotal != len(self.chunks) or not self.chunks:
            raise ValueError("FAISS index and chunks.jsonl disagree or are empty")
        if self.index.d != self.manifest["dimension"]:
            raise ValueError("FAISS dimension and manifest disagree")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected normalized inner-product FAISS index")
        if len({chunk["chunk_id"] for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("Duplicate chunk IDs")
        self.model = SentenceTransformer(
            self.manifest["model"], cache_folder=settings.cache_folder,
            local_files_only=settings.local_files_only,
        )
        if self.model.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError("Embedding model and FAISS dimensions disagree")
        self.lock = Lock()

    def search(self, question: str, top_k: int) -> list[dict]:
        with self.lock:
            vector = self.model.encode([f"query: {question}"],
                                       normalize_embeddings=True, convert_to_numpy=True)
            scores, positions = self.index.search(
                np.ascontiguousarray(vector, dtype=np.float32),
                min(top_k, self.index.ntotal),
            )
        return [dict(self.chunks[int(position)], score=float(score))
                for score, position in zip(scores[0], positions[0]) if position >= 0]
