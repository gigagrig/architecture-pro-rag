"""Single RAG pipeline used by both user interfaces."""

import time
from threading import BoundedSemaphore

from rag_bot.config import Settings
from rag_bot.prompts import messages_for
from rag_bot.schemas import Answer, GeneratedAnswer, Source, UNKNOWN_ANSWER
from rag_bot.safety import filter_documents, sanitize_question, unsafe_text


class ServiceBusy(Exception):
    """One generation is already in progress on this educational server."""


class RagService:
    def __init__(self, settings: Settings, retriever, generator, query_log=None):
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.slot = BoundedSemaphore(1)
        self.query_log = query_log

    def unknown(self, question: str, count: int) -> Answer:
        return Answer(question=question, answer=UNKNOWN_ANSWER, explanation=[],
                      unknown=True, sources=[], retrieved_count=count,
                      model=self.settings.llm_model)

    def ask(self, question: str) -> Answer:
        started = time.monotonic()
        trace = {}
        result = None
        error = None
        acquired = self.slot.acquire(blocking=False)
        try:
            if not acquired:
                raise ServiceBusy
            result = self._ask(question, trace)
            return result
        except Exception as caught:
            error = caught
            raise
        finally:
            if self.query_log is not None:
                self.query_log.record(question, started, trace, result, error)
            if acquired:
                self.slot.release()

    def _ask(self, question: str, trace: dict) -> Answer:
        found = self.retriever.search(question, self.settings.top_k)
        trace['retrieved'] = [{k: chunk[k] for k in ('chunk_id', 'source', 'title', 'score')} for chunk in found]
        trace['index_revision'] = str(getattr(self.retriever, 'path', self.settings.index_dir))
        documents = filter_documents(
            [chunk for chunk in found if chunk["score"] >= self.settings.min_score])
        trace['eligible'] = [c['chunk_id'] for c in documents]
        if not documents:
            trace['reason'] = 'no_eligible_chunks'
            return self.unknown(question, len(found))
        messages = messages_for(sanitize_question(question), documents)
        generated: GeneratedAnswer = self.generator.generate(messages)
        if unsafe_text(generated.model_dump_json()):
            trace['reason'] = 'unsafe_output'
            return self.unknown(question, len(found))
        by_id = {chunk["chunk_id"]: chunk for chunk in documents}
        if generated.unknown or not generated.evidence or not generated.explanation:
            trace['reason'] = 'model_unknown_or_missing_evidence'
            return self.unknown(question, len(found))
        if not self.valid_evidence(generated, by_id):
            trace['reason'] = 'invalid_evidence'
            return self.unknown(question, len(found))
        trace['reason'] = 'grounded_answer'
        sources = []
        for evidence in generated.evidence:
            chunk = by_id[evidence.chunk_id]
            sources.append(Source(chunk_id=chunk["chunk_id"], title=chunk["title"],
                                  source=chunk["source"], score=round(chunk["score"], 4),
                                  quote=evidence.quote))
        return Answer(question=question, answer=generated.answer,
                      explanation=generated.explanation, unknown=False,
                      sources=sources, retrieved_count=len(found), model=self.settings.llm_model)

    @staticmethod
    def valid_evidence(generated: GeneratedAnswer, by_id: dict[str, dict]) -> bool:
        return all(
            evidence.chunk_id in by_id and bool(evidence.quote.strip())
            for evidence in generated.evidence
        )
