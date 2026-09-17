"""Single RAG pipeline used by both user interfaces."""

from threading import BoundedSemaphore

from rag_bot.config import Settings
from rag_bot.prompts import messages_for
from rag_bot.schemas import Answer, GeneratedAnswer, Source, UNKNOWN_ANSWER


class ServiceBusy(Exception):
    """One generation is already in progress on this educational server."""


class RagService:
    def __init__(self, settings: Settings, retriever, generator):
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.slot = BoundedSemaphore(1)

    def unknown(self, question: str, count: int) -> Answer:
        return Answer(question=question, answer=UNKNOWN_ANSWER, explanation=[],
                      unknown=True, sources=[], retrieved_count=count,
                      model=self.settings.llm_model)

    def ask(self, question: str) -> Answer:
        if not self.slot.acquire(blocking=False):
            raise ServiceBusy
        try:
            return self._ask(question)
        finally:
            self.slot.release()

    def _ask(self, question: str) -> Answer:
        found = self.retriever.search(question, self.settings.top_k)
        documents = [chunk for chunk in found if chunk["score"] >= self.settings.min_score]
        if not documents:
            return self.unknown(question, len(found))
        messages = messages_for(question, documents)
        generated: GeneratedAnswer = self.generator.generate(messages)
        by_id = {chunk["chunk_id"]: chunk for chunk in documents}
        if generated.unknown or not generated.evidence or not generated.explanation:
            return self.unknown(question, len(found))
        if not self.valid_evidence(generated, by_id):
            return self.unknown(question, len(found))
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
