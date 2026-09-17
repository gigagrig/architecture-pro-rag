"""FastAPI application; initialize the index and encoder once per process."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from rag_bot.config import Settings
from rag_bot.llm import ChatGenerator, GenerationError
from rag_bot.schemas import Answer, Question
from rag_bot.service import RagService, ServiceBusy


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            from rag_bot.retrieval import Retriever

            settings = Settings.from_env()
            logging.getLogger("uvicorn.error").info("Loading FAISS index and embedding model")
            app.state.rag = RagService(settings, Retriever(settings), ChatGenerator(settings))
        else:
            app.state.rag = service
        yield

    app = FastAPI(title="Knowledge base RAG API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        rag = app.state.rag
        return {"status": "ok", "index_loaded": True,
                "chunks": rag.retriever.index.ntotal,
                "embedding_model": rag.retriever.manifest["model"],
                "llm_model": rag.settings.llm_model,
                "llm_status": "not_checked"}

    @app.post("/ask", response_model=Answer)
    def ask(body: Question) -> Answer:
        try:
            return app.state.rag.ask(body.question)
        except ServiceBusy:
            raise HTTPException(429, "Сервис занят. Повторите запрос чуть позже.",
                                headers={"Retry-After": "5"}) from None
        except GenerationError:
            raise HTTPException(503, "Модель временно недоступна или вернула некорректный ответ.") from None

    return app


app = create_app()
