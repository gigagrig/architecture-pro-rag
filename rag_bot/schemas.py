"""Validated public API and model output contracts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


UNKNOWN_ANSWER = "Я не знаю. В базе знаний недостаточно информации для ответа."


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    quote: str = Field(min_length=1, max_length=1200)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=4000)
    explanation: list[str] = Field(max_length=3)
    unknown: bool
    evidence: list[Evidence] = Field(max_length=5)


class Source(BaseModel):
    chunk_id: str
    title: str
    source: str
    score: float
    quote: str


class Answer(BaseModel):
    question: str
    answer: str
    explanation: list[str]
    unknown: bool
    sources: list[Source]
    retrieved_count: int
    model: str
