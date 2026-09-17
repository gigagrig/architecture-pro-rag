"""Heuristic filters for the educational prompt-injection demonstration."""

import re
import unicodedata


INSTRUCTIONS = re.compile(
    r"ignore\s+(?:all\s+|previous\s+|prior\s+)*(?:instructions|prompts)"
    r"|игнорируй\s+(?:(?:все|предыдущие)\s+)*(?:инструкции|правила)"
    r"|<\|(?:system|im_start|im_end)\|>|\[/?INST\]|</?system>"
    r"|(?:system|developer)\s*:", re.IGNORECASE,
)
SENSITIVE = re.compile(r"swordfish|суперпароль\s+root", re.IGNORECASE)


def normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def unsafe_text(text: str) -> bool:
    text = normalized(text)
    return bool(INSTRUCTIONS.search(text) or SENSITIVE.search(text))


def sanitize_question(question: str) -> str:
    return INSTRUCTIONS.sub("", normalized(question)).strip()


def filter_documents(documents: list[dict]) -> list[dict]:
    # Drop the whole chunk: removing only the command would leave its payload.
    return [chunk for chunk in documents if not any(
        unsafe_text(str(chunk.get(field, "")))
        for field in ("text", "title", "source", "chunk_id")
    )]
