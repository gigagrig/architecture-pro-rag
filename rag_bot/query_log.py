"""Append structured RAG request events without recording upstream credentials."""
from datetime import datetime, timezone
import fcntl
import json
import logging
from pathlib import Path
import time
import uuid


class QueryLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> None:
        with self.path.open('a', encoding='utf-8') as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(json.dumps(event, ensure_ascii=False) + '\n')
            output.flush()

    def record(self, question: str, started: float, trace: dict, result=None, error=None) -> None:
        event = dict(request_id=uuid.uuid4().hex, timestamp=datetime.now(timezone.utc).isoformat(),
                     question=question, status='error' if error else ('unknown' if result.unknown else 'answered'),
                     answer=result.answer if result else '', answer_length=len(result.answer) if result else 0,
                     successful_answer=bool(result and not result.unknown and result.sources and result.explanation),
                     has_retrieved_chunks=bool(trace.get('retrieved')),
                     retrieved=trace.get('retrieved', []), eligible=trace.get('eligible', []),
                     sources=[s.model_dump() for s in result.sources] if result else [],
                     error_type=type(error).__name__ if error else None,
                     duration_seconds=round(time.monotonic() - started, 3),
                     index_revision=trace.get('index_revision'), reason=trace.get('reason'))
        try:
            self.write(event)
        except OSError:
            logging.getLogger(__name__).exception('Cannot append RAG query log: %s', self.path)
