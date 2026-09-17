#!/usr/bin/env python3
"""Run reproducible RAG acceptance questions against the running HTTP API."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import requests

from rag_bot.schemas import Answer
from rag_bot.cli import ScriptParser


CASES = [
    ("Кто отец Kael Ardyn?", False, ["Xarn Velgor"]),
    ("Кто сестра-близнец Kael Ardyn?", False, ["Lyra Voss"]),
    ("На какой планете вырос Kael Ardyn?", False, ["Draxos"]),
    ("Какие объекты способен уничтожать Void Core?", False, ["планет"]),
    ("Что за оружие arcblade?", False, ["клин", "меч"]),
    ("Какой любимый десерт Kael Ardyn?", True, []),
    ("Какой пароль от Wi-Fi в офисе QuantumForge?", True, []),
]


def main() -> int:
    parser = ScriptParser(
        description="Check five grounded answers and two unknowns through the real RAG API.",
        epilog="Example: python check_api.py --output task4_api_results.json\n"
               "Output: JSON report and stdout progress. Exit codes: 0 passed, 1 failed, 2 invalid CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="API URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--output", type=Path, default=Path("task4_api_results.json"), help="JSON report path (default: task4_api_results.json)")
    parser.add_argument("--timeout", type=float, default=330, help="Timeout per question, seconds (default: 330)")
    args = parser.parse_args()
    print(f"API checks started; output: {args.output.resolve()}", flush=True)
    results = []
    for question, unknown, terms in CASES:
        print(f"Question: {question}", flush=True)
        started = time.monotonic()
        try:
            response = requests.post(f"{args.api_url.rstrip('/')}/ask", json={"question": question},
                                     timeout=(10, args.timeout))
            response.raise_for_status()
            answer = Answer.model_validate(response.json())
            passed = answer.unknown == unknown
            if not unknown:
                passed = passed and bool(answer.sources) and bool(answer.explanation)
                passed = passed and any(term.casefold() in answer.answer.casefold() for term in terms)
            else:
                passed = passed and answer.answer.startswith("Я не знаю") and not answer.sources
            results.append({"question": question, "passed": passed,
                            "seconds": round(time.monotonic() - started, 2),
                            "response": answer.model_dump()})
            print(f"{'PASS' if passed else 'FAIL'}: {answer.answer}", flush=True)
        except (requests.RequestException, ValueError):
            results.append({"question": question, "passed": False, "error": "API unavailable or invalid response"})
            print("FAIL: API unavailable or invalid response", flush=True)
    failures = sum(not item["passed"] for item in results)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "passed": len(results) - failures, "failed": failures, "results": results}
    try:
        if not args.output.parent.exists():
            args.output.parent.mkdir(parents=True)
            print(f"Created directory: {args.output.parent}")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        print(f"Cannot write report: {args.output}")
        return 1
    print(f"Created report: {args.output}")
    print(f"API checks finished: {len(results) - failures} passed, {failures} failed")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
