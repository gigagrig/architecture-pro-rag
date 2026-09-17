#!/usr/bin/env python3
"""Start the Telegram polling worker or check its credentials."""

import argparse
import os
from pathlib import Path
import re
import sys

from rag_bot.config import load_environment, secret_from_env
from rag_bot.cli import ScriptParser
from rag_bot.telegram import TelegramBot, TelegramClient, TelegramError


def main() -> int:
    parser = ScriptParser(
        description="Run the Telegram RAG bot against an existing HTTP API.",
        epilog="Example: python run_bot.py --api-url http://127.0.0.1:8000\n"
               "Check credentials: python run_bot.py --check\n"
               "Output: status on stdout, no files. Exit codes: 0 success, 1 error, 2 invalid CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-url", help="RAG API URL (default: RAG_API_URL or localhost:8000)")
    parser.add_argument("--token-file", type=Path, help="Token file (default: env or telegram_bot_token.txt)")
    parser.add_argument("--check", action="store_true", help="Check getMe/webhook status without receiving messages")
    args = parser.parse_args()
    print("Telegram bot startup", flush=True)
    load_environment()
    try:
        token = (args.token_file.read_text(encoding="utf-8").strip() if args.token_file
                 else secret_from_env("TELEGRAM_BOT_TOKEN", "telegram_bot_token.txt"))
        if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", token):
            print("Invalid Telegram token format; check the configured token file")
            return 1
        client = TelegramClient(token)
        identity = client.call("getMe")
        print(f"Bot: https://t.me/{identity['username']}", flush=True)
        webhook = client.call("getWebhookInfo")
        if webhook.get("url"):
            print("A webhook is configured. Polling cannot start until its owner disables it.")
            return 1
        if args.check:
            print("Telegram credentials checked successfully; webhook is not configured")
            return 0
        allowed = {int(value.strip()) for value in
                   os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").split(",") if value.strip()}
        api_url = args.api_url or os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
        print("RAG API configured; questions and credentials are not logged", flush=True)
        TelegramBot(client, api_url, allowed,
                    float(os.getenv("BOT_API_TIMEOUT", "150"))).run()
    except KeyboardInterrupt:
        print("Telegram bot stopped")
        return 0
    except OSError as error:
        print(f"Cannot read token file: {error.filename}")
        return 1
    except TelegramError as error:
        print(f"Telegram connection failed (code {error.code}). Check network/token or another polling process.")
        return 1
    except ValueError:
        print("Invalid bot configuration; check user IDs and timeout in .env")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
