"""Private-chat Telegram adapter; every question goes through the HTTP API."""

import time

import requests

from rag_bot.schemas import Answer


HELP = (
    "Я отвечаю по базе знаний вымышленного мира.\n"
    "Напишите вопрос на русском или английском, например:\n"
    "Кто отец Kael Ardyn?\n"
    "Какие объекты способен уничтожать Void Core?\n\n"
    "Я покажу ответ, краткое обоснование и источники. "
    "Если фактов не хватает, отвечу: «Я не знаю».\n"
    "Каждый вопрос независим: повторяйте имя сущности вместо «он» или «она».\n\n"
    "/start — начало работы\n/help — справка\n/health — состояние API"
)


class TelegramError(Exception):
    def __init__(self, code: int = 0, retry_after: int = 5):
        super().__init__(f"Telegram request failed (code {code})")
        self.code = code
        self.retry_after = retry_after


class TelegramClient:
    def __init__(self, token: str):
        self._base = f"https://api.telegram.org/bot{token}"

    def call(self, method: str, **payload):
        try:
            response = requests.post(f"{self._base}/{method}", json=payload, timeout=(10, 40))
            result = response.json()
            if not response.ok or not result.get("ok"):
                raise TelegramError(result.get("error_code", response.status_code),
                                    result.get("parameters", {}).get("retry_after", 5))
            return result["result"]
        except (requests.RequestException, ValueError, KeyError):
            # Request exceptions contain the token-bearing URL; never log them.
            raise TelegramError() from None

    def send(self, chat_id: int, text: str) -> None:
        # 2000 Unicode code points also fit the 4096 UTF-16-unit Telegram limit.
        for start in range(0, len(text), 2000):
            self.call("sendMessage", chat_id=chat_id, text=text[start:start + 2000])


def format_answer(answer: Answer) -> str:
    parts = [answer.answer]
    if answer.explanation:
        parts.append("Обоснование:\n" + "\n".join(
            f"{number}. {step}" for number, step in enumerate(answer.explanation, 1)))
    if answer.sources:
        parts.append("Источники:\n" + "\n".join(
            f"• {source.title} — {source.source} ({source.chunk_id})\n«{source.quote}»"
            for source in answer.sources))
    return "\n\n".join(parts)


class TelegramBot:
    def __init__(self, telegram: TelegramClient, api_url: str,
                 allowed_users: set[int], api_timeout: float = 150):
        self.telegram = telegram
        self.api_url = api_url.rstrip("/")
        self.allowed_users = allowed_users
        self.api_timeout = api_timeout

    def handle(self, update: dict) -> None:
        message = update.get("message")
        if not message or message.get("chat", {}).get("type") != "private":
            return
        chat_id = message["chat"]["id"]
        if self.allowed_users and message.get("from", {}).get("id") not in self.allowed_users:
            self.telegram.send(chat_id, "Доступ к учебному боту ограничен владельцем.")
            return
        text = message.get("text", "").strip()
        command = text.split(maxsplit=1)[0].split("@")[0] if text else ""
        if command in {"/start", "/help"}:
            self.telegram.send(chat_id, HELP)
            return
        if command == "/health":
            try:
                result = requests.get(f"{self.api_url}/health", timeout=10)
                result.raise_for_status()
                status = result.json()
                reply = (f"API работает. В индексе {status['chunks']} фрагментов.\n"
                         f"Модель: {status['llm_model']}.\n"
                         "Для проверки генерации отправьте вопрос по базе знаний.")
            except (requests.RequestException, ValueError, KeyError):
                reply = "API недоступно. Попросите владельца проверить запуск сервера."
            self.telegram.send(chat_id, reply)
            return
        if command.startswith("/"):
            self.telegram.send(chat_id, "Неизвестная команда. Список команд: /help")
            return
        if not text:
            self.telegram.send(chat_id, "Отправьте вопрос текстом: файлы и голосовые сообщения не поддерживаются.")
            return
        if len(text) > 2000:
            self.telegram.send(chat_id, "Сократите вопрос до 2000 символов.")
            return
        try:
            self.telegram.call("sendChatAction", chat_id=chat_id, action="typing")
        except TelegramError:
            pass
        try:
            result = requests.post(f"{self.api_url}/ask", json={"question": text},
                                   timeout=(10, self.api_timeout))
            if result.status_code == 429:
                reply = "Сервис занят. Повторите вопрос через несколько секунд."
            else:
                result.raise_for_status()
                reply = format_answer(Answer.model_validate(result.json()))
        except (requests.RequestException, ValueError):
            reply = "Не удалось получить ответ от API. Попробуйте позже; это техническая ошибка."
        self.telegram.send(chat_id, reply)

    def run(self) -> None:
        offset = 0
        print("Telegram polling started; press Ctrl+C to stop", flush=True)
        while True:
            try:
                updates = self.telegram.call("getUpdates", offset=offset, timeout=30,
                                             allowed_updates=["message"])
                for update in updates:
                    try:
                        self.handle(update)
                    except TelegramError as error:
                        if error.code != 403:
                            raise
                        print("Message skipped: Telegram delivery forbidden", flush=True)
                    offset = update["update_id"] + 1
            except TelegramError as error:
                if error.code in {401, 404, 409}:
                    raise
                print(f"Telegram temporarily unavailable (code {error.code}); retrying", flush=True)
                time.sleep(min(max(error.retry_after, 1), 60))
