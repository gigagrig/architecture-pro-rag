# architecture-pro-rag

Проектная работа по созданию RAG-бота для QuantumForge Software.

## Документы

- [Описание проектной работы](ProjectWorkDescription.md)
- [Решения и ссылки на материалы заданий](Project_template.md)
- [Отчёт о проделанной работе и проблемах](report.md)

## Задания 2 и 3

Установка зависимостей с CPU-сборкой PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Подготовка базы, построение индекса и пример поиска:

```bash
python prepare_knowledge_base.py
python build_index.py
python search_index.py "What is the Void Core capable of?"
```

Первый запуск `build_index.py` загружает модель размером около 1,1 ГБ. Последующие запуски используют локальный кеш. Для полностью автономного запуска можно передать сохранённый каталог через `--cache-folder` и добавить `--local-files-only`.

## Задание 4: Telegram-бот и API

- [Настройка LLM, запуск API/бота и автоматическая проверка](task4_rag_bot.md)
- [Инструкция для человека: проверка через Telegram](TELEGRAM_TESTING.md)

После заполнения `.env` запустите в двух терминалах:

```bash
python -m uvicorn rag_bot.api:app --host 127.0.0.1 --port 8000
python run_bot.py
```

API: `http://127.0.0.1:8000/docs`. Проверка ответов: `python check_api.py`.
