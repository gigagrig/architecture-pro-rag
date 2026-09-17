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

Используется Yandex Alice AI LLM Flash. Скопируйте `.env.example` в `.env`, положите ключ Yandex в `yandex_ai_api_key.txt`, токен Telegram — в `telegram_bot_token.txt`. После подготовки индекса запустите оба сервиса:

```bash
docker compose up -d --build
```

Для запуска без Docker используйте два терминала:

```bash
python -m uvicorn rag_bot.api:app --host 127.0.0.1 --port 8000
python run_bot.py
```

API: `http://127.0.0.1:8000/docs`. Проверка ответов: `python check_api.py`.

## Задание 5: демонстрация и prompt injection

[Описание защиты и команды запуска](task5_demonstration.md). Реальные ответы сохранены в [текстовом логе](task5_run.log) и [JSON](tasks_artifacts/task5_results.json). Повторная проверка после индексации тестового документа: `python demonstrate_task5.py`.

## Задание 6: ежедневное обновление индекса

[Описание и проверка](task6_index_updates.md), [диаграмма](task6_architecture.puml).
Сервис `index-updater` из Docker Compose обновляет индекс при запуске и ежедневно в 06:00 по Москве.
Новые материалы помещайте в `knowledge_base/entity_*.md`; API подхватывает новую версию перед поиском.
