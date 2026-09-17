# Решения проектной работы

## Задание 1. Исследование моделей и инфраструктуры

[Результат исследования моделей и инфраструктуры](task1_model_and_infrastructure_research.md)

## Задание 2. Подготовка базы знаний

В качестве исходной предметной области использована вселенная Star Wars. Скрипт [prepare_knowledge_base.py](prepare_knowledge_base.py) загружает открытые англоязычные статьи Wikipedia через MediaWiki API, удаляет служебные разделы и сведения о создании произведений, а затем заменяет ключевые имена, организации, планеты и технологии.

Результат:

- [каталог базы знаний](knowledge_base/) содержит 38 уникальных документов `entity_*.md`;
- [terms_map.json](knowledge_base/terms_map.json) содержит 162 подмены исходных терминов;
- [SOURCES.md](knowledge_base/SOURCES.md) связывает документы с исходными статьями и содержит сведения о лицензии;
- каждый документ описывает одну сущность, содержит не менее 100 слов и не включает исходные термины из словаря;
- `SOURCES.md` и `terms_map.json` не индексируются, поэтому исходные названия не передаются RAG-боту.

Повторная подготовка базы:

```bash
python prepare_knowledge_base.py --output-dir knowledge_base --min-documents 30
```

## Задание 3. Создание векторного индекса базы знаний

Для эмбеддингов использована модель [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base), выбранная в задании 1. Размерность эмбеддинга — 768.

Документы разделены на чанки по 100–250 слов с перекрытием 40 слов. Для каждого чанка сохранены путь к документу, заголовок, идентификатор и позиции первого и последнего слова. Перед добавлением в FAISS векторы нормализуются, поэтому `IndexFlatIP` выполняет точный поиск по cosine similarity.

Результат построения:

- документов: 38;
- чанков: 189;
- время построения на CPU после загрузки модели: 98,05 секунды;
- [FAISS-индекс](vector_index/faiss.index);
- [метаданные чанков](vector_index/chunks.jsonl);
- [манифест индекса](vector_index/manifest.json);
- [скрипт построения](build_index.py);
- [скрипт поиска](search_index.py).

Построение и поиск:

```bash
python build_index.py --knowledge-dir knowledge_base --output-dir vector_index
python search_index.py "What is the Void Core capable of?" --index-dir vector_index
```

Проверочные запросы:

| Запрос | Первый результат | Score |
|---|---|---:|
| `Which Nullbound Lord is Kael Ardyn's father?` | Xarn Velgor | 0,8499 |
| `What kind of weapon is an arcblade?` | arcblade | 0,8480 |
| `What is the Void Core capable of?` | Void Core | 0,8607 |

## Задание 4. Реализация RAG-бота с техниками промптинга

Реализован [модуль RAG](rag_bot/): загрузка существующего FAISS-индекса, запрос с тем же E5-энкодером, поиск фрагментов, few-shot примеры, краткое пошаговое обоснование по источникам и генерация через Yandex Alice AI LLM Flash (Chat Completions API). Проверяется принадлежность ID источников найденным фрагментам; дословность цитат не проверяется. При недостатке данных возвращается «Я не знаю».

Интерфейсы: [Telegram-бот](run_bot.py) и FastAPI (`POST /ask`, `GET /health`, `/docs`). Бот обращается к тому же API, которое проверяется [скриптом приёмки](check_api.py).

- [Настройка, архитектура и запуск](task4_rag_bot.md).
- [Инструкция проверки через Telegram](TELEGRAM_TESTING.md): пять вопросов с ответами и два без ответа.
- [Автоматические тесты](tests/test_rag_bot.py).
- [Dockerfile](Dockerfile) и [Compose](docker-compose.yml).

Реальные примеры запросов и ответов через API сохранены в [диалогах задания 4](task4_dialogues.md), полный результат с цитатами и временем — в [JSON-отчёте](task4_api_results.json). API и Telegram-бот запускаются одной командой `docker compose up -d --build`.

Скриншоты проверки задания 4:

- [Скриншот 1](screenshots/Task4_1.png).
- [Скриншот 2](screenshots/Task4_2.png).
- [Скриншот 3](screenshots/Task4_3.png).

## Задание 5. Запуск и демонстрация работы бота

Результат ещё не добавлен.

## Задание 6. Автоматическое ежедневное обновление базы знаний

Результат ещё не добавлен.

## Задание 7. Аналитика покрытия и качества базы знаний

Результат ещё не добавлен.
