FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY rag_bot ./rag_bot
COPY run_bot.py ./run_bot.py
CMD ["python", "-m", "uvicorn", "rag_bot.api:app", "--host", "0.0.0.0", "--port", "8000"]
