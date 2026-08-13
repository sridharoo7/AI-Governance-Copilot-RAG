FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config
COPY data ./data
CMD ["uvicorn", "rag_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]

