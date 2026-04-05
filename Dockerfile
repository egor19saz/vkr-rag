FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости (слой кешируется если requirements не изменился)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY . .

# Директории для данных
RUN mkdir -p data/pdfs data/xml data/chroma

# Порты: 8000 (API) и 7860 (Gradio)
EXPOSE 8000 7860

# По умолчанию запускается API
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
