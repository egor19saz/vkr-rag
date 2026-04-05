"""
Конфигурация проекта.

Чувствительные данные (токены, пароли) лучше хранить в .env файле:
    GIGACHAT_CREDENTIALS=ваш_токен_base64

Загрузка из .env:
    pip install python-dotenv
    from dotenv import load_dotenv; load_dotenv()
"""

import os

# ---------------------------------------------------------------------------
# GigaChat
# ---------------------------------------------------------------------------

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "YOUR_BASE64_CREDENTIALS_HERE")
"""
Как получить:
  1. Зарегистрируйтесь: https://developers.sber.ru/portal/products/gigachat
  2. Создайте проект → получите Client ID и Client Secret
  3. Закодируйте в base64: base64("ClientID:ClientSecret")
  4. Вставьте результат сюда или в .env файл
"""

GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
# Варианты: GIGACHAT_API_PERS (личный) | GIGACHAT_API_B2B | GIGACHAT_API_CORP

GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
# Варианты: GigaChat | GigaChat-Plus | GigaChat-Pro (Pro — самый мощный)

# ---------------------------------------------------------------------------
# GROBID
# ---------------------------------------------------------------------------

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
"""
Запуск GROBID через Docker (выполните в терминале):

    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0

После запуска проверьте: http://localhost:8070/api/isalive
"""
GROBID_TIMEOUT = 120  # секунды

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDER_CONFIG = {
    "type":       "sentence_transformers",   # "sentence_transformers" | "gigachat"
    "model_name": "intfloat/multilingual-e5-large",
    # Альтернативы:
    # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  — быстрее, хуже качество
    # "ai-forever/sbert-large-nlu-ru"  — только русский, хорошее качество
    "device": None,  # None = авто (GPU если доступен), "cpu", "cuda", "mps" (Apple)
}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 512   # символов
CHUNK_OVERLAP = 64    # символов перекрытия между чанками

# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

VECTOR_STORE_DIR = "./data/chroma"
COLLECTION_NAME  = "papers"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K   = 5    # сколько чанков использовать как контекст
VECTOR_WEIGHT     = 0.7  # вес векторного поиска (BM25 weight = 1 - VECTOR_WEIGHT)

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

LLM_TEMPERATURE = 0.2   # низкая температура = более предсказуемые, точные ответы
LLM_MAX_TOKENS  = 2048

# ---------------------------------------------------------------------------
# Пути к данным
# ---------------------------------------------------------------------------

PDF_INPUT_DIR  = "./data/pdfs"
XML_OUTPUT_DIR = "./data/xml"

# ---------------------------------------------------------------------------
# Сборка config dict для RAGPipeline
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = {
    "gigachat_credentials": GIGACHAT_CREDENTIALS,
    "gigachat_scope":       GIGACHAT_SCOPE,
    "gigachat_model":       GIGACHAT_MODEL,
    "grobid_url":           GROBID_URL,
    "grobid_timeout":       GROBID_TIMEOUT,
    "embedder":             EMBEDDER_CONFIG,
    "chunk_size":           CHUNK_SIZE,
    "chunk_overlap":        CHUNK_OVERLAP,
    "vector_store_dir":     VECTOR_STORE_DIR,
    "collection_name":      COLLECTION_NAME,
    "retrieval_top_k":      RETRIEVAL_TOP_K,
    "vector_weight":        VECTOR_WEIGHT,
    "llm_temperature":      LLM_TEMPERATURE,
}
