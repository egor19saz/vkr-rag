# Advanced RAG-система для анализа научных публикаций

> **ВКР**: Разработка Advanced RAG-системы для анализа научных публикаций
> Стек: GROBID · ChromaDB · BM25 · NetworkX · GigaChat · Streamlit · FastAPI

---

## Описание каждого файла

### Корневой уровень

| Файл | Назначение |
|------|-----------|
| `streamlit_app.py` | Главное веб-приложение. 4 вкладки: вопрос-ответ, анализ документа, граф знаний, справка. Запуск: `streamlit run streamlit_app.py` |
| `api_server.py` | FastAPI REST API. Все функции через HTTP. Swagger UI на `/docs` |
| `main.py` | CLI-интерфейс. Три режима: `parse`, `query`, `interactive` |
| `config.py` | Единый файл конфигурации: токен GigaChat, URL GROBID, размер чанков |
| `requirements.txt` | Python-зависимости |
| `Dockerfile` | Docker-образ приложения |
| `docker-compose.yml` | Запуск GROBID + API + Streamlit одной командой |
| `.env.example` | Шаблон файла с секретами |
| `.gitignore` | Исключает `.env`, ChromaDB из git |

### src/parsers/ — Парсинг PDF

| Файл | Назначение |
|------|-----------|
| `grobid_client.py` | HTTP-клиент GROBID. PDF → TEI XML. Батч-обработка, retry |
| `xml_processor.py` | Разбор TEI XML. Метаданные, секции, параграфы. Классифицирует роли: hypothesis / method / result / related_work / general |

### src/embeddings/

| Файл | Назначение |
|------|-----------|
| `embedder.py` | SentenceTransformerEmbedder (локально) + GigaChatEmbedder (API). TextChunker — разбивка текста на перекрывающиеся чанки |

### src/storage/

| Файл | Назначение |
|------|-----------|
| `vector_store.py` | ChromaDB: хранение векторов на диск, косинусный поиск, фильтры |
| `knowledge_graph.py` | NetworkX граф: документы, авторы, ключевые слова, ссылки. PageRank, поиск связанных статей. Экспорт в Gephi |

### src/retrieval/

| Файл | Назначение |
|------|-----------|
| `hybrid_retriever.py` | BM25 + Vector search + RRF fusion. Настраиваемые веса |

### src/llm/

| Файл | Назначение |
|------|-----------|
| `gigachat_client.py` | GigaChat: ответы, стриминг, резюме статьи |

### src/

| Файл | Назначение |
|------|-----------|
| `pipeline.py` | RAGPipeline — оркестратор: `ingest_pdf()` + `query()` |

### tests/

| Файл | Назначение |
|------|-----------|
| `tests/test_all.py` | 30+ unit-тестов: XML парсер, чанкер, VectorStore, ретривер, граф |

---

## Установка и запуск

### 1. Зависимости

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. GROBID (нужен Docker)

```bash
docker run -t --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

Проверка: http://localhost:8070

### 3. Токен GigaChat

1. Зарегистрироваться: https://developers.sber.ru/portal/products/gigachat
2. Получить Client ID + Client Secret
3. Закодировать:

```bash
echo -n "CLIENT_ID:CLIENT_SECRET" | base64
```

### 4. Файл .env

```bash
cp .env.example .env
# Вставить токен в GIGACHAT_CREDENTIALS=...
```

### 5. Запуск Streamlit

```bash
streamlit run streamlit_app.py
# http://localhost:8501
```

### Альтернативные запуски

```bash
# REST API + Swagger UI
uvicorn api_server:app --reload
# http://localhost:8000/docs

# CLI интерактивный режим
python main.py interactive

# Только парсинг PDF -> XML
python main.py parse --pdf paper.pdf --verbose

# Загрузить + спросить
python main.py query --pdf paper.pdf --query "Какова гипотеза авторов?"
```

### Тесты

```bash
pytest tests/ -v
```

### Docker Compose (всё сразу)

```bash
cp .env.example .env  # заполнить токен
docker-compose up --build
```

---

## Настройки (config.py)

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| GIGACHAT_MODEL | GigaChat | GigaChat / GigaChat-Plus / GigaChat-Pro |
| EMBEDDER model_name | multilingual-e5-large | Модель эмбеддингов |
| CHUNK_SIZE | 512 | Размер чанка в символах |
| CHUNK_OVERLAP | 64 | Перекрытие соседних чанков |
| RETRIEVAL_TOP_K | 5 | Фрагментов в контексте LLM |
| VECTOR_WEIGHT | 0.7 | Вес векторного поиска в RRF |
| LLM_TEMPERATURE | 0.2 | Температура (0=детерминировано) |
