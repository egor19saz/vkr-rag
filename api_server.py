"""
FastAPI REST Server — HTTP API для RAG-системы.

Запуск:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Документация: http://localhost:8000/docs  (Swagger UI)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import PIPELINE_CONFIG
from src.pipeline import RAGPipeline
from src.storage.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ВКР: RAG для научных публикаций",
    description="Анализ научных PDF через GROBID + GigaChat",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные объекты (инициализируются при старте)
pipeline: RAGPipeline | None = None
graph: KnowledgeGraph | None = None


@app.on_event("startup")
async def startup():
    global pipeline, graph
    logger.info("Инициализация RAG Pipeline...")
    pipeline = RAGPipeline(PIPELINE_CONFIG)
    graph = KnowledgeGraph()
    logger.info("Готово.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    role_filter: Optional[str] = None
    source_file: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: list[str]
    model: str
    tokens_used: int


class DocumentInfo(BaseModel):
    source_file: str
    title: str
    authors: list[str]
    year: str
    doi: str
    keywords: list[str]
    sections: list[str]
    paragraphs: int
    references: int


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    documents: int
    authors: int
    keywords: int
    references: int


# ---------------------------------------------------------------------------
# Endpoints: Documents
# ---------------------------------------------------------------------------

@app.post("/documents/upload", response_model=DocumentInfo, summary="Загрузить PDF")
async def upload_pdf(
    file: Annotated[UploadFile, File(description="PDF-файл для загрузки")],
    save_xml: bool = Query(True, description="Сохранить TEI XML на диск"),
):
    """
    Загрузить PDF, обработать через GROBID, проиндексировать.
    Возвращает метаданные документа.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Файл должен быть в формате PDF")

    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        doc = pipeline.ingest_pdf(tmp_path.rename(
            tmp_path.parent / file.filename
        ), save_xml=save_xml)

        # Добавляем в граф знаний
        graph.add_document(doc)

        return DocumentInfo(
            source_file=doc.source_file,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            keywords=doc.keywords,
            sections=[s.title for s in doc.sections],
            paragraphs=len(doc.all_paragraphs()),
            references=len(doc.references),
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/documents", response_model=list[DocumentInfo], summary="Список документов")
async def list_documents():
    """Получить список всех загруженных документов."""
    docs = []
    for name, doc in pipeline._parsed_docs.items():
        info = pipeline.get_document_info(name)
        if info:
            docs.append(DocumentInfo(
                source_file=name,
                title=info["title"],
                authors=info["authors"],
                year=info["year"],
                doi=info["doi"],
                keywords=info["keywords"],
                sections=info["sections"],
                paragraphs=info["paragraphs"],
                references=info["references"],
            ))
    return docs


@app.get("/documents/{filename}", response_model=DocumentInfo, summary="Метаданные документа")
async def get_document(filename: str):
    """Получить метаданные конкретного документа."""
    info = pipeline.get_document_info(filename)
    if not info:
        raise HTTPException(404, f"Документ не найден: {filename}")
    return DocumentInfo(source_file=filename, **info)


@app.get("/documents/{filename}/summary", summary="Резюме документа")
async def get_summary(filename: str):
    """Сгенерировать краткое резюме документа через GigaChat."""
    try:
        summary = pipeline.summarize(filename)
        return {"filename": filename, "summary": summary}
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.get("/documents/{filename}/paragraphs", summary="Параграфы по роли")
async def get_paragraphs(
    filename: str,
    role: Optional[str] = Query(None, description="hypothesis|method|result|related_work|general"),
    limit: int = Query(10, ge=1, le=100),
):
    """Получить параграфы документа, опционально отфильтровав по роли."""
    doc = pipeline._parsed_docs.get(filename)
    if not doc:
        raise HTTPException(404, f"Документ не найден: {filename}")

    if role:
        paragraphs = doc.paragraphs_by_role(role)
    else:
        paragraphs = doc.all_paragraphs()

    return {
        "filename": filename,
        "role_filter": role,
        "total": len(paragraphs),
        "paragraphs": [
            {"section": p.section, "role": p.role, "text": p.text[:300]}
            for p in paragraphs[:limit]
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints: Query / RAG
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, summary="Задать вопрос")
async def query(req: QueryRequest):
    """
    Ответить на вопрос по загруженным документам.
    Использует Hybrid Retrieval + GigaChat.
    """
    if pipeline.vector_store.count() == 0:
        raise HTTPException(400, "База пуста. Сначала загрузите PDF через /documents/upload")

    try:
        report = pipeline.query(
            question=req.question,
            top_k=req.top_k,
            role_filter=req.role_filter,
            source_file=req.source_file,
        )
        return QueryResponse(
            question=report.query,
            answer=report.answer,
            source_documents=report.source_documents,
            model=report.model,
            tokens_used=report.tokens_used,
        )
    except Exception as exc:
        logger.error("Ошибка при обработке запроса: %s", exc)
        raise HTTPException(500, f"Ошибка генерации: {exc}")


@app.post("/query/stream", summary="Стриминг ответа")
async def query_stream(req: QueryRequest):
    """
    Стриминг ответа по токенам (Server-Sent Events).
    Используйте для real-time вывода в UI.
    """
    # Сначала получаем контекст через ретривер
    from src.retrieval.hybrid_retriever import RetrievedChunk
    chunks: list[RetrievedChunk] = pipeline.retriever.retrieve(
        query=req.question, top_k=req.top_k
    )
    context_texts = [c.text for c in chunks]

    def generate():
        for token in pipeline.llm.stream(req.question, context_texts):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Endpoints: Knowledge Graph
# ---------------------------------------------------------------------------

@app.get("/graph/stats", response_model=GraphStats, summary="Статистика графа")
async def graph_stats():
    """Статистика графа знаний."""
    return GraphStats(**graph.stats())


@app.get("/graph/pagerank", summary="Топ документов по PageRank")
async def graph_pagerank(top_n: int = Query(10, ge=1, le=50)):
    """Ранжирование документов по PageRank (цитируемость)."""
    return {"ranked_documents": graph.pagerank_documents(top_n=top_n)}


@app.get("/graph/cited", summary="Самые цитируемые работы")
async def most_cited(top_n: int = Query(10, ge=1, le=50)):
    """Самые цитируемые ссылки в загруженных документах."""
    return {"references": graph.most_cited_references(top_n=top_n)}


@app.get("/graph/authors", summary="Сеть авторов")
async def author_network():
    """Топ авторов по числу публикаций."""
    return {"authors": graph.author_network()}


@app.get("/graph/keywords", summary="Частота ключевых слов")
async def keywords():
    """Ключевые слова и их частота встречаемости."""
    return {"keywords": graph.keyword_cooccurrence()}


@app.get("/graph/related/{filename}", summary="Связанные документы")
async def related_documents(filename: str):
    """Найти документы, связанные с данным через авторов и ключевые слова."""
    return {"related": graph.find_related_documents(filename)}


@app.post("/graph/export/gephi", summary="Экспорт в Gephi")
async def export_gephi():
    """Экспортировать граф в GEXF формат для Gephi."""
    path = Path("./data/graph.gexf")
    graph.export_gephi(path)
    return {"message": f"Граф экспортирован: {path}"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Состояние сервисов")
async def health():
    """Проверить доступность GROBID и GigaChat."""
    grobid_alive = pipeline.grobid.is_alive()
    return {
        "status": "ok",
        "grobid": "up" if grobid_alive else "down",
        "documents_indexed": pipeline.vector_store.count(),
        "graph_nodes": graph.G.number_of_nodes(),
    }
