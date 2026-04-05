"""
RAG Pipeline — оркестратор всей системы.

Объединяет все компоненты:
  PDF → GROBID → XML → Embeddings → VectorStore + BM25 → HybridRetriever → GigaChat → Отчёт
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.parsers.grobid_client import GROBIDClient
from src.parsers.xml_processor import TEIXMLProcessor, ParsedDocument
from src.embeddings.embedder import BaseEmbedder, TextChunker
from src.storage.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.llm.gigachat_client import GigaChatLLM, AnalyticsReport

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Полный RAG-пайплайн для анализа научных PDF.

    Пример использования:
        pipeline = RAGPipeline(config)
        pipeline.ingest_pdf("paper.pdf")
        report = pipeline.query("Какова гипотеза авторов?")
        print(report.answer)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Словарь настроек (см. config.py).
        """
        self.config = config

        # 1. GROBID
        self.grobid = GROBIDClient(
            grobid_url=config.get("grobid_url", "http://localhost:8070"),
            timeout=config.get("grobid_timeout", 120),
        )

        # 2. XML Parser
        self.xml_processor = TEIXMLProcessor()

        # 3. Chunker
        self.chunker = TextChunker(
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 64),
        )

        # 4. Embedder (lazy init — не инициализируем дважды)
        self._embedder: BaseEmbedder | None = None
        self._embedder_config = config.get("embedder", {})

        # 5. VectorStore
        self.vector_store = VectorStore(
            persist_dir=config.get("vector_store_dir", "./data/chroma"),
            collection_name=config.get("collection_name", "papers"),
        )

        # 6. Retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=config.get("retrieval_top_k", 5),
            vector_weight=config.get("vector_weight", 0.7),
        )

        # 7. LLM
        self.llm = GigaChatLLM(
            credentials_token=config["gigachat_credentials"],
            scope=config.get("gigachat_scope", "GIGACHAT_API_PERS"),
            model=config.get("gigachat_model", "GigaChat"),
            temperature=config.get("llm_temperature", 0.2),
        )

        # Кеш разобранных документов
        self._parsed_docs: dict[str, ParsedDocument] = {}
        self._all_chunks:  list[str] = []
        self._all_metas:   list[dict] = []

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    def _create_embedder(self) -> BaseEmbedder:
        cfg = self._embedder_config
        kind = cfg.get("type", "sentence_transformers")

        if kind == "gigachat":
            from src.embeddings.embedder import GigaChatEmbedder
            return GigaChatEmbedder(
                credentials_token=self.config["gigachat_credentials"],
                scope=self.config.get("gigachat_scope", "GIGACHAT_API_PERS"),
            )
        else:
            from src.embeddings.embedder import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder(
                model_name=cfg.get("model_name", "intfloat/multilingual-e5-large"),
                device=cfg.get("device", None),
            )

    # ------------------------------------------------------------------
    # Ingestion: загрузка PDF в базу знаний
    # ------------------------------------------------------------------

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        save_xml: bool = True,
        xml_output_dir: str | Path = "./data/xml",
    ) -> ParsedDocument:
        """
        Полный цикл загрузки PDF:
          1. Парсинг через GROBID → TEI XML
          2. Разбор XML → структурированный документ
          3. Чанкинг параграфов
          4. Создание эмбеддингов
          5. Сохранение в VectorStore + обновление BM25

        Returns:
            ParsedDocument — структурированное представление документа.
        """
        pdf_path = Path(pdf_path)
        logger.info("═══ Начало загрузки: %s ═══", pdf_path.name)

        # Шаг 1: GROBID
        logger.info("[1/5] GROBID парсинг...")
        if not self.grobid.is_alive():
          raise RuntimeError(
            "GROBID недоступен. Убедитесь, что сервис запущен "
            "или используйте публичный инстанс (kermitt2-grobid.hf.space)."
          )
        xml_text = self.grobid.process_pdf(pdf_path)

        # Опционально: сохранить XML
        if save_xml:
            xml_dir = Path(xml_output_dir)
            xml_dir.mkdir(parents=True, exist_ok=True)
            xml_path = xml_dir / (pdf_path.stem + ".tei.xml")
            xml_path.write_text(xml_text, encoding="utf-8")
            logger.info("    XML сохранён: %s", xml_path)

        # Шаг 2: XML → ParsedDocument
        logger.info("[2/5] Разбор XML...")
        doc = self.xml_processor.parse_string(xml_text, source_file=pdf_path.name)
        self._parsed_docs[pdf_path.name] = doc

        # Шаг 3: Чанкинг
        logger.info("[3/5] Чанкинг параграфов...")
        paragraphs = [p.text for p in doc.all_paragraphs()]
        if doc.abstract:
            paragraphs.insert(0, f"[Abstract] {doc.abstract}")

        chunks, source_indices = self.chunker.split_documents(paragraphs)
        metas = []
        all_paragraphs = doc.all_paragraphs()
        for idx in source_indices:
            if idx < len(all_paragraphs):
                p = all_paragraphs[idx]
                meta = {
                    "source_file": pdf_path.name,
                    "section":     p.section,
                    "role":        p.role,
                    "title":       doc.title[:200],
                    "year":        doc.year,
                    "authors":     ", ".join(doc.authors[:3]),
                }
            else:
                meta = {"source_file": pdf_path.name, "role": "abstract"}
            metas.append(meta)

        logger.info("    Создано %d чанков", len(chunks))

        # Шаг 4: Эмбеддинги
        logger.info("[4/5] Создание эмбеддингов...")
        embeddings = self.embedder.embed(chunks)

        # Шаг 5: Сохранение
        logger.info("[5/5] Сохранение в VectorStore...")
        self.vector_store.add(texts=chunks, embeddings=embeddings, metadatas=metas)

        # Обновляем BM25
        self._all_chunks.extend(chunks)
        self._all_metas.extend(metas)
        self.retriever.build_bm25_index(self._all_chunks, self._all_metas)

        logger.info("═══ Загрузка завершена: %s (%d чанков) ═══", pdf_path.name, len(chunks))
        return doc

    def ingest_directory(self, pdf_dir: str | Path, **kwargs) -> list[ParsedDocument]:
        """Загрузить все PDF из директории."""
        pdf_dir = Path(pdf_dir)
        docs = []
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            try:
                doc = self.ingest_pdf(pdf_file, **kwargs)
                docs.append(doc)
            except Exception as exc:
                logger.error("Ошибка при загрузке %s: %s", pdf_file.name, exc)
        return docs

    # ------------------------------------------------------------------
    # Query: запрос к базе знаний
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
        role_filter: str | None = None,
        source_file: str | None = None,
    ) -> AnalyticsReport:
        """
        Ответить на вопрос по загруженным статьям.

        Args:
            question:    Вопрос исследователя.
            top_k:       Количество контекстных фрагментов.
            role_filter: Фильтровать по роли ('hypothesis', 'result', 'method', …).
            source_file: Ограничить поиск одним файлом.

        Returns:
            AnalyticsReport с ответом и метаданными.
        """
        # Формируем фильтр метаданных
        where: dict | None = None
        if role_filter or source_file:
            conditions = {}
            if role_filter:
                conditions["role"] = role_filter
            if source_file:
                conditions["source_file"] = source_file
            where = conditions if len(conditions) == 1 else {"$and": [{"k": v} for k, v in conditions.items()]}

        # Ретривал
        chunks: list[RetrievedChunk] = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            metadata_filter=where,
        )

        if not chunks:
            return AnalyticsReport(
                query=question,
                answer="По данному запросу релевантных фрагментов не найдено. "
                       "Пожалуйста, загрузите статьи через pipeline.ingest_pdf().",
                context_chunks=[],
                source_documents=[],
            )

        context_texts = [c.text for c in chunks]
        source_docs = list({c.metadata.get("source_file", "") for c in chunks})

        # Генерация через GigaChat
        report = self.llm.generate(
            query=question,
            context_chunks=context_texts,
            source_documents=source_docs,
        )

        logger.info(
            "Запрос обработан: '%s' → %d символов ответа",
            question[:50], len(report.answer),
        )
        return report

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summarize(self, pdf_path_or_name: str | Path) -> str:
        """Получить краткое резюме загруженной статьи."""
        name = Path(pdf_path_or_name).name
        doc = self._parsed_docs.get(name)
        if doc is None:
            raise ValueError(f"Документ не найден: {name}. Сначала вызовите ingest_pdf().")
        text = doc.to_plain_text()
        return self.llm.summarize_paper(text)

    def get_document_info(self, pdf_name: str) -> dict | None:
        doc = self._parsed_docs.get(pdf_name)
        if doc is None:
            return None
        return {
            "title":      doc.title,
            "authors":    doc.authors,
            "year":       doc.year,
            "doi":        doc.doi,
            "keywords":   doc.keywords,
            "sections":   [s.title for s in doc.sections],
            "paragraphs": len(doc.all_paragraphs()),
            "references": len(doc.references),
        }
