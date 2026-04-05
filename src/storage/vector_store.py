"""
Vector Store — хранилище векторных представлений на основе ChromaDB.

ChromaDB — простое встроенное векторное хранилище, не требует отдельного сервера.
Документация: https://docs.trychroma.com/
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Обёртка над ChromaDB для хранения и поиска эмбеддингов.

    Данные сохраняются на диск, поэтому переживают перезапуск программы.
    """

    def __init__(
        self,
        persist_dir: str | Path = "./data/chroma",
        collection_name: str = "papers",
    ):
        import chromadb
        from chromadb.config import Settings

        self._persist_dir = str(persist_dir)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore: коллекция '%s' (%d документов)",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Добавление документов
    # ------------------------------------------------------------------

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Добавить тексты с готовыми эмбеддингами.

        Returns:
            Список присвоенных идентификаторов.
        """
        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        logger.info("Добавлено %d чанков в VectorStore", len(texts))
        return ids

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Найти top_k ближайших документов по косинусному расстоянию.

        Returns:
            Список словарей {id, text, metadata, score}.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            hits.append({
                "id":       doc_id,
                "text":     doc,
                "metadata": meta,
                "score":    1.0 - dist,  # cosine similarity
            })

        return hits

    # ------------------------------------------------------------------
    # Служебные методы
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def delete_all(self) -> None:
        """Очистить коллекцию."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.warning("VectorStore очищен")

    def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        result = self._collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if result["ids"]:
            return {"id": doc_id, "text": result["documents"][0], "metadata": result["metadatas"][0]}
        return None
