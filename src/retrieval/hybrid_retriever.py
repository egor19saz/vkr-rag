"""
Hybrid Retriever — комбинирует BM25 (keyword search) и семантический (vector) поиск.

Финальный рейтинг вычисляется через Reciprocal Rank Fusion (RRF),
что даёт стабильный результат без необходимости нормализовывать скоры.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Найденный фрагмент документа."""
    text:        str
    score:       float
    metadata:    dict[str, Any]
    source:      str = ""   # "vector" | "bm25" | "hybrid"
    doc_id:      str = ""


class HybridRetriever:
    """
    Гибридный ретривер: BM25 + Vector + RRF fusion.

    Args:
        vector_store:   Экземпляр VectorStore (ChromaDB).
        embedder:       Экземпляр BaseEmbedder для кодирования запроса.
        top_k:          Сколько результатов возвращать итого.
        vector_weight:  Вес векторного поиска (0..1). BM25 weight = 1 - vector_weight.
        rrf_k:          Параметр RRF (обычно 60).
    """

    def __init__(
        self,
        vector_store,
        embedder,
        top_k: int = 5,
        vector_weight: float = 0.7,
        rrf_k: int = 60,
    ):
        self._vs = vector_store
        self._embedder = embedder
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = 1.0 - vector_weight
        self.rrf_k = rrf_k

        # BM25 индекс (строится при вызове build_bm25_index)
        self._bm25 = None
        self._bm25_docs: list[str] = []
        self._bm25_metas: list[dict] = []

    # ------------------------------------------------------------------
    # Построение BM25 индекса
    # ------------------------------------------------------------------

    def build_bm25_index(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Построить BM25 индекс по переданным текстам.

        Должен быть вызван после загрузки всех документов.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 не установлен. BM25 поиск отключён. pip install rank-bm25")
            return

        self._bm25_docs = texts
        self._bm25_metas = metadatas or [{} for _ in texts]

        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 индекс построен: %d документов", len(texts))

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Найти релевантные фрагменты по запросу.

        Returns:
            Список RetrievedChunk, отсортированных по убыванию релевантности.
        """
        k = top_k or self.top_k

        # 1. Векторный поиск
        vector_results = self._vector_search(query, top_k=k * 2, where=metadata_filter)

        # 2. BM25 поиск
        bm25_results = self._bm25_search(query, top_k=k * 2)

        # 3. RRF fusion
        if vector_results and bm25_results:
            fused = self._rrf_fusion(vector_results, bm25_results, top_k=k)
        elif vector_results:
            fused = vector_results[:k]
        else:
            fused = bm25_results[:k]

        logger.debug(
            "Retrieval query='%s' → вектор:%d BM25:%d → итог:%d",
            query[:50], len(vector_results), len(bm25_results), len(fused),
        )
        return fused

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int, where: dict | None) -> list[RetrievedChunk]:
        try:
            q_vec = self._embedder.embed_one(query)
            hits = self._vs.search(q_vec, top_k=top_k, where=where)
            return [
                RetrievedChunk(
                    text=h["text"],
                    score=h["score"],
                    metadata=h["metadata"],
                    source="vector",
                    doc_id=h["id"],
                )
                for h in hits
            ]
        except Exception as exc:
            logger.error("Ошибка векторного поиска: %s", exc)
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        try:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append(RetrievedChunk(
                        text=self._bm25_docs[idx],
                        score=float(scores[idx]),
                        metadata=self._bm25_metas[idx],
                        source="bm25",
                        doc_id=str(idx),
                    ))
            return results
        except Exception as exc:
            logger.error("Ошибка BM25 поиска: %s", exc)
            return []

    def _rrf_fusion(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results:   list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, RetrievedChunk] = {}

        def _rrf_score(rank: int) -> float:
            return 1.0 / (self.rrf_k + rank + 1)

        # Vector ranks
        for rank, chunk in enumerate(vector_results):
            key = chunk.text[:100]  # текст как ключ дедупликации
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.vector_weight * _rrf_score(rank)
            chunk_map[key] = chunk

        # BM25 ranks
        for rank, chunk in enumerate(bm25_results):
            key = chunk.text[:100]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.bm25_weight * _rrf_score(rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

        # Сортировка и возврат top_k
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        results = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            chunk.score = rrf_scores[key]
            chunk.source = "hybrid"
            results.append(chunk)

        return results
