"""
Embedder — превращает текст в векторные представления.

Поддерживает:
  - sentence-transformers (локально, без интернета)
  - GigaChat Embeddings (через API Сбера)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Базовый интерфейс для всех эмбеддеров."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Преобразовать список текстов в матрицу эмбеддингов.

        Returns:
            ndarray shape (N, dim)
        """

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность вектора."""


# ---------------------------------------------------------------------------
# sentence-transformers (рекомендуется для локальной работы)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Эмбеддер на базе sentence-transformers.

    Рекомендуемые модели:
      - 'intfloat/multilingual-e5-large'        — хорошо для RU+EN
      - 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
      - 'ai-forever/sbert-large-nlu-ru'          — только RU
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        batch_size: int = 32,
        device: str | None = None,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        logger.info("Загрузка модели эмбеддингов: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._normalize = normalize
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Модель загружена, размерность=%d", self._dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 50,
        )
        return np.array(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# GigaChat Embeddings (Сбер)
# ---------------------------------------------------------------------------

class GigaChatEmbedder(BaseEmbedder):
    """
    Эмбеддер через GigaChat API (langchain-gigachat или прямые запросы).

    Требует установки:
        pip install gigachat

    Получите credentials_token на https://developers.sber.ru/portal/products/gigachat
    """

    def __init__(
        self,
        credentials_token: str,
        scope: str = "GIGACHAT_API_PERS",   # GIGACHAT_API_CORP для корпоративных
        model: str = "Embeddings",
        verify_ssl: bool = False,
    ):
        from gigachat import GigaChat

        self._client = GigaChat(
            credentials=credentials_token,
            scope=scope,
            verify_ssl_certs=verify_ssl,
        )
        self._model = model
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self.dimension or 1024
            return np.empty((0, dim), dtype=np.float32)

        result = self._client.embeddings(texts)
        vecs = [item.embedding for item in result.data]
        arr = np.array(vecs, dtype=np.float32)

        if self._dim is None:
            self._dim = arr.shape[1]

        return arr

    @property
    def dimension(self) -> int:
        return self._dim or 1024


# ---------------------------------------------------------------------------
# Chunker — разбивает длинные тексты на куски для индексации
# ---------------------------------------------------------------------------

class TextChunker:
    """Разбивает длинные тексты на перекрывающиеся чанки."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """Разбить текст на чанки по символам."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Пытаемся не разрывать посередине слова
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(" ")
                if last_space > self.chunk_size // 2:
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def split_documents(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """
        Разбить список текстов.

        Returns:
            (chunks, source_indices) — каждый чанк и индекс исходного текста.
        """
        all_chunks: list[str] = []
        source_indices: list[int] = []
        for i, text in enumerate(texts):
            for chunk in self.split(text):
                all_chunks.append(chunk)
                source_indices.append(i)
        return all_chunks, source_indices
