"""
GigaChat Client — взаимодействие с LLM GigaChat от Сбера.

Документация API: https://developers.sber.ru/docs/ru/gigachat/api/overview
SDK:              pip install gigachat

Получение токена:
  1. Зарегистрируйтесь на https://developers.sber.ru/portal/products/gigachat
  2. Создайте проект и получите credentials (Client ID + Secret → base64)
  3. Передайте его как GIGACHAT_CREDENTIALS в .env или в config.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AnalyticsReport:
    """Аналитический отчёт, сгенерированный LLM."""
    query:            str
    answer:           str
    context_chunks:   list[str]
    source_documents: list[str]
    model:            str = ""
    tokens_used:      int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — научный аналитик, работающий с академическими публикациями.
Твоя задача — отвечать на вопросы исследователя на основе предоставленного контекста.

Правила:
1. Отвечай только на основе контекста. Не придумывай факты.
2. Если в контексте нет ответа — так и скажи честно.
3. Разделяй: что пишет АВТОР статьи (его гипотеза/результаты) и что говорят ДРУГИЕ работы.
4. Ссылайся на конкретные фрагменты контекста.
5. Отвечай на русском языке, структурированно.
"""

USER_PROMPT_TEMPLATE = """Контекст из научных публикаций:
{context}

---
Вопрос исследователя: {query}

Дай аналитический ответ на основе контекста выше."""


def _build_context(chunks: list[str]) -> str:
    """Собрать строку контекста из чанков."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Фрагмент {i}]\n{chunk}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GigaChat Client
# ---------------------------------------------------------------------------

class GigaChatLLM:
    """
    Клиент для генерации аналитических отчётов через GigaChat.

    Args:
        credentials_token:  Base64-токен (Client ID:Secret).
        scope:              'GIGACHAT_API_PERS' (личный) или 'GIGACHAT_API_CORP' (корпоративный).
        model:              Модель GigaChat: 'GigaChat', 'GigaChat-Plus', 'GigaChat-Pro'.
        temperature:        Температура генерации (0.0–1.0).
        max_tokens:         Максимальное количество токенов в ответе.
        verify_ssl:         Проверять SSL (отключите для тестовых сред).
    """

    def __init__(
        self,
        credentials_token: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        verify_ssl: bool = False,
    ):
        try:
            from gigachat import GigaChat
            from gigachat.models import Chat, Messages, MessagesRole
        except ImportError as exc:
            raise ImportError(
                "Установите GigaChat SDK: pip install gigachat"
            ) from exc

        self._GigaChat = GigaChat
        self._Chat = Chat
        self._Messages = Messages
        self._MessagesRole = MessagesRole

        self._credentials = credentials_token
        self._scope = scope
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._verify_ssl = verify_ssl

    def _get_client(self):
        return self._GigaChat(
            credentials=self._credentials,
            scope=self._scope,
            model=self._model,
            verify_ssl_certs=self._verify_ssl,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        source_documents: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> AnalyticsReport:
        """
        Сгенерировать аналитический ответ по вопросу и контексту.

        Args:
            query:            Вопрос пользователя.
            context_chunks:   Релевантные фрагменты из ретривера.
            source_documents: Имена исходных файлов (для метаданных).
            system_prompt:    Переопределить системный промпт.

        Returns:
            AnalyticsReport с ответом и метаданными.
        """
        context_str = _build_context(context_chunks)
        user_message = USER_PROMPT_TEMPLATE.format(context=context_str, query=query)
        sys_prompt = system_prompt or SYSTEM_PROMPT

        payload = self._Chat(
            messages=[
                self._Messages(role=self._MessagesRole.SYSTEM, content=sys_prompt),
                self._Messages(role=self._MessagesRole.USER,   content=user_message),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        logger.info("GigaChat запрос: query='%s' контекст=%d чанков", query[:50], len(context_chunks))

        with self._get_client() as client:
            response = client.chat(payload)

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        logger.info("GigaChat ответ: %d символов, %d токенов", len(answer), tokens)

        return AnalyticsReport(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            source_documents=source_documents or [],
            model=self._model,
            tokens_used=tokens,
        )

    def stream(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """
        Стриминг ответа (генерация по токенам).
        Используйте в FastAPI / Gradio / Streamlit.

        Пример:
            for token in llm.stream(query, chunks):
                print(token, end="", flush=True)
        """
        context_str = _build_context(context_chunks)
        user_message = USER_PROMPT_TEMPLATE.format(context=context_str, query=query)
        sys_prompt = system_prompt or SYSTEM_PROMPT

        payload = self._Chat(
            messages=[
                self._Messages(role=self._MessagesRole.SYSTEM, content=sys_prompt),
                self._Messages(role=self._MessagesRole.USER,   content=user_message),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )

        with self._get_client() as client:
            for chunk in client.stream(payload):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def summarize_paper(self, text: str, max_length: int = 500) -> str:
        """
        Краткое резюме научной статьи.

        Удобно для предпросмотра документа перед детальным анализом.
        """
        prompt = (
            f"Кратко (не более {max_length} слов) изложи суть этой научной работы, "
            f"выделив: цель, методы, ключевые результаты.\n\n{text[:4000]}"
        )
        payload = self._Chat(
            messages=[
                self._Messages(role=self._MessagesRole.SYSTEM, content="Ты — научный редактор."),
                self._Messages(role=self._MessagesRole.USER, content=prompt),
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        with self._get_client() as client:
            resp = client.chat(payload)
        return resp.choices[0].message.content
