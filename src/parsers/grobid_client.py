"""
GROBID Client — отправляет PDF на сервер GROBID и получает TEI XML.

GROBID запускается локально через Docker:
  docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
"""

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class GROBIDClient:
    """HTTP-клиент для взаимодействия с GROBID REST API."""

    # Доступные эндпоинты GROBID
    ENDPOINTS = {
        "full_text":        "/api/processFulltextDocument",
        "header":           "/api/processHeaderDocument",
        "references":       "/api/processReferences",
        "citations":        "/api/processCitationList",
    }

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = grobid_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """Проверить, запущен ли GROBID."""
        try:
            resp = requests.get(f"{self.base_url}/api/isalive", timeout=30)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def process_pdf(
        self,
        pdf_path: str | Path,
        mode: str = "full_text",
        consolidate_header: bool = True,
        consolidate_citations: bool = False,
        include_raw_citations: bool = False,
        segment_sentences: bool = True,
    ) -> str:
        """
        Отправить PDF в GROBID и вернуть TEI XML.

        Args:
            pdf_path:               Путь к PDF-файлу.
            mode:                   Режим парсинга: 'full_text' | 'header' | 'references'.
            consolidate_header:     Обогащать метаданные заголовка через CrossRef.
            consolidate_citations:  Обогащать библиографические ссылки.
            include_raw_citations:  Включать сырые строки цитирований.
            segment_sentences:      Сегментировать абзацы на предложения.

        Returns:
            Строка с TEI XML.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        endpoint = self.ENDPOINTS.get(mode)
        if endpoint is None:
            raise ValueError(f"Неизвестный режим: {mode}. Допустимые: {list(self.ENDPOINTS)}")

        url = self.base_url + endpoint

        params = {
            "consolidateHeader":    "1" if consolidate_header else "0",
            "consolidateCitations": "1" if consolidate_citations else "0",
            "includeRawCitations":  "1" if include_raw_citations else "0",
            "segmentSentences":     "1" if segment_sentences else "0",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                with pdf_path.open("rb") as f:
                    response = requests.post(
                        url,
                        files={"input": (pdf_path.name, f, "application/pdf")},
                        data=params,
                        timeout=self.timeout,
                    )
                response.raise_for_status()
                logger.info("GROBID успешно обработал %s (режим=%s)", pdf_path.name, mode)
                return response.text

            except requests.RequestException as exc:
                logger.warning(
                    "Попытка %d/%d не удалась для %s: %s",
                    attempt, self.max_retries, pdf_path.name, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise RuntimeError(
                        f"GROBID не ответил после {self.max_retries} попыток: {exc}"
                    ) from exc

    def process_batch(
        self,
        pdf_dir: str | Path,
        output_dir: str | Path,
        mode: str = "full_text",
        **kwargs,
    ) -> dict[str, str]:
        """
        Батч-обработка всех PDF в директории.

        Returns:
            Словарь {имя_файла: путь_к_xml}.
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, str] = {}
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info("Найдено %d PDF для обработки", len(pdf_files))

        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info("[%d/%d] Обрабатываю %s", i, len(pdf_files), pdf_path.name)
            try:
                xml_content = self.process_pdf(pdf_path, mode=mode, **kwargs)
                xml_path = output_dir / (pdf_path.stem + ".tei.xml")
                xml_path.write_text(xml_content, encoding="utf-8")
                results[pdf_path.name] = str(xml_path)
                logger.info("  → Сохранено: %s", xml_path)
            except Exception as exc:
                logger.error("  ✗ Ошибка при обработке %s: %s", pdf_path.name, exc)
                results[pdf_path.name] = ""

        return results
