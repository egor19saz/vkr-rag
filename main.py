"""
main.py — точка входа в проект ВКР.

Запуск:
    python main.py --pdf paper.pdf --query "Какова гипотеза авторов?"
    python main.py --dir ./data/pdfs --query "Какие методы используются?"
    python main.py --interactive  # диалоговый режим
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Настройка логирования
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_pipeline():
    """Создать и вернуть RAGPipeline с настройками из config.py."""
    from config import PIPELINE_CONFIG
    from src.pipeline import RAGPipeline
    return RAGPipeline(PIPELINE_CONFIG)


# ---------------------------------------------------------------------------
# CLI режимы
# ---------------------------------------------------------------------------

def cmd_ingest_and_query(args: argparse.Namespace) -> None:
    """Загрузить PDF и выполнить запрос."""
    pipeline = build_pipeline()

    # Загрузка
    if args.pdf:
        pdf_path = Path(args.pdf)
        logger.info("Загрузка файла: %s", pdf_path)
        doc = pipeline.ingest_pdf(pdf_path, save_xml=True)
        print(f"\n✓ Загружено: {doc.title or pdf_path.name}")
        print(f"  Авторы: {', '.join(doc.authors[:3]) or '—'}")
        print(f"  Год: {doc.year or '—'}")
        print(f"  Секций: {len(doc.sections)}, Параграфов: {len(doc.all_paragraphs())}\n")

    elif args.dir:
        pdf_dir = Path(args.dir)
        docs = pipeline.ingest_directory(pdf_dir)
        print(f"\n✓ Загружено {len(docs)} документов из {pdf_dir}\n")

    # Запрос
    if args.query:
        print(f"📋 Вопрос: {args.query}\n")
        report = pipeline.query(
            args.query,
            role_filter=args.role,
        )
        print("=" * 60)
        print(report.answer)
        print("=" * 60)
        if report.source_documents:
            print(f"\nИсточники: {', '.join(report.source_documents)}")
        if report.tokens_used:
            print(f"Токенов использовано: {report.tokens_used}")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Интерактивный диалог с базой знаний."""
    pipeline = build_pipeline()

    print("=" * 60)
    print("  RAG-система для анализа научных публикаций")
    print("  Команды: :load <путь> | :info <файл> | :summary <файл> | :exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nВопрос > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        if user_input == ":exit":
            print("До свидания!")
            break

        elif user_input.startswith(":load "):
            pdf_path = Path(user_input[6:].strip())
            try:
                doc = pipeline.ingest_pdf(pdf_path, save_xml=True)
                print(f"✓ Загружено: {doc.title or pdf_path.name}")
            except Exception as exc:
                print(f"✗ Ошибка: {exc}")

        elif user_input.startswith(":info "):
            name = user_input[6:].strip()
            info = pipeline.get_document_info(name)
            if info:
                print(f"\nНазвание: {info['title']}")
                print(f"Авторы:   {', '.join(info['authors'][:5])}")
                print(f"Год:      {info['year']}")
                print(f"DOI:      {info['doi']}")
                print(f"Секции:   {', '.join(info['sections'][:5])}")
            else:
                print(f"Документ не найден: {name}")

        elif user_input.startswith(":summary "):
            name = user_input[9:].strip()
            try:
                summary = pipeline.summarize(name)
                print(f"\n{summary}")
            except Exception as exc:
                print(f"✗ Ошибка: {exc}")

        else:
            # Обычный вопрос
            try:
                report = pipeline.query(user_input)
                print(f"\n{report.answer}")
                if report.source_documents:
                    print(f"\n[Источники: {', '.join(report.source_documents)}]")
            except Exception as exc:
                print(f"✗ Ошибка при генерации ответа: {exc}")


def cmd_parse_only(args: argparse.Namespace) -> None:
    """Только разобрать PDF в XML, без LLM."""
    from config import GROBID_URL, GROBID_TIMEOUT, XML_OUTPUT_DIR
    from src.parsers.grobid_client import GROBIDClient
    from src.parsers.xml_processor import TEIXMLProcessor

    grobid = GROBIDClient(grobid_url=GROBID_URL, timeout=GROBID_TIMEOUT)

    if not grobid.is_alive():
        logger.error("GROBID недоступен по адресу %s", GROBID_URL)
        sys.exit(1)

    pdf_path = Path(args.pdf)
    xml_text = grobid.process_pdf(pdf_path)

    xml_dir = Path(XML_OUTPUT_DIR)
    xml_dir.mkdir(parents=True, exist_ok=True)
    xml_path = xml_dir / (pdf_path.stem + ".tei.xml")
    xml_path.write_text(xml_text, encoding="utf-8")
    print(f"✓ XML сохранён: {xml_path}")

    if args.verbose:
        processor = TEIXMLProcessor()
        doc = processor.parse_string(xml_text, source_file=pdf_path.name)
        print(f"\nНазвание: {doc.title}")
        print(f"Авторы:   {', '.join(doc.authors)}")
        print(f"Год:      {doc.year}")
        print(f"Секций:   {len(doc.sections)}")
        print(f"Ссылок:   {len(doc.references)}")
        print("\nПервые параграфы:")
        for p in doc.all_paragraphs()[:3]:
            print(f"  [{p.role}] {p.text[:120]}...")


# ---------------------------------------------------------------------------
# Аргументы CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ВКР: RAG-система для анализа научных публикаций (PDF → GigaChat)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # query — загрузить и спросить
    q_parser = subparsers.add_parser("query", help="Загрузить PDF и задать вопрос")
    q_parser.add_argument("--pdf",   help="Путь к одному PDF")
    q_parser.add_argument("--dir",   help="Директория с PDF")
    q_parser.add_argument("--query", help="Вопрос к базе знаний")
    q_parser.add_argument("--role",  help="Фильтр по роли: hypothesis|method|result|related_work")

    # interactive — диалог
    subparsers.add_parser("interactive", help="Интерактивный режим")

    # parse — только PDF→XML
    p_parser = subparsers.add_parser("parse", help="Только PDF → XML (без LLM)")
    p_parser.add_argument("--pdf", required=True, help="Путь к PDF")
    p_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.command == "query":
        cmd_ingest_and_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "parse":
        cmd_parse_only(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
