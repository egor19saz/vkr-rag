"""
XML Processor — разбирает TEI XML, который возвращает GROBID.

Извлекает:
  - Метаданные (авторы, название, аннотация, год, DOI)
  - Структурированные секции (Introduction, Methods, Results, …)
  - Параграфы с ролями (hypothesis / related_work / result / method / general)
  - Библиографические ссылки
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)

# TEI namespace
TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """Библиографическая ссылка."""
    ref_id:  str
    title:   str = ""
    authors: list[str] = field(default_factory=list)
    year:    str = ""
    doi:     str = ""
    raw:     str = ""


@dataclass
class Paragraph:
    """Параграф с семантической ролью."""
    text:       str
    section:    str = ""
    role:       str = "general"   # hypothesis | related_work | result | method | general
    ref_ids:    list[str] = field(default_factory=list)  # цитируемые ссылки


@dataclass
class Section:
    """Секция документа."""
    title:      str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Результат разбора одного документа."""
    source_file: str
    title:       str = ""
    abstract:    str = ""
    authors:     list[str] = field(default_factory=list)
    year:        str = ""
    doi:         str = ""
    keywords:    list[str] = field(default_factory=list)
    sections:    list[Section] = field(default_factory=list)
    references:  list[Reference] = field(default_factory=list)

    # -----------------------------------------------------------------------

    def all_paragraphs(self) -> list[Paragraph]:
        """Все параграфы документа (плоский список)."""
        return [p for sec in self.sections for p in sec.paragraphs]

    def paragraphs_by_role(self, role: str) -> list[Paragraph]:
        return [p for p in self.all_paragraphs() if p.role == role]

    def to_plain_text(self) -> str:
        """Весь текст документа одной строкой (для чанкинга)."""
        parts = [self.title, self.abstract]
        for sec in self.sections:
            parts.append(sec.title)
            parts.extend(p.text for p in sec.paragraphs)
        return "\n\n".join(filter(None, parts))


# ---------------------------------------------------------------------------
# Classifier: определяем роль параграфа по секции + ключевым словам
# ---------------------------------------------------------------------------

_ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("hypothesis", [
        r"hypothes", r"proposes?\s+that", r"we\s+assume", r"предполага",
        r"гипотез", r"мы\s+предполаг",
    ]),
    ("related_work", [
        r"related\s+work", r"background", r"literature", r"previous\s+work",
        r"обзор\s+литератур", r"смежные\s+работ", r"предыдущие\s+исследован",
    ]),
    ("result", [
        r"result", r"finding", r"показал", r"результат", r"мы\s+обнаружил",
        r"наши\s+результат",
    ]),
    ("method", [
        r"method", r"approach", r"algorithm", r"метод", r"алгоритм",
        r"подход", r"архитектур",
    ]),
]

_SECTION_ROLE_MAP: dict[str, str] = {
    "introduction":    "general",
    "related work":    "related_work",
    "background":      "related_work",
    "methodology":     "method",
    "methods":         "method",
    "approach":        "method",
    "experiments":     "result",
    "results":         "result",
    "evaluation":      "result",
    "discussion":      "result",
    "conclusion":      "general",
    # Russian
    "введение":        "general",
    "обзор литературы": "related_work",
    "методология":     "method",
    "методы":          "method",
    "эксперименты":    "result",
    "результаты":      "result",
    "обсуждение":      "result",
    "заключение":      "general",
}


def _classify_paragraph(text: str, section_title: str) -> str:
    """Определить роль параграфа."""
    # 1. По заголовку секции
    norm_sec = section_title.lower()
    for key, role in _SECTION_ROLE_MAP.items():
        if key in norm_sec:
            return role

    # 2. По тексту параграфа
    for role, patterns in _ROLE_PATTERNS:
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return role

    return "general"


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class TEIXMLProcessor:
    """Разбирает TEI XML, сформированный GROBID."""

    def parse_file(self, xml_path: str | Path) -> ParsedDocument:
        xml_path = Path(xml_path)
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        return self._parse_root(root, source_file=xml_path.name)

    def parse_string(self, xml_text: str, source_file: str = "<string>") -> ParsedDocument:
        root = etree.fromstring(xml_text.encode("utf-8"))
        return self._parse_root(root, source_file=source_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_root(self, root: etree._Element, source_file: str) -> ParsedDocument:
        doc = ParsedDocument(source_file=source_file)

        # Метаданные заголовка
        header = root.find(".//tei:teiHeader", NS)
        if header is not None:
            doc.title    = self._extract_title(header)
            doc.authors  = self._extract_authors(header)
            doc.year     = self._extract_year(header)
            doc.doi      = self._extract_doi(header)
            doc.keywords = self._extract_keywords(header)
            doc.abstract = self._extract_abstract(header)

        # Тело документа
        body = root.find(".//tei:body", NS)
        if body is not None:
            doc.sections = self._extract_sections(body)

        # Библиографические ссылки
        back = root.find(".//tei:back", NS)
        if back is not None:
            doc.references = self._extract_references(back)

        logger.info(
            "Разобран документ: '%s' (%d секций, %d ссылок)",
            doc.title[:60] if doc.title else source_file,
            len(doc.sections),
            len(doc.references),
        )
        return doc

    # ---------- Метаданные ----------

    def _extract_title(self, header: etree._Element) -> str:
        el = header.find(".//tei:titleStmt/tei:title[@type='main']", NS)
        if el is None:
            el = header.find(".//tei:titleStmt/tei:title", NS)
        return self._text(el)

    def _extract_authors(self, header: etree._Element) -> list[str]:
        authors = []
        for pers in header.findall(".//tei:sourceDesc//tei:persName", NS):
            forename = self._text(pers.find("tei:forename", NS))
            surname  = self._text(pers.find("tei:surname", NS))
            name = f"{forename} {surname}".strip()
            if name:
                authors.append(name)
        return authors

    def _extract_year(self, header: etree._Element) -> str:
        el = header.find(".//tei:date[@type='published']", NS)
        if el is None:
            el = header.find(".//tei:date", NS)
        if el is not None:
            return el.get("when", "") or self._text(el)
        return ""

    def _extract_doi(self, header: etree._Element) -> str:
        el = header.find(".//tei:idno[@type='DOI']", NS)
        return self._text(el)

    def _extract_keywords(self, header: etree._Element) -> list[str]:
        kws = []
        for el in header.findall(".//tei:keywords/tei:term", NS):
            t = self._text(el)
            if t:
                kws.append(t)
        return kws

    def _extract_abstract(self, header: etree._Element) -> str:
        el = header.find(".//tei:abstract", NS)
        return self._inner_text(el)

    # ---------- Тело ----------

    def _extract_sections(self, body: etree._Element) -> list[Section]:
        sections: list[Section] = []

        for div in body.findall(".//tei:div", NS):
            head = div.find("tei:head", NS)
            sec_title = self._text(head) if head is not None else "Untitled"
            paragraphs: list[Paragraph] = []

            for p_el in div.findall("tei:p", NS):
                text = self._inner_text(p_el)
                if len(text) < 20:
                    continue
                ref_ids = [
                    r.get("{http://www.w3.org/XML/1998/namespace}id", "")
                    for r in p_el.findall(".//tei:ref[@type='bibr']", NS)
                ]
                role = _classify_paragraph(text, sec_title)
                paragraphs.append(Paragraph(text=text, section=sec_title, role=role, ref_ids=ref_ids))

            if paragraphs:
                sections.append(Section(title=sec_title, paragraphs=paragraphs))

        return sections

    # ---------- Ссылки ----------

    def _extract_references(self, back: etree._Element) -> list[Reference]:
        refs: list[Reference] = []
        for bibl in back.findall(".//tei:listBibl/tei:biblStruct", NS):
            ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
            analytic = bibl.find("tei:analytic", NS)
            monogr   = bibl.find("tei:monogr", NS)

            title = ""
            if analytic is not None:
                title = self._text(analytic.find("tei:title", NS))
            if not title and monogr is not None:
                title = self._text(monogr.find("tei:title", NS))

            authors = []
            src = analytic if analytic is not None else monogr
            if src is not None:
                for pers in src.findall(".//tei:persName", NS):
                    forename = self._text(pers.find("tei:forename", NS))
                    surname  = self._text(pers.find("tei:surname", NS))
                    name = f"{forename} {surname}".strip()
                    if name:
                        authors.append(name)

            year = ""
            date_el = bibl.find(".//tei:date[@type='published']", NS)
            if date_el is None:
                date_el = bibl.find(".//tei:date", NS)
            if date_el is not None:
                year = date_el.get("when", "")[:4]

            doi_el = bibl.find(".//tei:idno[@type='DOI']", NS)
            doi = self._text(doi_el)

            raw_el = bibl.find(".//tei:note[@type='raw_reference']", NS)
            raw = self._text(raw_el)

            refs.append(Reference(
                ref_id=ref_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                raw=raw,
            ))
        return refs

    # ---------- Утилиты ----------

    @staticmethod
    def _text(el: Optional[etree._Element]) -> str:
        if el is None:
            return ""
        return (el.text or "").strip()

    @staticmethod
    def _inner_text(el: Optional[etree._Element]) -> str:
        """Извлечь весь текст элемента вместе с дочерними тегами."""
        if el is None:
            return ""
        return " ".join(
            t.strip() for t in el.itertext() if t.strip()
        )
