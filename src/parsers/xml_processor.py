from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


@dataclass
class Reference:
    ref_id:  str
    title:   str = ""
    authors: list[str] = field(default_factory=list)
    year:    str = ""
    doi:     str = ""
    journal: str = ""
    raw:     str = ""


@dataclass
class Paragraph:
    text:    str
    section: str = ""
    role:    str = "general"
    ref_ids: list[str] = field(default_factory=list)
    is_caption: bool = False 


@dataclass
class Section:
    title:      str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_file:   str
    title:         str = ""
    abstract:      str = ""
    authors:       list[str] = field(default_factory=list)
    affiliations:  list[str] = field(default_factory=list)
    year:          str = ""
    doi:           str = ""
    keywords:      list[str] = field(default_factory=list)
    sections:      list[Section] = field(default_factory=list)
    references:    list[Reference] = field(default_factory=list)
    figure_captions: list[str] = field(default_factory=list)

    def all_paragraphs(self) -> list[Paragraph]:
        return [p for sec in self.sections for p in sec.paragraphs]

    def paragraphs_by_role(self, role: str) -> list[Paragraph]:
        return [p for p in self.all_paragraphs() if p.role == role]

    def to_plain_text(self) -> str:
        parts = [self.title, self.abstract]
        for sec in self.sections:
            parts.append(sec.title)
            parts.extend(p.text for p in sec.paragraphs if not p.is_caption)
        return "\n\n".join(filter(None, parts))

    def to_plain_text_full(self) -> str:
        parts = [self.title, self.abstract]
        for sec in self.sections:
            parts.append(sec.title)
            parts.extend(p.text for p in sec.paragraphs)
        if self.figure_captions:
            parts.extend(self.figure_captions)
        return "\n\n".join(filter(None, parts))



_ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("hypothesis", [
        r"\bhypothes\w+", r"\bwe\s+propose\b", r"\bproposes?\s+that\b",
        r"\bwe\s+assume\b", r"\bour\s+hypothesis\b", r"\bwe\s+conjecture\b",
        r"\bwe\s+argue\s+that\b", r"\bwe\s+claim\s+that\b",
        r"\bпредполага\w+", r"\bгипотез\w+", r"\bмы\s+предполаг\w+",
    ]),
    ("related_work", [
        r"\brelated\s+work\b", r"\bprior\s+work\b", r"\bprevious\s+work\b",
        r"\bexisting\s+approach\w*\b", r"\brecently\s+proposed\b",
        r"\bhave\s+been\s+proposed\b", r"\bstate.of.the.art\b",
        r"\bprevious\s+stud\w+", r"\bhas\s+been\s+shown\b",
        r"\bwas\s+introduced\b", r"\bwere\s+first\b",
        r"\bобзор\s+литератур\w+", r"\bсмежные\s+работ\w+",
        r"\bпредыдущие\s+исследован\w+",
    ]),
    ("result", [
        r"\bwe\s+found\b", r"\bwe\s+observe\b", r"\bour\s+model\s+achiev\w+",
        r"\boutperform\w*\b", r"\baccurac\w+\b", r"\bperformance\b",
        r"\bimprove\w+\b", r"\bdemonstrat\w+\b", r"\bshow\w*\s+that\b",
        r"\bresults?\s+show\b", r"\bresults?\s+indicate\b",
        r"\bachiev\w+\s+\w+\s*%", r"\bdice\s+score\b", r"\bauc\b",
        r"\bspearman\b", r"\bicc\b", r"\bp\s*=\s*[\d.]+\b",
        r"\bнаши\s+результат\w+", r"\bпоказал\w+", r"\bрезультат\w+",
        r"\bмы\s+обнаружил\w+",
    ]),
    ("method", [
        r"\bwe\s+use\b", r"\bwe\s+employ\b", r"\bwe\s+apply\b",
        r"\bwe\s+train\b", r"\bour\s+approach\b", r"\bour\s+method\b",
        r"\barchitecture\b", r"\bpipeline\b", r"\bframework\b",
        r"\balgorithm\b", r"\bmodel\s+consist\w*\b", r"\bproposed\s+method\b",
        r"\bwe\s+implement\b", r"\bwe\s+design\b", r"\bis\s+defined\s+as\b",
        r"\bequation\b", r"\bformula\b", r"\bparameter\w*\b",
        r"\bнаш\s+метод\b", r"\bалгоритм\b", r"\bархитектур\w+",
        r"\bпредложенный\s+метод\b",
    ]),
]

_SECTION_ROLE_MAP: dict[str, str] = {
    "introduction":      "general",
    "background":        "related_work",
    "related":           "related_work",
    "literature":        "related_work",
    "prior":             "related_work",
    "method":            "method",
    "methods":           "method",
    "methodology":       "method",
    "approach":          "method",
    "model":             "method",
    "architecture":      "method",
    "proposed":          "method",
    "implementation":    "method",
    "training":          "method",
    "material":          "method",
    "dataset":           "method",
    "data":              "method",
    "experiment":        "result",
    "experimental":      "result",
    "result":            "result",
    "results":           "result",
    "evaluation":        "result",
    "performance":       "result",
    "validation":        "result",
    "numerical":         "result",
    "clinical":          "result",
    "ablation":          "result",
    "analysis":          "result",
    "discussion":        "result",
    "finding":           "result",
    "benchmark":         "result",
    "comparison":        "result",
    "quantitative":      "result",
    "qualitative":       "result",
    "conclusion":        "general",
    "conclusions":       "general",
    "summary":           "general",
    "limitation":        "general",
    "future":            "general",
    "appendix":          "method",
    "введение":          "general",
    "обзор":             "related_work",
    "методы":            "method",
    "метод":             "method",
    "методология":       "method",
    "предложен":         "method",
    "реализация":        "method",
    "обучение":          "method",
    "эксперимент":       "result",
    "результат":         "result",
    "валидация":         "result",
    "оценка":            "result",
    "обсуждение":        "result",
    "заключение":        "general",
    "вывод":             "general",
    "ограничени":        "general",
}


def _classify_paragraph(text: str, section_title: str) -> str:
    """Определить семантическую роль параграфа."""
    norm_sec = section_title.lower()

    text_lower = text.lower()
    scores = {r: 0 for r, _ in _ROLE_PATTERNS}
    for role, patterns in _ROLE_PATTERNS:
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                scores[role] += 1

    best_text_role = max(scores, key=lambda r: scores[r])
    text_score = scores[best_text_role]

    if text_score == 0 or (best_text_role != "hypothesis" and text_score <= 1):
        for key, role in _SECTION_ROLE_MAP.items():
            if key in norm_sec:
                if role == "general" and text_score > 0:
                    break
                return role

    if text_score > 0:
        return best_text_role

    return "general"



_FORMULA_GARBAGE = re.compile(
    r"^[\s\d\+\-\*\/\=\.\,\(\)\[\]\{\}\^\_\|\\<>±×÷∑∫∂∇αβγδεζηθλμπρσφψω]+$",
    re.IGNORECASE
)
_MOSTLY_SYMBOLS = re.compile(r"^[^\w\s]{3,}$")
_EQUATION_LABEL = re.compile(r"^\s*\(\s*\d+\s*\)\s*$")


def _clean_text(text: str) -> str:
    """Очистить текст от артефактов парсинга."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\(\s*\d{1,3}\s*\)\s*$", "", text)
    return text


def _is_garbage(text: str) -> bool:
    """Проверить, является ли текст мусором (обрывки формул, номера и т.д.)."""
    t = text.strip()
    if len(t) < 15:
        return True
    if _FORMULA_GARBAGE.match(t):
        return True
    if _MOSTLY_SYMBOLS.match(t):
        return True
    if _EQUATION_LABEL.match(t):
        return True
    letter_ratio = len(re.findall(r"[a-zA-Za-yA-Yа-яА-Я]", t)) / max(len(t), 1)
    if letter_ratio < 0.25 and len(t) < 80:
        return True
    if len(t) < 60:
        words = re.findall(r"[a-zA-Zа-яА-Я]{3,}", t)
        has_math = bool(re.search(r"[∂∫∑∇∆∏√∞±×÷≤≥≠≈λμσπωθφψαβγδεζ]", t))
        if has_math and len(words) == 0:
            return True
    if len(t) < 80:
        has_equals = "=" in t
        has_greek = bool(re.search(r"[λμσπωθφψαβγδεζ]", t))
        if has_equals or has_greek:
            all_words = re.findall(r"[a-zA-Zа-яА-Я]+", t)
            avg_word_len = sum(len(w) for w in all_words) / max(len(all_words), 1)
            if avg_word_len <= 2.5:
                return True
    return False


def _strip_section_number(title: str) -> str:
    """Удалить числовой префикс секции: '2.3 Methods' → 'Methods'."""
    return re.sub(r"^[\d\.\s]+", "", title).strip()



class TEIXMLProcessor:
    """Разбирает TEI XML, сформированный GROBID."""

    def parse_file(self, xml_path: str | Path) -> ParsedDocument:
        xml_path = Path(xml_path)
        tree = etree.parse(str(xml_path))
        return self._parse_root(tree.getroot(), source_file=xml_path.name)

    def parse_string(self, xml_text: str, source_file: str = "<string>") -> ParsedDocument:
        root = etree.fromstring(xml_text.encode("utf-8"))
        return self._parse_root(root, source_file=source_file)

    def _parse_root(self, root: etree._Element, source_file: str) -> ParsedDocument:
        doc = ParsedDocument(source_file=source_file)

        header = root.find(".//tei:teiHeader", NS)
        if header is not None:
            doc.title        = self._extract_title(header)
            doc.authors      = self._extract_authors(header)
            doc.affiliations = self._extract_affiliations(header)
            doc.year         = self._extract_year(header)
            doc.doi          = self._extract_doi(header)
            doc.keywords     = self._extract_keywords(header)
            doc.abstract     = self._extract_abstract(header)

        body = root.find(".//tei:body", NS)
        if body is not None:
            doc.sections, doc.figure_captions = self._extract_sections(body)

        back = root.find(".//tei:back", NS)
        if back is not None:
            doc.references = self._extract_references(back)
            if not doc.abstract:
                abs_el = back.find(".//tei:div[@type='abstract']", NS)
                if abs_el is not None:
                    doc.abstract = self._inner_text(abs_el)

        if not doc.keywords:
            doc.keywords = self._keywords_from_title(doc.title)

        logger.info(
            "Разобран: '%s' | %d секций | %d параграфов | %d ссылок | %d ключевых слов",
            (doc.title[:55] + "…") if len(doc.title) > 55 else doc.title,
            len(doc.sections),
            sum(len(s.paragraphs) for s in doc.sections),
            len(doc.references),
            len(doc.keywords),
        )
        return doc


    def _extract_title(self, header: etree._Element) -> str:
        for sel in [
            ".//tei:titleStmt/tei:title[@type='main']",
            ".//tei:titleStmt/tei:title[@level='a']",
            ".//tei:titleStmt/tei:title",
        ]:
            el = header.find(sel, NS)
            if el is not None:
                t = self._inner_text(el)
                if t:
                    return t
        return ""

    def _extract_authors(self, header: etree._Element) -> list[str]:
        authors = []
        seen = set()
        for pers in header.findall(".//tei:sourceDesc//tei:persName", NS):
            forename = " ".join(
                self._text(e) for e in pers.findall("tei:forename", NS) if self._text(e)
            )
            surname = self._text(pers.find("tei:surname", NS))
            name = f"{forename} {surname}".strip()
            if name and name not in seen:
                seen.add(name)
                authors.append(name)
        return authors

    def _extract_affiliations(self, header: etree._Element) -> list[str]:
        """Извлечь аффилиации (организации авторов)."""
        affils = []
        seen = set()
        for aff in header.findall(".//tei:affiliation", NS):
            parts = []
            for tag in ["tei:orgName[@type='department']", "tei:orgName[@type='institution']",
                        "tei:orgName"]:
                for el in aff.findall(tag, NS):
                    t = self._text(el)
                    if t:
                        parts.append(t)
            text = ", ".join(parts) if parts else self._inner_text(aff)
            text = text.strip()
            if text and text not in seen and len(text) > 3:
                seen.add(text)
                affils.append(text)
        return affils[:8]

    def _extract_year(self, header: etree._Element) -> str:
        for el in header.findall(".//tei:date[@type='published']", NS):
            when = el.get("when", "")
            m = re.search(r"(19\d\d|20[0-3]\d)", when)
            if m:
                return m.group(1)
        for el in header.findall(".//tei:date", NS):
            when = el.get("when", "") + " " + self._text(el)
            m = re.search(r"(19\d\d|20[0-3]\d)", when)
            if m:
                return m.group(1)
        return ""

    def _extract_doi(self, header: etree._Element) -> str:
        el = header.find(".//tei:idno[@type='DOI']", NS)
        if el is None:
            el = header.find(".//tei:idno[@type='doi']", NS)
        doi = self._text(el)
        # Normalize: убрать URL-префикс
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
        return doi

    def _extract_keywords(self, header: etree._Element) -> list[str]:
        kws = []
        seen = set()
        selectors = [
            ".//tei:keywords/tei:term",
            ".//tei:keywords/tei:item",
            ".//tei:textClass/tei:keywords/tei:term",
            ".//tei:textClass/tei:keywords",
        ]
        for sel in selectors:
            for el in header.findall(sel, NS):
                raw = self._inner_text(el)
                for kw in re.split(r"[;,·•]", raw):
                    kw = kw.strip().strip(".")
                    if kw and 2 < len(kw) < 60 and kw.lower() not in seen:
                        seen.add(kw.lower())
                        kws.append(kw)
        return kws[:20]

    def _extract_abstract(self, header: etree._Element) -> str:
        """Извлечь аннотацию — несколько способов."""
        for sel in [
            ".//tei:profileDesc/tei:abstract",
            ".//tei:abstract",
        ]:
            el = header.find(sel, NS)
            if el is not None:
                paras = el.findall(".//tei:p", NS)
                if paras:
                    parts = [self._inner_text(p) for p in paras]
                    text = " ".join(p for p in parts if p)
                else:
                    text = self._inner_text(el)
                if text and len(text) > 50:
                    return _clean_text(text)
        return ""

    def _keywords_from_title(self, title: str) -> list[str]:
        """Извлечь ключевые слова из названия как fallback."""
        if not title:
            return []
        stopwords = {
            "with", "from", "using", "based", "for", "the", "and", "deep",
            "via", "into", "over", "toward", "towards", "that", "this",
            "than", "more", "when", "where", "which", "their", "have",
        }
        words = re.findall(r"[A-Za-z]{5,}", title)
        kws = [w for w in words if w.lower() not in stopwords]
        return list(dict.fromkeys(kws))[:8]

    # ──────────────────────── Тело документа ─────────────────

    def _extract_sections(
        self, body: etree._Element
    ) -> tuple[list[Section], list[str]]:
        """
        Извлечь секции и подписи к рисункам/таблицам.

        Returns:
            (sections, figure_captions)
        """
        sections: list[Section] = []
        figure_captions: list[str] = []
        last_title = "General"

        divs = body.findall("tei:div", NS)
        if not divs:
            divs = body.findall(".//tei:div", NS)

        for div in divs:
            head_el = div.find("tei:head", NS)
            if head_el is not None:
                raw = self._inner_text(head_el)
                stripped = _strip_section_number(raw)
                sec_title = stripped if stripped else raw
                if sec_title:
                    last_title = sec_title
            else:
                sec_title = last_title

            paragraphs: list[Paragraph] = []

            for p_el in div.findall("tei:p", NS):
                para = self._process_paragraph(p_el, sec_title)
                if para:
                    paragraphs.append(para)

            for fig in div.findall(".//tei:figure", NS):
                caption = self._extract_figure_caption(fig)
                if caption:
                    figure_captions.append(caption)
                    paragraphs.append(Paragraph(
                        text=caption,
                        section=sec_title,
                        role="general",
                        is_caption=True,
                    ))

            for sub_div in div.findall("tei:div", NS):
                sub_head = sub_div.find("tei:head", NS)
                sub_title = sec_title
                if sub_head is not None:
                    raw = self._inner_text(sub_head)
                    sub_title = _strip_section_number(raw) or raw or sec_title
                for p_el in sub_div.findall("tei:p", NS):
                    para = self._process_paragraph(p_el, sub_title)
                    if para:
                        paragraphs.append(para)

            if paragraphs:
                sections.append(Section(title=sec_title, paragraphs=paragraphs))

        return sections, figure_captions

    def _process_paragraph(
        self, p_el: etree._Element, section_title: str
    ) -> Optional[Paragraph]:
        """Обработать один параграф TEI XML."""
        raw_text = self._inner_text(p_el)
        text = _clean_text(raw_text)

        if _is_garbage(text):
            return None

        ref_ids = []
        for r in p_el.findall(".//tei:ref[@type='bibr']", NS):
            target = r.get("target", "")
            if target.startswith("#"):
                ref_ids.append(target[1:])

        role = _classify_paragraph(text, section_title)

        return Paragraph(
            text=text,
            section=section_title,
            role=role,
            ref_ids=list(dict.fromkeys(ref_ids)),
        )

    def _extract_figure_caption(self, fig: etree._Element) -> str:
        """Извлечь подпись к рисунку или таблице."""
        desc = fig.find("tei:figDesc", NS)
        if desc is not None:
            text = _clean_text(self._inner_text(desc))
            if text and len(text) > 10:
                return text
        head = fig.find("tei:head", NS)
        if head is not None:
            text = _clean_text(self._inner_text(head))
            if text:
                return text
        return ""


    def _extract_references(self, back: etree._Element) -> list[Reference]:
        refs: list[Reference] = []
        for bibl in back.findall(".//tei:listBibl/tei:biblStruct", NS):
            ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
            analytic = bibl.find("tei:analytic", NS)
            monogr   = bibl.find("tei:monogr", NS)

            title = ""
            if analytic is not None:
                for t_el in analytic.findall("tei:title", NS):
                    t = self._inner_text(t_el)
                    if t:
                        title = t
                        break
            if not title and monogr is not None:
                t_el = monogr.find("tei:title", NS)
                if t_el is not None:
                    title = self._inner_text(t_el)

            authors = []
            src = analytic if analytic is not None else monogr
            if src is not None:
                for pers in src.findall(".//tei:persName", NS):
                    forename = " ".join(
                        self._text(e) for e in pers.findall("tei:forename", NS)
                        if self._text(e)
                    )
                    surname = self._text(pers.find("tei:surname", NS))
                    name = f"{forename} {surname}".strip()
                    if name:
                        authors.append(name)

            year = self._extract_year_from_bibl(bibl)

            doi = ""
            for idno in bibl.findall(".//tei:idno", NS):
                if idno.get("type", "").upper() == "DOI":
                    doi = self._text(idno)
                    break

            journal = ""
            if monogr is not None:
                title_el = monogr.find("tei:title[@level='j']", NS)
                if title_el is None:
                    title_el = monogr.find("tei:title[@level='m']", NS)
                if title_el is not None:
                    journal = self._inner_text(title_el)

            raw_el = bibl.find(".//tei:note[@type='raw_reference']", NS)
            raw = self._text(raw_el)

            refs.append(Reference(
                ref_id=ref_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                journal=journal,
                raw=raw,
            ))
        return refs

    def _extract_year_from_bibl(self, bibl: etree._Element) -> str:
        for date_el in bibl.findall(".//tei:date[@type='published']", NS):
            when = date_el.get("when", "")
            m = re.search(r"(19\d\d|20[0-3]\d)", when)
            if m:
                return m.group(1)
        for date_el in bibl.findall(".//tei:date", NS):
            when = date_el.get("when", "") + " " + self._text(date_el)
            m = re.search(r"(19\d\d|20[0-3]\d)", when)
            if m:
                return m.group(1)
        return ""


    @staticmethod
    def _text(el: Optional[etree._Element]) -> str:
        if el is None:
            return ""
        return (el.text or "").strip()

    @staticmethod
    def _inner_text(el: Optional[etree._Element]) -> str:
        """Весь текстовый контент элемента (включая дочерние теги)."""
        if el is None:
            return ""
        parts = []
        for t in el.itertext():
            t = t.strip()
            if t:
                parts.append(t)
        return " ".join(parts)
