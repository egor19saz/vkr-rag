"""
XML Processor v4 — финальная версия парсера TEI XML от GROBID.

Улучшения v4 (на основе анализа 8 реальных статей):
  - Римские цифры в заголовках (I., II., III., IV., V., VI.)
  - IEEE "Index Terms" как keywords
  - Новые секции: Preliminaries, Proposed Scheme, Training details,
    Evaluation metrics, Statistical Analysis, Clinical utility,
    Quantitative/Qualitative results, Example scenarios
  - Статистические статьи: hypothesize → hypothesis role
  - Годы 1900-2030
  - Объединение параграфов одной секции
  - Корректные ref_ids для графа цитирований
  - Keywords fallback из текста с расширенным списком терминов
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import Counter

from lxml import etree

logger = logging.getLogger(__name__)

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}

# Римские цифры для нормализации заголовков
_ROMAN = r"^[IVXivx]+\.\s*"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    ref_id:  str
    title:   str = ""
    authors: list[str] = field(default_factory=list)
    year:    str = ""
    doi:     str = ""
    raw:     str = ""


@dataclass
class Paragraph:
    text:    str
    section: str = ""
    role:    str = "general"
    ref_ids: list[str] = field(default_factory=list)


@dataclass
class Section:
    title:      str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_file: str
    title:       str = ""
    abstract:    str = ""
    authors:     list[str] = field(default_factory=list)
    year:        str = ""
    doi:         str = ""
    keywords:    list[str] = field(default_factory=list)
    sections:    list[Section] = field(default_factory=list)
    references:  list[Reference] = field(default_factory=list)

    def all_paragraphs(self) -> list[Paragraph]:
        return [p for sec in self.sections for p in sec.paragraphs]

    def paragraphs_by_role(self, role: str) -> list[Paragraph]:
        return [p for p in self.all_paragraphs() if p.role == role]

    def to_plain_text(self) -> str:
        parts = [self.title, self.abstract]
        for sec in self.sections:
            parts.append(sec.title)
            parts.extend(p.text for p in sec.paragraphs)
        return "\n\n".join(filter(None, parts))


# ---------------------------------------------------------------------------
# Role classifier
# ---------------------------------------------------------------------------

_SECTION_ROLE_MAP: list[tuple[str, str]] = [
    # related work
    ("related work",         "related_work"),
    ("related",              "related_work"),
    ("background",           "related_work"),
    ("literature",           "related_work"),
    ("prior work",           "related_work"),
    ("previous work",        "related_work"),
    ("state of the art",     "related_work"),
    ("preliminaries",        "related_work"),   # IEEE стиль
    ("preliminary",          "related_work"),
    ("обзор",                "related_work"),
    ("смежные",              "related_work"),
    # methods
    ("method",               "method"),
    ("approach",             "method"),
    ("methodology",          "method"),
    ("materials",            "method"),
    ("architecture",         "method"),
    ("model",                "method"),
    ("implementation",       "method"),
    ("hardware",             "method"),
    ("software",             "method"),
    ("system",               "method"),
    ("framework",            "method"),
    ("proposed",             "method"),
    ("proposed scheme",      "method"),
    ("algorithm",            "method"),
    ("training",             "method"),
    ("training detail",      "method"),
    ("evaluation metric",    "method"),
    ("statistical analysis", "method"),
    ("data collection",      "method"),
    ("dataset",              "method"),
    ("corpus",               "method"),
    ("preprocessing",        "method"),
    ("appendix",             "method"),
    ("pipeline",             "method"),
    ("filtering",            "method"),
    ("feature",              "method"),
    ("network",              "method"),
    ("mono block",           "method"),
    ("mono gate",            "method"),
    ("object detection",     "method"),
    ("prompt segmentation",  "method"),
    ("diffusion isolation",  "method"),
    ("encryption",           "method"),
    ("метод",                "method"),
    ("архитектур",           "method"),
    ("реализация",           "method"),
    ("обучение",             "method"),
    # results
    ("result",               "result"),
    ("experiment",           "result"),
    ("evaluation",           "result"),
    ("performance",          "result"),
    ("validation",           "result"),
    ("clinical",             "result"),
    ("numerical",            "result"),
    ("discussion",           "result"),
    ("ablation",             "result"),
    ("analysis",             "result"),
    ("summary",              "result"),
    ("finding",              "result"),
    ("benchmark",            "result"),
    ("comparison",           "result"),
    ("calibration",          "result"),
    ("generalization",       "result"),
    ("quantitative",         "result"),
    ("qualitative",          "result"),
    ("example scenario",     "result"),
    ("example",              "result"),
    ("case study",           "result"),
    ("clinical utility",     "result"),
    ("utility",              "result"),
    ("performance analysis", "result"),
    ("distortion",           "result"),
    ("результат",            "result"),
    ("эксперимент",          "result"),
    ("оценка",               "result"),
    ("валидация",            "result"),
    # hypothesis — статистические/методологические статьи
    ("challenge",            "hypothesis"),
    ("problem",              "hypothesis"),
    ("motivation",           "hypothesis"),
    ("the challenge",        "hypothesis"),
    # general
    ("introduction",         "general"),
    ("conclusion",           "general"),
    ("future",               "general"),
    ("limitation",           "general"),
    ("strength",             "general"),
    ("acknowledgment",       "general"),
    ("interactive demo",     "general"),
    ("demo",                 "general"),
    ("beyond existing",      "general"),
    ("going beyond",         "general"),
    ("information needed",   "general"),
    ("введение",             "general"),
    ("заключение",           "general"),
    ("вывод",                "general"),
    ("ограничени",           "general"),
]

_TEXT_PATTERNS: list[tuple[str, list[str]]] = [
    ("hypothesis", [
        r"\bwe\s+hypothesize\b",
        r"\bwe\s+propose\s+that\b",
        r"\bour\s+hypothesis\b",
        r"\bwe\s+conjecture\b",
        r"\bwe\s+argue\s+that\b",
        r"\bwe\s+claim\s+that\b",
        r"\bhypothesize\s+(an?\s+)?effect",
        r"\bгипотез",
        r"\bмы\s+предполагаем\b",
    ]),
    ("related_work", [
        r"\bprevious(ly)?\s+(work|studies|research|approach)\b",
        r"\bprior\s+work\b",
        r"\bexisting\s+(method|approach|work)\b",
        r"\bhave\s+been\s+(proposed|studied|shown)\b",
        r"\b\[[\d,\s]+\]\s+(show|propose|present|demonstrate)",
        r"\brelated\s+work\b",
        r"\bliteratur",
        r"\bпредыдущ",
        r"\bсмежн",
    ]),
    ("result", [
        r"\bwe\s+(found|observe|show|demonstrate|achieve|obtain)\b",
        r"\bour\s+(model|method|approach|system|results?)\s+(achieve|outperform|show|yield)\b",
        r"\baccuracy\s+(of|is|was|=)\s+[\d.]+",
        r"\bimprove(ment|s)?\s+(of|by|over)\b",
        r"\boutperform",
        r"\bpearson\s+correlation\b",
        r"\bspearman\b",
        r"\broc\s+auc\b",
        r"\bdice\s+(score|=)",
        r"\bicc\b",
        r"\bpsnr\b",
        r"\bssim\b",
        r"\biou\b",
        r"\berr\b.*\bencrypt",
        r"\br\s*=\s*0\.\d+",
        r"\bresults?\s+show\b",
        r"\bresults?\s+indicate\b",
        r"\bperformance\s+(of|is|was)\b",
        r"\btable\s+[IVX\d]",
        r"\bfigure\s+\d",
        r"\bрезультат",
        r"\bпоказывает",
        r"\bдостигает",
    ]),
    ("method", [
        r"\bwe\s+(use|employ|apply|adopt|propose|develop|train|implement)\b",
        r"\bour\s+(model|method|approach|system|network|framework|architecture|scheme)\b",
        r"\bwe\s+describe\b",
        r"\bthe\s+proposed\s+(method|approach|model|framework|scheme|algorithm)\b",
        r"\bconsists?\s+of\b",
        r"\barchitecture\b",
        r"\bneural\s+network\b",
        r"\btrained\s+(on|with|using)\b",
        r"\bdataset\s+(consists?|contains?)\b",
        r"\bwe\s+collect\b",
        r"\bwe\s+measure\b",
        r"\bequation\s+\(",
        r"\beq\.\s*\(",
        r"\balgorithm\b",
        r"\bparameter\b",
        r"\bencoder\b",
        r"\bdecoder\b",
        r"\bsyntax\s+element\b",
        r"\bалгоритм",
        r"\bметод",
        r"\bмодел",
    ]),
]


def _normalize_title(raw: str) -> str:
    """Нормализует заголовок секции: убирает нумерацию любого формата."""
    title = raw.strip()
    # Убираем римские цифры: "IV. PROPOSED SCHEME" → "PROPOSED SCHEME"
    title = re.sub(_ROMAN, "", title).strip()
    # Убираем арабские цифры: "3.1.2", "2.4" → ""
    title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title).strip()
    # Убираем буквенные подзаголовки: "A. Hardware" → "Hardware"
    title = re.sub(r"^[A-Za-z]\.\s+", "", title).strip()
    return title if title else raw.strip()


def _classify_paragraph(text: str, section_title: str) -> str:
    norm_sec = section_title.lower().strip()
    clean_sec = _normalize_title(norm_sec)

    for keyword, role in _SECTION_ROLE_MAP:
        if keyword in clean_sec or keyword in norm_sec:
            return role

    scores: dict[str, int] = {
        "hypothesis": 0, "related_work": 0, "result": 0, "method": 0
    }
    text_lower = text.lower()
    for role, patterns in _TEXT_PATTERNS:
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                scores[role] += 1

    best_role = max(scores, key=scores.get)
    if scores[best_role] >= 1:
        return best_role

    return "general"


# ---------------------------------------------------------------------------
# Keyword extractor
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "were", "have", "has", "been", "which", "they", "their", "more",
    "also", "into", "via", "using", "based", "such", "each", "both",
    "about", "when", "than", "these", "those", "where", "while", "how",
    "can", "may", "will", "our", "its", "not", "but", "however",
    "although", "therefore", "thus", "hence", "further", "moreover",
    "between", "within", "through", "during", "after", "before",
    "over", "under", "above", "below", "along", "across", "toward",
    "towards", "used", "show", "shown", "showed", "paper", "work",
    "study", "approach", "method", "model", "system", "data", "results",
    "proposed", "present", "presented", "provides", "provide", "given",
    "number", "large", "small", "high", "low", "different", "several",
    "many", "other", "first", "second", "third", "new", "well", "good",
    "here", "then", "also", "only", "even", "most", "some", "any",
    "average", "effect", "size", "effect", "treatment", "study",
}

_TECH_TERMS = re.compile(
    r"\b("
    # ML/AI/CV
    r"neural\s+network|deep\s+learning|machine\s+learning|"
    r"natural\s+language|reinforcement\s+learning|"
    r"convolutional|transformer|attention\s+mechanism|"
    r"graph\s+neural\s+network|GNN|diffusion\s+model|"
    r"U-Net|UNet|encoder|decoder|backbone|"
    r"RAG|LLM|NLP|CNN|RNN|LSTM|BERT|GPT|YOLO|SAM|"
    # Medical imaging
    r"ultrasound|MRI|CT\s+scan|POCUS|segmentation|"
    r"cartilage|osteoarthritis|knee|femoral|"
    r"dice\s+score|intraclass\s+correlation|ICC|"
    r"point-of-care|handheld|"
    # Drug discovery / chemistry
    r"drug\s+discovery|small\s+molecule|molecular\s+generation|"
    r"binding\s+affinity|oncology|cancer|lymphoma|"
    r"scaffold|SMILES|ChEMBL|Tanimoto|"
    r"BCL6|EZH2|kinase|enzyme|inhibitor|"
    r"activity\s+cliff|SALI|matched\s+molecular\s+pair|MMP|"
    r"medicinal\s+chemistry|"
    # Medical
    r"blood\s+pressure|heart\s+rate|breathing\s+rate|"
    r"clinical\s+trial|hemodynamic|cardiovascular|"
    r"infrared|sensor|breathing|respiratory|"
    # Video / security
    r"H\.265|HEVC|H\.264|video\s+encryption|"
    r"region\s+of\s+interest|ROI|coding\s+unit|"
    r"prompt\s+segmentation|diffusion\s+isolation|"
    # Statistics
    r"Bayesian|prior\s+distribution|effect\s+size|"
    r"causal\s+inference|treatment\s+effect|"
    r"experimental\s+design|meta-analysis|"
    r"statistical\s+power|hypothesis\s+test|"
    # General science
    r"PINN|physics.informed|"
    r"morpholog|syntax|corpus|language\s+model"
    r")\b",
    re.IGNORECASE
)


def _extract_keywords_from_text(text: str, max_kw: int = 10) -> list[str]:
    tech_found = []
    seen_lower = set()
    for m in _TECH_TERMS.finditer(text):
        kw = m.group(0).strip()
        if kw.lower() not in seen_lower:
            tech_found.append(kw)
            seen_lower.add(kw.lower())

    words = re.findall(r"\b[A-Za-z]{6,}\b", text)
    word_freq = Counter(
        w.lower() for w in words
        if w.lower() not in _STOPWORDS and len(w) >= 6
    )
    top_words = [
        w for w, _ in word_freq.most_common(max_kw * 2)
        if not any(w in t.lower() for t in tech_found)
    ]

    combined = tech_found + top_words
    return list(dict.fromkeys(combined))[:max_kw]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class TEIXMLProcessor:

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
            doc.title    = self._extract_title(header)
            doc.authors  = self._extract_authors(header)
            doc.year     = self._extract_year(header)
            doc.doi      = self._extract_doi(header)
            doc.abstract = self._extract_abstract(header)
            doc.keywords = self._extract_keywords(header)

        body = root.find(".//tei:body", NS)
        if body is not None:
            doc.sections = self._extract_sections(body)

        back = root.find(".//tei:back", NS)
        if back is not None:
            doc.references = self._extract_references(back)

        # Fallback keywords из текста статьи
        if not doc.keywords:
            text_for_kw = " ".join(
                [doc.title, doc.abstract] +
                [p.text for p in doc.all_paragraphs()[:15]]
            )
            doc.keywords = _extract_keywords_from_text(text_for_kw)

        logger.info(
            "Разобран: '%s' | %d секций | %d параграфов | %d ссылок | год=%s | %d ключ.слов",
            (doc.title[:50] if doc.title else source_file),
            len(doc.sections),
            len(doc.all_paragraphs()),
            len(doc.references),
            doc.year or "—",
            len(doc.keywords),
        )
        return doc

    # ---------- Метаданные ----------

    def _extract_title(self, header: etree._Element) -> str:
        for xpath in [
            ".//tei:titleStmt/tei:title[@type='main']",
            ".//tei:titleStmt/tei:title[@level='a']",
            ".//tei:titleStmt/tei:title",
        ]:
            el = header.find(xpath, NS)
            if el is not None:
                t = self._inner_text(el)
                if t and len(t) > 3:
                    return t
        return ""

    def _extract_authors(self, header: etree._Element) -> list[str]:
        authors = []
        seen = set()
        for pers in header.findall(".//tei:sourceDesc//tei:persName", NS):
            forename = self._text(pers.find("tei:forename", NS))
            surname  = self._text(pers.find("tei:surname", NS))
            name = f"{forename} {surname}".strip()
            if name and name not in seen:
                authors.append(name)
                seen.add(name)
        return authors

    def _parse_year(self, text: str) -> str:
        m = re.search(r"\b(19\d{2}|20[012]\d)\b", text)
        if m:
            return m.group(1)
        return ""

    def _extract_year(self, header: etree._Element) -> str:
        # 1. Дата публикации с when
        for el in header.findall(".//tei:date[@type='published']", NS):
            y = self._parse_year(el.get("when", "") or self._inner_text(el))
            if y:
                return y
        # 2. Любая дата с when
        for el in header.findall(".//tei:date", NS):
            y = self._parse_year(el.get("when", ""))
            if y:
                return y
        # 3. Текст даты
        for el in header.findall(".//tei:date", NS):
            y = self._parse_year(self._inner_text(el))
            if y:
                return y
        # 4. arXiv ID: YYMM.NNNNN → год
        for el in header.findall(".//tei:idno", NS):
            t = self._text(el)
            m = re.search(r"\b(2[0-9])(\d{2})\.\d{4,}", t)
            if m:
                year = 2000 + int(m.group(1))
                if 1900 <= year <= 2030:
                    return str(year)
            y = self._parse_year(t)
            if y:
                return y
        return ""

    def _extract_doi(self, header: etree._Element) -> str:
        for el in header.findall(".//tei:idno[@type='DOI']", NS):
            t = self._text(el)
            if t:
                return t
        return ""

    def _extract_keywords(self, header: etree._Element) -> list[str]:
        kws = []
        seen = set()
        # Стандартные места для keywords + IEEE "Index Terms"
        for xpath in [
            ".//tei:keywords/tei:term",
            ".//tei:keywords/tei:item",
            ".//tei:textClass/tei:keywords/tei:term",
            ".//tei:textClass/tei:keywords",
            ".//tei:keywords",
            # IEEE Index Terms часто попадают в note
            ".//tei:note[@type='keywords']",
            ".//tei:note[@type='index terms']",
        ]:
            for el in header.findall(xpath, NS):
                raw = self._inner_text(el)
                if not raw or len(raw) > 300:
                    continue
                for kw in re.split(r"[;,·•\n]", raw):
                    kw = kw.strip(" .-")
                    if kw and 2 < len(kw) < 80 and kw.lower() not in seen:
                        kws.append(kw)
                        seen.add(kw.lower())
        return kws[:15]

    def _extract_abstract(self, header: etree._Element) -> str:
        el = header.find(".//tei:abstract", NS)
        return self._inner_text(el)

    # ---------- Тело документа ----------

    def _extract_sections(self, body: etree._Element) -> list[Section]:
        sections: list[Section] = []
        section_map: dict[str, Section] = {}
        last_known_title = "General"

        for div in body.findall(".//tei:div", NS):
            head = div.find("tei:head", NS)

            if head is not None:
                raw_title = self._inner_text(head)
                sec_title = _normalize_title(raw_title)
                if sec_title:
                    last_known_title = sec_title
                else:
                    sec_title = last_known_title
            else:
                sec_title = last_known_title

            paragraphs: list[Paragraph] = []

            for p_el in div.findall("tei:p", NS):
                text = self._inner_text(p_el)
                if len(text.strip()) < 30:
                    continue

                ref_ids = []
                for ref in p_el.findall(".//tei:ref[@type='bibr']", NS):
                    target = ref.get("target", "")
                    if target.startswith("#"):
                        ref_ids.append(target[1:])

                role = _classify_paragraph(text, sec_title)
                paragraphs.append(Paragraph(
                    text=text,
                    section=sec_title,
                    role=role,
                    ref_ids=list(set(ref_ids)),
                ))

            if not paragraphs:
                continue

            # Объединяем секции с одинаковым заголовком
            if sec_title in section_map:
                section_map[sec_title].paragraphs.extend(paragraphs)
            else:
                sec = Section(title=sec_title, paragraphs=paragraphs)
                section_map[sec_title] = sec
                sections.append(sec)

        return sections

    # ---------- Ссылки ----------

    def _extract_references(self, back: etree._Element) -> list[Reference]:
        refs: list[Reference] = []
        for bibl in back.findall(".//tei:listBibl/tei:biblStruct", NS):
            ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
            analytic = bibl.find("tei:analytic", NS)
            monogr   = bibl.find("tei:monogr", NS)

            title = ""
            for src in [analytic, monogr]:
                if src is not None:
                    for t_el in src.findall(".//tei:title", NS):
                        t = self._inner_text(t_el)
                        if t and len(t) > 3:
                            title = t
                            break
                if title:
                    break

            authors = []
            seen_a = set()
            src = analytic if analytic is not None else monogr
            if src is not None:
                for pers in src.findall(".//tei:persName", NS):
                    forename = self._text(pers.find("tei:forename", NS))
                    surname  = self._text(pers.find("tei:surname", NS))
                    name = f"{forename} {surname}".strip()
                    if name and name not in seen_a:
                        authors.append(name)
                        seen_a.add(name)

            year = ""
            for date_el in bibl.findall(".//tei:date", NS):
                y = self._parse_year(date_el.get("when", "") or self._text(date_el))
                if y:
                    year = y
                    break

            doi = ""
            for idno in bibl.findall(".//tei:idno[@type='DOI']", NS):
                doi = self._text(idno)
                if doi:
                    break

            raw_el = bibl.find(".//tei:note[@type='raw_reference']", NS)
            raw = self._text(raw_el) if raw_el is not None else ""

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
        if el is None:
            return ""
        return " ".join(t.strip() for t in el.itertext() if t.strip())
