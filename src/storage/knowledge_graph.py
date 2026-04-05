"""
Knowledge Graph — граф связей между документами, авторами и понятиями.

Использует NetworkX. Позволяет:
  - Находить документы по теме через обход графа
  - Выявлять ключевые работы (PageRank)
  - Строить сети цитирований
  - Находить авторов, работающих в смежных областях
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Граф знаний для научных публикаций.

    Узлы:
      - document: статья (атрибуты: title, year, authors, doi)
      - author:   автор
      - keyword:  ключевое слово
      - reference: цитируемая работа

    Рёбра:
      - wrote:      автор → документ
      - cites:      документ → ссылка
      - has_keyword: документ → ключевое слово
      - co_author:  автор ↔ автор (совместные публикации)
    """

    def __init__(self):
        try:
            import networkx as nx
            self._nx = nx
        except ImportError:
            raise ImportError("pip install networkx")

        self.G: Any = self._nx.DiGraph()
        self._doc_count = 0

    # ------------------------------------------------------------------
    # Построение графа
    # ------------------------------------------------------------------

    def add_document(self, doc) -> str:
        """
        Добавить ParsedDocument в граф.

        Returns:
            node_id документа.
        """
        from src.parsers.xml_processor import ParsedDocument
        assert isinstance(doc, ParsedDocument)

        doc_id = f"doc::{doc.source_file}"

        # Узел документа
        self.G.add_node(doc_id, type="document", **{
            "title":   doc.title[:200],
            "year":    doc.year,
            "doi":     doc.doi,
            "abstract": doc.abstract[:500],
            "source_file": doc.source_file,
        })

        # Авторы
        for i, author in enumerate(doc.authors):
            author_id = f"author::{author.lower()}"
            self.G.add_node(author_id, type="author", name=author)
            self.G.add_edge(author_id, doc_id, relation="wrote", order=i)

            # Связи соавторства
            for j, other_author in enumerate(doc.authors):
                if i != j:
                    other_id = f"author::{other_author.lower()}"
                    if not self.G.has_edge(author_id, other_id):
                        self.G.add_edge(author_id, other_id,
                                        relation="co_author", count=1)
                    else:
                        self.G[author_id][other_id]["count"] += 1

        # Ключевые слова
        for kw in doc.keywords:
            kw_id = f"kw::{kw.lower()}"
            self.G.add_node(kw_id, type="keyword", label=kw)
            self.G.add_edge(doc_id, kw_id, relation="has_keyword")

        # Библиографические ссылки
        for ref in doc.references:
            if not ref.title:
                continue
            ref_id = f"ref::{ref.doi or ref.title[:80].lower()}"
            self.G.add_node(ref_id, type="reference", **{
                "title": ref.title[:200],
                "year":  ref.year,
                "doi":   ref.doi,
                "authors": ", ".join(ref.authors[:3]),
            })
            self.G.add_edge(doc_id, ref_id, relation="cites")

            # Если цитируемая работа тоже загружена — связать
            for existing_id in self.G.nodes:
                if existing_id.startswith("doc::"):
                    node_data = self.G.nodes[existing_id]
                    if (ref.doi and node_data.get("doi") == ref.doi) or \
                       (ref.title and node_data.get("title", "")[:60] == ref.title[:60]):
                        self.G.add_edge(doc_id, existing_id,
                                        relation="cites_loaded_doc")

        self._doc_count += 1
        logger.info("Граф: добавлен документ '%s' (%d узлов всего)",
                    doc.title[:50], self.G.number_of_nodes())
        return doc_id

    # ------------------------------------------------------------------
    # Аналитика
    # ------------------------------------------------------------------

    def pagerank_documents(self, top_n: int = 10) -> list[dict]:
        """Ранжирование документов по PageRank (цитируемость)."""
        pr = self._nx.pagerank(self.G, weight=None)
        doc_scores = [
            (node_id, score)
            for node_id, score in pr.items()
            if self.G.nodes[node_id].get("type") == "document"
        ]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "node_id": nid,
                "title":   self.G.nodes[nid].get("title", nid),
                "score":   round(score, 6),
            }
            for nid, score in doc_scores[:top_n]
        ]

    def most_cited_references(self, top_n: int = 10) -> list[dict]:
        """Самые цитируемые ссылки (in-degree в граф цитирований)."""
        ref_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("type") == "reference"
        ]
        ref_counts = [
            (n, self.G.in_degree(n))
            for n in ref_nodes
        ]
        ref_counts.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "title":  self.G.nodes[n].get("title", n),
                "doi":    self.G.nodes[n].get("doi", ""),
                "year":   self.G.nodes[n].get("year", ""),
                "cited_by": count,
            }
            for n, count in ref_counts[:top_n]
        ]

    def find_related_documents(self, doc_source_file: str, depth: int = 2) -> list[dict]:
        """
        Найти документы, связанные с данным через общих авторов / ключевые слова.

        Returns:
            Список связанных документов с описанием связи.
        """
        doc_id = f"doc::{doc_source_file}"
        if doc_id not in self.G:
            return []

        related = []
        visited = {doc_id}

        for neighbor in self._nx.neighbors(self.G, doc_id):
            node_type = self.G.nodes[neighbor].get("type", "")
            if node_type in ("author", "keyword"):
                for second in self._nx.neighbors(self.G, neighbor):
                    if second not in visited and second.startswith("doc::"):
                        visited.add(second)
                        relation_label = (
                            f"Общий автор: {self.G.nodes[neighbor].get('name', '')}"
                            if node_type == "author"
                            else f"Общая тема: {self.G.nodes[neighbor].get('label', '')}"
                        )
                        related.append({
                            "node_id": second,
                            "title":   self.G.nodes[second].get("title", second),
                            "relation": relation_label,
                        })

        return related

    def keyword_cooccurrence(self) -> dict[str, int]:
        """Частота встречаемости ключевых слов."""
        counts: dict[str, int] = defaultdict(int)
        for n, d in self.G.nodes(data=True):
            if d.get("type") == "keyword":
                counts[d.get("label", n)] = self.G.in_degree(n)
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def author_network(self) -> list[dict]:
        """Топ авторов по числу публикаций."""
        authors = []
        for n, d in self.G.nodes(data=True):
            if d.get("type") == "author":
                pub_count = sum(
                    1 for _, _, rel in self.G.out_edges(n, data="relation")
                    if rel == "wrote"
                )
                if pub_count > 0:
                    authors.append({"name": d.get("name", n), "publications": pub_count})
        return sorted(authors, key=lambda x: x["publications"], reverse=True)

    def stats(self) -> dict:
        """Статистика графа."""
        node_types: dict[str, int] = defaultdict(int)
        for _, d in self.G.nodes(data=True):
            node_types[d.get("type", "unknown")] += 1
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "documents":   node_types.get("document", 0),
            "authors":     node_types.get("author", 0),
            "keywords":    node_types.get("keyword", 0),
            "references":  node_types.get("reference", 0),
        }

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Сохранить граф в JSON (node-link format)."""
        data = self._nx.node_link_data(self.G)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Граф сохранён: %s (%d узлов)", path, self.G.number_of_nodes())

    def load(self, path: str | Path) -> None:
        """Загрузить граф из JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.G = self._nx.node_link_graph(data)
        logger.info("Граф загружен: %s (%d узлов)", path, self.G.number_of_nodes())

    def export_gephi(self, path: str | Path) -> None:
        """Экспорт в формат GEXF для Gephi (визуализация графа)."""
        self._nx.write_gexf(self.G, str(path))
        logger.info("Экспортировано для Gephi: %s", path)
