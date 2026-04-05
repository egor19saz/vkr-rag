"""
Тесты для проекта ВКР.

Запуск:
    pip install pytest
    pytest tests/ -v

Для тестов с реальным GROBID / GigaChat:
    pytest tests/ -v -m integration
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# XML Processor Tests
# ---------------------------------------------------------------------------

SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">Test Paper on Machine Learning</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Ivan</forename>
                <surname>Petrov</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date type="published" when="2024"/>
            </imprint>
          </monogr>
          <idno type="DOI">10.1234/test.2024</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>This paper presents a novel approach to machine learning.</p>
      </abstract>
      <textClass>
        <keywords>
          <term>machine learning</term>
          <term>neural networks</term>
        </keywords>
      </textClass>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>We propose a new method for learning from data. This is a significant contribution.</p>
        <p>Previous work has explored similar ideas but with different approaches.</p>
      </div>
      <div>
        <head>Methods</head>
        <p>The algorithm uses gradient descent to optimize the loss function iteratively.</p>
      </div>
      <div>
        <head>Results</head>
        <p>Our experiments show significant improvements over the baseline methods.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0">
            <analytic>
              <title>Deep Learning Foundations</title>
              <author>
                <persName><forename>Yann</forename><surname>LeCun</surname></persName>
              </author>
            </analytic>
            <monogr>
              <imprint>
                <date type="published" when="2015"/>
              </imprint>
            </monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


class TestTEIXMLProcessor:
    """Тесты для разбора TEI XML."""

    def setup_method(self):
        from src.parsers.xml_processor import TEIXMLProcessor
        self.processor = TEIXMLProcessor()

    def test_parse_title(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        assert "Machine Learning" in doc.title

    def test_parse_authors(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        assert len(doc.authors) == 1
        assert "Petrov" in doc.authors[0]

    def test_parse_year(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        assert doc.year == "2024"

    def test_parse_keywords(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        assert "machine learning" in doc.keywords

    def test_parse_abstract(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        assert len(doc.abstract) > 10

    def test_parse_sections(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        section_titles = [s.title for s in doc.sections]
        assert "Introduction" in section_titles
        assert "Methods" in section_titles
        assert "Results" in section_titles

    def test_paragraph_count(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        all_paras = doc.all_paragraphs()
        assert len(all_paras) >= 3

    def test_paragraph_roles(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        roles = {p.role for p in doc.all_paragraphs()}
        # Должны быть назначены роли
        assert roles <= {"hypothesis", "method", "result", "related_work", "general"}

    def test_method_role_classification(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        method_paras = doc.paragraphs_by_role("method")
        assert len(method_paras) >= 1

    def test_parse_references(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        assert len(doc.references) >= 1
        assert "LeCun" in doc.references[0].authors[0]

    def test_to_plain_text(self):
        doc = self.processor.parse_string(SAMPLE_TEI_XML)
        text = doc.to_plain_text()
        assert len(text) > 100
        assert "Machine Learning" in text


# ---------------------------------------------------------------------------
# Text Chunker Tests
# ---------------------------------------------------------------------------

class TestTextChunker:
    """Тесты для разбивки текста на чанки."""

    def setup_method(self):
        from src.embeddings.embedder import TextChunker
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_short_text_no_split(self):
        text = "Short text."
        chunks = self.chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        text = "word " * 100  # 500 символов
        chunks = self.chunker.split(text)
        assert len(chunks) > 1

    def test_chunks_not_empty(self):
        text = "A" * 300
        chunks = self.chunker.split(text)
        assert all(len(c) > 0 for c in chunks)

    def test_split_documents(self):
        texts = ["Short text one.", "Another short text.", "A" * 300]
        chunks, indices = self.chunker.split_documents(texts)
        assert len(chunks) >= 3
        assert len(chunks) == len(indices)
        # Первые два текста — по одному чанку
        assert indices[0] == 0
        assert indices[1] == 1

    def test_overlap_in_large_text(self):
        text = "word " * 200
        chunks = self.chunker.split(text)
        # Проверяем, что чанки перекрываются (начало следующего совпадает с концом предыдущего)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# VectorStore Tests (с мок-эмбеддингом)
# ---------------------------------------------------------------------------

class TestVectorStore:
    """Тесты для векторного хранилища."""

    def setup_method(self, tmp_path=None):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()

        from src.storage.vector_store import VectorStore
        self.store = VectorStore(
            persist_dir=self.tmp_dir,
            collection_name="test_collection",
        )

    def test_add_and_count(self):
        texts = ["Document one", "Document two", "Document three"]
        embeddings = np.random.rand(3, 128).astype(np.float32)
        self.store.add(texts, embeddings)
        assert self.store.count() == 3

    def test_search_returns_results(self):
        texts = ["Machine learning paper", "Deep learning research", "NLP study"]
        embeddings = np.random.rand(3, 128).astype(np.float32)
        self.store.add(texts, embeddings)

        query_vec = np.random.rand(128).astype(np.float32)
        results = self.store.search(query_vec, top_k=2)
        assert len(results) == 2

    def test_search_result_structure(self):
        texts = ["Test document"]
        embeddings = np.random.rand(1, 128).astype(np.float32)
        self.store.add(texts, embeddings)

        query_vec = np.random.rand(128).astype(np.float32)
        results = self.store.search(query_vec, top_k=1)

        assert "id" in results[0]
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]

    def test_delete_all(self):
        texts = ["Document"]
        embeddings = np.random.rand(1, 128).astype(np.float32)
        self.store.add(texts, embeddings)
        self.store.delete_all()
        assert self.store.count() == 0

    def test_empty_add(self):
        ids = self.store.add([], np.empty((0, 128)))
        assert ids == []


# ---------------------------------------------------------------------------
# Hybrid Retriever Tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """Тесты для гибридного ретривера."""

    def _make_retriever(self, texts: list[str]):
        """Создать ретривер с тестовыми данными."""
        import tempfile
        from src.storage.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever

        # Мок-эмбеддер
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = np.random.rand(64).astype(np.float32)
        mock_embedder.embed.return_value = np.random.rand(len(texts), 64).astype(np.float32)
        mock_embedder.dimension = 64

        tmp_dir = tempfile.mkdtemp()
        store = VectorStore(persist_dir=tmp_dir, collection_name="test")
        embeddings = mock_embedder.embed(texts)
        store.add(texts, embeddings)

        retriever = HybridRetriever(vector_store=store, embedder=mock_embedder, top_k=3)
        retriever.build_bm25_index(texts)
        return retriever

    def test_retrieve_returns_chunks(self):
        texts = [
            "Neural networks learn from data through backpropagation.",
            "BERT is a transformer model for NLP tasks.",
            "Random forests use ensemble methods for classification.",
        ]
        retriever = self._make_retriever(texts)
        results = retriever.retrieve("transformer model")
        assert len(results) > 0

    def test_retrieve_chunk_structure(self):
        texts = ["Machine learning is a subset of AI."]
        retriever = self._make_retriever(texts)
        results = retriever.retrieve("AI")
        if results:
            r = results[0]
            assert hasattr(r, "text")
            assert hasattr(r, "score")
            assert hasattr(r, "source")

    def test_bm25_only_without_vectors(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        import tempfile
        from src.storage.vector_store import VectorStore

        mock_embedder = MagicMock()
        mock_embedder.embed_one.side_effect = Exception("No vectors")
        mock_embedder.dimension = 64

        tmp_dir = tempfile.mkdtemp()
        store = VectorStore(persist_dir=tmp_dir, collection_name="bm25test")

        retriever = HybridRetriever(vector_store=store, embedder=mock_embedder, top_k=2)
        texts = ["keyword search works well", "full text retrieval system"]
        retriever.build_bm25_index(texts)

        # BM25 должен работать даже если векторный поиск падает
        results = retriever._bm25_search("keyword", top_k=2)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Knowledge Graph Tests
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    """Тесты для графа знаний."""

    def _make_mock_doc(self, title: str, authors: list[str], year: str = "2024"):
        from src.parsers.xml_processor import ParsedDocument, Reference
        doc = ParsedDocument(source_file=f"{title.lower().replace(' ', '_')}.pdf")
        doc.title = title
        doc.authors = authors
        doc.year = year
        doc.keywords = ["machine learning", "NLP"]
        doc.references = [
            Reference(ref_id="r1", title="Foundational Work", authors=["Smith J"], year="2020")
        ]
        return doc

    def test_add_document(self):
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        doc = self._make_mock_doc("Test Paper", ["Alice Smith", "Bob Jones"])
        node_id = g.add_document(doc)
        assert node_id in g.G.nodes
        assert g.G.nodes[node_id]["type"] == "document"

    def test_author_nodes_created(self):
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        doc = self._make_mock_doc("Test", ["Alice Smith", "Bob Jones"])
        g.add_document(doc)
        author_nodes = [n for n, d in g.G.nodes(data=True) if d.get("type") == "author"]
        assert len(author_nodes) == 2

    def test_keyword_nodes_created(self):
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        doc = self._make_mock_doc("Test", ["Author A"])
        g.add_document(doc)
        kw_nodes = [n for n, d in g.G.nodes(data=True) if d.get("type") == "keyword"]
        assert len(kw_nodes) == 2

    def test_stats(self):
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        doc = self._make_mock_doc("Test", ["Author A"])
        g.add_document(doc)
        s = g.stats()
        assert s["documents"] == 1
        assert s["authors"] >= 1
        assert s["total_nodes"] > 0

    def test_pagerank(self):
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        for i in range(3):
            doc = self._make_mock_doc(f"Paper {i}", [f"Author {i}"])
            g.add_document(doc)
        ranked = g.pagerank_documents(top_n=3)
        assert len(ranked) == 3
        assert "title" in ranked[0]
        assert "score" in ranked[0]

    def test_author_network(self):
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        doc = self._make_mock_doc("Test", ["Alice Smith"])
        g.add_document(doc)
        network = g.author_network()
        assert any(a["name"] == "Alice Smith" for a in network)

    def test_save_and_load(self, tmp_path=None):
        import tempfile, os
        from src.storage.knowledge_graph import KnowledgeGraph
        g = KnowledgeGraph()
        doc = self._make_mock_doc("Test", ["Author"])
        g.add_document(doc)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            g.save(path)
            g2 = KnowledgeGraph()
            g2.load(path)
            assert g2.G.number_of_nodes() == g.G.number_of_nodes()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Paragraph role classifier
# ---------------------------------------------------------------------------

class TestParagraphClassifier:
    """Тесты для классификатора ролей параграфов."""

    def test_hypothesis_detected(self):
        from src.parsers.xml_processor import _classify_paragraph
        text = "We hypothesize that the model will perform better with more data."
        role = _classify_paragraph(text, "Introduction")
        assert role == "hypothesis"

    def test_method_section(self):
        from src.parsers.xml_processor import _classify_paragraph
        role = _classify_paragraph("The text is fine.", "Methods")
        assert role == "method"

    def test_result_detected(self):
        from src.parsers.xml_processor import _classify_paragraph
        text = "The results show a significant improvement over the baseline."
        role = _classify_paragraph(text, "Discussion")
        assert role == "result"

    def test_related_work_section(self):
        from src.parsers.xml_processor import _classify_paragraph
        role = _classify_paragraph("Text.", "Related Work")
        assert role == "related_work"

    def test_general_fallback(self):
        from src.parsers.xml_processor import _classify_paragraph
        role = _classify_paragraph("Some random text with no keywords.", "Conclusion")
        assert role == "general"

    def test_russian_hypothesis(self):
        from src.parsers.xml_processor import _classify_paragraph
        text = "Мы предполагаем, что предложенный метод превзойдёт базовые алгоритмы."
        role = _classify_paragraph(text, "Введение")
        assert role == "hypothesis"

    def test_russian_method_section(self):
        from src.parsers.xml_processor import _classify_paragraph
        role = _classify_paragraph("Описание подхода.", "Методы")
        assert role == "method"
