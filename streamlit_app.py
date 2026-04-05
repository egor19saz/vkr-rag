"""
Advanced RAG-система для анализа научных публикаций
ВКР: Разработка Advanced RAG-системы для анализа научных публикаций

Работает локально и на Streamlit Community Cloud.
Запуск локально: streamlit run streamlit_app.py
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Advanced RAG — Научные публикации",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');
[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif}
.page-title{font-size:1.55rem;font-weight:600;letter-spacing:-.02em;line-height:1.2;margin-bottom:2px}
.page-sub{color:#6B7280;font-size:.85rem;margin-bottom:1.2rem}
.kpi-card{background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:14px 16px;text-align:center}
.kpi-val{font-size:1.8rem;font-weight:600;color:#111827;line-height:1}
.kpi-lbl{font-size:.7rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:.07em;margin-top:4px}
.answer-box{background:#F0F9FF;border-left:4px solid #0EA5E9;border-radius:0 10px 10px 0;
            padding:1.1rem 1.4rem;font-size:.93rem;line-height:1.75;margin-top:1rem}
.role-pill{display:inline-block;border-radius:20px;padding:2px 10px;font-size:.68rem;font-weight:600;
           text-transform:uppercase;letter-spacing:.05em;font-family:'JetBrains Mono',monospace}
.rp-hypothesis{background:#EDE9FE;color:#5B21B6}
.rp-method{background:#DBEAFE;color:#1D4ED8}
.rp-result{background:#D1FAE5;color:#065F46}
.rp-related_work{background:#FEF3C7;color:#92400E}
.rp-general{background:#F3F4F6;color:#374151}
.src-chip{display:inline-block;background:#E0F2FE;color:#0369A1;border-radius:6px;
           padding:2px 9px;font-size:.75rem;margin:2px;font-family:'JetBrains Mono',monospace}
.step-num{background:#111827;color:#fff;border-radius:50%;width:26px;height:26px;
          display:inline-flex;align-items:center;justify-content:center;
          font-size:.75rem;font-weight:600;margin-right:8px}
.ok-dot{color:#059669;font-size:.85rem}
.err-dot{color:#DC2626;font-size:.85rem}
.warn-dot{color:#D97706;font-size:.85rem}
</style>
""", unsafe_allow_html=True)


# ─── Конфигурация ─────────────────────────────────────────────
def _get_cfg() -> dict:
    creds, scope, model, grobid_url = "", "GIGACHAT_API_PERS", "GigaChat", "http://localhost:8070"
    try:
        creds      = st.secrets["gigachat"]["credentials"]
        scope      = st.secrets["gigachat"].get("scope", scope)
        model      = st.secrets["gigachat"].get("model", model)
        grobid_url = st.secrets["grobid"].get("url", grobid_url)
    except Exception:
        pass
    if not creds:
        creds = os.getenv("GIGACHAT_CREDENTIALS", "")
    grobid_url = os.getenv("GROBID_URL", grobid_url)
    if not creds:
        try:
            from config import PIPELINE_CONFIG
            creds = PIPELINE_CONFIG.get("gigachat_credentials", "")
        except ImportError:
            pass
    return {
        "gigachat_credentials": creds,
        "gigachat_scope": scope,
        "gigachat_model": model,
        "grobid_url": grobid_url,
        "grobid_timeout": 120,
        "embedder": {"type": "sentence_transformers", "model_name": "intfloat/multilingual-e5-large"},
        "chunk_size": 512, "chunk_overlap": 64,
        "vector_store_dir": "./data/chroma",
        "collection_name": "papers",
        "retrieval_top_k": 5,
        "vector_weight": 0.7,
        "llm_temperature": 0.2,
    }


@st.cache_resource(show_spinner="Загрузка модели эмбеддингов...")
def _init_pipeline():
    from src.pipeline import RAGPipeline
    return RAGPipeline(_get_cfg())


@st.cache_resource(show_spinner=False)
def _init_graph():
    from src.storage.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph()


def _badge(role):
    return f'<span class="role-pill rp-{role}">{role}</span>'


def _chips(names):
    return " ".join(f'<span class="src-chip">{n}</span>' for n in names)


pipeline = _init_pipeline()
graph = _init_graph()
cfg = _get_cfg()

# ─── Боковая панель ────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="page-title">📚 Advanced RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Анализ научных публикаций · ВКР</div>', unsafe_allow_html=True)

    st.markdown("##### Статус сервисов")
    grobid_ok = False
    try:
        grobid_ok = pipeline.grobid.is_alive()
    except Exception:
        pass

    if grobid_ok:
        st.markdown('<span class="ok-dot">● GROBID — работает</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="err-dot">● GROBID — недоступен</span>', unsafe_allow_html=True)
        with st.expander("Как запустить?"):
            st.code("docker run -t --rm --init \\\n  -p 8070:8070 \\\n  lfoppiano/grobid:0.8.0", language="bash")

    token_ok = cfg.get("gigachat_credentials", "") not in ("", "YOUR_BASE64_CREDENTIALS_HERE", "ВАШ_BASE64_ТОКЕН")
    if token_ok:
        st.markdown('<span class="ok-dot">● GigaChat — токен задан</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-dot">● GigaChat — токен не задан</span>', unsafe_allow_html=True)
        with st.expander("Как задать токен?"):
            st.code('[gigachat]\ncredentials = "ваш_токен"', language="toml")
            st.caption("Файл: .streamlit/secrets.toml")

    st.divider()
    st.markdown("##### Загрузить статью")
    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if st.button("🚀 Индексировать", use_container_width=True, type="primary"):
            if not token_ok:
                st.error("Добавьте токен GigaChat в secrets.toml.")
            else:
                if not grobid_ok:
                    st.warning("GROBID отвечает медленно, пробуем всё равно...")
                pdf_dir = Path("./data/pdfs")
                pdf_dir.mkdir(parents=True, exist_ok=True)
                sp = pdf_dir / uploaded.name
                sp.write_bytes(uploaded.getbuffer())
                with st.spinner(f"Обрабатываю {uploaded.name}..."):
                    try:
                        doc = pipeline.ingest_pdf(sp, save_xml=True)
                        graph.add_document(doc)
                        st.success("✅ Загружено!")
                        t = doc.title[:50] + "…" if len(doc.title) > 50 else doc.title
                        st.caption(f"**{t or uploaded.name}**")
                        st.caption(f"{len(doc.sections)} секций · {len(doc.all_paragraphs())} параграфов")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

    st.divider()
    st.markdown("##### База знаний")
    n_docs = len(pipeline._parsed_docs)
    n_chunks = pipeline.vector_store.count()
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_docs}</div><div class="kpi-lbl">Статей</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_chunks}</div><div class="kpi-lbl">Чанков</div></div>', unsafe_allow_html=True)

    if pipeline._parsed_docs:
        st.divider()
        st.markdown("##### Загруженные статьи")
        for fname, doc in pipeline._parsed_docs.items():
            yr = f" · {doc.year}" if doc.year else ""
            st.caption(f"📄 **{fname}**{yr}")


# ─── Вкладки ──────────────────────────────────────────────────
ROLES = {
    "Все": None, "Гипотезы": "hypothesis", "Методы": "method",
    "Результаты": "result", "Обзор литературы": "related_work", "Общий текст": "general",
}

tab_qa, tab_doc, tab_graph, tab_deploy = st.tabs([
    "💬 Вопрос-ответ", "🔍 Анализ документа", "🕸️ Граф знаний", "🚀 Деплой на облако"
])

# ── Вопрос-ответ ──────────────────────────────────────────────
with tab_qa:
    st.markdown("## Задать вопрос по загруженным статьям")
    if n_chunks == 0:
        st.info("☝️ Загрузите хотя бы одну статью PDF через боковую панель слева.")
    else:
        col_q, col_o = st.columns([3, 1])
        with col_q:
            question = st.text_area("Вопрос", placeholder="Например: Какова основная гипотеза авторов?", height=90, label_visibility="collapsed")
        with col_o:
            top_k = st.slider("Фрагментов", 1, 10, 5)
            role_lbl = st.selectbox("Роль", list(ROLES.keys()))

        st.markdown("**Примеры:**")
        examples = [
            "Какова основная гипотеза авторов?",
            "Какие методы используются?",
            "В чём новизна подхода?",
            "Каковы результаты экспериментов?",
            "Чем отличается от смежных работ?",
            "Какие ограничения метода?",
        ]
        ecols = st.columns(3)
        for i, ex in enumerate(examples):
            if ecols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
                question = ex

        if st.button("📋 Получить ответ", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Введите вопрос.")
            else:
                with st.spinner("Поиск + генерация..."):
                    try:
                        t0 = time.time()
                        report = pipeline.query(question, top_k=top_k, role_filter=ROLES[role_lbl])
                        elapsed = round(time.time() - t0, 1)
                        st.markdown(f'<div class="answer-box">{report.answer}</div>', unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Время", f"{elapsed} с")
                        m2.metric("Токенов", report.tokens_used or "—")
                        m3.metric("Модель", report.model or "GigaChat")
                        if report.source_documents:
                            st.markdown("**Источники:** " + _chips(report.source_documents), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

# ── Анализ документа ──────────────────────────────────────────
with tab_doc:
    st.markdown("## Детальный анализ документа")
    if not pipeline._parsed_docs:
        st.info("Загрузите PDF через боковую панель.")
    else:
        doc_name = st.selectbox("Документ", list(pipeline._parsed_docs.keys()))
        doc = pipeline._parsed_docs[doc_name]
        with st.expander("Метаданные", expanded=True):
            mc = st.columns(4)
            mc[0].metric("Год", doc.year or "—")
            mc[1].metric("Секций", len(doc.sections))
            mc[2].metric("Параграфов", len(doc.all_paragraphs()))
            mc[3].metric("Ссылок", len(doc.references))
            if doc.title: st.markdown(f"**Название:** {doc.title}")
            if doc.authors: st.markdown(f"**Авторы:** {', '.join(doc.authors[:5])}")
            if doc.doi: st.markdown(f"**DOI:** `{doc.doi}`")
            if doc.keywords: st.markdown("**Ключевые слова:** " + _chips(doc.keywords[:12]), unsafe_allow_html=True)
        if doc.abstract:
            with st.expander("Аннотация"):
                st.write(doc.abstract)
        if st.button("🤖 Сгенерировать резюме через GigaChat", use_container_width=True):
            with st.spinner("Читаю статью..."):
                try:
                    st.markdown(f'<div class="answer-box">{pipeline.summarize(doc_name)}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))
        st.divider()
        st.markdown("### Параграфы по семантическим ролям")
        role_map = {"Гипотезы":"hypothesis","Методы":"method","Результаты":"result","Обзор литературы":"related_work","Общие":"general"}
        sel = st.radio("Роль:", list(role_map.keys()), horizontal=True)
        paras = doc.paragraphs_by_role(role_map[sel])
        st.caption(f"Найдено: **{len(paras)}**")
        for p in paras[:8]:
            st.markdown(f'{_badge(p.role)} <small style="color:#9CA3AF">§ {p.section}</small>', unsafe_allow_html=True)
            st.write(p.text[:450] + ("…" if len(p.text) > 450 else ""))
            st.divider()
        with st.expander("Структура документа"):
            for sec in doc.sections:
                roles_set = set(p.role for p in sec.paragraphs)
                st.markdown(f"**{sec.title}** · {len(sec.paragraphs)} параграфов · `{', '.join(sorted(roles_set))}`")
        if doc.references:
            with st.expander(f"Библиографические ссылки ({len(doc.references)})"):
                for ref in doc.references[:15]:
                    st.markdown(f"- **{ref.title[:80] or '—'}** — {', '.join(ref.authors[:2]) or '—'} ({ref.year or '?'})")

# ── Граф знаний ───────────────────────────────────────────────
with tab_graph:
    st.markdown("## Граф знаний")
    gs = graph.stats()
    if gs["total_nodes"] == 0:
        st.info("Граф пуст. Загрузите документы.")
    else:
        cols = st.columns(5)
        for col, key, lbl, ico in [
            (cols[0],"documents","Документов","📄"),
            (cols[1],"authors","Авторов","👤"),
            (cols[2],"keywords","Тем","🏷"),
            (cols[3],"references","Ссылок","📚"),
            (cols[4],"total_edges","Рёбер","🔗"),
        ]:
            col.markdown(f'<div class="kpi-card"><div style="font-size:1.1rem">{ico}</div><div class="kpi-val">{gs[key]}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.divider()
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("### Ключевые документы (PageRank)")
            ranked = graph.pagerank_documents(top_n=8)
            ms = max((r["score"] for r in ranked), default=1)
            for i, item in enumerate(ranked, 1):
                t = item["title"][:55]+"…" if len(item["title"])>55 else item["title"]
                bw = int(item["score"]/ms*100)
                st.markdown(f'<div style="margin:5px 0"><small style="color:#9CA3AF">#{i}</small> <b style="font-size:13px">{t}</b><br><div style="background:#E5E7EB;border-radius:4px;height:5px;margin-top:3px"><div style="background:#0EA5E9;width:{bw}%;height:5px;border-radius:4px"></div></div></div>', unsafe_allow_html=True)
        with gc2:
            st.markdown("### Самые цитируемые")
            for ref in graph.most_cited_references(top_n=8):
                t = ref["title"][:55]+"…" if len(ref["title"])>55 else ref["title"]
                st.markdown(f'<div style="margin:5px 0;padding:8px;background:#F9FAFB;border-radius:8px;font-size:13px"><b>{t}</b> <span style="color:#9CA3AF">({ref["year"] or "?"})</span><br><span style="color:#0EA5E9">цитирований: {ref["cited_by"]}</span></div>', unsafe_allow_html=True)
        st.divider()
        kws = list(graph.keyword_cooccurrence().items())[:20]
        if kws:
            st.markdown("### Ключевые слова")
            kw_html = " ".join(f'<span class="role-pill rp-method" style="font-size:{0.65+v*0.05:.2f}rem;margin:3px">{k}</span>' for k,v in kws)
            st.markdown(kw_html, unsafe_allow_html=True)

# ── Деплой на облако ──────────────────────────────────────────
with tab_deploy:
    st.markdown("## Развернуть проект на Streamlit Community Cloud")
    st.caption("Бесплатный хостинг — ваш проект будет доступен по ссылке из любой точки мира.")

    with st.expander("1. Зарегистрироваться на GitHub", expanded=True):
        st.markdown("Перейдите на **github.com** → Sign up → создайте аккаунт.")
        st.markdown("GitHub — хранилище файлов проекта. Streamlit Cloud читает код именно оттуда.")

    with st.expander("2. Создать репозиторий"):
        st.markdown("На GitHub нажмите **New repository** → название `vkr-rag` → **Create repository**")

    with st.expander("3. Загрузить файлы проекта на GitHub"):
        st.code("""cd C:\\projects\\vkr_rag

git init
git add .
git commit -m "Initial commit — Advanced RAG VKR"
git remote add origin https://github.com/ВАШ_ЛОГИН/vkr-rag.git
git push -u origin main""", language="bash")
        st.warning("Убедитесь что в .gitignore есть строки `.env` и `.streamlit/secrets.toml` — не загружайте токены на GitHub!")

    with st.expander("4. Зарегистрироваться на Streamlit Cloud"):
        st.markdown("Перейдите на **share.streamlit.io** → Sign up → войдите через GitHub")

    with st.expander("5. Создать приложение"):
        st.markdown("Нажмите **New app** → выберите репозиторий `vkr-rag`")
        st.code("Main file path: streamlit_app.py", language="text")
        st.markdown("Нажмите **Deploy!** Первый запуск займёт 5-10 минут.")

    with st.expander("6. Добавить токен GigaChat в Secrets"):
        st.markdown("В настройках приложения: **Settings → Secrets** → вставьте:")
        st.code("""[gigachat]
credentials = "ВАШ_BASE64_ТОКЕН"
scope = "GIGACHAT_API_PERS"
model = "GigaChat"

[grobid]
url = "https://kermitt2-grobid.hf.space"  """, language="toml")
        st.info("В облаке используйте публичный GROBID: **https://kermitt2-grobid.hf.space** (бесплатный, может быть медленным)")

    with st.expander("7. Обновлять проект"):
        st.code("""git add .
git commit -m "Описание изменений"
git push""", language="bash")
        st.markdown("Streamlit Cloud автоматически обновит приложение через 1-2 минуты.")

    st.divider()
    st.markdown("### Ограничения бесплатного тарифа Streamlit Cloud")
    ca, cb = st.columns(2)
    ca.markdown("**Ограничения:**\n- 1 ГБ RAM\n- Засыпает через 7 дней неактивности\n- При пробуждении ждать 1-2 минуты\n- Векторная база сбрасывается при рестарте")
    cb.markdown("**Что работает:**\n- Загрузка и индексация PDF\n- Вопрос-ответ через GigaChat\n- Граф знаний\n- Анализ документов")
