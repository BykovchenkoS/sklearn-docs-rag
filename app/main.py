import html as html_lib
import os
import time
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException

from app.rag.chain import build_rag_chain
from app.schemas.chat import ChatRequest, ChatResponse, Source

_chain = None
_retriever = None

ROOT_PATH = os.getenv("ROOT_PATH", "")

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever
    _chain, _retriever = build_rag_chain()
    print("RAG chain ready")
    yield
    _chain = None
    _retriever = None


app = FastAPI(title="RAG service", lifespan=lifespan, root_path=ROOT_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    docs = _retriever.invoke(payload.question)
    try:
        answer = _chain.invoke(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider temporarily unavailable. "
                f"Try again in 30-60 seconds. Raw: {type(exc).__name__}"
            ),
        ) from exc
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
            full_context=doc.page_content.strip(),
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


def _status(text: str, detail: str = "", kind: str = "idle") -> str:
    detail_html = (
        f'<span class="status-detail">{html_lib.escape(detail)}</span>' if detail else ""
    )
    return (
        f'<div class="status-badge {kind}">'
        f'<span class="status-dot"></span>'
        f'<span class="status-text">{html_lib.escape(text)}</span>'
        f"{detail_html}</div>"
    )


def _timing_card(label: str, value: str, kind: str = "") -> str:
    return (
        f'<div class="timing-card {kind}">'
        f'<span class="timing-label">{html_lib.escape(label)}</span>'
        f'<span class="timing-value">{html_lib.escape(value)}</span>'
        f"</div>"
    )


def _format_timings(
    retrieval_ms: float,
    ttft_ms: float | None = None,
    llm_ms: float | None = None,
    streaming_chars: int | None = None,
    error: str | None = None,
) -> str:
    cards = [_timing_card("Retrieval", f"{retrieval_ms:.0f} ms", "retrieval")]
    if ttft_ms is not None:
        cards.append(_timing_card("TTFT", f"{ttft_ms:.0f} ms", "ttft"))
    if streaming_chars is not None:
        cards.append(_timing_card("LLM", f"{streaming_chars} симв.", "llm"))
    if llm_ms is not None:
        cards.append(_timing_card("LLM", f"{llm_ms:.0f} ms", "llm"))
        cards.append(_timing_card("Total", f"{retrieval_ms + llm_ms:.0f} ms", "total"))
    if error:
        cards.append(_timing_card("LLM", f"❌ {error}", "error"))
    return '<div class="timing-badges">' + "".join(cards) + "</div>"


def _module_label(url: str) -> str:
    u = url.lower()
    if "linear_model" in u:
        return "Linear models"
    if "tree" in u:
        return "Decision trees"
    if "model_evaluation" in u:
        return "Metrics"
    if u.startswith("local://"):
        return "About"
    return "Docs"


def _format_sources(docs: list) -> str:
    if not docs:
        return '<div class="panel-empty">Ничего не найдено.</div>'
    cards = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:220].strip().replace("\n", " ")
        module = _module_label(url)
        if url.startswith("http"):
            link = (
                f'<a class="source-link" href="{html_lib.escape(url, quote=True)}" '
                f'target="_blank" rel="noopener">открыть ↗</a>'
            )
        else:
            link = f'<span class="source-link muted">{html_lib.escape(url)}</span>'
        cards.append(
            '<div class="source-card">'
            '<div class="source-head">'
            f'<span class="source-num">{i}</span>'
            f'<span class="source-module">{html_lib.escape(module)}</span>'
            f"{link}"
            "</div>"
            f'<div class="source-snippet">{html_lib.escape(snippet)}…</div>'
            "</div>"
        )
    return '<div class="sources-list">' + "".join(cards) + "</div>"


INITIAL_STATUS = _status("💤 Готов к работе", kind="idle")
EMPTY_TIMINGS = '<div class="panel-empty">Здесь появятся тайминги запроса.</div>'
EMPTY_SOURCES = '<div class="panel-empty">Здесь появятся источники ответа.</div>'


def respond(message: str, history: list):
    if not message or not message.strip():
        yield history, "", INITIAL_STATUS, EMPTY_TIMINGS, EMPTY_SOURCES
        return

    history = history + [{"role": "user", "content": message}]

    yield (
        history,
        "",
        _status("⏳ Ищу в документации…", kind="busy"),
        EMPTY_TIMINGS,
        EMPTY_SOURCES,
    )

    t0 = time.perf_counter()
    docs = _retriever.invoke(message)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    sources_panel = _format_sources(docs)

    history.append({"role": "assistant", "content": ""})
    yield (
        history,
        "",
        _status("🤖 LLM пишет ответ…", "генерация", "busy"),
        _format_timings(retrieval_ms),
        sources_panel,
    )

    t1 = time.perf_counter()
    ttft_ms: float | None = None
    accumulated = ""
    try:
        for chunk in _chain.stream(message):
            if not chunk:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t1) * 1000
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield (
                history,
                "",
                _status("🤖 LLM пишет ответ…", f"{len(accumulated)} символов", "busy"),
                _format_timings(
                    retrieval_ms,
                    ttft_ms=ttft_ms,
                    streaming_chars=len(accumulated),
                ),
                sources_panel,
            )

        llm_total_ms = (time.perf_counter() - t1) * 1000
        total_ms = retrieval_ms + llm_total_ms
        yield (
            history,
            "",
            _status("✅ Готово", f"{total_ms / 1000:.1f} с", "done"),
            _format_timings(
                retrieval_ms,
                ttft_ms=ttft_ms,
                llm_ms=llm_total_ms,
            ),
            sources_panel,
        )
    except Exception as exc:
        history[-1]["content"] = (
            f"⚠️ LLM-провайдер сейчас недоступен ({type(exc).__name__}). "
            f"Попробуй через 30-60 секунд."
        )
        yield (
            history,
            "",
            _status("⚠️ Провайдер недоступен", type(exc).__name__, "error"),
            _format_timings(retrieval_ms, error=type(exc).__name__),
            sources_panel,
        )


def clear_all():
    return [], "", INITIAL_STATUS, EMPTY_TIMINGS, EMPTY_SOURCES


HEADER_HTML = f"""
<div class="app-header">
  <div class="brand">
    <div class="logo">📖</div>
    <div>
      <h1>scikit-learn docs RAG</h1>
      <p>Ответы по официальной документации с цитатами · RU / EN</p>
    </div>
  </div>
  <div class="header-links">
    <a href="https://github.com/BykovchenkoS/sklearn-docs-rag" target="_blank" rel="noopener">GitHub</a>
    <a href="{ROOT_PATH}/docs" target="_blank" rel="noopener">API</a>
  </div>
</div>
"""

ABOUT_MD = """
## ℹ️ О сервисе

Внутренний AI-ассистент для команды Data Science: отвечает на вопросы по
официальной документации **scikit-learn** и показывает источники.

### Что умеет
- **Linear models** - OLS, Ridge, Lasso, ElasticNet, логистическая регрессия
- **Decision trees** - критерии, переобучение, гиперпараметры
- **Metrics** - confusion matrix, precision/recall/F1, ROC AUC, MSE/MAE/R²
- Понимает **русский и английский**, отвечает на языке вопроса
- Формулы рендерятся, ответ приходит потоком, вне корпуса

### Корпус
3 раздела документации scikit-learn + `about.md` → **265 чанков**
(нарезка 1000 символов, перекрытие 200).

### Метрики (RAGAS, golden dataset из 10 вопросов)

| Метрика | Значение |
|---|---|
| Recall@4 | **1.00** |
| Faithfulness | **0.854** |
| Answer Relevancy | **0.862** |

### Стек
FastAPI · LangChain LCEL · Qdrant · `multilingual-e5-small` ·
`llama-3.3-70b-instruct` (OpenRouter) · Gradio · Docker · GitHub Actions.

[Исходный код на GitHub](https://github.com/BykovchenkoS/sklearn-docs-rag)
"""

FOOTER_MD = """
---
<sub>FastAPI · LangChain · Qdrant · Gradio — RAG над документацией scikit-learn</sub>
"""

CSS = """
:root {
  --brand: #6366f1;
  --brand-2: #8b5cf6;
  --brand-soft: rgba(99, 102, 241, .14);
}

.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 1rem !important;
}

/* --- шапка --- */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem 1.25rem;
  margin-bottom: .85rem;
  border-radius: 18px;
  color: #fff;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #a855f7 100%);
  box-shadow: 0 12px 30px rgba(99, 102, 241, .25);
}
.app-header h1 {
  margin: 0;
  font-size: 1.32rem;
  font-weight: 750;
  letter-spacing: -.02em;
}
.app-header p {
  margin: .2rem 0 0;
  font-size: .88rem;
  opacity: .9;
}
.brand { display: flex; align-items: center; gap: .85rem; }
.logo {
  width: 46px; height: 46px;
  display: grid; place-items: center;
  border-radius: 14px;
  font-size: 1.45rem;
  background: rgba(255, 255, 255, .18);
  border: 1px solid rgba(255, 255, 255, .25);
  backdrop-filter: blur(6px);
}
.header-links { display: flex; gap: .5rem; }
.header-links a {
  color: #fff;
  text-decoration: none;
  font-size: .82rem;
  font-weight: 600;
  padding: .45rem .8rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, .16);
  border: 1px solid rgba(255, 255, 255, .28);
  transition: .18s ease;
}
.header-links a:hover {
  background: rgba(255, 255, 255, .3);
  transform: translateY(-1px);
}

/* --- статус --- */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: .55rem;
  padding: .55rem .9rem;
  margin-bottom: .6rem;
  border-radius: 999px;
  font-size: .88rem;
  font-weight: 600;
  background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary);
  color: var(--body-text-color);
}
.status-dot {
  width: 9px; height: 9px;
  border-radius: 50%;
  background: #94a3b8;
  flex: none;
}
.status-badge.busy .status-dot {
  background: #f59e0b;
  animation: pulse 1s ease-in-out infinite;
}
.status-badge.done .status-dot { background: #10b981; }
.status-badge.error .status-dot { background: #ef4444; }
.status-detail { opacity: .62; font-weight: 500; }
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: .4; transform: scale(.78); }
}

/* --- чат --- */
#chatbot {
  height: calc(100vh - 360px) !important;
  min-height: 420px !important;
  border-radius: 18px !important;
}

/* --- чипы вопросов --- */
.chip-row { flex-wrap: wrap !important; gap: .5rem !important; }
.chip-row button {
  border-radius: 999px !important;
  font-size: .82rem !important;
  padding: .35rem .85rem !important;
  border: 1px solid var(--border-color-primary) !important;
  background: var(--background-fill-secondary) !important;
  transition: .18s ease;
}
.chip-row button:hover {
  border-color: var(--brand) !important;
  background: var(--brand-soft) !important;
  transform: translateY(-1px);
}

/* --- тайминги --- */
.timing-badges { display: flex; flex-wrap: wrap; gap: .5rem; }
.timing-card {
  flex: 1 1 110px;
  min-width: 110px;
  padding: .55rem .7rem;
  border-radius: 14px;
  background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary);
}
.timing-label {
  display: block;
  font-size: .68rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  opacity: .6;
}
.timing-value {
  display: block;
  margin-top: .18rem;
  font-size: .98rem;
  font-weight: 700;
}
.timing-card.retrieval { border-left: 3px solid #6366f1; }
.timing-card.ttft { border-left: 3px solid #f59e0b; }
.timing-card.llm { border-left: 3px solid #8b5cf6; }
.timing-card.total { border-left: 3px solid #10b981; }
.timing-card.error { border-left: 3px solid #ef4444; }

/* --- источники --- */
.sources-list { display: flex; flex-direction: column; gap: .6rem; }
.source-card {
  padding: .7rem .8rem;
  border-radius: 14px;
  background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary);
  transition: .18s ease;
}
.source-card:hover {
  transform: translateY(-1px);
  border-color: var(--brand);
  box-shadow: 0 8px 20px rgba(0, 0, 0, .08);
}
.source-head {
  display: flex;
  align-items: center;
  gap: .5rem;
  margin-bottom: .35rem;
}
.source-num {
  width: 22px; height: 22px;
  display: grid; place-items: center;
  border-radius: 50%;
  background: var(--brand);
  color: #fff;
  font-size: .72rem;
  font-weight: 700;
  flex: none;
}
.source-module {
  font-size: .72rem;
  font-weight: 600;
  padding: .14rem .5rem;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--body-text-color);
}
.source-link {
  margin-left: auto;
  font-size: .76rem;
  font-weight: 600;
  text-decoration: none;
  color: var(--brand);
}
.source-link:hover { text-decoration: underline; }
.source-link.muted { color: var(--body-text-color); opacity: .55; }
.source-snippet {
  font-size: .8rem;
  line-height: 1.35;
  opacity: .8;
}
.panel-empty {
  padding: .5rem 0;
  font-size: .84rem;
  opacity: .55;
}

/* --- мобильная версия --- */
@media (max-width: 768px) {
  .app-header { flex-direction: column; align-items: flex-start; }
  .app-header h1 { font-size: 1.12rem; }
  .header-links { width: 100%; }
  #chatbot { height: 52vh !important; min-height: 320px !important; }
  .timing-card { flex: 1 1 45%; }
  .source-snippet { font-size: .78rem; }
}
"""

with gr.Blocks(title="scikit-learn docs RAG", theme=THEME, css=CSS, fill_height=True) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.Tab("💬 Чат"):
            with gr.Row():
                with gr.Column(scale=3, min_width=420):
                    status_html = gr.HTML(INITIAL_STATUS)
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        type="messages",
                        latex_delimiters=LATEX_DELIMITERS,
                        show_copy_button=True,
                        avatar_images=(None, None),
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                            scale=8,
                            container=False,
                            autofocus=True,
                        )
                        send = gr.Button("Отправить", scale=1, variant="primary")
                        clear = gr.Button("🧹", scale=0, variant="secondary", min_width=48)

                    with gr.Row(elem_classes=["chip-row"]):
                        for _label, _question in [
                            ("Ridge", "How does Ridge regression work?"),
                            ("Lasso vs Ridge", "What is the difference between Lasso and Ridge?"),
                            ("Метрики", "Объясни разницу между precision и recall с формулами"),
                            ("Что умеешь?", "Что ты умеешь?"),
                        ]:
                            _chip = gr.Button(
                                _label,
                                size="sm",
                                variant="secondary",
                                elem_classes=["chip"],
                            )
                            _chip.click(
                                fn=lambda q=_question: q,
                                inputs=None,
                                outputs=msg,
                            )

                with gr.Column(scale=1, min_width=280, elem_id="side-panel"):
                    gr.Markdown("### ⏱ Тайминги")
                    timings_html = gr.HTML(EMPTY_TIMINGS)
                    gr.Markdown("### 📚 Источники")
                    sources_html = gr.HTML(EMPTY_SOURCES)

        with gr.Tab("ℹ️ О сервисе"):
            gr.Markdown(ABOUT_MD)

    gr.Markdown(FOOTER_MD)

    _outputs = [chatbot, msg, status_html, timings_html, sources_html]
    msg.submit(respond, [msg, chatbot], _outputs)
    send.click(respond, [msg, chatbot], _outputs)
    clear.click(clear_all, None, _outputs)


app = gr.mount_gradio_app(app, demo, path="/", root_path=ROOT_PATH)
