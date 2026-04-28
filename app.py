"""
Counsel from the Dead — Streamlit App (atmospheric redesign)
Backend logic (matching, retrieval, conversation, session state) unchanged.
"""

import os
# Must be set before protobuf is imported (chromadb → opentelemetry → protobuf)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import html as html_module
import json
import re
import time
from pathlib import Path

import openai
import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from figures import FIGURES

load_dotenv(Path(__file__).parent / ".env")

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "figures"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MATCHER_MODEL = "gpt-4o-mini"
CHAT_MODEL = "gpt-4o"
TOP_K = 3

# ── Figure themes ──────────────────────────────────────────────────────────────
THEMES = {
    "tolstoy": {
        "bg": "radial-gradient(ellipse at center, #2a1f17 0%, #14100b 100%)",
        "text": "#d9c9a8",
        "accent": "#c8965a",
        "muted": "rgba(217,201,168,0.5)",
        "body_font": "'EB Garamond', Georgia, serif",
        "display_font": "'Cormorant Garamond', Georgia, serif",
        "texture_css": """
    html::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
        background-size: 200px 200px;
        opacity: 0.04;
        pointer-events: none;
        mix-blend-mode: overlay;
        z-index: 0;
    }""",
    },
    "austen": {
        "bg": "linear-gradient(160deg, #f4ede0 0%, #eeddd0 100%)",
        "text": "#3a2e26",
        "accent": "#8b5a4a",
        "muted": "rgba(58,46,38,0.5)",
        "body_font": "'Cormorant Garamond', Georgia, serif",
        "display_font": "'Playfair Display', Georgia, serif",
    },
    "epictetus": {
        "bg": "linear-gradient(180deg, #2d2925 0%, #201e1b 100%)",
        "text": "#c4b89a",
        "accent": "#8b6f3f",
        "muted": "rgba(196,184,154,0.5)",
        "body_font": "'Cormorant Garamond', Georgia, serif",
        "display_font": "'Cinzel', serif",
    },
    "montaigne": {
        "bg": "linear-gradient(160deg, #2b2419 0%, #1e1a12 100%)",
        "text": "#d4c4a0",
        "accent": "#a0794a",
        "muted": "rgba(212,196,160,0.5)",
        "body_font": "'EB Garamond', Georgia, serif",
        "display_font": "'Cormorant Garamond', Georgia, serif",
    },
    "douglass": {
        "bg": "#0d0d0d",
        "text": "#f0ece4",
        "accent": "#c89a4a",
        "muted": "rgba(240,236,228,0.5)",
        "body_font": "'Crimson Pro', Georgia, serif",
        "display_font": "'Playfair Display', Georgia, serif",
    },
    "nietzsche": {
        "bg": "radial-gradient(ellipse at 30% 20%, #1a1a22 0%, #0a0a0c 70%)",
        "text": "#e8e8ea",
        "accent": "#9a9aa8",
        "muted": "rgba(232,232,234,0.45)",
        "body_font": "'Cormorant Garamond', Georgia, serif",
        "display_font": "'Cormorant SC', Georgia, serif",
    },
    "rumi": {
        "bg": "linear-gradient(160deg, #15132e 0%, #0d0b1e 100%)",
        "text": "#e8d9a8",
        "accent": "#d4a04a",
        "muted": "rgba(232,217,168,0.5)",
        "body_font": "'Cormorant Garamond', Georgia, serif",
        "display_font": "'Cormorant Garamond', Georgia, serif",
        "texture_css": """
    html::after {
        content: '';
        position: fixed;
        inset: 0;
        background:
            linear-gradient(rgba(212,160,74,.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(212,160,74,.04) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
        z-index: 0;
    }""",
    },
    "confucius": {
        "bg": "linear-gradient(180deg, #ebe5d6 0%, #ddd7c8 100%)",
        "text": "#2a2520",
        "accent": "#8b3a2a",
        "muted": "rgba(42,37,32,0.5)",
        "body_font": "'Noto Serif', Georgia, serif",
        "display_font": "'Noto Serif', Georgia, serif",
        "texture_css": """
    html::after {
        content: '';
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(
            90deg,
            transparent 0, transparent 52px,
            rgba(42,37,32,.03) 52px, rgba(42,37,32,.03) 53px
        );
        pointer-events: none;
        z-index: 0;
    }""",
    },
}

# ── CSS ────────────────────────────────────────────────────────────────────────

FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&"
    "family=Cormorant+SC:wght@400;500&"
    "family=EB+Garamond:ital,wght@0,400;0,500;1,400&"
    "family=Playfair+Display:ital,wght@0,400;0,700;1,400&"
    "family=Cinzel:wght@400;500&"
    "family=Crimson+Pro:ital,wght@0,400;0,600;1,400&"
    "family=Noto+Serif:ital,wght@0,400;0,700;1,400&"
    "display=swap"
)

GLOBAL_CSS = """
/* Chrome removal */
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton { display: none !important; }

[data-testid="stSidebar"] { display: none !important; }

/* Reset */
html, body {
    margin: 0 !important;
    padding: 0 !important;
}

html::after { content: none; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
}

/* ── Landing screen ─────────────────────────────────── */
.landing-outer {
    text-align: center;
    width: 100%;
    animation: fadeIn 0.7s ease forwards;
}

/* Sticky bottom bar — transparent so body gradient shows through */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stChatInputContainer"],
.stChatInputContainer {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

.landing-prompt {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 28px;
    font-weight: 400;
    color: #d4cfc4;
    text-align: center;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
    line-height: 1.4;
}

.landing-hint {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 13px;
    font-style: italic;
    color: #6b6660;
    text-align: center;
    margin-top: 0.75rem;
}

/* Landing textarea */
div[data-testid="stTextArea"] textarea {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(212,207,196,0.25) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    color: #d4cfc4 !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 20px !important;
    text-align: center !important;
    resize: none !important;
    caret-color: #d4cfc4 !important;
    min-height: 80px !important;
}

div[data-testid="stTextArea"] textarea:focus {
    box-shadow: none !important;
    border-bottom-color: rgba(212,207,196,0.5) !important;
}

div[data-testid="stTextArea"] label { display: none !important; }
div[data-testid="stTextArea"] [data-testid="InputInstructions"] { display: none !important; }

/* Landing submit button */
div[data-testid="stFormSubmitButton"] button {
    background: transparent !important;
    border: none !important;
    color: #6b6660 !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 13px !important;
    font-style: italic !important;
    letter-spacing: 0.12em !important;
    box-shadow: none !important;
    padding: 0.5rem 1rem !important;
    cursor: pointer !important;
    transition: color 0.3s ease !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    color: #d4cfc4 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Conversation screen ──────────────────────────── */
.convo-header {
    margin-bottom: 3rem;
    padding-top: 3rem;
    animation: fadeIn 0.6s ease forwards;
}

.figure-name {
    font-size: 44px;
    font-weight: 400;
    letter-spacing: 0.05em;
    line-height: 1.1;
    margin: 0 0 0.6rem 0;
}

.figure-reason {
    font-size: 16px;
    font-style: italic;
    line-height: 1.6;
    margin: 0;
    opacity: 0.65;
}

/* ── Messages ──────────────────────────────────────── */
.msg-block {
    margin-bottom: 2rem;
    animation: fadeIn 0.35s ease forwards;
}

.msg-user {
    text-align: right;
    padding-left: 25%;
}

.msg-user p {
    display: inline;
    font-size: 17px;
    line-height: 1.75;
    padding-right: 0.85rem;
    border-right: 2px solid;
}

.msg-figure {
    padding-right: 15%;
}

.msg-figure p {
    font-size: 19px;
    line-height: 1.85;
    margin: 0 0 1em 0;
}

.msg-figure p:last-child { margin-bottom: 0; }

/* ── Passages (marginal notes) ─────────────────────── */
.passages-block {
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
    padding-left: 1rem;
    border-left: 2px solid;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
}

.passage-item {
    font-size: 13.5px;
    font-style: italic;
    line-height: 1.65;
}

.passage-work {
    font-style: normal;
    font-size: 10.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.3rem;
    font-family: inherit;
}

.passage-excerpt {
    line-height: 1.65;
    opacity: 0.8;
}

/* Style the expanders inside the passage block */
.passages-block + div [data-testid="stExpander"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* ── Begin again ───────────────────────────────────── */
.begin-again-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.5rem;
}

.begin-again-wrap button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-style: italic !important;
    font-size: 12px !important;
    opacity: 0.35 !important;
    padding: 0.25rem 0 !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease !important;
    letter-spacing: 0.05em !important;
}

.begin-again-wrap button:hover {
    opacity: 0.85 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Chat input ────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid rgba(128,128,128,0.15) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 17px !important;
    font-style: italic !important;
}

[data-testid="stChatInputSubmitButton"] {
    opacity: 0.15 !important;
}
"""


def figure_css(theme: dict) -> str:
    css = f"""
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {{
        background: {theme['bg']} !important;
        background-attachment: fixed !important;
    }}

    body, p, div, span, li {{
        color: {theme['text']};
    }}

    .figure-name {{
        font-family: {theme['display_font']};
        color: {theme['accent']};
    }}

    .figure-reason,
    .convo-header {{
        font-family: {theme['body_font']};
        color: {theme['text']};
    }}

    .msg-user {{
        color: {theme['muted']};
        font-family: {theme['body_font']};
    }}

    .msg-user p {{
        border-right-color: {theme['accent']};
    }}

    .msg-figure {{
        color: {theme['text']};
        font-family: {theme['body_font']};
    }}

    .passages-block {{
        border-left-color: {theme['accent']}55;
    }}

    .passage-item {{
        color: {theme['accent']};
        font-family: {theme['body_font']};
    }}

    .passage-work {{
        color: {theme['accent']};
        font-family: {theme['display_font']};
    }}

    .begin-again-wrap button {{
        color: {theme['text']} !important;
        font-family: {theme['body_font']} !important;
    }}

    [data-testid="stChatInput"] textarea {{
        color: {theme['text']} !important;
        font-family: {theme['body_font']} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {theme['muted']} !important;
        font-style: italic !important;
    }}

    [data-testid="stChatInput"] {{
        border-top-color: {theme['accent']}22 !important;
    }}

    /* Sticky bottom bar inherits page background */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div,
    [data-testid="stChatInputContainer"],
    .stChatInputContainer {{
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }}
    """
    return css + theme.get("texture_css", "")


def landing_css() -> str:
    return """
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background: #1a1a1a !important;
    }
    .block-container {
        min-height: 100vh;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    [data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        max-width: 600px;
        gap: 0.25rem;
    }
    [data-testid="stForm"],
    div[data-testid="stTextArea"] {
        width: 100%;
        max-width: 600px;
    }
    """


def inject_css(fig_key: str | None):
    if fig_key and fig_key in THEMES:
        extra = figure_css(THEMES[fig_key])
    else:
        extra = landing_css()

    st.markdown(
        f'<style>@import url("{FONTS_URL}");\n{GLOBAL_CSS}\n{extra}</style>',
        unsafe_allow_html=True,
    )


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource
def _chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def load_chroma():
    client = _chroma_client()
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        st.error(
            "ChromaDB collection not found. "
            "Run `python setup_corpus.py` in the project directory, then refresh."
        )
        st.stop()


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        st.stop()
    return openai.OpenAI(api_key=api_key)


# ── Backend: figure profiles (unchanged) ──────────────────────────────────────

def build_profiles_string() -> str:
    lines = []
    for key, fig in FIGURES.items():
        lines.append(f'Figure key: "{key}"\nName: {fig["name"]}\nProfile: {fig["profile"]}')
        lines.append("")
    return "\n".join(lines)


PROFILES_STRING = build_profiles_string()

MATCHER_SYSTEM = f"""You are a literary counsellor matching people to the right guide for their situation.

Here are the eight figures available:

{PROFILES_STRING}

Your task: read the person's concern and select the single figure whose work most directly addresses what they are actually wrestling with.

Do not default to the most famous figure. Consider what the person is actually wrestling with beneath the surface phrasing, not just topic keywords. Pick the figure whose specific characteristic moves and concerns most closely match this person's situation.

Respond ONLY with valid JSON containing exactly two keys:
- "figure": the lowercase figure key (one of the exact keys listed above)
- "reason": a single sentence in plain language addressed to the user explaining why this figure was chosen (e.g. "Epictetus spent his life teaching how to maintain inner freedom when the world refuses to cooperate with your plans.")

Do not include any text outside the JSON object."""

STRICT_MATCHER_SYSTEM = MATCHER_SYSTEM + "\n\nIMPORTANT: Your entire response must be a single JSON object and nothing else. No markdown, no explanation, no code fences."


# ── Backend: match figure (unchanged) ─────────────────────────────────────────

def match_figure(concern: str, client: openai.OpenAI, strict: bool = False) -> dict:
    system = STRICT_MATCHER_SYSTEM if strict else MATCHER_SYSTEM
    response = client.chat.completions.create(
        model=MATCHER_MODEL,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": concern},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def safe_match_figure(concern: str, client: openai.OpenAI) -> tuple[str, str]:
    """Returns (figure_key, reason). Falls back to montaigne on parse failure."""
    try:
        result = match_figure(concern, client, strict=False)
        key = result.get("figure", "").lower()
        reason = result.get("reason", "")
        if key in FIGURES and reason:
            return key, reason
        raise ValueError(f"Invalid figure key or missing reason: {result}")
    except Exception as e:
        try:
            result = match_figure(concern, client, strict=True)
            key = result.get("figure", "").lower()
            reason = result.get("reason", "")
            if key in FIGURES and reason:
                return key, reason
            raise ValueError(f"Still invalid: {result}")
        except Exception:
            return "montaigne", (
                "Michel de Montaigne spent his life sitting with questions that resist easy answers — "
                "he seemed like the right companion for what you've described."
            )


# ── Backend: retrieval (unchanged) ────────────────────────────────────────────

def retrieve_passages(query: str, figure_key: str, collection, embedder, k: int = TOP_K) -> list[dict]:
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"figure": {"$eq": figure_key}},
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    for doc, meta in zip(docs, metas):
        passages.append({
            "text": doc,
            "work": meta.get("work", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return passages


# ── Backend: conversation system prompt (unchanged) ───────────────────────────

def build_conversation_system(figure_key: str, passages: list[dict]) -> str:
    fig = FIGURES[figure_key]
    fig_name = fig["name"]
    passage_block = ""
    for i, p in enumerate(passages, 1):
        passage_block += (
            f"\n\n--- Passage {i} | Source: {p['work']} (chunk {p['chunk_index']}) ---\n"
            f"{p['text']}"
        )
    return (
        f"You are {fig_name}. Respond as yourself — in your voice, from your worldview, "
        f"drawing on your own writing."
        f"{passage_block}"
        f"\n\nUse these passages naturally. When it fits, refer to them — for example, "
        f"'as I wrote in {{work}}' — or paraphrase their ideas in your own voice. "
        f"Do not quote them at length, since the user can see the passages alongside our conversation. "
        f"Speak as the figure, not as a summarizer of the figure. "
        f"Never refer to yourself as an AI, language model, or assistant. "
        f"If asked about events or technology after your death, respond with the curiosity, confusion, "
        f"or refusal that fits your character. "
        f"Do not moralize beyond what these passages support. "
        f"Be willing to push back, disagree, or sit with uncertainty — you are not a comfort-bot. "
        f"Keep responses to two to four paragraphs unless the user explicitly asks for more."
    )


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _to_html_paras(text: str) -> str:
    """Convert plain text with paragraph breaks into HTML <p> tags."""
    escaped = html_module.escape(text)
    paras = re.split(r"\n\s*\n", escaped)
    return "".join(f"<p>{p.strip()}</p>" for p in paras if p.strip())


def passage_excerpt(text: str, max_chars: int = 260) -> str:
    """Return roughly 2 sentences, capped at max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    sub = text[:max_chars]
    # Find last sentence boundary
    for punct in [". ", "! ", "? "]:
        idx = sub.rfind(punct)
        if idx > max_chars // 3:
            return text[: idx + 1]
    return sub.rstrip() + "\u2026"


def render_user_msg(text: str, theme: dict):
    escaped = html_module.escape(text)
    st.markdown(
        f'<div class="msg-block msg-user"><p style="border-right-color:{theme["accent"]}">'
        f"{escaped}</p></div>",
        unsafe_allow_html=True,
    )


def render_figure_msg(text: str):
    body_html = _to_html_paras(text)
    st.markdown(
        f'<div class="msg-block msg-figure">{body_html}</div>',
        unsafe_allow_html=True,
    )


def render_passages(passages: list[dict], theme: dict):
    if not passages:
        return

    items_html = ""
    for p in passages:
        work = html_module.escape(p["work"])
        excerpt = html_module.escape(passage_excerpt(p["text"]))
        items_html += (
            f'<div class="passage-item">'
            f'<div class="passage-work">{work}</div>'
            f'<div class="passage-excerpt">{excerpt}</div>'
            f"</div>"
        )

    st.markdown(
        f'<div class="passages-block" style="border-left-color:{theme["accent"]}55;">'
        f"{items_html}</div>",
        unsafe_allow_html=True,
    )


# ── Streaming response (display updated; API call logic unchanged) ─────────────

def generate_figure_response(
    figure_key: str,
    messages: list[dict],
    passages: list[dict],
    client: openai.OpenAI,
    theme: dict | None = None,
) -> str:
    system_prompt = build_conversation_system(figure_key, passages)

    openai_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    full_response = ""
    placeholder = st.empty()

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        messages=openai_messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            placeholder.markdown(
                f'<div class="msg-block msg-figure">'
                f"{_to_html_paras(full_response)}"
                f'<span style="opacity:0.4"> |</span></div>',
                unsafe_allow_html=True,
            )

    placeholder.markdown(
        f'<div class="msg-block msg-figure">{_to_html_paras(full_response)}</div>',
        unsafe_allow_html=True,
    )
    return full_response


# ── Session state ──────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "messages": [],           # [{role, content, passages?}]
        "selected_figure": None,
        "match_reason": None,
        "retrieved_passages": [],
        "initial_concern": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Landing screen ─────────────────────────────────────────────────────────────

def render_landing(openai_client, collection, embedder):
    st.markdown(
        '<div class="landing-prompt">What\'s weighing on you?</div>',
        unsafe_allow_html=True,
    )

    with st.form("concern_form", clear_on_submit=False):
        concern = st.text_area(
            "",
            height=80,
            placeholder="",
            key="concern_input",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("begin", use_container_width=True)

    st.markdown(
        '<div class="landing-hint">ctrl + enter to begin</div>',
        unsafe_allow_html=True,
    )

    if submitted and concern.strip():
        with st.spinner(""):
            fig_key, reason = safe_match_figure(concern.strip(), openai_client)

        passages = retrieve_passages(concern.strip(), fig_key, collection, embedder)

        st.session_state.selected_figure = fig_key
        st.session_state.match_reason = reason
        st.session_state.initial_concern = concern.strip()
        st.session_state.retrieved_passages = passages
        st.session_state.messages = [{"role": "user", "content": concern.strip()}]

        st.rerun()
    elif submitted:
        st.markdown(
            '<div style="text-align:center;color:#6b6660;font-style:italic;'
            'font-family:\'Cormorant Garamond\',serif;font-size:14px;">'
            "write something first</div>",
            unsafe_allow_html=True,
        )


# ── Conversation screen ────────────────────────────────────────────────────────

def render_conversation(fig_key: str, openai_client, collection, embedder):
    theme = THEMES[fig_key]
    fig_name = FIGURES[fig_key]["name"]
    reason = st.session_state.match_reason or ""

    # "begin again" — top right
    _, col_ba = st.columns([10, 2])
    with col_ba:
        st.markdown('<div class="begin-again-wrap">', unsafe_allow_html=True)
        if st.button("begin again", key="begin_again"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Figure header
    display_font = theme["display_font"]
    body_font = theme["body_font"]
    accent = theme["accent"]
    text_color = theme["text"]

    st.markdown(
        f'<div class="convo-header">'
        f'<div class="figure-name" style="font-family:{display_font};color:{accent};">'
        f"{html_module.escape(fig_name)}</div>"
        f'<div class="figure-reason" style="font-family:{body_font};color:{text_color};">'
        f"{html_module.escape(reason)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Render message history
    messages = st.session_state.messages
    for i, msg in enumerate(messages):
        is_last = i == len(messages) - 1
        if msg["role"] == "user":
            render_user_msg(msg["content"], theme)
        else:
            render_figure_msg(msg["content"])
            # Show passages for this assistant turn
            turn_passages = msg.get("passages") or []
            if turn_passages:
                render_passages(turn_passages, theme)

    # Generate response if last message is from user
    if messages and messages[-1]["role"] == "user":
        current_passages = st.session_state.retrieved_passages
        response_text = generate_figure_response(
            fig_key, messages, current_passages, openai_client, theme
        )
        # Append with passages attached
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "passages": current_passages}
        )
        # Show passages immediately below the just-streamed response
        render_passages(current_passages, theme)

    # Chat input
    user_input = st.chat_input("speak")
    if user_input and user_input.strip():
        # Use the most recent assistant response + new user message for richer embedding context
        context_parts = []
        for msg in reversed(messages[-6:]):
            if msg["role"] == "assistant":
                context_parts.append(msg["content"][:600])
                break
        context_parts.append(user_input.strip())
        retrieval_query = " ".join(context_parts)

        passages = retrieve_passages(retrieval_query, fig_key, collection, embedder)
        st.session_state.retrieved_passages = passages
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    fig_key = st.session_state.selected_figure

    # CSS first, always
    inject_css(fig_key)

    collection = load_chroma()
    embedder = load_embedder()
    openai_client = load_openai_client()

    if not fig_key:
        render_landing(openai_client, collection, embedder)
    else:
        render_conversation(fig_key, openai_client, collection, embedder)


if __name__ == "__main__":
    main()
