# Counsel from the Dead

A Streamlit app for grounded conversations with literary figures, where every response is anchored in real passages from their writing via vector search.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and paste your Anthropic API key
   ```

3. **Download and index the corpus** (one-time, ~5–10 minutes)
   ```bash
   python setup_corpus.py
   ```
   This downloads public-domain texts from Project Gutenberg, chunks them, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and stores them in a local ChromaDB database. The script is idempotent — safe to re-run if interrupted.

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```
   Open the URL shown in your terminal (typically http://localhost:8501).

## How it works

1. Describe what's on your mind in the text box.
2. Claude Haiku matches you to one of eight literary figures whose work speaks to your concern.
3. You have a conversation with that figure using Claude Sonnet.
4. Every response is grounded in three passages retrieved from the figure's actual writing via vector search — visible in the sidebar.

## Figures available

- Leo Tolstoy — mortality, meaning, spiritual crisis
- Jane Austen — relationships, self-deception, social navigation
- Epictetus — control, loss, freedom of mind
- Michel de Montaigne — doubt, friendship, human contradiction
- Frederick Douglass — oppression, dignity, moral courage
- Friedrich Nietzsche — values, self-overcoming, meaning-making
- Jalal ad-Din Rumi — love, longing, spiritual path
- Confucius — character, duty, daily practice

## Project structure

```
.
├── app.py              # Streamlit application
├── setup_corpus.py     # One-time corpus download and indexing
├── figures.py          # Figure roster and thematic profiles
├── corpus/             # Downloaded texts (created by setup_corpus.py)
├── chroma_db/          # Vector store (created by setup_corpus.py)
├── requirements.txt
├── .env.example
└── README.md
```
