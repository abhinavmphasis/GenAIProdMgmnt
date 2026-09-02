# 📚 AI Document RAG Assistant (Powered by OpenRouter)

An intelligent, interactive Retrieval-Augmented Generation (RAG) web application built with **Streamlit** and **OpenRouter**. Attach your documents (PDF, DOCX, TXT, CSV, Markdown, JSON), index their contents into vector representations, and query your knowledge base using manual natural language queries with cited sources and streaming responses.

---

## 🌟 Key Features

- 📎 **Multi-Format Document Ingestion**: Upload `.pdf`, `.docx`, `.txt`, `.csv`, `.md`, and `.json` documents seamlessly.
- ⚡ **High-Speed Vector Retrieval**: Fast semantic & lexical vector retrieval with cosine similarity scoring.
- 🤖 **OpenRouter LLM Integration**:
  - Connect to hundreds of state-of-the-art LLMs (OpenAI GPT-4o / GPT-4o-mini, Google Gemini 2.0 Flash, DeepSeek V3 / R1, Anthropic Claude 3.5 Sonnet / Haiku, Meta Llama 3.3 70B, Mistral, Qwen, etc.).
  - Real-time token streaming.
  - Test connection tool built directly into the UI.
- 🔍 **Transparent Source Citations**: Every AI response includes an expandable citation drawer showing the exact document source, page number, similarity score, and excerpt.
- 📑 **Document & Vector Inspector**: Search and explore indexed chunks without making LLM API calls.
- ⚙️ **Configurable Parameters**: Adjust chunk size, chunk overlap, Top-K retrieved documents, LLM temperature, and custom system prompts.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. Install the requirements via pip:

```bash
pip install -r requirements.txt
```

### 2. Configure OpenRouter API Key (Optional)
You can provide your API key directly in the web UI sidebar, or create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_DEFAULT_MODEL=openai/gpt-4o-mini
```

*(Get your OpenRouter API key at [openrouter.ai/keys](https://openrouter.ai/keys))*

### 3. Launch the Application
Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Attach Files**: Expand the **Attach & Manage Files** section in the main view or sidebar and upload one or more documents.
2. **Configure Model**: Select your preferred LLM model from the sidebar dropdown (e.g. `openai/gpt-4o-mini` or `google/gemini-2.0-flash-001`).
3. **Ask Questions**: Type your question into the chat input bar.
4. **Inspect Citations**: Click **View Retrieved Source Citations** beneath any response to inspect the chunks retrieved from your files.

---

## 📂 Project Structure

```
.
├── app.py                     # Streamlit frontend & interactive chat interface
├── core/
│   ├── __init__.py            # Core module exports
│   ├── document_loader.py     # Parser for PDF, DOCX, CSV, JSON, TXT, MD
│   ├── chunking.py            # Recursive text chunking with metadata
│   ├── vector_store.py        # Vector indexing and similarity search
│   └── llm.py                 # OpenRouter API client & RAG pipeline
├── .env.example               # Environment variables template
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```
