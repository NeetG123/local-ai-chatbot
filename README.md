# Local AI Chatbot

A fully offline, privacy-focused chatbot with document Q&A (RAG), voice input/output, and a web UI — all running locally on your machine with no data sent to external services.

## Features

- **Retrieval-Augmented Generation (RAG)** — Upload PDFs, DOCX, or TXT files and ask questions against them
- **Conversational Chat** — Falls back to plain chat when no documents are relevant
- **Voice Input** — Record audio or upload audio files; transcribed locally via OpenAI Whisper
- **Text-to-Speech** — Optionally speaks responses aloud using pyttsx3
- **100% Offline** — LLM inference via LM Studio, embeddings via Sentence Transformers, no external API calls

## Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Gradio 4.42 |
| LLM orchestration | LangChain 0.2 |
| LLM inference | LM Studio (local server) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | ChromaDB |
| Speech-to-text | OpenAI Whisper (base) |
| Text-to-speech | pyttsx3 |
| Audio capture | sounddevice |
| Document parsing | PyPDF, python-docx, unstructured |

## Project Structure

```
local-ai-chatbot/
├── app.py          # Gradio UI and event handlers
├── rag_chain.py    # RAGChatbot class and LLM + retriever logic
├── ingest.py       # Document loading, chunking, and embedding
├── voice.py        # SpeechToText and TextToSpeech classes
├── config.py       # All configuration settings
├── docs/           # Place your documents here
├── vectorstore/    # ChromaDB persisted embeddings (auto-created)
└── audio_temp/     # Temporary audio files (auto-created)
```

## Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) installed and running locally
  - Start the Local Server (default: `http://localhost:1234/v1`)
  - Load a model (default config assumes `Mistral-7B-Instruct-v0.3`)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd local-ai-chatbot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install PyTorch (CPU build)**

   ```bash
   pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

   For GPU support, install the appropriate CUDA build from [pytorch.org](https://pytorch.org) instead.

4. **Install remaining dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
python app.py
```

Open `http://localhost:7860` in your browser. On first run, the Whisper and Sentence Transformers models are downloaded automatically (~225 MB total).

## Adding Documents

**Via web UI:** Upload files using the document panel, then click **Ingest Documents**.

**Via command line:**

```bash
# Copy your files to the docs/ folder first
python ingest.py
```

Supported formats: `.pdf`, `.docx`, `.txt`

## Configuration

All settings are in [config.py](config.py). Key options:

| Setting | Default | Description |
|---|---|---|
| `LM_STUDIO_MODEL` | `mistral-7b-instruct-v0.3` | Model identifier from LM Studio |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic |
| `LLM_MAX_TOKENS` | `1024` | Maximum response length |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `CHUNK_SIZE` | `1000` | Document chunk size in characters |
| `RETRIEVER_K` | `4` | Number of document chunks retrieved per query |
| `RELEVANCE_SCORE_THRESHOLD` | `0.30` | Minimum similarity score to trigger RAG |
| `TTS_RATE` | `175` | Text-to-speech speed in words per minute |

## How It Works

1. **Ingestion** — Documents are split into 1000-character chunks with 150-character overlap, embedded with `all-MiniLM-L6-v2`, and persisted in ChromaDB.
2. **Query** — Each user message is embedded and compared against stored chunks. If the top match exceeds the relevance threshold, the most similar chunks are injected as context into the LLM prompt.
3. **Fallback** — When no documents meet the threshold, the chatbot answers using conversation history alone.
4. **Voice** — Audio is transcribed locally by Whisper before being sent to the chat pipeline. TTS runs on a background thread so the UI stays responsive.

## License

MIT
