import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# ── LM Studio ────────────────────────────────────────────────────
# Start the Local Server in LM Studio, then copy the model identifier
# shown at localhost:1234/v1/models and paste it as LM_STUDIO_MODEL.
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"          # any non-empty string works
LM_STUDIO_MODEL    = "mistral-7b-instruct-v0.3"

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 1024

# ── Embeddings ───────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # ~80 MB, auto-cached by HuggingFace

# ── ChromaDB ─────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(BASE_DIR / "vectorstore")
CHROMA_COLLECTION  = "documents"

# ── Document Ingestion ───────────────────────────────────────────
DOCS_DIR              = str(BASE_DIR / "docs")
CHUNK_SIZE            = 1000
CHUNK_OVERLAP         = 150
SUPPORTED_EXTENSIONS  = [".pdf", ".docx", ".txt"]

# ── RAG Retrieval ────────────────────────────────────────────────
RETRIEVER_K               = 4
RELEVANCE_SCORE_THRESHOLD = 0.30   # queries scoring below this skip RAG context

# ── Voice / Whisper ──────────────────────────────────────────────
# WHISPER_MODEL options: tiny (~75 MB), base (~145 MB), small (~460 MB),
#                        medium (~1.5 GB), large (~3 GB)
# Use "base" for a good speed/accuracy balance on CPU.
WHISPER_MODEL        = "base"
AUDIO_SAMPLE_RATE    = 16000       # Hz — required by Whisper
AUDIO_RECORD_SECONDS = 7
AUDIO_TEMP_DIR       = str(BASE_DIR / "audio_temp")

# ── TTS ──────────────────────────────────────────────────────────
TTS_RATE   = 175    # words per minute (pyttsx3)
TTS_VOLUME = 0.9    # 0.0 – 1.0
