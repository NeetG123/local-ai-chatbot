"""
Document ingestion pipeline.

Usage:
    python ingest.py                  # ingests everything in docs/
    python ingest.py path/to/file.pdf # ingests a single file

Place PDF, DOCX, or TXT files in the docs/ directory, then run this script.
The resulting ChromaDB vector store is persisted to vectorstore/ and reused
by rag_chain.py at query time.
"""

import sys
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def load_documents(source: str) -> list:
    """Load all supported files from a directory or a single file path."""
    source_path = Path(source)
    paths = (
        [source_path]
        if source_path.is_file()
        else list(source_path.rglob("*"))
    )

    docs = []
    for path in tqdm(paths, desc="Loading files"):
        suffix = path.suffix.lower()
        if suffix not in config.SUPPORTED_EXTENSIONS:
            continue
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as exc:
            print(f"[WARN] Could not load {path.name}: {exc}")

    print(f"[INFO] Loaded {len(docs)} document pages/sections.")
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks


def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and persist to ChromaDB.

    If the collection already exists, new chunks are added to it.
    To start fresh, delete the vectorstore/ directory and re-run.
    """
    embeddings = _make_embeddings()

    # Chroma.from_documents adds to an existing collection when
    # persist_directory already contains that collection name.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"[INFO] Vectorstore persisted to: {config.CHROMA_PERSIST_DIR}")
    return vectorstore


def run_ingestion(source: str | None = None) -> None:
    target = source or config.DOCS_DIR
    docs = load_documents(target)
    if not docs:
        print("[WARN] No documents found. Place files in the docs/ folder.")
        return
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(path_arg)
