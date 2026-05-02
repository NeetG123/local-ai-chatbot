"""
RAG orchestrator.

RAGChatbot.query(user_input) → (answer: str, sources: list[str])

When ChromaDB returns zero chunks above the relevance threshold, the
call bypasses context injection and routes to the LLM as plain chat.
"""

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import config

_RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context below. If the context does not contain enough information, say "
    "you don't know rather than guessing.\n\nContext:\n{context}"
)

_CHAT_SYSTEM = (
    "You are a helpful, friendly assistant. Answer the user's question "
    "concisely and accurately."
)


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.LM_STUDIO_BASE_URL,
        api_key=config.LM_STUDIO_API_KEY,
        model=config.LM_STUDIO_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _make_retriever(embeddings: HuggingFaceEmbeddings):
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": config.RETRIEVER_K,
            "score_threshold": config.RELEVANCE_SCORE_THRESHOLD,
        },
    )


class RAGChatbot:
    def __init__(self):
        self.llm        = _make_llm()
        self.embeddings = _make_embeddings()
        self.retriever  = _make_retriever(self.embeddings)
        self.chat_history: list[tuple[str, str]] = []

    def refresh_retriever(self):
        """Call after new documents are ingested to pick up fresh embeddings."""
        self.retriever = _make_retriever(self.embeddings)

    def _recent_history(self) -> list:
        messages = []
        for role, content in self.chat_history[-6:]:   # last 3 turns
            cls = HumanMessage if role == "human" else AIMessage
            messages.append(cls(content=content))
        return messages

    def query(self, user_input: str) -> tuple[str, list[str]]:
        """Return (answer, list_of_source_filenames)."""
        docs = self.retriever.invoke(user_input)

        if docs:
            context  = "\n\n---\n\n".join(d.page_content for d in docs)
            sources  = list({d.metadata.get("source", "unknown") for d in docs})
            prompt   = ChatPromptTemplate.from_messages([
                ("system", _RAG_SYSTEM),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ])
            chain_in = {"context": context, "history": self._recent_history(), "question": user_input}
        else:
            sources  = []
            prompt   = ChatPromptTemplate.from_messages([
                ("system", _CHAT_SYSTEM),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ])
            chain_in = {"history": self._recent_history(), "question": user_input}

        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke(chain_in)

        self.chat_history.append(("human", user_input))
        self.chat_history.append(("ai", answer))

        return answer, sources

    def clear_history(self):
        self.chat_history = []
