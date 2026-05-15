from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "rag" / "medical_docs"
VECTOR_DIR = BASE_DIR / "rag" / "vector_db"


def build_vector_store() -> str:
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFDirectoryLoader(str(DOCS_DIR))
        docs = loader.load()
        if not docs:
            return "No PDFs found in rag/medical_docs."

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
        ).split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=str(VECTOR_DIR),
        ).persist()

        return f"Knowledge base indexed {len(chunks)} text chunks from {len(docs)} PDF pages."

    except Exception as exc:
        return f"RAG libraries are not ready: {exc}"


def retrieve_context(query: str, k: int = 3) -> str:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        if not VECTOR_DIR.exists():
            return ""

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )

        docs = db.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)

    except Exception:
        return ""
