"""RAG retriever over framework notes (MECE, Pyramid Principle, RICE, etc.).

Uses a local sentence-transformers embedding model (no extra API key) and
a Chroma vector store persisted to data/chroma_db/. The index is built
lazily on first use and reused afterward.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

NOTES_DIR = Path(__file__).resolve().parent.parent / "data" / "framework_notes"
PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "framework_notes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _load_notes() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for path in sorted(NOTES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path.name}))
    return docs


def _get_vectorstore(rebuild: bool = False) -> Chroma:
    global _vectorstore
    if _vectorstore is not None and not rebuild:
        return _vectorstore

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    if rebuild or store._collection.count() == 0:
        docs = _load_notes()
        if docs:
            store.add_documents(docs)

    _vectorstore = store
    return store


def get_retriever(k: int = 4):
    """Return a LangChain retriever over the framework notes collection."""
    return _get_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve_framework_notes(query: str, k: int = 4) -> list[dict]:
    """Retrieve the top-k framework-note chunks relevant to `query`.

    Returns a list of dicts: {content, source}.
    """
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    return [{"content": doc.page_content, "source": doc.metadata.get("source")} for doc in results]
