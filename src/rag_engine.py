"""
RAG Engine for Singapore Travel Assistant
Handles document loading, chunking, embedding, vector store, and retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Embedding model — loaded lazily so the provider can be set at runtime
_embedding_model = None


def get_embedding_model(provider: str = "google", api_key: str = None):
    """Return the embedding model for the given provider."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        import os
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        _embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    else:
        from langchain_openai import OpenAIEmbeddings
        import os
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        _embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    return _embedding_model


KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
VECTOR_STORE_DIR = Path(__file__).parent.parent / "vector_store"

# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_source_metadata(file_path: str) -> dict:
    """
    Extract source title and URL from the first two lines of the markdown file.
    Expected format:
      # Title
      Source: <name>
      URL: <url>
    """
    metadata = {"source_file": os.path.basename(file_path)}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[:5]:
            line = line.strip()
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
            elif line.startswith("Source:"):
                metadata["source_name"] = line.replace("Source:", "").strip()
            elif line.startswith("URL:"):
                metadata["source_url"] = line.replace("URL:", "").strip()
    except Exception:
        pass
    return metadata


def load_documents() -> List[Document]:
    """
    Load all markdown files from the knowledge base directory.
    Splits by markdown headers first, then by character count.
    """
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")

    markdown_files = list(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found in {KNOWLEDGE_BASE_DIR}")

    # Header-based splitter (preserves section context)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    # Character splitter for long sections
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_documents = []

    for file_path in markdown_files:
        source_meta = extract_source_metadata(str(file_path))

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # First split by headers
        try:
            header_splits = header_splitter.split_text(content)
        except Exception:
            header_splits = [Document(page_content=content)]

        # Then split long sections by character count
        for doc in header_splits:
            # Combine header metadata with source metadata
            doc.metadata.update(source_meta)

            if len(doc.page_content) > CHUNK_SIZE:
                sub_chunks = char_splitter.split_documents([doc])
                all_documents.extend(sub_chunks)
            else:
                all_documents.append(doc)

    return all_documents


def build_vector_store(documents: List[Document], provider: str = "google", api_key: str = None) -> FAISS:
    """Build FAISS vector store from documents."""
    embeddings = get_embedding_model(provider, api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS) -> None:
    """Persist vector store to disk."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"✅ Vector store saved to {VECTOR_STORE_DIR}")


def load_vector_store(provider: str = "google", api_key: str = None) -> Optional[FAISS]:
    """Load vector store from disk if it exists."""
    index_path = VECTOR_STORE_DIR / "index.faiss"
    if not index_path.exists():
        return None
    embeddings = get_embedding_model(provider, api_key)
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def get_or_build_vector_store(force_rebuild: bool = False, provider: str = "google", api_key: str = None) -> FAISS:
    """
    Load existing vector store or build a new one from knowledge base documents.
    """
    global _embedding_model
    _embedding_model = None  # reset so new provider/key is used

    if not force_rebuild:
        vs = load_vector_store(provider, api_key)
        if vs is not None:
            print("✅ Loaded existing vector store from disk.")
            return vs

    print("📚 Building vector store from knowledge base documents...")
    documents = load_documents()
    print(f"   Loaded {len(documents)} chunks from {KNOWLEDGE_BASE_DIR}")

    vector_store = build_vector_store(documents, provider, api_key)
    save_vector_store(vector_store)
    return vector_store


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve the top-k most relevant document chunks for a query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke(query)


def format_retrieved_sources(docs: List[Document]) -> str:
    """
    Format retrieved document chunks into a context string for the LLM,
    including source citations.
    """
    if not docs:
        return "No relevant information found in the knowledge base."

    sections = []
    seen_sources = set()

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        title = meta.get("source_name", meta.get("title", "Singapore Travel Guide"))
        url = meta.get("source_url", "")

        source_key = f"{title}|{url}"
        citation = f"[Source: {title}" + (f" — {url}]" if url else "]")

        sections.append(f"--- Excerpt {i} {citation} ---\n{doc.page_content.strip()}")

        seen_sources.add(source_key)

    return "\n\n".join(sections)


def get_source_references(docs: List[Document]) -> List[dict]:
    """
    Return a deduplicated list of source references from retrieved documents.
    """
    seen = set()
    sources = []
    for doc in docs:
        meta = doc.metadata
        title = meta.get("source_name", meta.get("title", "Singapore Travel Guide"))
        url = meta.get("source_url", "")
        key = f"{title}|{url}"
        if key not in seen:
            seen.add(key)
            sources.append({"title": title, "url": url})
    return sources
