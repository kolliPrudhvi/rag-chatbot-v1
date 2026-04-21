"""
ingest.py
---------
Indexing pipeline for the RAG chatbot.

Pipeline:
    PDF files -> Text chunks -> Embeddings -> ChromaDB

Run this script once whenever documents in data/ change.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "resume_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# -----------------------------------------------------------------------------
# STEP 1: LOAD PDFs
# -----------------------------------------------------------------------------
def load_pdfs(data_dir: Path) -> list:
    """Load every PDF in data_dir into LangChain Document objects."""
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}/")

    all_docs = []
    for pdf in pdf_files:
        print(f"  Loading {pdf.name}...")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        all_docs.extend(docs)
        print(f"    -> {len(docs)} pages")
    return all_docs


# -----------------------------------------------------------------------------
# STEP 2: SPLIT INTO CHUNKS
# -----------------------------------------------------------------------------
def split_documents(docs: list) -> list:
    """Split full-page Documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


# -----------------------------------------------------------------------------
# STEP 3: EMBED AND STORE
# -----------------------------------------------------------------------------
def build_vector_store(chunks: list):
    """Embed each chunk and persist to ChromaDB."""
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"  Building Chroma index with {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vector_store


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RAG INDEXING PIPELINE")
    print("=" * 60)

    print("\n[1/3] Loading PDFs...")
    docs = load_pdfs(DATA_DIR)
    print(f"  Total pages loaded: {len(docs)}")

    print("\n[2/3] Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  Total chunks created: {len(chunks)}")
    if chunks:
        print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    print("\n[3/3] Embedding and storing in ChromaDB...")
    build_vector_store(chunks)

    print("\n" + "=" * 60)
    print(f"DONE. Index saved to ./{PERSIST_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()