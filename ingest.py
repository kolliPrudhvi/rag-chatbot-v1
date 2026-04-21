"""
ingest.py
---------
Indexing pipeline for the RAG chatbot.

Pipeline:
    PDF files -> Clean text -> Chunk -> Embed -> ChromaDB

Run this script once whenever documents in data/ change.
"""

import re
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
# TEXT CLEANING
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Fix common PyPDF extraction artifacts.

    - Rejoin single-letter splits (T exas -> Texas, T ools -> Tools)
    - Add missing spaces after colons and parens
    - Collapse multiple spaces
    """
    # Rejoin capital-letter splits: "T exas" -> "Texas"
    text = re.sub(r'\b([A-Z]) ([a-z]+)', r'\1\2', text)
    # Add space after colon before uppercase: "Tools:Docker" -> "Tools: Docker"
    text = re.sub(r':([A-Z])', r': \1', text)
    # Add space after closing paren before uppercase: ")Aug" -> ") Aug"
    text = re.sub(r'\)([A-Z])', r') \1', text)
    # Fix concatenated title-date: "InternMar" -> "Intern Mar" (two capitals joined)
    text = re.sub(r'([a-z])([A-Z][a-z]{2})', r'\1 \2', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text


# -----------------------------------------------------------------------------
# STEP 1: LOAD PDFs
# -----------------------------------------------------------------------------
def load_pdfs(data_dir: Path) -> list:
    """Load every PDF in data_dir and clean extraction artifacts."""
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}/")

    all_docs = []
    for pdf in pdf_files:
        print(f"  Loading {pdf.name}...")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
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
    """Embed each chunk and persist to ChromaDB.

    Deletes any existing collection first to avoid duplicate chunks
    from repeated runs.
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"  Resetting existing collection (if any)...")
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    existing.delete_collection()

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