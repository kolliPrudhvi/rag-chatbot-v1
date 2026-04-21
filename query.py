"""
query.py
--------
Query-time pipeline for the RAG chatbot.

Flow:
    User question -> Embed -> Retrieve top-k chunks -> Build prompt -> LLM -> Answer
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "resume_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama3.2:3b"   # must match what you pulled with `ollama pull`
LLM_TEMPERATURE = 0.1        # low = deterministic, factual. High = creative.
TOP_K = 8                    # how many chunks to retrieve per question


# -----------------------------------------------------------------------------
# PROMPT TEMPLATE
# -----------------------------------------------------------------------------
# This is arguably the most important piece of code in the whole project.
# It tells the LLM:
#   1. Its role and constraints
#   2. What the retrieved context is
#   3. What the user question is
#   4. How to handle missing info (say "I don't know" instead of hallucinating)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the provided context.

Rules:
- Answer using only the information in the context below.
- If the context does not contain the answer, say "I don't have enough information to answer that."
- Be concise and specific. Cite facts directly from the context.
- Do not make up information.

Context:
{context}

Question: {question}

Answer:"""


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def format_docs(docs: list) -> str:
    """Concatenate retrieved chunks into a single string for the prompt.

    Each chunk is separated by a divider so the LLM can tell them apart.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# -----------------------------------------------------------------------------
# BUILD THE PIPELINE
# -----------------------------------------------------------------------------
def build_rag_chain():
    """Assemble the full retrieval-augmented generation chain."""

    # 1. Load the persisted Chroma index
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # 2. Turn the vector store into a retriever
    # The retriever is the interface: "give me top-k chunks for a query"
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # 3. Set up the LLM (local, via Ollama)
    print(f"Connecting to Ollama ({LLM_MODEL})...")
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # 4. Build the prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 5. Wire everything together using LCEL (LangChain Expression Language)
    # The `|` operator pipes output from one step into the next.
    #
    # Flow:
    #   Input dict -> {context: retriever output, question: passthrough}
    #   -> prompt fills in template
    #   -> LLM generates
    #   -> parse to string
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RAG CHATBOT — CLI")
    print("=" * 60)

    chain, retriever = build_rag_chain()

    print("\nReady. Type your question (or 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        # Show what chunks were retrieved (useful for debugging/interviews)
        print("\n[Retrieving relevant chunks...]")
        retrieved = retriever.invoke(question)
        for i, doc in enumerate(retrieved, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] {source} (p.{page}): {preview}...")

        # Generate answer
        print("\n[Generating answer...]")
        answer = chain.invoke(question)
        print(f"\nBot: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()