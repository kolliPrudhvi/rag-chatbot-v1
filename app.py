"""
app.py
------
Streamlit web UI for the RAG chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------------------------------------------------------
# CONFIG (same as query.py)
# -----------------------------------------------------------------------------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "resume_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
LLM_TEMPERATURE = 0.1
TOP_K = 8


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


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# -----------------------------------------------------------------------------
# CACHED: build the chain ONCE and reuse across requests
# -----------------------------------------------------------------------------
# Streamlit re-runs the whole script on every interaction. @st.cache_resource
# makes sure we load the embedding model + LLM + vector store only once.
@st.cache_resource(show_spinner="Loading models and vector store...")
def build_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume RAG Chatbot")
st.caption("Ask questions about the indexed resume. Powered by Llama 3.2 + ChromaDB (fully local).")


# -----------------------------------------------------------------------------
# SIDEBAR — project info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Stack**
        - LLM: `llama3.2:3b` via Ollama
        - Embeddings: `all-MiniLM-L6-v2`
        - Vector DB: ChromaDB
        - Framework: LangChain
        - UI: Streamlit

        **How it works**
        1. Your question is embedded into a 384-dim vector.
        2. Chroma retrieves the top 8 most similar chunks from the resume.
        3. Chunks + your question are stuffed into an LLM prompt.
        4. Llama 3.2 generates a grounded answer.
        """
    )
    st.divider()
    st.caption("Tip: ask about skills, projects, experience, or education.")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# LOAD THE CHAIN
# -----------------------------------------------------------------------------
chain, retriever = build_rag_chain()


# -----------------------------------------------------------------------------
# CHAT STATE
# -----------------------------------------------------------------------------
# Streamlit session_state persists across re-runs within one session.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}** — page {src['page']}")
                    st.code(src["content"], language="text")


# -----------------------------------------------------------------------------
# CHAT INPUT
# -----------------------------------------------------------------------------
if question := st.chat_input("Ask a question about the resume..."):
    # Save + show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retriever.invoke(question)

        with st.spinner("Generating answer..."):
            answer = chain.invoke(question)

        st.markdown(answer)

        # Show sources in an expander
        sources = [
            {"page": d.metadata.get("page", "?"), "content": d.page_content}
            for d in retrieved
        ]
        with st.expander("Retrieved sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}** — page {src['page']}")
                st.code(src["content"], language="text")

        # Save assistant message with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })