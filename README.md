# Resume RAG Chatbot

A local, end-to-end Retrieval-Augmented Generation (RAG) system that answers questions about a resume using a local LLM — no API keys, no cloud, no cost.

## Demo

Ask questions like:
- "What is this person's education?"
- "Summarize their AI/ML experience"
- "What programming languages do they know?"

The system retrieves the most relevant chunks from the resume and grounds the LLM's answer in that context.

## Stack

| Component | Tech |
|---|---|
| LLM | Llama 3.2 (3B) via [Ollama](https://ollama.com/) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent, local) |
| Orchestration | LangChain (LCEL) |
| UI | Streamlit |

## Architecture
PDF → Clean text → Chunks (500 char, 100 overlap) → Embeddings → ChromaDB
↓
User question → Embed → Top-k retrieval → Prompt with context → Llama 3.2 → Answer

## Setup

```bash
# 1. Install Ollama and pull the model
ollama pull llama3.2:3b

# 2. Create venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add PDFs to data/
mkdir data
# drop your PDFs into data/

# 5. Build the index
python ingest.py

# 6. Run the CLI chatbot
python query.py

# OR run the web UI
streamlit run app.py
```

## Project Structure
rag-chatbot-v1/
├── data/              # PDF documents to index
├── chroma_db/         # Persistent vector store (gitignored)
├── ingest.py          # Indexing pipeline
├── query.py           # CLI chatbot
├── app.py             # Streamlit web UI
├── requirements.txt
└── README.md


## Key Design Decisions

- **Local-first:** Uses Ollama + HuggingFace embeddings to avoid API costs and keep data private. Swapping to a hosted API is a one-line change.
- **Chunking:** 500-char chunks with 100-char overlap, using `RecursiveCharacterTextSplitter` to preserve semantic boundaries.
- **Text cleaning:** PyPDF introduces artifacts like `"T exas"` and `"Tools:Docker"`. A regex-based cleaner normalizes these before chunking.
- **Transparent retrieval:** The UI shows which chunks fed each answer, making hallucinations easy to spot.

## Known Limitations & Next Steps

- **No chat memory:** Each question is independent. Next: add conversational memory.
- **Pure semantic retrieval:** Fails when specific keywords matter (e.g., "M.S." vs "Master's"). Next: add BM25 hybrid search.
- **No evaluation harness:** Next: add RAGAS or similar for retrieval precision/recall + answer faithfulness metrics.
- **Single-user Streamlit:** Not production-grade. Next: FastAPI backend + React frontend for multi-user.

## License

MIT