## Overview

This project is a Retrieval-Augmented Generation (RAG) API built to minimise LLM hallucinations by grounding answers strictly in uploaded documents. When a user uploads a document, it is chunked, embedded, and stored in a ChromaDB vector database. When a question is asked, the API retrieves the most semantically similar chunks and passes them as context to the LLM (via Groq), ensuring answers are based only on the provided documents and not the model's general knowledge.

Built to learn Python, FastAPI, and core RAG engineering concepts from scratch..

## Architecture

Client -> FastAPI -> ChromaDB (retrieval)
-> Groq LLM (generation)

FastAPI: Provides the backend API that can then be called
ChromaDB: In-memory vector database for adding and retrieving documents semantically
Groq LLM: Generates answers using only the retrieved chunks as context, preventing hallucination

## How to Run

1. Clone the repository
   git clone https://github.com/Rububzz/rag-agent.git
   cd rag-agent

2. Create a `.env` file in the root folder with your Groq API key
   GROQ_API_KEY=your-key-here

3. Start the server
   docker compose up

4. The API will be available at http://localhost:8000
   Visit http://localhost:8000/docs for the interactive API explorer

## API Endpoints

### GET /health

Check if the server is running.

- **Returns:** `{"status": "ok", "current time": ...}`
- **Test:** `curl http://localhost:8000/health`

### POST /upload

Upload a document to the knowledge base.

- **Accepts:** `.txt` file
- **Returns:** `{"message": "Uploaded and indexed N chunks", "duration_ms": ...}`
- **Test:** `curl -X POST http://localhost:8000/upload -F "file=@sample.txt"`

### POST /query

Ask a question against uploaded documents.

- **Accepts:** `{"question": "your question here"}`
- **Returns:** `{"question": ..., "answer": ..., "chunks_used": ..., "duration_ms": ...}`
- **Note:** Returns 400 if no documents have been uploaded
- **Test:** `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "How does RAG work?"}'`

## What I learnt

1. Precision vs Context
   - Precision = How much of the retrieved text is actually relevant to the user’s question?
   - Context = How much surrounding information is preserved in each chunk for the LLM to understand the answer
2. How chunking affects precision and context
   - **Small**: More precise but loses context
   - **Big**: More context but precision suffers
3. Vector Database Querying
   - How embedding creates vectors and the database is able to find semantically similar embedded text to return as context
4. Query phrasing affects retrieval quality
   - When "How does RAG work?" returned photosynthesis as the second result but the longer query returned correct results
5. Error handling matters
   - When ChromaDB was empty the LLM hallucinated a confident answer instead of saying it didn't know, which is why validation is critical in RAG systems
