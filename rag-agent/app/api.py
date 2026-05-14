import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.chunker import chunk_text
from app.llm import ask
from app.retriever import add_document, search

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    start = time.time()
    logger.info(f"Upload request received: {file.filename}")
    content = await file.read()
    text = content.decode("utf-8")
    chunks = chunk_text(text)
    add_document(chunks)
    duration = round((time.time() - start) * 1000)
    logger.info(f"Successfully indexed {len(chunks)} chunks in {duration}ms")
    return {
        "message": f"Uploaded and indexed {len(chunks)} chunks",
        "duration_ms": duration,
    }


@app.post("/query")
def query(req: QueryRequest):
    start = time.time()
    logger.info(f"Query received: {req.question}")
    results = search(req.question)
    if not results or all(r.strip() == "" for r in results):
        logger.warning("Query attempted but no documents in database")
        raise HTTPException(
            status_code=400, detail="No document found. Please upload a document first."
        )
    context = " ".join(results)
    answer = ask(req.question, context)
    duration = round((time.time() - start) * 1000)
    logger.info(
        f"Query answered successfully in {duration}ms, used {len(results)} chunks"
    )
    return {
        "question": req.question,
        "answer": answer,
        "chunks_used": results,
        "duration_ms": duration,
    }
