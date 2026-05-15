import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.chunker import chunk_text
from app.llm import ask
from app.retriever import add_documents, delete_document, search

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
    try:
        logger.info("Starting Upload")
        content = await file.read()
        text = content.decode("utf-8")
        chunks = chunk_text(text)
        add_documents(chunks)
        duration = round((time.time() - start) * 1000)
        return {
            "message": f"Uploaded and indexed {len(chunks)} chunks",
            "duration_ms": duration,
        }
    except RuntimeError as e:
        logger.warning(f"Upload Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    start = time.time()
    try:
        results = search(req.question)
        if not results or all(r.strip() == "" for r in results):
            raise HTTPException(
                status_code=400, detail="No document found. Please upload a document"
            )
        context = " ".join(results)
        answer = ask(req.question, context)
        duration = round((time.time() - start) * 1000)
        return {
            "question": req.question,
            "answer": answer,
            "chunks_used": results,
            "duration_ms": duration,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "current time": time.time(),
    }


@app.delete("/document")
def delete():
    start = time.time()
    try:
        delete_document()
        duration = (time.time() - start) * 1000
        return {"message": f"Cleared Collection", "duration": duration}
    except Exception as e:
        logger.warning(f"Failed to delete with message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
