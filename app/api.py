import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.cache import clear_cache, get_cached, set_cached
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
    top_k: int = 2


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    start = time.time()
    try:
        logger.info("Starting Upload")
        clear_cache()
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
        cache_result = get_cached(req.question)
        if cache_result is not None:
            duration = round((time.time() - start) * 1000)
            logger.info(f"Cache hit for question: {req.question}")
            return {
                "question": req.question,
                "answer": cache_result,
                "chunks_used": [],
                "duration_ms": duration,
                "cached": True,
            }
        results = search(req.question, n=req.top_k)
        if not results or all(r.strip() == "" for r in results):
            raise HTTPException(
                status_code=400, detail="No document found. Please upload a document"
            )
        context = " ".join(results)
        answer = ask(req.question, context)
        duration = round((time.time() - start) * 1000)
        set_cached(req.question, answer)
        return {
            "question": req.question,
            "answer": answer,
            "chunks_used": results,
            "duration_ms": duration,
            "cached": False,
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
        clear_cache()
        return {"message": f"Cleared Collection", "duration": duration}
    except Exception as e:
        logger.warning(f"Failed to delete with message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
