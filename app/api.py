import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.cache import clear_cache, get_cached, set_cached
from app.chunker import chunk_text
from app.llm import ask
from app.parser import extract_text
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
        text = extract_text(content, file.filename)
        chunks = chunk_text(text)
        add_documents(chunks, file.filename)
        duration = round((time.time() - start) * 1000)
        return {
            "message": f"Uploaded and indexed {len(chunks)} chunks",
            "filename": file.filename,
            "duration_ms": duration,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Upload Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    start = time.time()
    try:
        cache_result = get_cached(req.question, req.top_k)
        cache_hit = cache_result is not None
        if not cache_hit:
            results = search(req.question, n=req.top_k)
            documents = results["documents"]
            metadatas = results["metadatas"]
            if not documents or all(d.strip() == "" for d in documents):
                raise HTTPException(
                    status_code=400,
                    detail="No document found. Please upload a document",
                )
            cache_result = {
                "documents": documents,
                "metadatas": metadatas,
            }
            set_cached(req.question, req.top_k, cache_result)
        documents = cache_result["documents"]
        metadatas = cache_result["metadatas"]
        context = "\n\n".join(
            f"[Source {i + 1}: {metadata['filename']} | chunk {metadata['chunk_index']}]\n{doc}"
            for i, (doc, metadata) in enumerate(zip(documents, metadatas))
        )
        llm_result = ask(req.question, context)
        answer = llm_result["answer"]
        duration = round((time.time() - start) * 1000)
        logger.info(
            f"Query answered in {duration}ms, {llm_result['total_tokens']} tokens used (prompt: {llm_result['prompt_tokens']}, completion: {llm_result['completion_tokens']})"
        )
        return {
            "question": req.question,
            "answer": answer,
            "chunks_used": documents,
            "duration_ms": duration,
            "cached": cache_hit,
            "token_usage": {
                "prompt_tokens": llm_result["prompt_tokens"],
                "completion_tokens": llm_result["completion_tokens"],
                "total_tokens": llm_result["total_tokens"],
            },
            "sources": [
                {
                    "filename": metadata["filename"],
                    "chunk_index": metadata["chunk_index"],
                }
                for metadata in metadatas
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query: {e}")
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
        logger.error(f"Failed to delete with message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
