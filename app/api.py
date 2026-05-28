import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.cache import clear_cache, get_cached, set_cached
from app.chunker import chunk_text
from app.llm import ask
from app.parser import extract_text
from app.query_rewriter import rewrite_query
from app.reranker import rerank
from app.retriever import (
    add_documents,
    bm25_retrieve,
    delete_document,
    hybrid_search,
    multi_search,
    search,
)

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    search_k: int = 10
    rerank_top_k: int = 2
    use_rerank: bool = False
    use_rewrite_query: bool = False
    use_hybrid: bool = False


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
        if req.use_hybrid:
            cache_hit = False
            result = hybrid_search(req.question, req.search_k)
        elif req.use_rewrite_query:
            cache_hit = False
            queries = [req.question] + rewrite_query(req.question, n=3)
            result = multi_search(queries, req.search_k)
        else:
            result = get_cached(req.question, req.search_k)
            cache_hit = result is not None
            if not cache_hit:
                results = search(req.question, n=req.search_k)
                search_documents = results["documents"]
                search_metadatas = results["metadatas"]
                if not search_documents or all(
                    d.strip() == "" for d in search_documents
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="No document found. Please upload a document",
                    )
                result = {
                    "documents": search_documents,
                    "metadatas": search_metadatas,
                }
                set_cached(req.question, req.search_k, result)
        if req.use_rerank:
            reranked_result = rerank(req.question, result, req.rerank_top_k)
            documents = reranked_result["documents"]
            metadatas = reranked_result["metadatas"]
            scores = reranked_result["scores"]
        else:
            documents = result["documents"][: req.rerank_top_k]
            metadatas = result["metadatas"][: req.rerank_top_k]
            scores = [None] * len(documents)

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
                    "preview": metadata["preview"],
                    "score": score,
                }
                for metadata, score in zip(metadatas, scores)
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


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    try:
        cache_result = get_cached(req.question, req.search_k)
        if cache_result is None:
            content = search(req.question, req.search_k)
            search_documents = content["documents"]
            search_metadatas = content["metadatas"]
            if not search_documents or all(d.strip() == "" for d in search_documents):
                raise HTTPException(
                    status_code=400, detail="No document found. Please upload document"
                )
            cache_result = {
                "documents": search_documents,
                "metadatas": search_metadatas,
            }
            set_cached(req.question, req.search_k, cache_result)
        if req.use_rerank:
            reranked_result = rerank(req.question, cache_result, req.rerank_top_k)
            documents = reranked_result["documents"]
            metadatas = reranked_result["metadatas"]
            scores = reranked_result["scores"]
        else:
            documents = cache_result["documents"][: req.rerank_top_k]
            metadatas = cache_result["metadatas"][: req.rerank_top_k]
            scores = [None] * len(documents)
        return {
            "question": req.question,
            "sources": [
                {
                    "filename": metadata["filename"],
                    "chunk_index": metadata["chunk_index"],
                    "preview": metadata["preview"],
                    "text": doc,
                    "score": score,
                }
                for doc, metadata, score in zip(documents, metadatas, scores)
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieve failed with {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
