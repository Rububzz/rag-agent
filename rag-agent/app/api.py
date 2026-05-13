from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.chunker import chunk_text
from app.llm import ask
from app.retriever import add_document, search

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = chunk_text(text)
    add_document(chunks)
    return {"message": f"Uploaded and indexed {len(chunks)} chunks"}


@app.post("/query")
def query(req: QueryRequest):
    results = search(req.question)
    context = " ".join(results)
    answer = ask(req.question, context)
    return {"question": req.question, "answer": answer, "chunks_used": results}
