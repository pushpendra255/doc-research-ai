from fastapi import APIRouter, UploadFile, File
from app.services.pdf_processor import extract_text
from app.services.vector_store import store_documents, search_documents
from app.services.qa_engine import ask_groq

router = APIRouter()

@router.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...)):
    docs = []
    for file in files:
        text = extract_text(await file.read())
        docs.append((file.filename, text))
    store_documents(docs)
    return {"status": "Documents processed."}

@router.get("/query")
async def query_docs(q: str):
    matches = search_documents(q)
    answer = ask_groq(q, matches)
    return {"answer": answer, "matches": matches}
