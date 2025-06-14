import pdfplumber
from fastapi import APIRouter, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse

from app.utils.chunking import chunk_text
from app.utils.embedder import embed_chunks
from app.utils.embedder import embed_text
from app.utils.similarity import cosine_similarity

from app.utils.qa import answer_question
from fastapi import HTTPException

from app.services.vector_store import FaissStore
import uuid

faiss_store = FaissStore()

router = APIRouter()



@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1) Save incoming PDF
    file_location = f"savedFile/stemp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # 2) Extract raw text
    extracted_text = ""
    with pdfplumber.open(file_location) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            extracted_text += page_text + "\n\n"

    # 3) Chunk & embed
    chunks = chunk_text(extracted_text)
    embedded = embed_chunks(chunks)

    # 4) Persist into FAISS
    doc_id    = str(uuid.uuid4())
    vectors   = [item["embedding"] for item in embedded]
    texts     = [item["chunk"]     for item in embedded]
    metadatas = [(doc_id, txt) for txt in texts]
    faiss_store.add(vectors, metadatas)

    # 5) Return metadata (including doc_id!)
    return JSONResponse({
        "filename":     file.filename,
        "doc_id":       doc_id,
        "chunks_count": len(chunks)
    })

@router.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question")
    doc_id = body.get("doc_id")  # Optional

    if not question:
        raise HTTPException(status_code=400, detail="`question` is required")

    if faiss_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents available")

    query_vec = embed_text(question)

    # Retrieve top-k results
    top_chunks = faiss_store.search(query_vec, top_k=3, doc_id=doc_id)

    return {
        "question": question,
        "top_chunks": [
            {"score": round(score, 4), "chunk": chunk}
            for score, chunk in top_chunks
        ]
    }


@router.post("/answer")
async def answer(request: Request):
    body = await request.json()
    question = body.get("question")
    doc_id = body.get("doc_id")  # Optional

    if not question:
        raise HTTPException(status_code=400, detail="`question` is required")

    if faiss_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents available")

    query_vec = embed_text(question)

    top_chunks = faiss_store.search(query_vec, top_k=3, doc_id=doc_id)
    context = " ".join(chunk for _, chunk in top_chunks)

    answer = answer_question(question, context)

    return {
        "question": question,
        "answer": answer,
        "source_chunks": [
            {"score": round(score, 4), "chunk": chunk}
            for score, chunk in top_chunks
        ]
    }



# @router.get("/insight")
# async def get_insight(doc_id: str = Query(...), q: str = Query(...)):
#     return {"answer": f"Insight for {doc_id}: {q}"}
