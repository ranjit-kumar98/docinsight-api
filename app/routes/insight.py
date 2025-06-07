import pdfplumber
from fastapi import APIRouter, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse
from app.utils.chunking import chunk_text
from app.utils.embedder import embed_chunks
from app.utils.embedder import embed_text
from app.utils.similarity import cosine_similarity

router = APIRouter()

# In-memory storage
pdf_memory_store = []


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save file to temp location
    file_location = f"savedFile/stemp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Extract text using pdfplumber
    extracted_text = ""
    with pdfplumber.open(file_location) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() or ""  # avoid None

    # Clean up: delete temp file (optional for now)
    # os.remove(file_location)
     # Step 1: Chunk the text
    chunks = chunk_text(extracted_text)

    # Step 2: Get embeddings
    embedded_chunks = embed_chunks(chunks)

    pdf_memory_store.clear()
    pdf_memory_store.extend(embedded_chunks)

    return JSONResponse({
        "filename": file.filename,
        "chunks_count": len(chunks),
        "sample_chunk": embedded_chunks[0] if embedded_chunks else {}
    })

@router.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question")

    if not pdf_memory_store:
        return {"error": "No document uploaded yet."}

   
    query_vec = embed_text(question)

    # Score all chunks
    scored = []
    for item in pdf_memory_store:
        score = cosine_similarity(query_vec, item["embedding"])
        scored.append((score, item["chunk"]))

    # Sort by similarity score (descending)
    top_chunks = sorted(scored, key=lambda x: x[0], reverse=True)[:3]

    return {
        "question": question,
        "top_chunks": [
            {"score": round(score, 4), "chunk": chunk}
            for score, chunk in top_chunks
        ]
    }

@router.get("/insight")
async def get_insight(doc_id: str = Query(...), q: str = Query(...)):
    return {"answer": f"Insight for {doc_id}: {q}"}
