import pdfplumber
from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from app.utils.chunking import chunk_text
from app.utils.embedder import embed_chunks

router = APIRouter()

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

    return JSONResponse({
        "filename": file.filename,
        "chunks_count": len(chunks),
        "sample_chunk": embedded_chunks[0] if embedded_chunks else {}
    })

@router.get("/insight")
async def get_insight(doc_id: str = Query(...), q: str = Query(...)):
    return {"answer": f"Insight for {doc_id}: {q}"}
