import pdfplumber
from fastapi import APIRouter, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse

from app.utils.chunking import chunk_text
from app.utils.embedder import embed_chunks
from app.utils.embedder import embed_text
from app.utils.similarity import cosine_similarity

from app.utils.qa import answer_question
from fastapi import HTTPException

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
        # for page in pdf.pages:
        #     extracted_text += page.extract_text() or ""
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            print(f"\n--- Page {i+1} Content ---\n{page_text[:300]}...\n")
            extracted_text += page_text + "\n\n"  # avoid None

    # Clean up: delete temp file (optional for now)
    # os.remove(file_location)
     # Step 1: Chunk the text
    chunks = chunk_text(extracted_text)

    print(f"\n--- Total Chunks: {len(chunks)} ---")
    for i, ch in enumerate(chunks):
        print(f"\n--- Chunk {i+1} (Length: {len(ch.split())} tokens) ---\n{ch[:300]}...\n")


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


@router.post("/answer")
async def answer(request: Request):
    """
    Returns a precise answer to the user's question by:
     1. Embedding + retrieving top chunks (Day 4)
     2. Running a QA model over those chunks
    """
    body = await request.json()
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="`question` is required")

    if not pdf_memory_store:
        raise HTTPException(status_code=400, detail="No document uploaded")

    # 1. Embed & retrieve top chunks (reuse Day 4 logic)
    query_vec = embed_text(question)
    scores = [(cosine_similarity(query_vec, item["embedding"]), item["chunk"])
              for item in pdf_memory_store]
    top_chunks = sorted(scores, key=lambda x: x[0], reverse=True)[:3]
    context = " ".join(chunk for _, chunk in top_chunks)

    # 2. QA over the combined context
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
