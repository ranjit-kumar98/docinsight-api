from fastapi import APIRouter, File, UploadFile, Query

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    return {"message": f"Received file: {file.filename}"}

@router.get("/insight")
async def get_insight(doc_id: str = Query(...), q: str = Query(...)):
    return {"answer": f"Insight for {doc_id}: {q}"}
