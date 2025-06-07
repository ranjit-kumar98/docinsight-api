from fastapi import FastAPI
from app.routes.insight import router as insight_router
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="DocInsight API")

app.include_router(insight_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "DocInsight API is running ✅"}
