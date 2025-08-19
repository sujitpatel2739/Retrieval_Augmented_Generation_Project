from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from .config import Settings
from .logger.json_logger import JsonLogger
from .workflow import RAGWorkflow
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load settings
settings = Settings()
logger = JsonLogger()

class QueryRequest(BaseModel):
    query: str
    top_k: int

workflow = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow
    # Initialize workflow
    workflow = RAGWorkflow(
        completion_threshold=settings.completion_threshold,
        collection_name=str(settings.weaviate_primary_class),
        OPENAI_API_KEY=OPENAI_API_KEY
    )
    yield
    workflow.db_operator.close_connection()
    print("Workflow Lifespan Ended! All connections closed")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/")
def serve_home():
    return FileResponse("src/static/index.html")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query through the RAG workflow"""
    print("/query called")
    response, workflow_log = workflow.execute(request.query, request.top_k)
    logger.log_workflow(workflow_log)
    print(response)
    return response


@app.post('/document')
async def process_file(
    file: UploadFile = File(...),
    max_token_len: int = Form(...),
    min_token_len: int = Form(...)
    ) -> Dict:
    
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()

    if extension not in [".pdf", ".docx", ".txt", ".html", ".htm"]:
        return {"status": "ERROR", "status_code": 400, "detail": f"Unsupported file type: {extension}"}

    try:
        file_bytes = await file.read()
    except Exception as e:
        return {"status": "EXCEPTION", "status_code": 400, "detail": f"Cannot read Bytes: {e}"}

    response = workflow.process_document(
        file_bytes=file_bytes,
        extension=extension,
        max_token_len=max_token_len,
        min_token_len=min_token_len
    )
    return response


@app.get("/logs/workflows")
async def get_workflow_logs(
    workflow_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Get workflow logs with optional filtering"""
    return workflow.logger.get_workflow_logs(workflow_id, start_time, end_time)

@app.get("/logs/finetuning")
async def export_logs_for_finetuning(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Export logs in OpenAI finetuning format"""
    return workflow.logger.export_for_finetuning(start_time, end_time)


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
