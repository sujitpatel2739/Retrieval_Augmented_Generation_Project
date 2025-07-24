from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from .config import Settings
from .logger.json_logger import JsonLogger
from .workflow import RAGWorkflow

import os
from dotenv import load_dotenv

app = FastAPI()

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="src/static"), name="static")

logger = JsonLogger()

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load settings
settings = Settings()

# Initialize workflow
workflow = RAGWorkflow(
    completion_threshold=settings.completion_threshold,
    vector_db_class=settings.weaviate_primary_class,
    OPENAI_API_KEY=OPENAI_API_KEY
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query through the RAG workflow"""
    response, workflow_log = workflow.execute(request.query)
    logger.log_workflow(workflow_log)
    return response

@app.post('/document')
async def process_file(file: UploadFile = File(...)) -> Dict:
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()
    
    if extension not in [".pdf", ".docx", ".txt", ".html", ".htm"]:
        return {"status": "ERROR", "status_code": 400, "detail": f"Unsupported file type: {filename}"}
    
    response = workflow._process_document(file)
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