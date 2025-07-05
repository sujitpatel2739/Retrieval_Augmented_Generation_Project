from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from .config import Settings
from .logger.json_logger import JsonLogger
from .components import (
    LLMRequestRouter,
    LLMQueryReformulator,
    VectorRetriever,
    LLMCompletionChecker,
    LLMAnswerGenerator
)
from .workflow import RAGWorkflow
from .models import RAGResponse, Document
import os
from dotenv import load_dotenv

app = FastAPI()

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="src/static"), name="static")

logger = JsonLogger()

# Load environment variables from .env
load_dotenv()
# Read required values from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load settings
settings = Settings()

# Initialize retriever
retriever = VectorRetriever(
    class_name=settings.weaviate_class_name,
    url=settings.weaviate_url
)

# Initialize components
router = LLMRequestRouter(model=settings.router_model, api_key=OPENAI_API_KEY)
reformulator = LLMQueryReformulator(model=settings.reformulator_model, api_key=OPENAI_API_KEY)
completion_checker = LLMCompletionChecker(model=settings.completion_model, api_key=OPENAI_API_KEY)
answer_generator = LLMAnswerGenerator(model=settings.answer_model, api_key=OPENAI_API_KEY)

# Initialize workflow
workflow = RAGWorkflow(
    router=router,
    reformulator=reformulator,
    retriever=retriever,
    completion_checker=completion_checker,
    answer_generator=answer_generator,
    completion_threshold=settings.completion_threshold
)

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    documents: List[Document]

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query through the RAG workflow"""
    response, workflow_log = workflow.execute(request.query)
    logger.log_workflow(workflow_log)
    if not response:
        raise HTTPException(status_code=400, detail="Could not process query")
    return response

@app.post("/documents")
async def add_documents(request: DocumentRequest) -> dict:
    """Add documents to the retriever"""
    try:
        documents = [Document(text=doc.text, metadata=doc.metadata) for doc in request.documents]
        retriever.add_documents(documents)
        return {"status": "success", "message": f"Added {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))\
            
            
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

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