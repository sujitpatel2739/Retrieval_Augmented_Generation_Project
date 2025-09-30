from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .db.session import engine, Base
from .api import routes_users, routes_collections, routes_messages
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from src.config import Settings
from .logger.json_logger import JsonLogger
from .workflow import Workflow
import os
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv('.env') 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load settings
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize workflow
    Workflow.init(
        OPENAI_API_KEY=settings.OPENAI_API_KEY,
        completion_threshold=0.7
    )
    print("[LIFESPAN] Workflow initialized")
    yield
    # ---- SHUTDOWN ----
    Workflow.shutdown()
    print("[LIFESPAN] Workflow shut down")

app = FastAPI(title="FastAPI Backend",
              lifespan=lifespan,
              version="1.0.0"
              )
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Have to Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

# Include routers
app.include_router(routes_users.router)
app.include_router(routes_collections.router)
app.include_router(routes_messages.router)

# @app.get("/logs/workflows")
# async def get_workflow_logs(
#     workflow_id: Optional[str] = None,
#     start_time: Optional[datetime] = None,
#     end_time: Optional[datetime] = None
# ):
#     """Get workflow logs with optional filtering"""
#     return Workflow.logger.get_workflow_logs(workflow_id, start_time, end_time)

# @app.get("/logs/finetuning")
# async def export_logs_for_finetuning(
#     start_time: Optional[datetime] = None,
#     end_time: Optional[datetime] = None
# ):
#     """Export logs in OpenAI finetuning format"""
#     return Workflow.logger.export_for_finetuning(start_time, end_time)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)