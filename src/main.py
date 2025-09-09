from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .db.session import engine, Base
from .api import routes_users, routes_collections, routes_messages

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FastAPI Backend",
    description="FastAPI + PostgreSQL backend with CRUD APIs for Users, Collections, and Messages",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
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

# Add the collections route to users router
app.include_router(routes_collections.router, prefix="/users")

@app.get("/")
async def root():
    return {
        "message": "FastAPI Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)