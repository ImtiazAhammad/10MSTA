from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from typing import Dict, Any
from loguru import logger

from backend.core.rag_system import RAGSystem
from backend.models.schemas import (
    QueryRequest, QueryResponse, DocumentInfo, 
    HealthCheck, ErrorResponse, EvaluationMetrics
)
from config.settings import settings

# Configure logging
logger.add(
    settings.LOG_FILE,
    level=settings.LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual RAG System",
    description="A production-grade RAG system for Bengali and English text processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Multilingual RAG System API")
    
    # Check if there's a default PDF to load
    default_pdf = os.path.join(settings.RAW_DATA_DIR, "HSC26-Bangla1st-Paper.pdf")
    if os.path.exists(default_pdf):
        try:
            logger.info("Loading default PDF...")
            rag_system.initialize_from_pdf(default_pdf)
            logger.success("Default PDF loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load default PDF: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Multilingual RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        stats = rag_system.get_system_stats()
        
        services = {
            "rag_system": "initialized" if stats["is_initialized"] else "not_initialized",
            "embedding_service": "available" if any(
                status == "available" for status in stats["embedding_service_health"].values()
            ) else "unavailable",
            "llm_service": "available" if any(
                status == "available" for status in stats["llm_service_health"].values()
            ) else "unavailable"
        }
        
        return HealthCheck(
            status="healthy",
            services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            services={"error": str(e)}
        )

@app.post("/upload-pdf", response_model=DocumentInfo)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        file_path = os.path.join(settings.RAW_DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"PDF uploaded: {file.filename}")
        
        # Process PDF in background or immediately based on size
        doc_info = rag_system.initialize_from_pdf(file_path, force_reprocess=True)
        
        return doc_info
        
    except Exception as e:
        logger.error(f"PDF upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        if not rag_system.is_initialized:
            raise HTTPException(
                status_code=400, 
                detail="RAG system not initialized. Please upload a PDF first."
            )
        
        response = rag_system.query(
            query=request.query,
            use_ollama=request.use_ollama,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """Get system statistics."""
    try:
        return rag_system.get_system_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-document", response_model=DocumentInfo)
async def add_document(file: UploadFile = File(...)):
    """Add a new document to the existing knowledge base."""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        file_path = os.path.join(settings.RAW_DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to knowledge base
        doc_info = rag_system.add_document(file_path)
        
        logger.info(f"Document added: {file.filename}")
        return doc_info
        
    except Exception as e:
        logger.error(f"Add document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear the entire knowledge base."""
    try:
        rag_system.clear_knowledge_base()
        logger.info("Knowledge base cleared")
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        logger.error(f"Clear knowledge base failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/available")
async def get_available_models():
    """Get available models for embeddings and generation."""
    return {
        "embedding_models": {
            "openai": settings.OPENAI_EMBEDDING_MODEL,
            "ollama": settings.OLLAMA_EMBEDDING_MODEL
        },
        "generation_models": {
            "openai": settings.OPENAI_MODEL,
            "ollama": settings.OLLAMA_MODEL
        }
    }

@app.post("/evaluate", response_model=EvaluationMetrics)
async def evaluate_rag_system(request: QueryRequest):
    """Evaluate the RAG system performance (basic implementation)."""
    try:
        if not rag_system.is_initialized:
            raise HTTPException(
                status_code=400, 
                detail="RAG system not initialized. Please upload a PDF first."
            )
        
        # Get response from RAG system
        response = rag_system.query(
            query=request.query,
            use_ollama=request.use_ollama,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Basic evaluation metrics (can be enhanced)
        groundedness_score = 0.8 if len(response.retrieved_chunks) > 0 else 0.2
        relevance_score = min(1.0, len(response.retrieved_chunks) / request.top_k)
        coherence_score = 0.9 if "❌" not in response.answer else 0.3
        overall_score = (groundedness_score + relevance_score + coherence_score) / 3
        
        return EvaluationMetrics(
            groundedness_score=groundedness_score,
            relevance_score=relevance_score,
            coherence_score=coherence_score,
            overall_score=overall_score
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )