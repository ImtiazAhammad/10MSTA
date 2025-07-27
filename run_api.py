#!/usr/bin/env python3
"""
Main script to run the FastAPI backend server.
"""

import uvicorn
import sys
import os
from loguru import logger

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings

def main():
    """Run the FastAPI server."""
    logger.info("Starting Multilingual RAG System API Server")
    logger.info(f"Server will run on {settings.API_HOST}:{settings.API_PORT}")
    
    # Configure uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()