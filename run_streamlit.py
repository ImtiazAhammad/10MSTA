#!/usr/bin/env python3
"""
Main script to run the Streamlit frontend.
"""

import subprocess
import sys
import os
from loguru import logger

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings

def main():
    """Run the Streamlit frontend."""
    logger.info("Starting Multilingual RAG System Frontend")
    logger.info(f"Frontend will run on port {settings.STREAMLIT_PORT}")
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/streamlit/app.py",
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Streamlit server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()