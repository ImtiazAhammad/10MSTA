#!/usr/bin/env python3
"""
CLI script for testing the RAG system.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.rag_system import RAGSystem
from loguru import logger

def test_pdf_processing(pdf_path: str, use_ollama: bool = False):
    """Test PDF processing and querying."""
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    logger.info("Initializing RAG system...")
    rag_system = RAGSystem()
    
    try:
        # Initialize from PDF
        logger.info(f"Processing PDF: {pdf_path}")
        doc_info = rag_system.initialize_from_pdf(pdf_path, use_ollama=use_ollama)
        logger.success(f"Document processed: {doc_info.total_chunks} chunks created")
        
        # Test queries
        test_queries = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            response = rag_system.query(query, use_ollama=use_ollama)
            
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"Answer: {response.answer}")
            print(f"Processing time: {response.processing_time:.2f}s")
            print(f"Model used: {response.model_used}")
            print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
            
            if response.retrieved_chunks:
                print("\nTop retrieved chunk:")
                chunk = response.retrieved_chunks[0]
                print(f"Score: {chunk.score:.3f}")
                print(f"Text: {chunk.text[:200]}...")
            print(f"{'='*60}")
        
        # Get system stats
        stats = rag_system.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"- Initialized: {stats['is_initialized']}")
        print(f"- Total chunks: {stats.get('total_chunks', 0)}")
        print(f"- MCQ answers: {stats.get('total_mcq_answers', 0)}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test the Multilingual RAG System")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama instead of OpenAI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    logger.info("Starting RAG system test...")
    
    try:
        test_pdf_processing(args.pdf_path, use_ollama=args.ollama)
        logger.success("Test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()