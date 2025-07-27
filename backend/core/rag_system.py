import os
import time
from typing import List, Dict, Tuple, Any, Optional
from loguru import logger

from utils.preprocessing.pdf_processor import PDFProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.llm_service import LLMService
from backend.models.schemas import QueryResponse, ChunkInfo, DocumentInfo
from config.settings import settings


class RAGSystem:
    """
    Main RAG system that orchestrates PDF processing, embedding, vector storage, and answer generation.
    """
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        
        self.mcq_answers = {}
        self.full_text = ""
        self.document_info = None
        self.is_initialized = False
        
        # Create necessary directories
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(settings.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def initialize_from_pdf(self, pdf_path: str, use_ollama: bool = False, 
                           force_reprocess: bool = False) -> DocumentInfo:
        """
        Initialize the RAG system from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            use_ollama: Whether to use Ollama for embeddings
            force_reprocess: Whether to force reprocessing even if cache exists
            
        Returns:
            Document information
        """
        try:
            logger.info(f"Initializing RAG system from PDF: {pdf_path}")
            
            # Check if processed data exists
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            vector_store_path = os.path.join(settings.EMBEDDINGS_DIR, f"{pdf_name}_store.pkl")
            
            if not force_reprocess and os.path.exists(vector_store_path):
                logger.info("Loading existing processed data...")
                self.vector_store.load_store(vector_store_path)
                
                # Load metadata
                metadata_path = os.path.join(settings.PROCESSED_DATA_DIR, f"{pdf_name}_metadata.txt")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.full_text = f.read()
                
                self.mcq_answers = self.pdf_processor.extract_mcq_answers(self.full_text)
                self.is_initialized = True
                
                stats = self.vector_store.get_stats()
                self.document_info = DocumentInfo(
                    filename=pdf_name,
                    total_pages=0,  # Not available from cache
                    total_chunks=stats['total_chunks'],
                    processing_status="loaded_from_cache"
                )
                
                logger.success("RAG system initialized from cache")
                return self.document_info
            
            # Process PDF from scratch
            logger.info("Processing PDF from scratch...")
            
            # Step 1: Extract text using OCR
            raw_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Step 2: Clean and preprocess text
            cleaned_text = self.pdf_processor.clean_text(raw_text)
            self.full_text = cleaned_text
            
            # Save processed text
            text_path = os.path.join(settings.PROCESSED_DATA_DIR, f"{pdf_name}_text.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Step 3: Extract MCQ answers
            self.mcq_answers = self.pdf_processor.extract_mcq_answers(cleaned_text)
            
            # Step 4: Split into chunks
            chunks = self.pdf_processor.split_into_chunks(cleaned_text)
            
            # Step 5: Generate embeddings
            embeddings, model_used = self.embedding_service.embed_chunks(chunks, use_ollama)
            
            # Step 6: Initialize vector store
            metadata = {
                'pdf_path': pdf_path,
                'model_used': model_used,
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP,
                'total_mcq_answers': len(self.mcq_answers),
                'processing_time': time.time()
            }
            
            self.vector_store.initialize_store(chunks, embeddings, metadata)
            
            # Step 7: Save vector store
            self.vector_store.save_store(vector_store_path)
            
            # Save metadata
            metadata_path = os.path.join(settings.PROCESSED_DATA_DIR, f"{pdf_name}_metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(self.full_text)
            
            self.is_initialized = True
            
            # Create document info
            self.document_info = DocumentInfo(
                filename=pdf_name,
                total_pages=len(self.pdf_processor.extract_text_from_pdf(pdf_path).split('\n\n')),
                total_chunks=len(chunks),
                processing_status="processed_successfully"
            )
            
            logger.success(f"RAG system initialized successfully. {len(chunks)} chunks, {len(self.mcq_answers)} MCQ answers")
            return self.document_info
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
    
    def query(self, query: str, use_ollama: bool = False, top_k: int = None, 
              similarity_threshold: float = None) -> QueryResponse:
        """
        Query the RAG system.
        
        Args:
            query: User query
            use_ollama: Whether to use Ollama for generation
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Query response with answer and retrieved chunks
        """
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Please call initialize_from_pdf first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Check for direct MCQ answer first
            direct_answer = self.llm_service.extract_direct_mcq_answer(query, self.mcq_answers)
            if direct_answer:
                processing_time = time.time() - start_time
                return QueryResponse(
                    answer=direct_answer,
                    retrieved_chunks=[],
                    query=query,
                    processing_time=processing_time,
                    model_used="direct_mcq_lookup"
                )
            
            # Check for pattern-based MCQ match
            sl, answer_letter = self.pdf_processor.find_mcq_match(query, self.full_text)
            if sl and answer_letter:
                pattern_answer = f"প্রশ্ন {sl} এর উত্তর: {answer_letter}"
                processing_time = time.time() - start_time
                return QueryResponse(
                    answer=pattern_answer,
                    retrieved_chunks=[],
                    query=query,
                    processing_time=processing_time,
                    model_used="pattern_mcq_lookup"
                )
            
            # RAG-based retrieval
            top_k = top_k or settings.TOP_K_RETRIEVAL
            similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query, use_ollama)
            
            # Retrieve similar chunks
            retrieved_chunks, scores, indices = self.vector_store.search(
                query_embedding, 
                k=top_k, 
                metric=settings.FAISS_INDEX_TYPE,
                threshold=similarity_threshold
            )
            
            # Generate answer
            model_used = settings.OLLAMA_MODEL if use_ollama else settings.OPENAI_MODEL
            answer = self.llm_service.generate_answer(retrieved_chunks, query, use_ollama)
            
            # Create chunk info objects
            chunk_infos = [
                ChunkInfo(text=chunk, score=score, index=idx)
                for chunk, score, idx in zip(retrieved_chunks, scores, indices)
            ]
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                answer=answer,
                retrieved_chunks=chunk_infos,
                query=query,
                processing_time=processing_time,
                model_used=model_used
            )
            
            logger.success(f"Query processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            processing_time = time.time() - start_time
            return QueryResponse(
                answer=f"❌ প্রশ্ন প্রক্রিয়া করতে সমস্যা হয়েছে: {str(e)}",
                retrieved_chunks=[],
                query=query,
                processing_time=processing_time,
                model_used="error"
            )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "is_initialized": self.is_initialized,
            "document_info": self.document_info.dict() if self.document_info else None,
            "total_mcq_answers": len(self.mcq_answers),
            "embedding_service_health": self.embedding_service.check_service_health(),
            "llm_service_health": self.llm_service.check_service_health()
        }
        
        if self.is_initialized:
            stats.update(self.vector_store.get_stats())
        
        return stats
    
    def add_document(self, pdf_path: str, use_ollama: bool = False) -> DocumentInfo:
        """
        Add a new document to the existing knowledge base.
        
        Args:
            pdf_path: Path to the new PDF file
            use_ollama: Whether to use Ollama for embeddings
            
        Returns:
            Document information
        """
        try:
            logger.info(f"Adding new document: {pdf_path}")
            
            # Process new PDF
            raw_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            cleaned_text = self.pdf_processor.clean_text(raw_text)
            chunks = self.pdf_processor.split_into_chunks(cleaned_text)
            
            # Generate embeddings
            embeddings, model_used = self.embedding_service.embed_chunks(chunks, use_ollama)
            
            # Add to existing vector store
            if self.is_initialized:
                self.vector_store.add_chunks(chunks, embeddings)
            else:
                # Initialize if this is the first document
                self.vector_store.initialize_store(chunks, embeddings)
                self.is_initialized = True
            
            # Update MCQ answers and full text
            new_mcq_answers = self.pdf_processor.extract_mcq_answers(cleaned_text)
            self.mcq_answers.update(new_mcq_answers)
            self.full_text += "\n\n" + cleaned_text
            
            # Save updated vector store
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            vector_store_path = os.path.join(settings.EMBEDDINGS_DIR, f"combined_store.pkl")
            self.vector_store.save_store(vector_store_path)
            
            doc_info = DocumentInfo(
                filename=pdf_name,
                total_pages=len(cleaned_text.split('\n\n')),
                total_chunks=len(chunks),
                processing_status="added_successfully"
            )
            
            logger.success(f"Document added successfully: {len(chunks)} new chunks")
            return doc_info
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
    
    def clear_knowledge_base(self):
        """
        Clear the entire knowledge base.
        """
        self.vector_store.clear_store()
        self.mcq_answers = {}
        self.full_text = ""
        self.document_info = None
        self.is_initialized = False
        logger.info("Knowledge base cleared")