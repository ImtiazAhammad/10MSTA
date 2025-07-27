import pytest
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.rag_system import RAGSystem
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.llm_service import LLMService
from utils.preprocessing.pdf_processor import PDFProcessor


class TestBasicComponents:
    """Test basic component initialization."""
    
    def test_rag_system_initialization(self):
        """Test RAG system can be initialized."""
        rag_system = RAGSystem()
        assert rag_system is not None
        assert not rag_system.is_initialized
    
    def test_embedding_service_initialization(self):
        """Test embedding service can be initialized."""
        embedding_service = EmbeddingService()
        assert embedding_service is not None
    
    def test_vector_store_initialization(self):
        """Test vector store can be initialized."""
        vector_store = VectorStore()
        assert vector_store is not None
        assert len(vector_store.chunks) == 0
        assert len(vector_store.embeddings) == 0
    
    def test_llm_service_initialization(self):
        """Test LLM service can be initialized."""
        llm_service = LLMService()
        assert llm_service is not None
        assert llm_service.system_prompt is not None
    
    def test_pdf_processor_initialization(self):
        """Test PDF processor can be initialized."""
        pdf_processor = PDFProcessor()
        assert pdf_processor is not None
        assert pdf_processor.config is not None
        assert pdf_processor.languages is not None


class TestTextProcessing:
    """Test text processing functionality."""
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        pdf_processor = PDFProcessor()
        
        # Test text with excessive whitespace
        dirty_text = "This  is    a   test\n\n\n\nwith   extra   spaces।।।"
        cleaned = pdf_processor.clean_text(dirty_text)
        
        assert "  " not in cleaned  # No double spaces
        assert "।।" not in cleaned  # No double Bengali punctuation
    
    def test_chunking(self):
        """Test text chunking functionality."""
        pdf_processor = PDFProcessor()
        
        test_text = "This is sentence one. This is sentence two। This is sentence three?"
        chunks = pdf_processor.split_into_chunks(test_text, chunk_size=50, chunk_overlap=20)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_mcq_extraction(self):
        """Test MCQ answer extraction."""
        pdf_processor = PDFProcessor()
        
        test_text = """
        1. What is the answer? উত্তর: ক
        2. Another question? উত্তর: খ
        3 | গ
        4 | ঘ
        """
        
        answers = pdf_processor.extract_mcq_answers(test_text)
        assert isinstance(answers, dict)
        assert len(answers) >= 2  # Should find at least 2 answers


class TestVectorOperations:
    """Test vector operations."""
    
    def test_vector_store_operations(self):
        """Test basic vector store operations."""
        vector_store = VectorStore()
        
        # Test with dummy data
        chunks = ["This is chunk 1", "This is chunk 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        vector_store.initialize_store(chunks, embeddings)
        
        assert len(vector_store.chunks) == 2
        assert len(vector_store.embeddings) == 2
        assert len(vector_store.indices) > 0
    
    def test_get_stats(self):
        """Test getting vector store statistics."""
        vector_store = VectorStore()
        stats = vector_store.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_chunks" in stats
        assert "total_embeddings" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])