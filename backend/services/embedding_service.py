import time
import numpy as np
from typing import List, Tuple
import requests
from openai import OpenAI
from loguru import logger
from config.settings import settings


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI or Ollama.
    """
    
    def __init__(self):
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def get_openai_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Get embedding from OpenAI API.
        
        Args:
            text: Text to embed
            model: Model name (defaults to settings)
            
        Returns:
            Embedding vector
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        model = model or settings.OPENAI_EMBEDDING_MODEL
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def get_ollama_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Get embedding from Ollama API.
        
        Args:
            text: Text to embed
            model: Model name (defaults to settings)
            
        Returns:
            Embedding vector
        """
        model = model or settings.OLLAMA_EMBEDDING_MODEL
        
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Ollama embedding error: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[str], use_ollama: bool = False, model: str = None) -> Tuple[List[List[float]], str]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks
            use_ollama: Whether to use Ollama instead of OpenAI
            model: Model name to use
            
        Returns:
            Tuple of (embeddings list, model_name)
        """
        embeddings = []
        model_used = model
        
        if use_ollama:
            model_used = model or settings.OLLAMA_EMBEDDING_MODEL
            logger.info(f"Generating embeddings using Ollama model: {model_used}")
        else:
            model_used = model or settings.OPENAI_EMBEDDING_MODEL
            logger.info(f"Generating embeddings using OpenAI model: {model_used}")
        
        for i, chunk in enumerate(chunks):
            try:
                if use_ollama:
                    embedding = self.get_ollama_embedding(chunk, model_used)
                else:
                    embedding = self.get_openai_embedding(chunk, model_used)
                
                embeddings.append(embedding)
                
                # Rate limiting
                if not use_ollama:
                    time.sleep(0.1)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Error embedding chunk {i}: {str(e)}")
                # Add zero vector as fallback
                dim = 1536 if not use_ollama else 768  # Default dimensions
                embeddings.append([0.0] * dim)
        
        logger.success(f"Generated {len(embeddings)} embeddings")
        return embeddings, model_used
    
    def embed_query(self, query: str, use_ollama: bool = False, model: str = None) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            use_ollama: Whether to use Ollama instead of OpenAI
            model: Model name to use
            
        Returns:
            Query embedding as numpy array
        """
        try:
            if use_ollama:
                model_used = model or settings.OLLAMA_EMBEDDING_MODEL
                embedding = self.get_ollama_embedding(query, model_used)
            else:
                model_used = model or settings.OPENAI_EMBEDDING_MODEL
                embedding = self.get_openai_embedding(query, model_used)
            
            return np.array(embedding, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def check_service_health(self) -> dict:
        """
        Check the health of embedding services.
        
        Returns:
            Dictionary with service status
        """
        health = {
            "openai": "unavailable",
            "ollama": "unavailable"
        }
        
        # Check OpenAI
        if self.openai_client:
            try:
                self.get_openai_embedding("test", settings.OPENAI_EMBEDDING_MODEL)
                health["openai"] = "available"
            except:
                health["openai"] = "error"
        
        # Check Ollama
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                health["ollama"] = "available"
        except:
            health["ollama"] = "error"
        
        return health