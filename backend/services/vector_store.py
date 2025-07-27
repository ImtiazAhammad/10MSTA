import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from loguru import logger
from config.settings import settings


class VectorStore:
    """
    Vector store using FAISS for similarity search.
    Supports multiple similarity metrics (L2, dot product, cosine).
    """
    
    def __init__(self):
        self.indices = {}
        self.chunks = []
        self.embeddings = []
        self.metadata = {}
        
    def build_index(self, embeddings: List[List[float]], metric: str = None) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: List of embedding vectors
            metric: Similarity metric ('l2', 'dot', 'cosine')
            
        Returns:
            FAISS index
        """
        metric = metric or settings.FAISS_INDEX_TYPE
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        
        if metric == 'l2':
            index = faiss.IndexFlatL2(dimension)
        elif metric == 'dot':
            index = faiss.IndexFlatIP(dimension)
        elif metric == 'cosine':
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        index.add(embeddings_array)
        logger.info(f"Built {metric} index with {index.ntotal} vectors")
        return index
    
    def initialize_store(self, chunks: List[str], embeddings: List[List[float]], 
                        metadata: Dict[str, Any] = None):
        """
        Initialize the vector store with chunks and embeddings.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Additional metadata
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.metadata = metadata or {}
        
        # Build indices for different metrics
        metrics = ['l2', 'dot', 'cosine']
        for metric in metrics:
            try:
                self.indices[metric] = self.build_index(embeddings, metric)
                logger.success(f"Built {metric} index successfully")
            except Exception as e:
                logger.error(f"Failed to build {metric} index: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               metric: str = None, threshold: float = None) -> Tuple[List[str], List[float], List[int]]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            metric: Similarity metric to use
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (chunks, scores, indices)
        """
        metric = metric or settings.FAISS_INDEX_TYPE
        threshold = threshold or settings.SIMILARITY_THRESHOLD
        
        if metric not in self.indices:
            raise ValueError(f"Index for metric '{metric}' not available")
        
        index = self.indices[metric]
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        if metric == 'cosine':
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = index.search(query_vector, k)
        scores = scores[0]
        indices = indices[0]
        
        # Filter by threshold
        if metric == 'l2':
            # For L2, lower scores are better, so we invert the threshold logic
            valid_mask = scores <= (2.0 - threshold)  # Convert similarity to distance
        else:
            # For dot product and cosine, higher scores are better
            valid_mask = scores >= threshold
        
        valid_indices = indices[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Get corresponding chunks
        retrieved_chunks = [self.chunks[i] for i in valid_indices if i < len(self.chunks)]
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks using {metric} similarity")
        return retrieved_chunks, valid_scores.tolist(), valid_indices.tolist()
    
    def save_store(self, filepath: str):
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save the store
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            store_data = {
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            
            # Save main data
            with open(filepath, 'wb') as f:
                pickle.dump(store_data, f)
            
            # Save FAISS indices
            for metric, index in self.indices.items():
                index_path = filepath.replace('.pkl', f'_{metric}.index')
                faiss.write_index(index, index_path)
            
            logger.success(f"Vector store saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    def load_store(self, filepath: str):
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load the store from
        """
        try:
            # Load main data
            with open(filepath, 'rb') as f:
                store_data = pickle.load(f)
            
            self.chunks = store_data['chunks']
            self.embeddings = store_data['embeddings']
            self.metadata = store_data.get('metadata', {})
            
            # Load FAISS indices
            self.indices = {}
            metrics = ['l2', 'dot', 'cosine']
            
            for metric in metrics:
                index_path = filepath.replace('.pkl', f'_{metric}.index')
                if os.path.exists(index_path):
                    self.indices[metric] = faiss.read_index(index_path)
                    logger.info(f"Loaded {metric} index")
            
            logger.success(f"Vector store loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        stats = {
            'total_chunks': len(self.chunks),
            'total_embeddings': len(self.embeddings),
            'available_indices': list(self.indices.keys()),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0,
            'metadata': self.metadata
        }
        
        for metric, index in self.indices.items():
            stats[f'{metric}_index_size'] = index.ntotal
        
        return stats
    
    def add_chunks(self, new_chunks: List[str], new_embeddings: List[List[float]]):
        """
        Add new chunks and embeddings to the existing store.
        
        Args:
            new_chunks: List of new text chunks
            new_embeddings: List of new embedding vectors
        """
        if len(new_chunks) != len(new_embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Add to existing data
        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_embeddings)
        
        # Rebuild indices
        metrics = list(self.indices.keys())
        self.indices = {}
        
        for metric in metrics:
            self.indices[metric] = self.build_index(self.embeddings, metric)
        
        logger.info(f"Added {len(new_chunks)} new chunks to vector store")
    
    def clear_store(self):
        """
        Clear all data from the vector store.
        """
        self.chunks = []
        self.embeddings = []
        self.indices = {}
        self.metadata = {}
        logger.info("Vector store cleared")