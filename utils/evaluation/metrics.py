import re
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class RAGEvaluator:
    """
    Evaluation metrics for RAG system performance.
    """
    
    def __init__(self):
        self.sentence_model = None
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def evaluate_groundedness(self, answer: str, context_chunks: List[str]) -> float:
        """
        Evaluate how well the answer is grounded in the provided context.
        
        Args:
            answer: Generated answer
            context_chunks: Retrieved context chunks
            
        Returns:
            Groundedness score (0-1)
        """
        if not context_chunks or not answer.strip():
            return 0.0
        
        try:
            # Simple keyword overlap approach
            answer_words = set(re.findall(r'\w+', answer.lower()))
            context_text = " ".join(context_chunks).lower()
            context_words = set(re.findall(r'\w+', context_text))
            
            if not answer_words:
                return 0.0
            
            # Calculate overlap ratio
            overlap = len(answer_words.intersection(context_words))
            groundedness = overlap / len(answer_words)
            
            # Boost score if answer contains direct quotes from context
            for chunk in context_chunks:
                if any(phrase in chunk.lower() for phrase in answer.lower().split('.') if len(phrase.strip()) > 10):
                    groundedness = min(1.0, groundedness * 1.2)
            
            return min(1.0, groundedness)
            
        except Exception as e:
            logger.error(f"Error calculating groundedness: {e}")
            return 0.5
    
    def evaluate_relevance(self, query: str, context_chunks: List[str]) -> float:
        """
        Evaluate how relevant the retrieved chunks are to the query.
        
        Args:
            query: Original query
            context_chunks: Retrieved context chunks
            
        Returns:
            Relevance score (0-1)
        """
        if not context_chunks or not query.strip():
            return 0.0
        
        try:
            if self.sentence_model:
                # Use semantic similarity
                query_embedding = self.sentence_model.encode([query])
                chunk_embeddings = self.sentence_model.encode(context_chunks)
                
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                return float(np.mean(similarities))
            else:
                # Fallback to keyword overlap
                query_words = set(re.findall(r'\w+', query.lower()))
                relevance_scores = []
                
                for chunk in context_chunks:
                    chunk_words = set(re.findall(r'\w+', chunk.lower()))
                    if chunk_words:
                        overlap = len(query_words.intersection(chunk_words))
                        score = overlap / len(query_words.union(chunk_words))
                        relevance_scores.append(score)
                
                return float(np.mean(relevance_scores)) if relevance_scores else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5
    
    def evaluate_coherence(self, answer: str) -> float:
        """
        Evaluate the coherence and quality of the generated answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            Coherence score (0-1)
        """
        if not answer.strip():
            return 0.0
        
        try:
            coherence_score = 0.5  # Base score
            
            # Check for error messages
            if any(error_indicator in answer for error_indicator in ['❌', 'error', 'failed', 'সমস্যা']):
                coherence_score -= 0.3
            
            # Check length (too short or too long might indicate issues)
            answer_length = len(answer.strip())
            if 10 <= answer_length <= 500:
                coherence_score += 0.2
            elif answer_length < 5:
                coherence_score -= 0.3
            
            # Check for proper sentence structure
            sentences = re.split(r'[।.!?]', answer)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            
            if valid_sentences:
                coherence_score += 0.2
            
            # Check for Bengali/English mixed content appropriately
            bengali_chars = len(re.findall(r'[\u0980-\u09FF]', answer))
            english_chars = len(re.findall(r'[a-zA-Z]', answer))
            
            if bengali_chars > 0 or english_chars > 0:
                coherence_score += 0.1
            
            return min(1.0, max(0.0, coherence_score))
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.5
    
    def evaluate_answer_correctness(self, answer: str, expected_answer: str) -> float:
        """
        Evaluate answer correctness against expected answer (if available).
        
        Args:
            answer: Generated answer
            expected_answer: Expected correct answer
            
        Returns:
            Correctness score (0-1)
        """
        if not answer.strip() or not expected_answer.strip():
            return 0.0
        
        try:
            # Normalize answers
            answer_norm = re.sub(r'\s+', ' ', answer.lower().strip())
            expected_norm = re.sub(r'\s+', ' ', expected_answer.lower().strip())
            
            # Exact match
            if answer_norm == expected_norm:
                return 1.0
            
            # Substring match
            if expected_norm in answer_norm or answer_norm in expected_norm:
                return 0.8
            
            # Word overlap
            answer_words = set(re.findall(r'\w+', answer_norm))
            expected_words = set(re.findall(r'\w+', expected_norm))
            
            if answer_words and expected_words:
                overlap = len(answer_words.intersection(expected_words))
                union = len(answer_words.union(expected_words))
                return overlap / union if union > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correctness: {e}")
            return 0.0
    
    def comprehensive_evaluation(
        self, 
        query: str, 
        answer: str, 
        context_chunks: List[str],
        expected_answer: str = None
    ) -> Dict[str, float]:
        """
        Perform comprehensive evaluation of RAG system response.
        
        Args:
            query: Original query
            answer: Generated answer
            context_chunks: Retrieved context chunks
            expected_answer: Expected answer (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'groundedness': self.evaluate_groundedness(answer, context_chunks),
            'relevance': self.evaluate_relevance(query, context_chunks),
            'coherence': self.evaluate_coherence(answer)
        }
        
        if expected_answer:
            metrics['correctness'] = self.evaluate_answer_correctness(answer, expected_answer)
            metrics['overall'] = np.mean(list(metrics.values()))
        else:
            metrics['overall'] = np.mean([metrics['groundedness'], metrics['relevance'], metrics['coherence']])
        
        return metrics
    
    def batch_evaluate(
        self, 
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, List[float]]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries with keys:
                       'query', 'answer', 'context_chunks', 'expected_answer' (optional)
        
        Returns:
            Dictionary with lists of scores for each metric
        """
        all_metrics = {
            'groundedness': [],
            'relevance': [],
            'coherence': [],
            'overall': []
        }
        
        for case in test_cases:
            metrics = self.comprehensive_evaluation(
                query=case['query'],
                answer=case['answer'],
                context_chunks=case.get('context_chunks', []),
                expected_answer=case.get('expected_answer')
            )
            
            for metric, score in metrics.items():
                if metric in all_metrics:
                    all_metrics[metric].append(score)
        
        return all_metrics