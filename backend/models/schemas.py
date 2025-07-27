from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query in English or Bengali")
    use_ollama: bool = Field(default=False, description="Whether to use Ollama instead of OpenAI")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")


class ChunkInfo(BaseModel):
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score")
    index: int = Field(..., description="Chunk index in the document")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    retrieved_chunks: List[ChunkInfo] = Field(..., description="Retrieved relevant chunks")
    query: str = Field(..., description="Original query")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for generation")
    timestamp: datetime = Field(default_factory=datetime.now)


class MCQAnswer(BaseModel):
    question_number: int = Field(..., description="MCQ question number")
    answer_option: str = Field(..., description="Answer option (ক, খ, গ, ঘ)")
    answer_text: Optional[str] = Field(None, description="Full answer text if available")


class DocumentInfo(BaseModel):
    filename: str = Field(..., description="PDF filename")
    total_pages: int = Field(..., description="Total number of pages")
    total_chunks: int = Field(..., description="Total number of chunks")
    processing_status: str = Field(..., description="Processing status")
    created_at: datetime = Field(default_factory=datetime.now)


class EvaluationMetrics(BaseModel):
    groundedness_score: float = Field(..., description="How well the answer is grounded in context")
    relevance_score: float = Field(..., description="How relevant the retrieved chunks are")
    coherence_score: float = Field(..., description="How coherent the generated answer is")
    overall_score: float = Field(..., description="Overall evaluation score")


class HealthCheck(BaseModel):
    status: str = Field(..., description="API health status")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(..., description="Status of dependent services")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)