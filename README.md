# 📚 Multilingual RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed for Bengali and English text processing, specifically optimized for educational content and MCQ answering.

## 🌟 Features

- **Multilingual Support**: Handles both Bengali and English queries seamlessly
- **OCR Integration**: Extracts text from PDF documents using Tesseract OCR
- **Multiple LLM Support**: Compatible with both OpenAI and Ollama (local) models
- **Vector Search**: FAISS-based similarity search with multiple metrics (L2, dot product, cosine)
- **MCQ Processing**: Specialized handling of multiple-choice questions with answer extraction
- **Production Ready**: FastAPI backend with Streamlit frontend
- **Evaluation Metrics**: Built-in evaluation system for RAG performance assessment
- **Caching**: Intelligent caching of processed documents and embeddings

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│   RAG System    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                ┌──────▼──────┐                   ┌──────▼──────┐                 ┌──────▼──────┐
                │ PDF Processor│                   │Vector Store │                 │LLM Service  │
                │   (OCR)     │                   │  (FAISS)    │                 │(OpenAI/     │
                └─────────────┘                   └─────────────┘                 │ Ollama)     │
                                                                                  └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **System Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr tesseract-ocr-ben poppler-utils
   
   # macOS
   brew install tesseract tesseract-lang poppler
   
   # Windows
   # Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
   # Download and install Poppler from: https://github.com/oschwartz10612/poppler-windows
   ```

2. **Python 3.8+**

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/multilingual-rag-system.git
   cd multilingual-rag-system
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Optional: Install Ollama for local models**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull deepseek-r1:8b
   ollama pull nomic-embed-text
   ```

### Running the System

1. **Start the API server**:
   ```bash
   python run_api.py
   ```

2. **Start the Streamlit UI** (in another terminal):
   ```bash
   python run_streamlit.py
   ```

3. **Access the application**:
   - API Documentation: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

## 📖 Usage

### 1. Upload a PDF Document

- Navigate to the "Document Management" tab in the Streamlit UI
- Upload your PDF file (e.g., HSC26-Bangla1st-Paper.pdf)
- Wait for processing to complete

### 2. Ask Questions

- Go to the "Query System" tab
- Enter your question in Bengali or English
- Configure query settings (model choice, top-k, similarity threshold)
- Click "Search" to get answers

### 3. Sample Queries

**Bengali Questions:**
- অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
- কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
- বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?

**English Questions:**
- What is the main theme of the story?
- Who is the protagonist in the story?

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=500
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# Vector Database
FAISS_INDEX_TYPE=cosine  # Options: l2, dot, cosine
```

### Model Configuration

The system supports multiple models:

**Embedding Models:**
- OpenAI: `text-embedding-3-small`
- Ollama: `nomic-embed-text`

**Generation Models:**
- OpenAI: `gpt-4o`
- Ollama: `deepseek-r1:8b`

## 📊 API Documentation

### Core Endpoints

- `POST /query` - Submit a query to the RAG system
- `POST /upload-pdf` - Upload and process a PDF document
- `GET /stats` - Get system statistics
- `GET /health` - Health check
- `POST /evaluate` - Evaluate system performance

### Example API Usage

```python
import requests

# Query the system
response = requests.post("http://localhost:8000/query", json={
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "use_ollama": False,
    "top_k": 5,
    "similarity_threshold": 0.7
})

print(response.json())
```

## 🧪 Evaluation

The system includes comprehensive evaluation metrics:

- **Groundedness**: How well the answer is supported by retrieved context
- **Relevance**: How relevant the retrieved chunks are to the query
- **Coherence**: How coherent and well-formed the generated answer is

Access the evaluation interface through the "Evaluation" tab in the Streamlit UI.

## 🛠️ Development

### Project Structure

```
multilingual-rag-system/
├── backend/
│   ├── api/           # FastAPI application
│   ├── core/          # Core RAG system logic
│   ├── models/        # Pydantic schemas
│   └── services/      # Business logic services
├── frontend/
│   └── streamlit/     # Streamlit UI
├── utils/
│   ├── preprocessing/ # PDF processing utilities
│   └── evaluation/    # Evaluation metrics
├── config/            # Configuration management
├── data/              # Data storage
│   ├── raw/          # Raw PDF files
│   ├── processed/    # Processed text files
│   └── embeddings/   # Vector embeddings
└── tests/            # Test files
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black .
flake8 .
```

## 📋 Technical Answers

### 1. Text Extraction Method

**Method Used**: Tesseract OCR with pdf2image
- **Why**: Tesseract provides excellent Bengali text recognition
- **Challenges**: 
  - OCR accuracy varies with PDF quality
  - Bengali character recognition requires specific language models
  - Formatting preservation during text extraction

### 2. Chunking Strategy

**Strategy**: Sentence-aware chunking with overlap
- **Approach**: Split by Bengali (।) and English (.) sentence endings
- **Benefits**: Maintains semantic coherence within chunks
- **Parameters**: 1000 characters per chunk, 500 character overlap

### 3. Embedding Model

**Models Used**:
- **OpenAI**: `text-embedding-3-small` (1536 dimensions)
- **Ollama**: `nomic-embed-text` (768 dimensions)
- **Why**: Multilingual support and semantic understanding

### 4. Similarity Comparison

**Method**: FAISS vector similarity search
- **Metrics**: Cosine similarity (default), L2 distance, dot product
- **Storage**: In-memory FAISS indices with disk persistence
- **Why**: Fast similarity search with multiple metric support

### 5. Query-Document Matching

**Approach**: Multi-level matching strategy
1. Direct MCQ answer lookup (pattern matching)
2. Semantic embedding similarity
3. Fallback to keyword matching
- **Vague queries**: Lower similarity scores, broader context retrieval

### 6. Result Relevance

**Current Performance**: Good for specific factual questions
**Potential Improvements**:
- Better chunking strategies (paragraph-based)
- Domain-specific embedding models
- Larger document corpus
- Query expansion techniques
- Re-ranking mechanisms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT and embedding models
- Ollama for local model support
- Tesseract OCR for text extraction
- FAISS for vector similarity search
- FastAPI and Streamlit for the web framework

