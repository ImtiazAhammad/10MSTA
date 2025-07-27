import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .query-result {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chunk-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Make API request to the backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"error": str(e)}

def display_system_stats():
    """Display system statistics in the sidebar."""
    with st.sidebar:
        st.header("📊 System Status")
        
        # Health check
        health = make_api_request("/health")
        if "error" not in health:
            if health["status"] == "healthy":
                st.success("✅ System Healthy")
            else:
                st.error("❌ System Unhealthy")
            
            # Service status
            for service, status in health["services"].items():
                if status == "available" or status == "initialized":
                    st.success(f"✅ {service.replace('_', ' ').title()}")
                else:
                    st.warning(f"⚠️ {service.replace('_', ' ').title()}: {status}")
        
        # System stats
        stats = make_api_request("/stats")
        if "error" not in stats and stats.get("is_initialized"):
            st.subheader("📈 Statistics")
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("MCQ Answers", stats.get("total_mcq_answers", 0))
            st.metric("Embedding Dimension", stats.get("embedding_dimension", 0))
            
            # Available indices
            if "available_indices" in stats:
                st.write("**Available Indices:**")
                for idx in stats["available_indices"]:
                    st.write(f"• {idx.upper()}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Multilingual RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Bengali & English Question Answering System")
    
    # Display system stats in sidebar
    display_system_stats()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Query System", 
        "📄 Document Management", 
        "📊 System Analytics", 
        "🧪 Evaluation", 
        "⚙️ Settings"
    ])
    
    with tab1:
        query_interface()
    
    with tab2:
        document_management()
    
    with tab3:
        system_analytics()
    
    with tab4:
        evaluation_interface()
    
    with tab5:
        settings_interface()

def query_interface():
    """Query interface tab."""
    st.header("🔍 Ask Questions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Enter your question (Bengali or English):",
            height=100,
            placeholder="Example: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        )
    
    with col2:
        # Query settings
        st.subheader("Query Settings")
        use_ollama = st.checkbox("Use Ollama (Local)", value=False)
        top_k = st.slider("Top K Results", 1, 10, 5)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.1)
    
    # Sample questions
    st.subheader("📝 Sample Questions")
    sample_questions = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "What is the main theme of the story?",
        "Who is the protagonist in the story?"
    ]
    
    selected_question = st.selectbox("Select a sample question:", [""] + sample_questions)
    if selected_question:
        query = selected_question
        st.rerun()
    
    # Query button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
            return
        
        with st.spinner("Processing your question..."):
            # Make query request
            request_data = {
                "query": query,
                "use_ollama": use_ollama,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
            
            response = make_api_request("/query", "POST", request_data)
            
            if "error" not in response:
                display_query_result(response)
            else:
                st.error(f"Query failed: {response['error']}")

def display_query_result(response: Dict):
    """Display query results."""
    st.subheader("📋 Query Results")
    
    # Main answer
    st.markdown(f'<div class="query-result"><h4>Answer:</h4><p>{response["answer"]}</p></div>', 
                unsafe_allow_html=True)
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{response['processing_time']:.2f}s")
    with col2:
        st.metric("Model Used", response['model_used'])
    with col3:
        st.metric("Retrieved Chunks", len(response['retrieved_chunks']))
    
    # Retrieved chunks
    if response['retrieved_chunks']:
        st.subheader("📄 Retrieved Context")
        
        for i, chunk in enumerate(response['retrieved_chunks']):
            with st.expander(f"Chunk {i+1} (Score: {chunk['score']:.3f})"):
                st.markdown(f'<div class="chunk-card">{chunk["text"]}</div>', 
                           unsafe_allow_html=True)
                st.caption(f"Index: {chunk['index']}")

def document_management():
    """Document management tab."""
    st.header("📄 Document Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = make_api_request("/upload-pdf", "POST", files=files)
                    
                    if "error" not in response:
                        st.success(f"✅ Document processed successfully!")
                        st.json(response)
                    else:
                        st.error(f"Processing failed: {response['error']}")
    
    with col2:
        st.subheader("🗂️ Knowledge Base Management")
        
        if st.button("📊 Get System Stats", use_container_width=True):
            stats = make_api_request("/stats")
            if "error" not in stats:
                st.json(stats)
            else:
                st.error(f"Failed to get stats: {stats['error']}")
        
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_clear", False):
                response = make_api_request("/clear-knowledge-base", "DELETE")
                if "error" not in response:
                    st.success("✅ Knowledge base cleared!")
                    st.session_state["confirm_clear"] = False
                else:
                    st.error(f"Clear failed: {response['error']}")
            else:
                st.warning("Click again to confirm clearing the knowledge base.")
                st.session_state["confirm_clear"] = True

def system_analytics():
    """System analytics tab."""
    st.header("📊 System Analytics")
    
    # Get system stats
    stats = make_api_request("/stats")
    
    if "error" not in stats and stats.get("is_initialized"):
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        with col2:
            st.metric("MCQ Answers", stats.get("total_mcq_answers", 0))
        with col3:
            st.metric("Embedding Dimension", stats.get("embedding_dimension", 0))
        with col4:
            st.metric("Available Indices", len(stats.get("available_indices", [])))
        
        # Service health visualization
        st.subheader("🏥 Service Health")
        
        embedding_health = stats.get("embedding_service_health", {})
        llm_health = stats.get("llm_service_health", {})
        
        health_data = []
        for service, status in embedding_health.items():
            health_data.append({"Service": f"Embedding ({service})", "Status": status})
        for service, status in llm_health.items():
            health_data.append({"Service": f"LLM ({service})", "Status": status})
        
        if health_data:
            df = pd.DataFrame(health_data)
            fig = px.bar(df, x="Service", y="Status", color="Status",
                        title="Service Health Status")
            st.plotly_chart(fig, use_container_width=True)
        
        # Document metadata
        if "metadata" in stats:
            st.subheader("📄 Document Metadata")
            st.json(stats["metadata"])
    
    else:
        st.info("📋 No data available. Please upload a document first.")

def evaluation_interface():
    """Evaluation interface tab."""
    st.header("🧪 System Evaluation")
    
    st.markdown("""
    Evaluate the RAG system's performance using different metrics:
    - **Groundedness**: How well the answer is supported by retrieved context
    - **Relevance**: How relevant the retrieved chunks are to the query
    - **Coherence**: How coherent and well-formed the generated answer is
    """)
    
    # Test queries
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
    ]
    
    selected_query = st.selectbox("Select test query:", test_queries)
    
    col1, col2 = st.columns(2)
    with col1:
        use_ollama_eval = st.checkbox("Use Ollama for Evaluation", value=False)
    with col2:
        top_k_eval = st.slider("Top K for Evaluation", 1, 10, 5)
    
    if st.button("🧪 Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            request_data = {
                "query": selected_query,
                "use_ollama": use_ollama_eval,
                "top_k": top_k_eval,
                "similarity_threshold": 0.7
            }
            
            response = make_api_request("/evaluate", "POST", request_data)
            
            if "error" not in response:
                display_evaluation_results(response)
            else:
                st.error(f"Evaluation failed: {response['error']}")

def display_evaluation_results(metrics: Dict):
    """Display evaluation results."""
    st.subheader("📈 Evaluation Results")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Groundedness", f"{metrics['groundedness_score']:.2f}")
    with col2:
        st.metric("Relevance", f"{metrics['relevance_score']:.2f}")
    with col3:
        st.metric("Coherence", f"{metrics['coherence_score']:.2f}")
    with col4:
        st.metric("Overall Score", f"{metrics['overall_score']:.2f}")
    
    # Radar chart
    categories = ['Groundedness', 'Relevance', 'Coherence']
    values = [
        metrics['groundedness_score'],
        metrics['relevance_score'],
        metrics['coherence_score']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Evaluation Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="RAG System Performance Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def settings_interface():
    """Settings interface tab."""
    st.header("⚙️ System Settings")
    
    # Available models
    models = make_api_request("/models/available")
    
    if "error" not in models:
        st.subheader("🤖 Available Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Embedding Models:**")
            for provider, model in models["embedding_models"].items():
                st.write(f"• {provider.upper()}: `{model}`")
        
        with col2:
            st.write("**Generation Models:**")
            for provider, model in models["generation_models"].items():
                st.write(f"• {provider.upper()}: `{model}`")
    
    # Configuration info
    st.subheader("🔧 Configuration")
    st.info("""
    **To configure the system:**
    1. Copy `.env.example` to `.env`
    2. Set your API keys and preferences
    3. Restart the application
    
    **For Ollama setup:**
    1. Install Ollama from https://ollama.ai
    2. Pull the required models: `ollama pull deepseek-r1:8b`
    3. Pull embedding model: `ollama pull nomic-embed-text`
    """)
    
    # System requirements
    st.subheader("📋 System Requirements")
    requirements = {
        "Python": "3.8+",
        "RAM": "4GB+ (8GB+ recommended)",
        "Storage": "2GB+ free space",
        "Tesseract OCR": "Required for PDF processing",
        "Poppler": "Required for PDF to image conversion"
    }
    
    for req, desc in requirements.items():
        st.write(f"• **{req}**: {desc}")

if __name__ == "__main__":
    main()