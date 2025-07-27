import requests
from typing import List, Dict, Any
from openai import OpenAI
from loguru import logger
from config.settings import settings


class LLMService:
    """
    Service for generating answers using OpenAI or Ollama LLMs.
    """
    
    def __init__(self):
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        self.system_prompt = '''You are a helpful assistant for Bangla textbook MCQs. You will be given:

- A user-submitted multiple-choice question (MCQ) in Bangla from a textbook.
- A set of MCQs from the same page that includes:
  - Question texts
  - Four options labeled: (ক), (খ), (গ), (ঘ)
  - Answer keys, either:
    - Inline format: "উত্তর: গ"
    - Tabular format: SL | Ans, e.g., "6 | গ"

Your task is to return the full correct answer text (e.g., দুইটি) based only on the given MCQs and answers.

🧠 Steps to follow:

1. Find the question in the list that best matches the user's input.
2. Retrieve its corresponding answer option (e.g., খ or গ).
   - If the answer is from a table (e.g., 6 | গ), match SL 6 to Q6.
   - If the answer is inline (উত্তর: গ), use that.
3. Use the question's options to find the full answer text that corresponds to that option.

🛑 Rules & Constraints:

- Use only the given passage content—do not infer or use external knowledge.
- Return only the full answer text, not the option letter.
- Do not explain, summarize, or add commentary.

🎯 Output Format:

User Question: [Insert the user's question here]
Expected Answer: [Insert the full answer text]

✅ Example:

User Question: 'অপরিচিত' গল্পে রেলকার্টারি কর্তৃক টিকিট বেঁধে বুলিয়েছিল?
Expected Answer: দুইটি'''
    
    def generate_openai_response(self, context: str, query: str, model: str = None) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            context: Retrieved context chunks
            query: User query
            model: Model name (defaults to settings)
            
        Returns:
            Generated response
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        model = model or settings.OPENAI_MODEL
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"প্রসঙ্গ:\n{context}\n\nপ্রশ্ন: {query}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            raise
    
    def generate_ollama_response(self, context: str, query: str, model: str = None) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            context: Retrieved context chunks
            query: User query
            model: Model name (defaults to settings)
            
        Returns:
            Generated response
        """
        model = model or settings.OLLAMA_MODEL
        
        prompt = f"{self.system_prompt}\n\nপ্রসঙ্গ:\n{context}\n\nপ্রশ্ন: {query}"
        
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}")
            raise
    
    def generate_answer(self, context_chunks: List[str], query: str, 
                       use_ollama: bool = False, model: str = None) -> str:
        """
        Generate answer based on retrieved context.
        
        Args:
            context_chunks: List of retrieved context chunks
            query: User query
            use_ollama: Whether to use Ollama instead of OpenAI
            model: Model name to use
            
        Returns:
            Generated answer
        """
        if not context_chunks:
            return "❌ প্রাসঙ্গিক কোনো তথ্য পাওয়া যায়নি।"
        
        # Combine context chunks
        context = "\n".join(context_chunks)
        
        try:
            if use_ollama:
                model_used = model or settings.OLLAMA_MODEL
                logger.info(f"Generating answer using Ollama model: {model_used}")
                answer = self.generate_ollama_response(context, query, model_used)
            else:
                model_used = model or settings.OPENAI_MODEL
                logger.info(f"Generating answer using OpenAI model: {model_used}")
                answer = self.generate_openai_response(context, query, model_used)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"❌ উত্তর তৈরি করতে সমস্যা হয়েছে: {str(e)}"
    
    def extract_direct_mcq_answer(self, query: str, mcq_answers: Dict[int, str]) -> str:
        """
        Extract direct MCQ answer if question number is found in query.
        
        Args:
            query: User query
            mcq_answers: Dictionary of MCQ answers
            
        Returns:
            Direct answer or empty string if not found
        """
        import re
        
        # Try to extract question number from query
        match = re.search(r'\d+', query)
        if match:
            q_num = int(match.group())
            if q_num in mcq_answers:
                ans_letter = mcq_answers[q_num]
                return f"প্রশ্ন {q_num} এর উত্তর: {ans_letter}"
        
        return ""
    
    def check_service_health(self) -> Dict[str, str]:
        """
        Check the health of LLM services.
        
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
                self.openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                health["openai"] = "available"
            except:
                health["openai"] = "error"
        
        # Check Ollama
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=10
            )
            if response.status_code == 200:
                health["ollama"] = "available"
        except:
            health["ollama"] = "error"
        
        return health