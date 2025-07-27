import os
import re
import time
from typing import List, Dict, Tuple
import pytesseract
from pdf2image import convert_from_path
from loguru import logger
from config.settings import settings


class PDFProcessor:
    """
    PDF processing utility for extracting text using OCR.
    Supports both Bengali and English text extraction.
    """
    
    def __init__(self):
        self.config = settings.TESSERACT_CONFIG
        self.languages = "+".join(settings.OCR_LANGUAGES)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Starting OCR extraction for: {pdf_path}")
            images = convert_from_path(pdf_path)
            text = ""
            
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                raw_text = pytesseract.image_to_string(
                    img, 
                    lang=self.languages, 
                    config=self.config
                )
                # Clean up OCR artifacts
                cleaned_text = raw_text.replace('\x0c', '').strip()
                text += cleaned_text + "\n"
                
            logger.success(f"OCR extraction completed. Total characters: {len(text)}")
            return text
            
        except Exception as e:
            logger.error(f"Error during OCR extraction: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix common OCR errors for Bengali
        text = re.sub(r'।+', '।', text)  # Multiple sentence endings
        text = re.sub(r'\?+', '?', text)  # Multiple question marks
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def split_into_chunks(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into chunks for vector embedding.
        Uses sentence-aware chunking for better semantic coherence.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Split by Bengali and English sentence endings
        sentences = re.split(r'(।|\?|!|\n|\.)', text)
        
        # Merge sentences with their punctuation
        merged_sentences = []
        i = 0
        while i < len(sentences) - 1:
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1].strip()
            if sentence:
                merged_sentences.append(sentence)
            i += 2
        
        # Create chunks
        chunks = []
        current_chunk = ""
        
        for sentence in merged_sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous chunk
                    current_chunk = current_chunk[-chunk_overlap:] + sentence + " "
                else:
                    # Sentence is too long, split it
                    chunks.append(sentence[:chunk_size])
                    current_chunk = sentence[chunk_size-chunk_overlap:] + " "
            else:
                current_chunk += sentence + " "
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def extract_mcq_answers(self, text: str) -> Dict[int, str]:
        """
        Extract MCQ answers from text using regex patterns.
        
        Args:
            text: Input text containing MCQs
            
        Returns:
            Dictionary mapping question numbers to answer options
        """
        answers = {}
        
        # Pattern for inline answers: "১. ... উত্তর: ক"
        inline_pattern = r'(\d+)[.|।]\s*.*?[উত্তর|উ:]*[:：\-\s]+([কখগঘ])'
        inline_matches = re.findall(inline_pattern, text)
        
        # Pattern for tabular answers: "1 | ক"
        table_pattern = r'(\d+)\s*\|\s*([কখগঘ])'
        table_matches = re.findall(table_pattern, text)
        
        # Combine all matches
        all_matches = inline_matches + table_matches
        
        for num_str, option in all_matches:
            try:
                num = int(num_str)
                answers[num] = option
            except ValueError:
                continue
        
        logger.info(f"Extracted {len(answers)} MCQ answers")
        return answers
    
    def find_mcq_match(self, query: str, text: str) -> Tuple[int, str]:
        """
        Find matching MCQ in text for a given query.
        
        Args:
            query: User query
            text: Full text to search in
            
        Returns:
            Tuple of (question_number, answer_option) or (None, None)
        """
        # Pattern to match MCQ structure
        pattern = re.compile(
            r'(\d+)[.|।]\s*(.*?)\((ক\)|খ\)|গ\)|ঘ\)).*?উত্তর[:：\-\s]*([কখগঘ])', 
            re.DOTALL
        )
        
        matches = pattern.findall(text)
        query_start = query.strip()[:30]  # First 30 characters for matching
        
        for sl, question, _, correct_option in matches:
            if query_start in question.strip():
                return int(sl), correct_option
                
        return None, None