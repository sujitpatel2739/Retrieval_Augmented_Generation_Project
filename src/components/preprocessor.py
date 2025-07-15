from abc import ABC, abstractmethod
from ..logger.base import StepLog
from .base_component import BaseComponent
from typing import List, Dict, Tuple, Any
import re
import nltk
import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from ..config import Settings

from transformers import AutoTokenizer

# Ensure required NLTK models are downloaded
nltk.download('punkt')

class BasePreprocessor(ABC):
    """Abstract base class for all document preprocessors."""

    @abstractmethod
    def preprocess(self, document: Any, **kwargs) -> List[Dict]:
        """
        Process the input document and return a list of processed text chunks with metadata.
        """
        pass

    def _execute(self, document: Any, **kwargs) -> List[Dict]:
        """Internal execution to comply with BaseComponent."""
        return self.preprocess(document, **kwargs)


class UniversalExtractor():
    """Extracts clean text and metadata from various document types."""

    def __init__(self):
        super().__init__()

    def _execute(self, file_path: str, file_type: str = None) -> Tuple[str, Dict[str, Any]]:
        text = ""
        metadata = {}

        # Detect file type if not provided
        if not file_type:
            file_type = os.path.splitext(file_path)[1].lower().strip('.')

        if file_type == "pdf":
            text, metadata = self._extract_pdf(file_path)
        elif file_type in ["docx", "doc"]:
            text, metadata = self._extract_word(file_path)
        elif file_type in ["html", "htm"]:
            text, metadata = self._extract_html(file_path)
        elif file_type in ["txt", "md"]:
            text, metadata = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        cleaned_text = self._clean_text(text)
        return cleaned_text, metadata

    def _extract_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or '' for page in reader.pages])
            metadata = dict(reader.metadata or {})
            return text, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF: {e}")

    def _extract_word(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            metadata = {"num_paragraphs": len(doc.paragraphs)}
            return text, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to extract Word document: {e}")

    def _extract_html(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            metadata = {"title": soup.title.string if soup.title else ""}
            return text, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to extract HTML: {e}")

    def _extract_txt(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata = {"num_characters": len(text)}
            return text, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to extract TXT: {e}")

    def _clean_text(self, text: str) -> str:
        # Basic cleaning: remove multiple spaces, blank lines, control chars
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()



class NoiseRemoverCleaner():
    """
    Smart Noise Remover and Pre-Cleaner
    Inherits from BasePreprocessor (which inherits from BaseComponent).

    Purpose:
    - Remove headers, footers, page numbers, boilerplate text, and repeated noise
    - Preserve meaningful content, structure, and metadata
    """

    def __init__(self):
        super().__init__()

    def _execute(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        cleaned_text = self._clean_text(text)
        updated_metadata = metadata or {}

        return {
            "text": cleaned_text,
            "metadata": updated_metadata
        }

    def _clean_text(self, text: str) -> str:
        """
        Apply a series of regex and heuristic-based cleaning rules.
        """
        # Normalize whitespace
        text = re.sub(r'[\t\r\f]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove common headers/footers (detect repeated patterns)
        text = self._remove_repeated_lines(text)

        # Remove standalone page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove unwanted unicode junk (common in badly extracted PDFs)
        text = re.sub(r'[\u200B\uFEFF\u2028\u2029]', '', text)

        # Remove lines with mostly non-alphanumeric characters (garbage lines)
        text = re.sub(r'^\s*[^\w\s]{3,}\s*$', '', text, flags=re.MULTILINE)

        # Optionally collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    def _remove_repeated_lines(self, text: str) -> str:
        """
        Identify and remove repeated headers/footers based on frequency.
        """
        lines = text.splitlines()
        line_freq = {}

        for line in lines:
            line = line.strip()
            if line:
                line_freq[line] = line_freq.get(line, 0) + 1

        # Threshold: lines that appear on >10% of pages (approx)
        threshold = max(2, int(len(lines) * 0.1))

        noise_lines = {line for line, freq in line_freq.items() if freq > threshold and len(line) < 100}

        cleaned_lines = [line for line in lines if line.strip() not in noise_lines]

        return '\n'.join(cleaned_lines)


class SmartAdaptiveChunker():
    """
    Performs intelligent adaptive chunking on cleaned text for downstream processing.
    Includes delimiter-based splitting, token-aware chunking, semantic refinement, and metadata assignment.
    """

    def __init__(self, max_chunk_tokens: int = 300):
        # super().__init__(name="smart_adaptive_chunker")
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # or any lightweight model

    def _execute(self, text: str) -> List[Dict[str, Any]]:
        """Pipeline entry point"""
        initial_chunks = self.delimiter_based_splitting(text)
        return initial_chunks  # For now, return this as we build step by step.

    def delimiter_based_splitting(self, text: str) -> List[str]:
        """
        Step 3: Splits text based on logical delimiters (headings, paragraphs, etc.)
        """

        # Common delimiters: Headings, numbered sections, double newlines, bullet points
        delimiters = [
            r'\n\s*\d+\.\s+',               # Numbered lists or sections: "1. ", "2. "
            r'\n\s*[-*]\s+',                # Bulleted lists: "- ", "* "
            r'\n{2,}',                      # Double newlines (paragraph break)
            r'\n\s*[A-Z][^\n]{0,50}\n'      # Short uppercase-ish headings
        ]

        # Combine delimiters into one regex pattern
        pattern = '|'.join(delimiters)

        # Split the text
        splits = re.split(pattern, text)

        # Remove very small or empty splits
        clean_splits = [s.strip() for s in splits if s and len(s.strip()) > 20]

        return clean_splits


    def token_aware_chunking(self, splits: List[str]) -> List[str]:
        """
        Step 4: Further split any overlength chunks based on token count, respecting sentence boundaries where possible.
        """

        token_chunks = []

        for split in splits:
            tokens = self.tokenizer.tokenize(split)
            total_tokens = len(tokens)

            if total_tokens <= self.max_chunk_tokens:
                token_chunks.append(split)
            else:
                # Split into smaller chunks
                sentences = re.split(r'(?<=[.!?])\s+', split)  # sentence-level split
                current_chunk = ""
                current_tokens = 0

                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.tokenize(sentence))

                    if current_tokens + sentence_tokens <= self.max_chunk_tokens:
                        current_chunk += " " + sentence
                        current_tokens += sentence_tokens
                    else:
                        token_chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sentence_tokens

                if current_chunk.strip():
                    token_chunks.append(current_chunk.strip())

        return token_chunks
    
    