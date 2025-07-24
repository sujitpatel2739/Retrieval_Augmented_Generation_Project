from typing import List, Dict, Union, Any
import re
import torch
import fitz
import io
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from ..models import Document
import docx
import uuid
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

class UniversalExtractor():
    def __init__(self):
        pass

    def execute(self, file_bytes: bytes, filename: str) -> List[str]:
        ext = filename.lower()
        if ext.endswith(".pdf"):
            return self.extract_from_pdf(file_bytes)
        elif ext.endswith(".txt"):
            return self.extract_from_txt(file_bytes)
        elif ext.endswith(".docx"):
            return self.extract_from_docx(file_bytes)
        elif ext.endswith(".html") or ext.endswith(".htm"):
            return self.extract_from_html(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def extract_from_pdf(self, file_bytes: bytes) -> List[str]:
        buffer = io.BytesIO(file_bytes)
        doc = fitz.open(stream=buffer, filetype="pdf")
        extracted_blocks = []

        for page in doc:
            blocks = page.get_block("blocks")  # returns: (x0, y0, x1, y1, "block", block_no, block_type)
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right

            for block in sorted_blocks:
                block = block[4].strip()
                if block:
                    cleaned = self._clean_block(block)
                    if cleaned:
                        extracted_blocks.append(cleaned)

        return extracted_blocks

    def extract_from_txt(self, file_bytes: bytes) -> List[str]:
        content = file_bytes.decode('utf-8', errors='ignore')
        return content.split("\n\n")

    def extract_from_docx(self, file_bytes: bytes) -> List[str]:
        buffer = io.BytesIO(file_bytes)
        doc = docx.Document(buffer)
        extracted_paragraphs = []

        for para in doc.paragraphs:
            block = para.block.strip()
            if block:
                cleaned = self.clean_block(block)
                if cleaned:
                    extracted_paragraphs.append(cleaned)

        return extracted_paragraphs

    def extract_from_html(self, file_bytes: bytes) -> List[str]:
        raw_html = file_bytes.decode('utf-8', errors='ignore')
        # Replace <br> with newline so it can be split properly
        raw_html = raw_html.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')

        soup = BeautifulSoup(raw_html, 'html.parser')
        content_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote'])

        extracted_elements = []
        for tag in content_tags:
            block = tag.get_block(separator='\n', strip=True)
            # Now split into logical lines
            sub_blocks = re.split(r'\n+|•|- ', block)
            for sub in sub_blocks:
                cleaned = self.clean_block(sub)
                if cleaned:
                    extracted_elements.append(cleaned)

        return extracted_elements

    def clean_block(self, block: str) -> str:
        block = re.sub(r'\s+', ' ', block)  # collapse excessive whitespace
        block = block.strip()
        return block if len(block) > 5 else ''


class NoiseRemover:
    """
    Smart Noise Remover and Pre-Cleaner
    Inherits from BasePreprocessor (which inherits from BaseComponent).

    Purpose:
    - Remove headers, footers, page numbers, boilerplate block, and repeated noise
    - Preserve meaningful content, structure, and metadata
    """

    def execute(self, blocks: List[str], min_words: int = 4) -> List[str]:
        cleaned_blocks = []
        seen = set()

        for block in blocks:
            original = block

            if not block:
                continue
            block = block.strip()

            # Preserve dates like 2025-07-18
            block = re.sub(r'(?<=\d)\s*[-/]\s*(?=\d)', '-', block)

            # Preserve time like 12:04:52 or 12:04
            block = re.sub(r'(?<=\d)\s*[:]\s*(?=\d)', ':', block)

            # Remove common URLs, emails, phone numbers
            block = re.sub(r'\b(?:https?://|www\.)\S+\b', '', block)
            block = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', block)
            # Remove Indian phone numbers
            block = re.sub(r'\b(?:\+91[-\s]?|0)?[6-9]\d{9}\b', '', block)
            # Remove general international-style phone numbers
            block = re.sub(r'\b(?:\+[\d]{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '', block)

            # Remove leading/trailing HTML/JSON/Tags/brackets etc.
            block = re.sub(r'^<[^>]+>|<[^>]+>$', '', block)
            block = re.sub(r'^[\[{(]+|[\]})]+$', '', block)

            # Clean repeated punctuations but keep date/time/log symbols
            block = re.sub(r'[^\w\s:/\-,]', '', block)  # keeps : / - , for dates and logs
            block = re.sub(r'[^\w\s.,!?()\-+]', '', block) # other noisy symbols

            # Remove extra spaces
            block = re.sub(r'\s+', ' ', block).strip()

            # Remove if only digits or random characters
            if re.fullmatch(r'[\d\s]+', block) or re.fullmatch(r'[^\w\s]+', block):
                continue

            # Remove short blocks
            if len(block.split()) < min_words:
                continue

            # Remove near-duplicates (approx match)
            is_duplicate = False
            for seen_block in seen:
                similarity = SequenceMatcher(None, block, seen_block).ratio()
                if similarity > 0.90:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            cleaned_blocks.append(block)
            seen.add(block)

        return cleaned_blocks


class SmartAdaptiveChunker():
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 64,
        min_tokens: int = 4,
        similarity_threshold: float = 0.80,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.similarity_threshold = similarity_threshold
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def execute(self, blocks: List[str]) -> List[Dict[str, Any]]:
        logical_blocks = self.delimiter_split(blocks, max_bullet_length = 64)
        refined_chunks = self.semantic_refine(logical_blocks)
        return refined_chunks
    

    def delimiter_split(self, blocks: List[str], max_bullet_length: int = 128) -> List[str]:
        chunks = []
        for block in blocks:
            temp_lines, chunks = self.merge_bullets(block, chunks, max_bullet_length)
            # Now break temp_lines further using punctuation rules
            combined = " ".join(temp_lines)

            # Smart sentence splitter (handles ., !, ? but not decimals/abbreviations)
            sentences = re.split(r'(?<=[.!?])\s+', combined.strip())

            current_chunk = []
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = len(sentence.split())

                # If this single sentence is longer than max_tokens, flush what we have and split it directly
                if sentence_tokens > self.max_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    chunks.append(sentence.strip())  # Add long sentence as-is
                    continue
                
                # If adding this sentence keeps us within the limit, add it
                if current_tokens + sentence_tokens <= self.max_tokens:
                    current_chunk.append(sentence.strip())
                    current_tokens += sentence_tokens
                else:
                    # Else flush current and start a new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence.strip()]
                    current_tokens = sentence_tokens

            # Add any remaining sentences
            if current_chunk:
                chunks.append(' '.join(current_chunk))

        return chunks
    
    def merge_bullets(self, block: str, _chunks: List[Any], max_bullet_length: int) -> List[str]:
        chunks = _chunks
        temp_lines = []
        # Normalize line breaks and strip
        lines = block.strip().splitlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # If this is a heading ending in ':' or ',', start collecting bullets
            if line.endswith(":") or line.endswith(","):
                collected = [line]
                i += 1
                while i < len(lines):
                    bullet = lines[i].strip()
                    # Match bullets: -, *, •, numbered 1. 2. etc.
                    if re.match(r"^[-*•]\s+|^\d+\.\s+", bullet):
                        # If bullet is too long, treat it as standalone
                        if len(bullet) > max_bullet_length:
                            if collected:
                                chunks.append(" ".join(collected))
                                collected = []
                            chunks.append(bullet)
                        else:
                            collected.append(bullet)
                        i += 1
                    else:
                        break
                if collected:
                    chunks.append(" ".join(collected))
            else:
                temp_lines.append(line)
                i += 1
                    
        return temp_lines, chunks


    def make_chunk(self, text: str) -> Document:
        return Document(
            text= text.strip(),
            metadata= {"chunk_id": str(uuid.uuid4()),
                        "token_count": len(self.tokenizer.tokenize(text))}
        )

    def semantic_refine(self, chunks: List[str]) -> List[Document]:
        embeddings = self.get_embedding(chunks)

        refined_chunks = []
        i = 0
        while i < len(chunks) - 1:
            # Compute cosine similarity for adjacent pairs using tensor operations
            sim = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
            if sim > self.similarity_threshold:
                merged_text = chunks[i] + " " + chunks[i + 1]
                merged_chunk = self.make_chunk(merged_text)
                refined_chunks.append(merged_chunk)
                i += 2  # Skip next one because it's merged
            else:
                refined_chunks.append(self.make_chunk(chunks[i]))
                i += 1
        # Add last chunk if not processed
        if i == len(chunks) - 1:
            refined_chunks.append(self.make_chunk(chunks[i]))

        return refined_chunks
    

def get_embedding(self, texts: Union[str, List[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(texts, str):
        texts = [texts]

    return self.embedder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)