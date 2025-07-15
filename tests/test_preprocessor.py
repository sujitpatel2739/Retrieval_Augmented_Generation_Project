import os
import pytest
from src.components.preprocessor import UniversalExtractor, NoiseRemoverCleaner, SmartAdaptiveChunker
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
# from transformers import AutoTokenizer
from unittest.mock import patch


# Ensure the sample files exist in the current directory
SAMPLE_FILES = {
    "pdf": os.path.join("tests", "sample.pdf"),
    "docx": os.path.join("tests", "sample.docx"),
    "html": os.path.join("tests", "sample.html"),
    "txt": os.path.join("tests", "sample.txt")
}

@pytest.fixture(scope="module")
def check_sample_files():
    """Verify that all sample files exist in the current directory."""
    for file_path in SAMPLE_FILES.values():
        if not os.path.isfile(file_path):
            pytest.fail(f"Sample file {file_path} not found in current directory")
    return SAMPLE_FILES

@pytest.fixture
def universal_extractor():
    return UniversalExtractor()

@pytest.fixture
def noise_remover():
    return NoiseRemoverCleaner()

@pytest.fixture
def chunker():
    # Mock tokenizer to avoid loading BERT model during testing
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=type('Tokenizer', (), {
        'tokenize': lambda self, text: text.split()  # Simplified tokenizer for testing
    })()):
        return SmartAdaptiveChunker(max_chunk_tokens=50)

@pytest.fixture
def sample_text():
    """Sample text for testing cleaning and chunking independently of file extraction."""
    return """
    Header: Test Document
    This is a test paragraph.
    
    123
    
    Another paragraph with some content.
    Header: Test Document
    \u200B\uFEFF
    ====
    - Bullet point 1
    - Bullet point 2
    """

def test_universal_extractor_pdf(universal_extractor, check_sample_files):
    """Test extraction from sample.pdf."""
    text, metadata = universal_extractor._execute(SAMPLE_FILES["pdf"], file_type="pdf")
    
    assert isinstance(text, str) and len(text) > 0, "PDF text extraction failed"
    assert isinstance(metadata, dict), "PDF metadata should be a dictionary"
    assert text.strip(), "Extracted PDF text should not be empty after cleaning"
    # Check that cleaning removed excessive whitespace
    assert "\n\n\n" not in text, "Excessive newlines should be removed"
    assert "  " not in text, "Multiple spaces should be removed"

def test_universal_extractor_docx(universal_extractor, check_sample_files):
    """Test extraction from sample.docx."""
    text, metadata = universal_extractor._execute(SAMPLE_FILES["docx"], file_type="docx")
    
    assert isinstance(text, str) and len(text) > 0, "DOCX text extraction failed"
    assert isinstance(metadata, dict) and "num_paragraphs" in metadata, "DOCX metadata should include num_paragraphs"
    assert text.strip(), "Extracted DOCX text should not be empty after cleaning"
    assert "\n\n\n" not in text, "Excessive newlines should be removed"
    assert "  " not in text, "Multiple spaces should be removed"

def test_universal_extractor_html(universal_extractor, check_sample_files):
    """Test extraction from sample.html."""
    text, metadata = universal_extractor._execute(SAMPLE_FILES["html"], file_type="html")
    
    assert isinstance(text, str) and len(text) > 0, "HTML text extraction failed"
    assert isinstance(metadata, dict) and "title" in metadata, "HTML metadata should include title"
    assert text.strip(), "Extracted HTML text should not be empty after cleaning"
    assert "\n\n\n" not in text, "Excessive newlines should be removed"
    assert "  " not in text, "Multiple spaces should be removed"

def test_universal_extractor_txt(universal_extractor, check_sample_files):
    """Test extraction from sample.txt."""
    text, metadata = universal_extractor._execute(SAMPLE_FILES["txt"], file_type="txt")
    
    assert isinstance(text, str) and len(text) > 0, "TXT text extraction failed"
    assert isinstance(metadata, dict) and "num_characters" in metadata, "TXT metadata should include num_characters"
    assert text.strip(), "Extracted TXT text should not be empty after cleaning"
    assert "\n\n\n" not in text, "Excessive newlines should be removed"
    assert "  " not in text, "Multiple spaces should be removed"

def test_universal_extractor_file_type_detection(universal_extractor, check_sample_files):
    """Test automatic file type detection in _execute."""
    for ext, file_path in SAMPLE_FILES.items():
        text, metadata = universal_extractor._execute(file_path)  # No file_type provided
        assert isinstance(text, str) and len(text) > 0, f"File type detection failed for {ext}"
        assert isinstance(metadata, dict), f"Metadata should be a dictionary for {ext}"

def test_universal_extractor_unsupported_type(universal_extractor):
    """Test handling of unsupported file type."""
    with pytest.raises(ValueError, match="Unsupported file type: invalid"):
        universal_extractor._execute("sample.invalid", file_type="invalid")

def test_universal_extractor_clean_text(universal_extractor):
    """Test _clean_text method directly."""
    input_text = "  Test   text\n\n\nwith   spaces\r\n\tand\tcontrol\nchars  "
    cleaned = universal_extractor._clean_text(input_text)
    
    assert cleaned == "Test text\nwith spaces\nand control\nchars", "Cleaning failed to normalize whitespace"
    assert "\n\n" not in cleaned, "Multiple newlines should be collapsed"
    assert "  " not in cleaned, "Multiple spaces should be collapsed"
    assert "\t" not in cleaned, "Control characters should be removed"

def test_noise_remover_cleaner_execute(noise_remover, sample_text):
    """Test NoiseRemoverCleaner._execute with metadata."""
    metadata = {"source": "test"}
    result = noise_remover._execute(sample_text, metadata)
    
    assert isinstance(result, dict) and "text" in result and "metadata" in result, "Invalid result format"
    assert result["metadata"] == metadata, "Metadata should be preserved"
    assert "Header: Test Document" not in result["text"], "Headers should be removed"
    assert "123" not in result["text"], "Page numbers should be removed"
    assert "====" not in result["text"], "Garbage lines should be removed"
    assert "\u200B" not in result["text"], "Unicode junk should be removed"
    assert "This is a test paragraph" in result["text"], "Main content should be preserved"

def test_noise_remover_cleaner_no_metadata(noise_remover, sample_text):
    """Test NoiseRemoverCleaner._execute with no metadata."""
    result = noise_remover._execute(sample_text)
    
    assert isinstance(result["metadata"], dict) and not result["metadata"], "Empty metadata should be returned"
    assert "Header: Test Document" not in result["text"], "Headers should be removed"
    assert "This is a test paragraph" in result["text"], "Main content should be preserved"

def test_noise_remover_remove_repeated_lines(noise_remover):
    """Test _remove_repeated_lines directly."""
    input_text = """
    Header
    Content line 1
    Header
    Content line 2
    Header
    Content line 3
    """
    cleaned = noise_remover._remove_repeated_lines(input_text)
    
    assert "Header" not in cleaned, "Repeated headers should be removed"
    assert "Content line 1" in cleaned, "Content should be preserved"
    assert "Content line 3" in cleaned, "Content should be preserved"

def test_noise_remover_clean_text_edge_cases(noise_remover):
    """Test _clean_text with edge cases."""
    # Only noise
    noise_only = "Header\n123\n====\n\u200B"
    assert not noise_remover._clean_text(noise_only).strip(), "Only noise should result in empty text"
    
    # Complex page numbers
    complex_text = "Page 1 of 10\nContent\nPage 2 of 10"
    cleaned = noise_remover._clean_text(complex_text)
    assert "Page 1 of 10" not in cleaned, "Complex page numbers should be removed"
    assert "Content" in cleaned, "Content should be preserved"

def test_smart_adaptive_chunker_init():
    """Test SmartAdaptiveChunker initialization with mocked tokenizer."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        chunker = SmartAdaptiveChunker(max_chunk_tokens=50)
        assert mock_tokenizer.called, "Tokenizer should be initialized"
        assert chunker.max_chunk_tokens == 50, "Max chunk tokens should be set"

def test_smart_adaptive_chunker_execute(chunker, sample_text):
    """Test SmartAdaptiveChunker._execute."""
    chunks = chunker._execute(sample_text)
    
    assert isinstance(chunks, list) and len(chunks) > 0, "Chunks should be a non-empty list"
    assert any("This is a test paragraph" in chunk for chunk in chunks), "Main content should be in a chunk"
    assert any("Bullet point 1" in chunk for chunk in chunks), "Bullet points should be in chunks"

def test_smart_adaptive_chunker_delimiter_splitting(chunker):
    """Test delimiter_based_splitting with various delimiters."""
    input_text = """
    1. Section One
    First paragraph.
    
    2. Section Two
    Second paragraph.
    - Bullet 1
    - Bullet 2
    
    HEADING
    Final content.
    """
    chunks = chunker.delimiter_based_splitting(input_text)
    
    assert len(chunks) >= 4, "Should split on numbered sections, bullets, and heading"
    assert any("First paragraph" in chunk for chunk in chunks), "Section content should be preserved"
    assert any("Bullet 1" in chunk for chunk in chunks), "Bullet points should be preserved"
    assert any("Final content" in chunk for chunk in chunks), "Heading content should be preserved"

def test_smart_adaptive_chunker_delimiter_edge_cases(chunker):
    """Test delimiter_based_splitting edge cases."""
    # No delimiters
    no_delimiters = "This is a single block of text with no delimiters."
    chunks = chunker.delimiter_based_splitting(no_delimiters)
    assert len(chunks) == 1, "No delimiters should result in one chunk"
    
    # Small splits
    small_splits = "1. Tiny\nShort\n2. Another"
    chunks = chunker.delimiter_based_splitting(small_splits)
    assert all(len(chunk) >= 20 for chunk in chunks), "Small splits (<20 chars) should be filtered"

def test_smart_adaptive_chunker_token_aware(chunker):
    """Test token_aware_chunking with token limits."""
    input_splits = [
        "Short sentence.",
        "This is a very long sentence " * 20 + "end."
    ]
    chunks = chunker.token_aware_chunking(input_splits)
    
    assert len(chunks) > 1, "Long sentence should be split"
    assert chunks[0] == "Short sentence.", "Short sentences should be preserved"
    assert all(len(chunker.tokenizer.tokenize(chunk)) <= 50 for chunk in chunks), "Chunks should respect token limit"

def test_smart_adaptive_chunker_token_aware_edge_cases(chunker):
    """Test token_aware_chunking edge cases."""
    # Empty input
    assert chunker.token_aware_chunking([]) == [], "Empty input should return empty list"
    
    # Long sentence exceeding token limit
    long_sentence = "Word " * 100
    chunks = chunker.token_aware_chunking([long_sentence])
    assert all(len(chunker.tokenizer.tokenize(chunk)) <= 50 for chunk in chunks), "Long sentence should be split"

def test_integration_pipeline(universal_extractor, noise_remover, chunker, check_sample_files):
    """Test full pipeline for each file type."""
    for file_type, file_path in SAMPLE_FILES.items():
        # Step 1: Extract
        text, metadata = universal_extractor._execute(file_path)
        assert isinstance(text, str) and len(text) > 0, f"Extraction failed for {file_type}"
        assert isinstance(metadata, dict), f"Metadata invalid for {file_type}"
        
        # Step 2: Clean
        cleaned_result = noise_remover._execute(text, metadata)
        assert isinstance(cleaned_result["text"], str) and len(cleaned_result["text"]) > 0, f"Cleaning failed for {file_type}"
        assert cleaned_result["metadata"] == metadata, f"Metadata not preserved for {file_type}"
        
        # Step 3: Chunk
        chunks = chunker._execute(cleaned_result["text"])
        assert isinstance(chunks, list) and len(chunks) > 0, f"Chunking failed for {file_type}"

def test_integration_pipeline_failure(universal_extractor, noise_remover, chunker):
    """Test pipeline with a failing extraction."""
    with pytest.raises(ValueError, match="Unsupported file type: invalid"):
        universal_extractor._execute("sample.invalid", file_type="invalid")
    
    # Test empty text after cleaning
    empty_text = "Header\n123\n===="
    cleaned = noise_remover._execute(empty_text)
    chunks = chunker._execute(cleaned["text"])
    assert len(chunks) == 0 or all(not chunk.strip() for chunk in chunks), "Empty cleaned text should produce no chunks"

if __name__ == "__main__":
    pytest.main([__file__])