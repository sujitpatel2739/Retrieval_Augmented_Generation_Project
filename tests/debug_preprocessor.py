import os
import sys
import time
from datetime import datetime
from src.components.preprocessor import UniversalExtractor, NoiseRemoverCleaner, SmartAdaptiveChunker
from unittest.mock import patch

# Define sample files with absolute paths
BASE_DIR = "tests"
SAMPLE_FILES = {
    "pdf": os.path.join(BASE_DIR, "sample.pdf"),
    "docx": os.path.join(BASE_DIR, "sample.docx"),
    "html": os.path.join(BASE_DIR, "sample.html"),
    "txt": os.path.join(BASE_DIR, "sample.txt")
}

# Log file for detailed outputs
LOG_FILE = os.path.join(BASE_DIR, "pipeline_output.log")

def log_to_file(message, file_handle):
    """Write a message to both console and log file."""
    print(message)
    file_handle.write(message + "\n")

def validate_extracted_output(text, metadata, file_type):
    """Validate the output of UniversalExtractor."""
    issues = []
    if not isinstance(text, str) or not text.strip():
        issues.append(f"Text is empty or invalid for {file_type}")
    if not isinstance(metadata, dict):
        issues.append(f"Metadata is not a dictionary for {file_type}")
    if file_type == "pdf" and "/Title" not in metadata:
        issues.append(f"PDF metadata missing title for {file_type}")
    if file_type == "docx" and "num_paragraphs" not in metadata:
        issues.append(f"DOCX metadata missing num_paragraphs for {file_type}")
    if file_type == "html" and "title" not in metadata:
        issues.append(f"HTML metadata missing title for {file_type}")
    if file_type == "txt" and "num_characters" not in metadata:
        issues.append(f"TXT metadata missing num_characters for {file_type}")
    if "\n\n\n" in text:
        issues.append(f"Excessive newlines not removed for {file_type}")
    if "  " in text:
        issues.append(f"Multiple spaces not removed for {file_type}")
    return issues

def validate_cleaned_output(cleaned_result, original_metadata, file_type):
    """Validate the output of NoiseRemoverCleaner."""
    issues = []
    if not isinstance(cleaned_result, dict) or "text" not in cleaned_result or "metadata" not in cleaned_result:
        issues.append(f"Invalid cleaned result format for {file_type}")
    if cleaned_result["metadata"] != original_metadata:
        issues.append(f"Metadata not preserved for {file_type}")
    if not cleaned_result["text"].strip():
        issues.append(f"Cleaned text is empty for {file_type}")
    if "\u200B" in cleaned_result["text"] or "\uFEFF" in cleaned_result["text"]:
        issues.append(f"Unicode junk not removed for {file_type}")
    if "\n\n\n" in cleaned_result["text"]:
        issues.append(f"Excessive newlines not removed in cleaned text for {file_type}")
    if "  " in cleaned_result["text"]:
        issues.append(f"Multiple spaces not removed in cleaned text for {file_type}")
    return issues

def validate_chunked_output(chunks, file_type, tokenizer, max_tokens=300):
    """Validate the output of SmartAdaptiveChunker."""
    issues = []
    if not isinstance(chunks, list):
        issues.append(f"Chunks is not a list for {file_type}")
    if not chunks:
        issues.append(f"No chunks produced for {file_type}")
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, str) or not chunk.strip():
            issues.append(f"Chunk {i} is empty or invalid for {file_type}")
        token_count = len(tokenizer.tokenize(chunk))
        if token_count > max_tokens:
            issues.append(f"Chunk {i} exceeds {max_tokens} tokens ({token_count}) for {file_type}")
    return issues

def run_pipeline():
    """Run the preprocessor pipeline on all sample files and log detailed outputs."""
    extractor = UniversalExtractor()
    cleaner = NoiseRemoverCleaner()
    
    # Mock tokenizer for SmartAdaptiveChunker
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=type('Tokenizer', (), {
        'tokenize': lambda self, text: text.split()  # Simplified tokenizer
    })()) as mock_tokenizer:
        chunker = SmartAdaptiveChunker(max_chunk_tokens=300)
        tokenizer = chunker.tokenizer
    
    all_passed = True
    results = []

    # Open log file
    with open(LOG_FILE, 'w', encoding='utf-8') as log_handle:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_file(f"Pipeline Run Started: {timestamp}", log_handle)
        log_to_file(f"Current working directory: {os.getcwd()}", log_handle)
        log_to_file(f"Test directory files: {os.listdir(BASE_DIR)}", log_handle)
        log_to_file("\n", log_handle)

        for file_type, file_path in SAMPLE_FILES.items():
            log_to_file(f"{'='*50}\nProcessing {file_type} file: {file_path}\n{'='*50}", log_handle)
            issues = []

            # Check if file exists
            if not os.path.isfile(file_path):
                issues.append(f"File {file_path} not found")
                results.append((file_type, False, issues))
                all_passed = False
                log_to_file(f"FAIL: {file_type}\n  Issues:\n    - {issues[0]}", log_handle)
                continue

            try:
                # Step 1: Extract
                log_to_file("Step 1: Extracting with UniversalExtractor", log_handle)
                text, metadata = extractor._execute(file_path)
                extraction_issues = validate_extracted_output(text, metadata, file_type)
                issues.extend(extraction_issues)
                log_to_file(f"Extracted Text (first 500 chars):\n{text[:500]}{'...' if len(text) > 500 else ''}", log_handle)
                log_to_file(f"Extracted Metadata: {metadata}", log_handle)
                log_to_file(f"Extraction Validation Issues: {extraction_issues or 'None'}", log_handle)
                log_to_file(f"Extracted Text Length: {len(text)}", log_handle)
                log_to_file("\n", log_handle)

                # Step 2: Clean
                log_to_file("Step 2: Cleaning with NoiseRemoverCleaner", log_handle)
                cleaned_result = cleaner._execute(text, metadata)
                cleaning_issues = validate_cleaned_output(cleaned_result, metadata, file_type)
                issues.extend(cleaning_issues)
                log_to_file(f"Cleaned Text (first 500 chars):\n{cleaned_result['text'][:500]}{'...' if len(cleaned_result['text']) > 500 else ''}", log_handle)
                log_to_file(f"Cleaned Metadata: {cleaned_result['metadata']}", log_handle)
                log_to_file(f"Cleaning Validation Issues: {cleaning_issues or 'None'}", log_handle)
                log_to_file(f"Cleaned Text Length: {len(cleaned_result['text'])}", log_handle)
                log_to_file("\n", log_handle)

                # Step 3: Chunk
                log_to_file("Step 3: Chunking with SmartAdaptiveChunker", log_handle)
                chunks = chunker._execute(cleaned_result["text"])
                chunking_issues = validate_chunked_output(chunks, file_type, tokenizer)
                issues.extend(chunking_issues)
                log_to_file(f"Number of Chunks: {len(chunks)}", log_handle)
                for i, chunk in enumerate(chunks):
                    token_count = len(tokenizer.tokenize(chunk))
                    log_to_file(f"Chunk {i} (Tokens: {token_count}):\n{chunk[:200]}{'...' if len(chunk) > 200 else ''}", log_handle)
                log_to_file(f"Chunking Validation Issues: {chunking_issues or 'None'}", log_handle)
                log_to_file("\n", log_handle)

                # Log results
                passed = len(issues) == 0
                results.append((file_type, passed, issues))
                if not passed:
                    all_passed = False
                
                log_to_file(f"{'PASS' if passed else 'FAIL'}: {file_type}", log_handle)
                if issues:
                    log_to_file("Issues:", log_handle)
                    for issue in issues:
                        log_to_file(f"  - {issue}", log_handle)
                else:
                    log_to_file("  No issues found", log_handle)

            except Exception as e:
                issues.append(f"Error processing {file_type}: {str(e)}")
                results.append((file_type, False, issues))
                all_passed = False
                log_to_file(f"FAIL: {file_type}", log_handle)
                log_to_file(f"  Error: {str(e)}", log_handle)

            log_to_file("\n", log_handle)

        # Print summary
        log_to_file(f"{'='*50}\nSummary\n{'='*50}", log_handle)
        for file_type, passed, issues in results:
            log_to_file(f"{file_type}: {'PASS' if passed else 'FAIL'}", log_handle)
            if issues:
                log_to_file("  Issues:", log_handle)
                for issue in issues:
                    log_to_file(f"    - {issue}", log_handle)
        
        log_to_file(f"\nOverall result: {'All units working correctly' if all_passed else 'Some units failed'}", log_handle)
    
    return all_passed

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)