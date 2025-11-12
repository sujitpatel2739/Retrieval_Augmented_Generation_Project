import re
import logging
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict
import unicodedata

class Cleaner:
    """
    Smart Adaptive Noise Remover (v3)
    - Handles noise, duplication, and preserves important structure.
    - Replaces sensitive info with tags only in long blocks.
    """

    def __init__(self, min_words: int = 3, min_similarity: float = 0.88, strict: bool = False):
        self.min_words = min_words
        self.min_similarity = min_similarity
        self.strict = strict
        self.logger = logging.getLogger("Cleaner")

    def execute(self, blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        print("Cleaner v3 called")
        if not blocks:
            return []

        normalized_blocks = [{
                                'content': self._normalize_text(b['content']),
                                'section_type': b['section_type']
                              } for b in blocks if b['content'].strip()]
        
        frequent_lines = self._detect_repeated_lines(normalized_blocks)

        cleaned = []
        seen = []

        for block in normalized_blocks:
            content = block['content']
            if not content.strip():
                continue

            if self._is_noise(content, frequent_lines):
                continue

            # Decide if it's paragraph (multi-line or long)
            is_paragraph = '\n' in content or len(content.split()) > 8

            # Replace sensitive info only if paragraph/long
            if is_paragraph:
                content = self._mask_sensitive_info(content)
            else:
                # If short line and contains phone/urls — skip it entirely
                if self._contains_sensitive_info(content):
                    continue

            # Handle bullets: preserve if meaningful
            content = self._handle_bullets(content)

            # Remove too short results after cleaning
            if len(content.split()) < 3:
                continue

            # Deduplication
            if any(self._is_similar(content, s) for s in seen):
                continue

            cleaned.append({'section_type': block['section_type'], 'content': content})
            seen.append(content)

        return cleaned

    # ------------------------------------------------------
    # Internal Helper Functions
    # ------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).strip()

        # Fix OCR split words
        text = re.sub(r'(\b\w)\s+(\w\b)', r'\1\2', text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def _detect_repeated_lines(self, blocks: List[str]) -> set:
        counts = Counter(blocks)
        frequent = {b for b, c in counts.items() if c > 2 and len(b.split()) < 10}
        return frequent

    def _is_noise(self, text: str, frequent_lines: set) -> bool:
        if text in frequent_lines:
            return True
        if re.fullmatch(r'(Page|PAGE)?\s*\d{1,4}\s*(of\s*\d+)?', text, re.I):
            return True
        if re.fullmatch(r'[.,;:(){}\[\]]+', text):
            return True
        if any(kw in text.lower() for kw in ["copyright", "rights reserved", "confidential"]):
            return True
        if len(text.split()) <= 2 and not re.match(r'^[A-Za-z]+\s*\d*[:)?]?$', text):
            return True
        return False

    def _mask_sensitive_info(self, text: str) -> str:
        """Replace URLs, emails, phone numbers with tags in long paragraphs."""
        text = re.sub(r'\b(?:https?://|www\.)\S+\b', '[URL]', text)
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
        text = re.sub(r'\b(?:\+91[-\s]?|0)?[6-9]\d{9}\b', '[PHONE]', text)
        text = re.sub(r'\b(?:\+[\d]{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        
        return text

    def _contains_sensitive_info(self, text: str) -> bool:
        """Return True if text contains phone numbers, URLs, or emails."""
        return bool(re.search(r'(https?://|www\.|@|\+91|[6-9]\d{9})', text))

    def _handle_bullets(self, text: str) -> str:
        """Preserve bullets if long, else remove."""
        bullet_pattern = r'^[\-\*\•]\s*(.*)$'
        match = re.match(bullet_pattern, text.strip())
        if match:
            content = match.group(1).strip()
            if len(content.split()) < 4:
                return ""  # remove short bullet
            return content  # preserve long bullet
        return text

    def _is_similar(self, a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= self.min_similarity

