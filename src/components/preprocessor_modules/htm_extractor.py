import re
import logging
from bs4 import BeautifulSoup
from typing import List, Dict
import os


class HTMExtractor:
    """Improved extractor for HTML documents — detects headings, subtopics, bullet points, and normal text."""

    def __init__(self):
        pass

    def extract_from_htm(self, file_bytes: bytes) -> List[Dict[str, str]]:
        try:
            raw_html = file_bytes.decode('utf-8', errors='ignore')
            raw_html = re.sub(r'<br\s*/?>', '\n', raw_html, flags=re.IGNORECASE)

            soup = BeautifulSoup(raw_html, 'html.parser')

            for tag in soup(['header', 'footer', 'nav', 'script', 'style']):
                tag.decompose()
            content_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote'])
            
            results: List[Dict[str, str]] = []
            prev_text = ''
            is_merging_bullets = False
            
            for tag in content_tags:
                text = tag.get_text(separator=' ', strip=True)
                if not text:
                    continue
                
                cleaned = self.clean_block(text)
                section_type = self._infer_section_type(tag, cleaned)
                
                if section_type == "BulletPoint":
                    prev_text += " " + cleaned
                    is_merging_bullets = True
                elif is_merging_bullets:
                    results.append({
                        "section_type": 'BulletPoint',
                        "content": prev_text
                    })
                    prev_text = cleaned
                    is_merging_bullets = False
                    
                results.append({
                    "section_type": section_type,
                    "content": cleaned
                })

            return results 

        except Exception as e:
            logging.exception(f"SmartHTMExtractor.extract_from_html: Exception {str(e)}")
            raise

    def _infer_section_type(self, tag, text: str) -> str:
        """
        Determines the section type based on tag name, numbering, bullets, and style hints.
        """
        tag_name = tag.name.lower()

        if tag_name == "h1":
            return "Heading1"
        if tag_name == "h2":
            return "Heading2"
        if tag_name == "h3":
            return "SubTopic"
        if tag_name == "h4":
            return "SubSubTopic"

        if tag_name == "li":
            return "BulletPoint"

        if tag_name == "blockquote":
            return "Quote"

        if re.match(r'^\d+(?:\.\d+)+\s+[A-Za-z]', text):
            return "SubTopic"

        if text.isupper() and len(text.split()) <= 6:
            return "Heading1"

        if text.endswith(":") and len(text.split()) <= 8:
            return "Topic"

        return "NormalText"

    def clean_block(self, text: str) -> str:
        """Minimal text cleanup — remove extra spaces, html remnants, and repeated symbols."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u00A0\t\r]+', ' ', text)
        text = re.sub(r'(•|-){2,}', '-', text)
        return text.strip()
