import io
import re
import logging
from typing import List, Dict
import docx
import os
NUMBERED_HEADING_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s+[A-Za-z]')  
BULLET_RE = re.compile(r'^\s*(?:[-\u2022\u2023\u25E6\*•]|[a-zA-Z]\)|\d+\)|\(\d+\))\s+')

def _clean_text(s: str) -> str:
    s = s.replace('\u00A0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


class DOCXExtractor:
    """Improved extractor for DOCX files, preserving paragraph order & section types."""

    def __init__(self):
        pass

    def extract_from_docx(self, file_bytes: bytes) -> List[Dict[str, str]]:
        """
        Returns list of dicts:
        [
            {"section_type": "Heading1", "content": "Kinematics"},
            {"section_type": "SubTopic", "content": "1.4 Introduction"},
            {"section_type": "NormalText", "content": "This section introduces motion..."}
        ]
        """
        try:
            buffer = io.BytesIO(file_bytes)
            doc = docx.Document(buffer)

            results: List[Dict[str, str]] = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                cleaned = _clean_text(text)
                if not cleaned:
                    continue

                style_name = para.style.name if para.style else ""

                section_type = "NormalText"


                if NUMBERED_HEADING_RE.match(cleaned):
                    dots = cleaned.split()[0].count('.')
                    if dots == 1:
                        section_type = "Topic"
                    elif dots == 2:
                        section_type = "SubTopic"
                    else:
                        section_type = "SubSubTopic"
                elif style_name.startswith("Heading"):
                    try:
                        level = int(style_name.replace("Heading", "").strip())
                    except:
                        level = 1
                    section_type = f"Heading{level}"

                elif BULLET_RE.match(cleaned):
                    section_type = "BulletPoint"
                
                elif cleaned.isupper() and len(cleaned.split()) <= 6:
                    section_type = "Heading2"

                elif cleaned.endswith(":") and len(cleaned.split()) <= 10:
                    section_type = "Topic"

                results.append({
                    "section_type": section_type,
                    "content": cleaned
                })

            return results
            return self._merge_normaltext(results)

        except Exception as e:
            logging.exception(f"SmartDocxExtractor.extract_from_docx: Exception {str(e)}")
            raise

    def _merge_normaltext(self, blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Merge consecutive NormalText paragraphs into one block to preserve paragraph grouping.
        """
        merged = []
        buffer = []
        for block in blocks:
            if block["section_type"] == "NormalText":
                buffer.append(block["content"])
            else:
                if buffer:
                    merged.append({
                        "section_type": "NormalText",
                        "content": "\n\n".join(buffer)
                    })
                    buffer = []
                merged.append(block)
        if buffer:
            merged.append({
                "section_type": "NormalText",
                "content": "\n\n".join(buffer)
            })
        return merged