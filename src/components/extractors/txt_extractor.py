import io
import re
import logging
from typing import List, Dict
import os


# Regex helpers
NUMBERED_HEADING_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s+[A-Za-z]')   # e.g. 1.4 Introduction
BULLET_RE = re.compile(r'^\s*(?:[-\u2022\u2023\u25E6\*•]|[a-zA-Z]\)|\d+\)|\(\d+\))\s+')  # •, -, 1), etc.


def _clean_text(s: str) -> str:
    s = s.replace('\u00A0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


class TXTExtractor:
    """Improved extractor for TXT files — detects headings, topics, bullets, normal paragraphs."""

    def __init__(self):
        pass

    def extract_from_txt(self, file_bytes: bytes) -> List[Dict[str, str]]:
        """
        Returns list of dicts:
        [
            {"section_type": "Heading1", "content": "Kinematics"},
            {"section_type": "SubTopic", "content": "1.4 Introduction"},
            {"section_type": "NormalText", "content": "Motion is change of position..."},
        ]
        """
        try:
            content = file_bytes.decode("utf-8", errors="ignore")
            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]  # preserve paragraph separation

            results: List[Dict[str, str]] = []

            for para in paragraphs:
                cleaned = _clean_text(para)
                if not cleaned:
                    continue

                section_type = "NormalText"

                # 1️⃣ Detect numbered headings/topics/subtopics
                if NUMBERED_HEADING_RE.match(cleaned):
                    dots = cleaned.split()[0].count('.')
                    if dots == 0:
                        section_type = "Topic"
                    elif dots == 1:
                        section_type = "SubTopic"
                    else:
                        section_type = "SubSubTopic"

                # 2️⃣ Detect bullet points
                elif BULLET_RE.match(cleaned):
                    section_type = "BulletPoint"

                # 3️⃣ All-caps or short lines likely to be main headings
                elif cleaned.isupper() and len(cleaned.split()) <= 6:
                    section_type = "Heading1"

                # 4️⃣ Lines ending with colon (“Introduction:”)
                elif cleaned.endswith(":") and len(cleaned.split()) <= 10:
                    section_type = "Topic"

                # 5️⃣ One-line titles surrounded by whitespace (manual headings)
                elif len(cleaned.split()) <= 8 and cleaned[0].isalpha() and cleaned[-1].isalpha() and cleaned == cleaned.title():
                    section_type = "Heading2"

                results.append({
                    "section_type": section_type,
                    "content": cleaned
                })

            return results

        except Exception as e:
            logging.exception(f"SmartTxtExtractor.extract_from_txt: Exception {str(e)}")
            raise
