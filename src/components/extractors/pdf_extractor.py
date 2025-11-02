
import io
import re
import pymupdf
from typing import List, Dict, Tuple, Any
from collections import Counter
import os

NUMBERED_HEADING_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s+')
BULLET_RE = re.compile(r'^\s*(?:[-\u2022\u2023\u25E6\*•]|[a-zA-Z]\)|\d+\)|\(\d+\))\s+')
PAGE_NUM_RE = re.compile(r'^\s*\d+\s*$')

def _clean_text(s: str) -> str:
    s = s.replace('\u00A0', ' ')  
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class PDFExtractor:
    def __init__(self, ignore_header_footer_freq: int = 0.6):
        """
        ignore_header_footer_freq: fraction of pages a text must appear at top/bottom
                                   to be considered header/footer (0..1).
        """
        self.ignore_header_footer_freq = ignore_header_footer_freq

    def run(self, file_bytes: bytes) -> List[Dict[str, str]]:
        """
        Returns list of dicts: [{"section_type": "Heading1", "content": "Kinematics"}, ...]
        """
        buffer = io.BytesIO(file_bytes)
        doc = pymupdf.open(stream=buffer, filetype="pdf")

        top_texts = []
        bottom_texts = []
        pages_info = []
        for page in doc:
            page_dict = page.get_text("dict")
            spans = self._extract_spans_from_page_dict(page_dict)
            pages_info.append(spans)
            if spans:
                first = next((s['text'] for s in spans if s['text'].strip()), "")
                last = next((s['text'] for s in reversed(spans) if s['text'].strip()), "")
                if first:
                    top_texts.append(_clean_text(first.lower()))
                if last:
                    bottom_texts.append(_clean_text(last.lower()))

        header_candidates = self._frequent_texts(top_texts, doc.page_count, self.ignore_header_footer_freq)
        footer_candidates = self._frequent_texts(bottom_texts, doc.page_count, self.ignore_header_footer_freq)

        all_font_sizes = []
        for spans in pages_info:
            for s in spans:
                if s['text'].strip():
                    all_font_sizes.append(s['size'])
        if all_font_sizes:
            median_font = sorted(all_font_sizes)[len(all_font_sizes)//2]
        else:
            median_font = 10.0

        heading_size_threshold = median_font + 2.0
        footnote_size_threshold = max(6.0, median_font - 3.0)

        results: List[Dict[str, str]] = []
        for page_index, spans in enumerate(pages_info):
            for s in spans:
                raw = s['text'].strip()
                if not raw:
                    continue
                cleaned = _clean_text(raw)

                low = cleaned.lower()
                if low in header_candidates or low in footer_candidates:
                    continue
                if PAGE_NUM_RE.match(cleaned):
                    continue

                if s['size'] <= footnote_size_threshold and len(cleaned) < 200:
                    continue

                section_type = "NormalText"

                if NUMBERED_HEADING_RE.match(cleaned):
                    dots = cleaned.strip().split()[0].count('.')
                    if dots == 1:
                        section_type = "Topic"
                    elif dots == 2:
                        section_type = "SubTopic"
                    else:
                        section_type = "SubSubTopic"
                elif s['size'] >= heading_size_threshold and len(cleaned.split()) <= 12:
                    if s['size'] >= heading_size_threshold + 2.5:
                        section_type = "Heading1"
                    else:
                        section_type = "Heading2"
                elif cleaned.isupper() and len(cleaned.split()) <= 8:
                    section_type = "Heading2"
                elif BULLET_RE.match(cleaned):
                    section_type = "BulletPoint"
                elif cleaned.endswith(":") and len(cleaned.split()) <= 12:
                    section_type = "Topic"

                results.append({"section_type": section_type, "content": cleaned, "x0": s['x'], "y0": s['y']})

        return self._post_process_merge(results)

    def _extract_spans_from_page_dict(self, page_dict: dict) -> List[Dict[str, Any]]:
        """
        Convert page.get_text('dict') into a flat list of spans ordered top-to-bottom, left-to-right.
        Each span: {'text':..., 'size': float, 'x':float, 'y':float}
        """
        spans_out = []
        blocks = page_dict.get("blocks", [])
        sortable = []
        for block in blocks:
            bbox = block.get("bbox", [0,0,0,0])
            y0 = bbox[1]
            x0 = bbox[0]
            lines = block.get("lines", [])
            for line in lines:
                line_text_parts = []
                max_span_size = 0.0
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size = span.get("size", 0.0)
                    origin_x = span.get("bbox", [0,0,0,0])[0] if span.get("bbox") else x0
                    line_text_parts.append(text)
                    if size and size > max_span_size:
                        max_span_size = size
                line_text = "".join(line_text_parts)
                sortable.append((y0, x0, line_text, max_span_size))
        sortable_sorted = sorted(sortable, key=lambda t: (t[0], t[1]))
        for y0, x0, text, size in sortable_sorted:
            spans_out.append({"text": text, "size": float(size or 0.0), "x": float(x0), "y": float(y0)})
        return spans_out

    def _frequent_texts(self, text_list: List[str], page_count: int, freq_threshold: float) -> set:
        """
        Return a set of texts that appear in at least freq_threshold fraction of pages.
        Lowercases texts before counting.
        """
        if not text_list:
            return set()
        c = Counter(text_list)
        result = set()
        for text, count in c.items():
            if count / max(1, page_count) >= freq_threshold and len(text.strip()) > 0:
                result.add(text)
        return result

    def _post_process_merge(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Improved merge logic using y0 gaps to detect paragraph boundaries.

        - Merges adjacent NormalText lines into paragraphs.
        - Starts a new paragraph when vertical gap (y0 difference) is large.
        - Keeps non-NormalText (e.g., bullets, headings) separate.
        """
        merged = []
        buffer_par = []
        prev_y0 = None

        for it in items:
            t = it["section_type"]
            c = it["content"]
            y0 = it.get("y0", 0)

            if t == "NormalText":
                if buffer_par:
                    y_gap = abs(y0 - prev_y0) if prev_y0 is not None else 0

                    if y_gap > 25:  
                        merged.append({
                            "section_type": "NormalText",
                            "content": " ".join(buffer_par).strip()
                        })
                        buffer_par = [c]
                    else:
                        buffer_par.append(c)
                else:
                    buffer_par = [c]
                prev_y0 = y0
            else:
                if buffer_par:
                    merged.append({
                        "section_type": "NormalText",
                        "content": " ".join(buffer_par).strip()
                    })
                    buffer_par = []
                merged.append(it)
                prev_y0 = None  

        if buffer_par:
            merged.append({
                "section_type": "NormalText",
                "content": " ".join(buffer_par).strip()
            })

        return merged
