# dependencies: pip install pymupdf
import io
import re
import pymupdf  # PyMuPDF
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import os

NUMBERED_HEADING_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s+')
BULLET_RE = re.compile(r'^\s*(?:[-\u2022\u2023\u25E6\*•]|[a-zA-Z]\)|\d+\)|\(\d+\))\s+')
PAGE_NUM_RE = re.compile(r'^\s*\d+\s*$')

def _clean_text(s: str) -> str:
    s = s.replace('\u00A0', ' ')  # non-breaking spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class SmartPDFExtractor:
    def __init__(self, ignore_header_footer_freq: int = 0.6):
        """
        ignore_header_footer_freq: fraction of pages a text must appear at top/bottom
                                   to be considered header/footer (0..1).
        """
        self.ignore_header_footer_freq = ignore_header_footer_freq

    def extract_from_pdf(self, file_bytes: bytes) -> List[Dict[str, str]]:
        """
        Returns list of dicts: [{"section_type": "Heading1", "content": "Kinematics"}, ...]
        """
        # open document
        buffer = io.BytesIO(file_bytes)
        doc = pymupdf.open(stream=buffer, filetype="pdf")

        # 1) gather candidate header/footer strings by sampling first/last lines of pages
        top_texts = []
        bottom_texts = []
        pages_info = []
        for page in doc:
            page_dict = page.get_text("dict")
            spans = self._extract_spans_from_page_dict(page_dict)
            pages_info.append(spans)
            if spans:
                # top candidate: first non-empty span text on page
                first = next((s['text'] for s in spans if s['text'].strip()), "")
                last = next((s['text'] for s in reversed(spans) if s['text'].strip()), "")
                if first:
                    top_texts.append(_clean_text(first.lower()))
                if last:
                    bottom_texts.append(_clean_text(last.lower()))

        # frequency thresholding to detect headers/footers
        header_candidates = self._frequent_texts(top_texts, doc.page_count, self.ignore_header_footer_freq)
        footer_candidates = self._frequent_texts(bottom_texts, doc.page_count, self.ignore_header_footer_freq)

        # 2) compute global font statistics (to identify large headings and tiny footnotes)
        all_font_sizes = []
        for spans in pages_info:
            for s in spans:
                if s['text'].strip():
                    all_font_sizes.append(s['size'])
        if all_font_sizes:
            median_font = sorted(all_font_sizes)[len(all_font_sizes)//2]
        else:
            median_font = 10.0

        # heuristics thresholds
        heading_size_threshold = median_font + 2.0
        footnote_size_threshold = max(6.0, median_font - 3.0)

        # 3) classify per-page, aggregate into result blocks
        results: List[Dict[str, str]] = []
        for page_index, spans in enumerate(pages_info):
            for s in spans:
                raw = s['text'].strip()
                if not raw:
                    continue
                cleaned = _clean_text(raw)

                low = cleaned.lower()
                # ignore headers/footers and page numbers
                if low in header_candidates or low in footer_candidates:
                    continue
                if PAGE_NUM_RE.match(cleaned):
                    continue

                # tiny-font likely footnote: ignore
                if s['size'] <= footnote_size_threshold and len(cleaned) < 200:
                    # some footnotes may be important; skip by default
                    continue

                # classification rules (ordered by priority)
                section_type = "NormalText"

                # 1) Very large font or large and short => Heading1 / Heading2
                if s['size'] >= heading_size_threshold and len(cleaned.split()) <= 12:
                    # differentiate levels by font size gap
                    if s['size'] >= heading_size_threshold + 2.5:
                        section_type = "Heading1"
                    else:
                        section_type = "Heading2"
                # 2) Numbered headings: 1., 1.2, 1.2.3 etc => Topic/SubTopic
                elif NUMBERED_HEADING_RE.match(cleaned):
                    # count number of dots to decide level
                    dots = cleaned.strip().split()[0].count('.')
                    if dots == 1:
                        section_type = "Topic"
                    elif dots == 2:
                        section_type = "SubTopic"
                    else:
                        section_type = "SubSubTopic"
                # 3) Uppercase short line (often headings)
                elif cleaned.isupper() and len(cleaned.split()) <= 8:
                    section_type = "Heading2"
                # 4) Bullet points
                elif BULLET_RE.match(cleaned):
                    section_type = "BulletPoint"
                # 5) If line ends with ':' and short -> likely a section header introducing bullets
                elif cleaned.endswith(":") and len(cleaned.split()) <= 12:
                    section_type = "Topic"
                # else keep NormalText

                results.append({"section_type": section_type, "content": cleaned})

        # Optionally merge adjacent NormalText into paragraphs and group bullets under topics.
        merged = self._post_process_merge(results)
        return merged

    def _extract_spans_from_page_dict(self, page_dict: dict) -> List[Dict[str, Any]]:
        """
        Convert page.get_text('dict') into a flat list of spans ordered top-to-bottom, left-to-right.
        Each span: {'text':..., 'size': float, 'x':float, 'y':float}
        """
        spans_out = []
        blocks = page_dict.get("blocks", [])
        # blocks contain lines->spans; we will flatten preserving vertical order
        # Use block bbox y0 for sorting then x0
        sortable = []
        for block in blocks:
            bbox = block.get("bbox", [0,0,0,0])
            y0 = bbox[1]
            x0 = bbox[0]
            # some blocks have "lines"
            lines = block.get("lines", [])
            for line in lines:
                # get concatenated text for the line and approximate size as max span size
                line_text_parts = []
                max_span_size = 0.0
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size = span.get("size", 0.0)
                    origin_x = span.get("bbox", [0,0,0,0])[0] if span.get("bbox") else x0
                    line_text_parts.append(text)
                    if size and size > max_span_size:
                        max_span_size = size
                    # store each span individually? We'll collapse to line-level though
                line_text = "".join(line_text_parts)
                sortable.append((y0, x0, line_text, max_span_size))
        # sort by y then x
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
        Minor pass to:
         - merge adjacent NormalText lines into a single NormalText paragraph
         - keep bullets as individual items
         - optionally combine Heading + Bullet groupings (not implemented here)
        """
        merged = []
        buffer_par = []
        for it in items:
            t = it["section_type"]
            c = it["content"]
            if t == "NormalText":
                buffer_par.append(c)
            else:
                if buffer_par:
                    merged.append({"section_type": "NormalText", "content": " ".join(buffer_par)})
                    buffer_par = []
                merged.append(it)
        if buffer_par:
            merged.append({"section_type": "NormalText", "content": " ".join(buffer_par)})
        return merged


def main():
    # ✅ Path to your test PDF file
    pdf_path = "sample.pdf"  # change this to your test PDF path
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        return

    # Read the PDF as bytes
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # Initialize extractor
    extractor = SmartPDFExtractor()

    # Extract structured data
    print("[INFO] Extracting structured content from PDF...")
    extracted_sections = extractor.extract_from_pdf(file_bytes)

    # Print results
    print(f"\n[INFO] Extracted {len(extracted_sections)} blocks:\n")
    for i, section in enumerate(extracted_sections, start=1):
        section_type = section["section_type"]
        content = section["content"]
        print(f"{i:03d}. [{section_type}] {content[:120]}")  # limit text to 120 chars per line

    print("\n[INFO] Extraction complete!")

if __name__ == "__main__":
    main()