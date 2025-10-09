from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Document:
    text: str
    metadata: Dict = None

@dataclass
class SearchResult:
    text: str
    metadata: Optional[Dict] = None
    score: Optional[float] = None

@dataclass
class RAGResponse:
    answer: str
    confidence_score: float
    keywords: List[str]
    
@dataclass
class QueryStatus:
    status: str
    detail: str = None