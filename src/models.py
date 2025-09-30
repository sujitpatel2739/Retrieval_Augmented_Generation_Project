from dataclasses import dataclass
from typing import Dict

@dataclass
class Document:
    text: str
    metadata: Dict = None

@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float

@dataclass
class RAGResponse:
    content: str
    confidence_score: float
    
@dataclass
class QueryStatus:
    status: str
    detail: str = None