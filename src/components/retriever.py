from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import weaviate
import numpy as np
from sentence_transformers import SentenceTransformer
from ..models import SearchResult, Document
from ..config import Settings
from .base_component import BaseComponent

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

class BaseRetriever(BaseComponent):
    """Base class for retrieving relevant context"""

    def __init__(self):
        super().__init__(name="retriever")

    def _execute(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Execute retrieval"""
        return self.retrieve(query, keywords)

    @abstractmethod
    def retrieve(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Retrieve relevant context based on query and keywords."""
        pass


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        class_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        url: Optional[str] = None
    ):
        super().__init__()

        settings = Settings()
        self.client = weaviate.connect_to_local(port=settings.weaviate_port, skip_init_checks=True)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.class_name = class_name

    def retrieve(self, query: str, keywords: List[str]) -> List[SearchResult]:
        semantic_results = self.semantic_search(query)
        keyword_results = self.keyword_search(keywords)
        return self.rerank(semantic_results, keyword_results)

    def add_documents(self, documents: List[Document]) -> None:
        collection = self.client.collections.get(self.class_name)
    
        objects_to_add = []
        vectors_to_add = []
    
        for doc in documents:
            embedding = self._get_embedding(doc.text).tolist()
            properties = {"text": doc.text}
            if doc.metadata:
                properties.update(doc.metadata)
    
            objects_to_add.append(properties)
            vectors_to_add.append(embedding)
    
        # Batch import using v4 API
        collection.data.insert_many(objects=objects_to_add, vectors=vectors_to_add)

    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_vector = self._get_embedding(query).tolist()

        collection = self.client.collections.get(self.class_name)
        response = collection.query.near_vector(query_vector).with_limit(top_k).with_additional(["distance"]).fetch()

        results = response.objects

        return [
            SearchResult(
                text=obj.properties.get("text", ""),
                metadata=obj.properties,
                score=1.0 - obj.additional.get("distance", 1.0)  # Convert distance to similarity
            )
            for obj in results
        ]

    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[SearchResult]:
        if not keywords:
            return []
    
        # Build the where filter using v4 schema
        where_filter = {
            "operator": "Or",
            "operands": [
                {
                    "path": ["text"],
                    "operator": "Like",
                    "valueText": f"%{keyword}%"
                } for keyword in keywords
            ]
        }
    
        collection = self.client.collections.get(self.class_name)
        response = (
            collection.query
            .where(where_filter)
            .with_limit(top_k)
            .fetch()
        )
    
        results = response.objects
    
        keyword_results = []
        for obj in results:
            text = obj.properties.get("text", "")
            lower_text = text.lower()
            keyword_count = sum(lower_text.count(keyword.lower()) for keyword in keywords)
            score = min(1.0, keyword_count / 5) if keyword_count else 0.1
    
            keyword_results.append(
                SearchResult(
                    text=text,
                    metadata=obj.properties,
                    score=score
                )
            )
    
        return keyword_results

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def rerank(self, semantic_results: List[SearchResult], keyword_results: List[SearchResult]) -> List[SearchResult]:
        merged = {}

        for result in semantic_results:
            merged[result.text] = result

        for result in keyword_results:
            if result.text not in merged or result.score > merged[result.text].score:
                merged[result.text] = result

        return sorted(merged.values(), key=lambda x: x.score, reverse=True)
