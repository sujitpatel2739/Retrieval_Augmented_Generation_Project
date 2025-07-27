from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import numpy as np
from sentence_transformers import SentenceTransformer
from ..models import SearchResult, Document
from ..config import Settings
from .base_component import BaseComponent

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

class BaseDBOperator(BaseComponent):
    """Base class for retrieving relevant context"""

    def __init__(self):
        super().__init__(name="retriever")

    def _execute(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Execute retrieval"""
        return self.retrieve(query, top_k = top_k)

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Retrieve relevant context based on query"""
        pass


class DBOperator(BaseDBOperator):
    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()

        settings = Settings()
        self.client = weaviate.connect_to_local(port=settings.weaviate_port, skip_init_checks=True)
        self.embedder = SentenceTransformer(embedding_model_name)
        self.collection = None
        
        print("Weaviate connected: ", self.client.is_ready())
        if self.client.collections.exists(collection_name):
            self.collection = self.client.collections.get(collection_name)
        else:
            self.collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.OBJECT,
                    nested_properties=[
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="token_len", data_type=DataType.INT),
                ]),
                ],
            )
        
    def close(self):
        self.client.close()

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        semantic_results = self.semantic_search(query, top_k=top_k)
        # return self.rerank(semantic_results)
        return semantic_results

    def add_documents(self, documents: List[Document], embeddings: List[Any]) -> None:
        objects = list()
        
        for doc, embedding in zip(documents, embeddings):
            objects.append(wvc.data.DataObject(
            properties = {"text": doc.text,
                          'metadata': doc.metadata},
            vector = embedding.detach().cpu().numpy().tolist()
            ))
        
        self.collection.data.insert_many(objects)
        

    from weaviate.classes.query import MetadataQuery

    def semantic_search(self, query: str, top_k: int = 15) -> List[SearchResult]:
        query_vector = self.embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True)
        query_vector = query_vector.detach().cpu().numpy().tolist()

        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata = wvc.query.MetadataQuery(distance=True)
        )

        results = response.objects
        return [
            SearchResult(
                text=obj.properties['text'],
                metadata=obj.properties.get('metadata'),
                score=1.0 - obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else None
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
    
        collection = self.client.collections.get(self.collection_name)
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


    def rerank(self, semantic_results: List[SearchResult]) -> List[SearchResult]:
        merged = {}

        for result in semantic_results:
            merged[result.text] = result

        return sorted(merged.values(), key=lambda x: x.score, reverse=True)
    
    
