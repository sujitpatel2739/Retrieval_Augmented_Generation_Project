from abc import ABC, abstractmethod
from typing import List, Any, Dict
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from ..models import SearchResult, Document
from ..config import Settings
from .base_component import BaseComponent
from datetime import datetime
import os
import json
import uuid

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

class BaseVecOperator(BaseComponent):
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
    

class VecOperator(BaseVecOperator):
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()

        settings = Settings()
        self.client = weaviate.connect_to_local(port=settings.weaviate_port, skip_init_checks=True)
        self.embedder = SentenceTransformer(embedding_model_name)
        
        print("Weaviate connected: ", self.client.is_ready())
        
    
    def create_collection(self, user_id: str) -> Dict[str, Any]:
        """Create a new collection in Weaviate with specified schema."""
        if self.client.collections.exists(collection_name):
            return {"status": "ERROR", "detail": f"Collection {collection_name} already exists!"}
        
        collection_id = str(uuid.uuid4())
        if not user_id:
            collection_id = 'temp_' + str(uuid.uuid4())
            collection_name = collection_id
        else:
            collection_name = user_id[:12] + '_' + collection_id[:12]
        
        new_collection = self.client.collections.create(
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

        return {"status": "SUCCESS", "name": collection_name, "id": collection_id}
        
         
    def change_collection(self, curr_collection_name: str, collection_name: str, collection_id: str, user_id: str) -> None:
        """Switch to an existing collection in Weaviate."""
        if user_id[:6] =='guest_':
                self.delete_collection(curr_collection_name, collection_id, user_id)
        if self.client.collections.exists(collection_name):
            self.collections_cache[user_id][collection_id] = self.client.collections.get(collection_name)
            return 
        else:
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        
    def get_collections(self, collection_name) -> Dict[str, Any]:
        """Retrieve all collections from Weaviate."""
        if not self.client.is_ready():
            return {'status': 'INTERNAL_ERROR', 'detail': 'Weaviate client is not ready!'}
        
        collections = self.client.collections.get(collection_name)
        if not collections:
            return {'status': 'ERROR', 'detail': 'No collections found! Please create a new collection.'}
        
        return collections
        
        
    def delete_collection(self, collection_name: str) -> None:
        
        """Delete a collection in Weaviate."""
        
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)
            return {"status": "SUCCESS", "detail": f"Collection {collection_name} deleted."}
        else:
            return {"status": "ERROR", "detail": f"Collection {collection_name} does not exist."}
            

    def close_connection(self, collection_name: str, collection_id: str) -> None:
        """Close the Weaviate client connection."""
        if collection_id[:5] == 'temp_':
            self.delete_collection(collection_name)
        self.dump_collection()
        self.client.close()
        
        
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        semantic_results = self.semantic_search(query, top_k=top_k)
        # return self.rerank(semantic_results)
        return semantic_results

    def add_documents(self, collection_name: str, documents: List[Document], embeddings: List[Any]) -> Dict[str, Any]:
        """Add documents to the Weaviate collection."""
        
        objects = list()
        
        for doc, embedding in zip(documents, embeddings):
            objects.append(wvc.data.DataObject(
            properties = {"text": doc.text,
                          'metadata': doc.metadata},
            vector = embedding.detach().cpu().numpy().tolist()
            ))
        collection = self.client.collections.get(collection_name)
        if not collection:
            return {"status": "ERROR", "detail": f"Collection {collection_name} does not exist!"}
        self.client.insert_many(objects)
        
        return {"status": "SUCCESS", 'doc_count': len(documents), "last_updated": datetime.now().isoformat()}
        

    def semantic_search(self, query: str, top_k: int, collection_id: str, user_id: str) -> List[SearchResult]:
        """Perform semantic search using Weaviate."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True)
        query_vector = query_vector.detach().cpu().numpy().tolist()

        response = self.collections_cache[user_id][collection_id].query.near_vector(
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
        

    def keyword_search(self, keywords: List[str], top_k: int, user_id: str, collection_id: str) -> List[SearchResult]:
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
    
        response = (
            self.collections_cache[user_id][collection_id].query
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
        """Rerank results by merging semantic and keyword search results."""
        merged = {}

        for result in semantic_results:
            merged[result.text] = result

        return sorted(merged.values(), key=lambda x: x.score, reverse=True)
    
