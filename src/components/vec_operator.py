from abc import abstractmethod
from typing import List, Any, Dict, Optional
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from ..models import SearchResult, Document
from ..config import Settings
from .base_component import BaseComponent
from datetime import datetime
import uuid
from cachetools import TTLCache
from threading import Lock
import random
import string

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

class BaseVecOperator(BaseComponent):
    """Base class for retrieving relevant context"""

    def __init__(self):
        super().__init__(name="retriever")

    def _execute(self, query: str, top_k: int, collection_name: str) -> List[SearchResult]:
        """Execute retrieval"""
        return self.retrieve(query, top_k, collection_name)

    @abstractmethod
    def retrieve(self, query: str, top_k: int, collection_name: str) -> List[SearchResult]:
        """Retrieve relevant context based on query""" 
        pass
    

class VecOperator(BaseVecOperator):
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()

        settings = Settings()
        # connect to Weaviate
        self.client = weaviate.connect_to_local(port=settings.WEAVIATE_PORT, skip_init_checks=True)
        self.embedder = SentenceTransformer(embedding_model_name)
        # small in-process TTL cache for collection wrappers returned by the client
        self._collections_cache = TTLCache(maxsize=256, ttl=300)
        self._cache_lock = Lock()

        logging.info("Weaviate connected: %s", self.client.is_ready())
        
    
    def get_unique_name(self, length, seed=None):
        if seed is not None:
            random.seed(seed)
        gk = [
            "Alpha","Beta","Gamma","Delta","Epsilon","Zeta","Eta","Theta","Iota","Kappa",
            "Lambda","Mu","Nu","Xi","Omicron","Pi","Rho","Sigma","Tau","Upsilon",
            "Phi","Chi","Psi","Omega","Astra","Boreas","Helios","Nyx","Aether","Cosmos","Lyra","Vega","Nova","Quasar",
            "Zenith","Pulse","Novae","Nimbus","Echo","Vertex","Ion","Flux","Aegis","VertexPrime",
            "Hyperion","Nebula","Cosmo"
        ]   
        alphabet = string.ascii_letters + string.digits
        return random.choice(gk) + '_'.join(random.choice(alphabet) for _ in range(length))


    def create_collection(self, user_id: Optional[uuid.UUID]) -> Dict[str, Any]:
        """Create a new collection in Weaviate with specified schema."""
        collection_id = uuid.uuid4()
        collection_name = self.get_unique_name(16)
        if not user_id:
            collection_id = 'temp_' + str(collection_id)
            collection_name = 'temp_' + collection_name
        
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

        # cache the new collection wrapper so subsequent queries are fast
        try:
            self.get_or_cache_collection(collection_name)
            return {'status': 'SUCCESS', 'id': collection_id, 'name': collection_name}
        except Exception as e:
            # if fetching fails, ignore caching — operations will refetch on demand
            logging.exception("Failed to cache new collection %s", collection_name)
            return {'status': 'EXCEPTION', 'detail': f'{str(e)}'}
         
    def change_collection(self, collection_name: str):
        """Return collection wrapper for an existing collection (cached).

        Raises ValueError if collection doesn't exist.
        """
        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready")
        
        return self.get_or_cache_collection(collection_name) 
        
        
    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Delete a collection in Weaviate and invalidate cache."""
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)
            # invalidate cache
            self.invalidate_collection_cache(collection_name)
            return {"status": "SUCCESS", "detail": f"Collection {collection_name} deleted."}
        else:
            return {"status": "ERROR", "detail": f"Collection {collection_name} does not exist."}
            

    def close_connection(self, collection_name: Optional[str]) -> None:
        """Close the Weaviate client connection."""
        if collection_name:
            self.delete_collection(collection_name)
        self.client.close()
        
        
    def retrieve(self, query: str, top_k: int, collection_name: str) -> List[SearchResult]:
        semantic_results = self.semantic_search(query, top_k=top_k, collection_name=collection_name)
        # return self.rerank(semantic_results)
        return semantic_results

    def add_documents(self, collection_name: str, documents: List[Document], embeddings: List[Any]) -> Dict[str, Any]:
        """Add documents to the Weaviate collection."""
        
        objects = list()
        for doc, embedding in zip(documents, embeddings):
            # Only include allowed properties, pass vector separately
            objects.append({
                'properties': {"text": doc.text,
                              'metadata': doc.metadata},
                'vector': embedding.detach().cpu().numpy().tolist()
            })
        # ensure collection exists and use cached wrapper
        try:
            collection = self.get_or_cache_collection(collection_name)
        except Exception:
            if not self.client.collections.exists(collection_name):
                return {"status": "ERROR", "detail": f"Collection {collection_name} does not exist!"}
        
        # Collection-level insert
        try:
            with collection.batch.fixed_size(batch_size=50) as batch:
                for obj in objects:
                    # Pass vector as separate argument, not in properties
                    batch.add_object(obj['properties'], vector=obj['vector'])
                    if batch.number_errors > 50:
                        return {"status": "ERROR", "detail": "Batch import stopped due to excessive errors."}
            failed_objects = collection.batch.failed_objects
            if failed_objects:
                return {
                    "status": "ERROR",
                    "detail": f"Number of failed imports: {len(failed_objects)}."
                }
        except Exception as e:
            logging.exception(f"Failed to insert objects into collection %s {collection_name}; error: %s: {str(e)}")
            return {"status": "EXCEPTION", "detail": f"{str(e)}"}
        return {"status": "SUCCESS", 'doc_count': len(documents), "last_updated": datetime.now().isoformat()}
        

    def semantic_search(self, query: str, top_k: int, collection_name: str) -> List[SearchResult]:
        """Perform semantic search using Weaviate."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True)
        query_vector = query_vector.detach().cpu().numpy().tolist()

        collection = self.get_or_cache_collection(collection_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True)
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
        

    def keyword_search(self, keywords: List[str], top_k: int, user_id: str, collection_name: str) -> List[SearchResult]:
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
    
        # get cached collection wrapper by collection_id (collection name string expected)
        collection = self.get_or_cache_collection(collection_name)
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
        """Rerank results by merging semantic and keyword search results."""
        merged = {}

        for result in semantic_results:
            merged[result.text] = result

        return sorted(merged.values(), key=lambda x: x.score, reverse=True)
    
    # ----------------------------------------------------------------
    # Cache helpers
    # ----------------------------------------------------------------
    def get_or_cache_collection(self, collection_name: str):
        """Return a cached collection wrapper or fetch and cache it."""
        # fetch and cache
        if self.client.collections.exists(collection_name):
            with self._cache_lock:
                cached = self._collections_cache.get(collection_name)
                if cached is not None:
                    return cached
                coll = self.client.collections.use(collection_name)
                self._collections_cache[collection_name] = coll
                return coll
            
        raise ValueError(f"Collection {collection_name} does not exist.")


    def invalidate_collection_cache(self, collection_name: str):
        with self._cache_lock:
            self._collections_cache.pop(collection_name, None)
    
