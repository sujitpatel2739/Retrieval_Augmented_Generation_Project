from abc import abstractmethod
from openai import OpenAI
from typing import List, Dict, Any
from .base_component import BaseComponent
from ..config import Settings
from ..models import SearchResult

class BaseCompletionChecker(BaseComponent):
    """Base class for checking if query can be answered with context"""
    def __init__(self):
        super().__init__(name="completion_checker")
    
    def _execute(self, query: str, context: List[Dict[str, Any]]) -> float:
        """Execute completion check"""
        return self.check_completion(query, context)
    
    @abstractmethod
    def check_completion(self, query: str, context: List[Dict[str, Any]]) -> float:
        """Check if the query can be answered with the given context."""
        pass

class LLMCompletionChecker(BaseCompletionChecker):
    def __init__(self, model: str = "deepseek/deepseek-chat-v3-0324:free", api_key: str = None):
        super().__init__()
        self.client = OpenAI(base_url = Settings().openrouter_base_url, api_key=api_key)
        self.model = model
    
    def check_completion(self, query: str, context: List[SearchResult]) -> float:
        formatted_context = "\n\n".join([
            f"Context {i+1}:\n{result.text}"
            for i, result in enumerate(context)
        ])
        
        prompt = """Analyze if the given context contains sufficient information to answer the query.
        Return ONLY a float number between 0 and 1, where:
        - 1.0 means the context perfectly contains all needed information
        - 0.0 means the context has no relevant information
        - Values in between represent partial information
        
        Consider:
        - Relevance of context to the query
        - Completeness of information
        - Reliability of information
        
        Context:
        {context}
        
        Query: {query}
        
        Score (0.0-1.0):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user", 
                "content": prompt.format(context=formatted_context, query=query)
            }],
            temperature=0,
            max_tokens=10
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError as e:
            print(f"ERROR: {e}")
            return 0.0  # Default to 0