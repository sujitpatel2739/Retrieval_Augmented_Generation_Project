from abc import abstractmethod
from typing import List, Dict, Any
from openai import OpenAI
from .base_component import BaseComponent
from ..config import Settings
from ..models import RAGResponse, Citation, SearchResult
import json

class BaseAnswerGenerator(BaseComponent):
    """Base class for generating answers from context"""
    def __init__(self):
        super().__init__(name="answer_generator")
    
    def _execute(self, query: str, context: List[Dict[str, Any]]) -> RAGResponse:
        """Execute answer generation"""
        return self.generate_answer(query, context)
    
    @abstractmethod
    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> RAGResponse:
        """Generate an answer using the retrieved context."""
        pass

class LLMAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model: str = "deepseek/deepseek-chat-v3-0324:free", api_key: str = None):
        super().__init__()
        self.client = OpenAI(base_url = Settings().openrouter_base_url, api_key=api_key)
        self.model = model
    
    def generate_answer(self, query: str, context: List[SearchResult]) -> RAGResponse:
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Context {i+1}:\n{result.text}"
            for i, result in enumerate(context)
        ])
        
        prompt = """Using the provided context, answer the question. Your response must be in JSON format with these fields:
        1. "answer": <Your detailed response>
        2. "citations": <A list of objects, each with:
           - "text": The relevant quote from the context
           - "relevance_score": A float between 0-1 indicating how relevant this citation is>
        3. "confidence_score": <A float between 0-1 indicating your overall confidence>
        
        Only use information from the provided context. If you're unsure, reflect that in the confidence score.
        
        Context:
        {context}
        
        Question: {query}
        
        Respond with only the JSON object, no other text."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user", 
                "content": prompt.format(context=formatted_context, query=query)
            }],
            temperature=0,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        parsed = json.loads(result)
        
        citations = [
            Citation(
                text=cite["text"],
                metadata={},  # Could be enhanced with context metadata/Either  remove this entirely
                relevance_score=cite["relevance_score"]
            )
            for cite in parsed["citations"]
        ]
        
        return RAGResponse(
            answer=parsed["answer"],
            citations=citations,
            confidence_score=parsed["confidence_score"]
        ) 