from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
import json
from .base_component import BaseComponent
from ..config import Settings

@dataclass
class ReformulatedQuery:
    refined_text: str
    keywords: List[str]

class BaseQueryReformulator(BaseComponent, ABC):
    """Base class for query reformulation"""
    def __init__(self):
        super().__init__(name="reformulator")
    
    def _execute(self, query: str) -> ReformulatedQuery:
        """Execute reformulation"""
        return self.reformulate(query)
    
    @abstractmethod
    def reformulate(self, query: str) -> ReformulatedQuery:
        """Reformulate the query and generate keywords."""
        pass

class LLMQueryReformulator(BaseQueryReformulator):
    def __init__(self, model: str = "deepseek/deepseek-chat-v3-0324:free", api_key: str = None):
        super().__init__()
        self.client = OpenAI(base_url = Settings().OPENROUTER_BASE_URL, api_key=api_key)
        self.model = model
    
    def reformulate(self, query: str) -> ReformulatedQuery:
        import re
        prompt = f"""Given the user query, reformulate it to correct any typos and extract key search terms.\nReturn your response in this JSON format:\n{{\n    \"refined_query\": <reformulated question>,\n    \"keywords\": <[\"key1\", \"key2\", \"key3\"]>\n}}\nOnly return the JSON object, no other text.\nUser Query: {query}"""
        last_exception = None
        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            try:
                return ReformulatedQuery(
                    refined_text=json.loads(content)["refined_query"],
                    keywords=json.loads(content)["keywords"]
                )
            except Exception as e:
                last_exception = e
                # Try to extract JSON substring using regex
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(0))
                        return ReformulatedQuery(
                            refined_text=result.get("refined_query", query),
                            keywords=result.get("keywords", [])
                        )
                    except Exception:
                        continue
        # If all attempts fail, raise last exception
        raise ValueError(f"Could not parse LLM output as JSON after retries: {last_exception}\nOutput: {content}")