from abc import abstractmethod
from typing import List, Dict, Any
from openai import OpenAI
from .base_component import BaseComponent
from ..config import Settings
from ..models import RAGResponse, SearchResult
import json
from pydantic import ValidationError, BaseModel

class BaseAnswerGenerator(BaseComponent):
    """Base class for generating answers from context"""
    def __init__(self):
        super().__init__(name="answer_generator")

    def _execute(self, query: str, context: str, temperature: float, max_tokens: int) -> RAGResponse:
        """Execute answer generation"""
        return self.generate_answer(query, context, temperature, max_tokens)

    @abstractmethod
    def generate_answer(self, query: str, context: str, temperature: float, max_tokens: int) -> RAGResponse:
        """Generate an answer using the retrieved context."""
        pass

class LLMAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model: str = "deepseek/deepseek-chat-v3-0324:free", api_key: str = None):
        super().__init__()
        self.client = OpenAI(base_url = Settings().OPENROUTER_BASE_URL, api_key=api_key)
        self.model = model

    def generate_answer(self, query: str, context: str, temperature: float, max_tokens: int) -> RAGResponse:
        """
        Generate an answer using the LLM with safe JSON parsing, schema validation,
        retry on failure, and graceful fallback.
        """

        base_prompt = """Using the provided context, answer the question. 
            Your response must be in JSON format with these fields:

            {{
                "answer": <Your detailed text response>,
                "confidence_score": <A float between 0-1 indicating your overall confidence of the accurate answer (only if answer is possible from context, otherwise empty)>,
                "keywords": <A python list containing important keywords from the answer (only if answer is possible from context, otherwise empty)>
            }}

            Only use information from the provided context. 
            If you're unsure, reflect that in the confidence score.

            Context:
            {context}

            Query: {query}

            Respond with only the JSON object, no other text."""

        def call_llm(prompt: str) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=None
            )
            result = response.choices[0].message.content
            return result.replace("```json", "").replace("```", "").strip()

        # === First Attempt ===
        raw_output = call_llm(base_prompt.format(context=context, query=query))

        try:
            parsed = json.loads(raw_output)
            return RAGResponse(**parsed)

        except (json.JSONDecodeError, ValidationError):
            # === Retry Attempt ===
            fix_prompt = f"""The following text was supposed to be valid JSON but was not: 

            {raw_output}

            Please fix it so that it strictly matches this schema:
            {{
                "answer": string,
                "confidence_score": float between 0 and 1,
                "keywords": Python list
            }}
            Respond with only the corrected JSON object.
            """

            raw_retry = call_llm(fix_prompt)

            try:
                parsed_retry = json.loads(raw_retry)
                return RAGResponse(**parsed_retry)
            except (json.JSONDecodeError, ValidationError):
                # === Final Fallback ===
                return RAGResponse(
                    answer="Unable to generate a valid structured response.",
                    confidence_score=0.0,
                    keywords=[]
                )
        