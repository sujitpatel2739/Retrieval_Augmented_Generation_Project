from typing import Optional, Tuple, List, Dict, Any
from ..models import RAGResponse, QueryIntent
from ..components import (
    UniversalExtractor,
    NoiseRemover,
    SmartAdaptiveChunker,
    LLMRequestRouter,
    LLMQueryReformulator,
    DBOperator,
    LLMCompletionChecker,
    LLMAnswerGenerator
)
from .base import BaseWorkflow
from ..logger.base import StepLog
from ..logger.json_logger import JsonLogger
from ..config import Settings

# Configure logging to write to stdout
logger = JsonLogger()

class RAGWorkflow(BaseWorkflow):
    def __init__(
        self,
        completion_threshold: float = 0.7,
        vector_db_class = Settings().weaviate_primary_class,
        OPENAI_API_KEY = None,
        tokenizer_model_name = "bert-base-uncased",
        max_tokens = 32,
        min_tokens = 4,
        similarity_threshold = 0.80
    ):
        super().__init__(name="rag_workflow_primary")
        # Initialize components
        settings = Settings(),
        self.extractor = UniversalExtractor(),
        self.noise_remover = NoiseRemover(),
        self.chunker = SmartAdaptiveChunker(tokenizer_model_name, max_tokens, min_tokens, similarity_threshold),
        self.router = LLMRequestRouter(model=settings.router_model, api_key=OPENAI_API_KEY)
        self.reformulator = LLMQueryReformulator(model=settings.reformulator_model, api_key=OPENAI_API_KEY)
        self.completion_checker = LLMCompletionChecker(model=settings.completion_model, api_key=OPENAI_API_KEY)
        self.db_operator = DBOperator(class_name=settings.weaviate_class_name, url=settings.weaviate_url)
        self.answer_generator = LLMAnswerGenerator(model=settings.answer_model, api_key=OPENAI_API_KEY)
        self.completion_threshold = completion_threshold,
        self.vector_db_class = vector_db_class
    
    def _execute(self, query: str) -> Tuple[Optional[RAGResponse], List[StepLog]]:
        step_logs: List[StepLog] = []
        
        # Route
        intent, route_log = self.router.execute(query)
        logger.log_step(route_log)

        step_logs.append(route_log)
        if intent != QueryIntent.ANSWER:
            return {"status": "ERROR", "status_code": 400, "detail": f"INDETIFIED_QUERY_INTENT: {intent}"}, step_logs
        
        # Reformulate
        reformulated, reform_log = self.reformulator.execute(query)
        logger.log_step(reform_log)
        step_logs.append(reform_log)
        
        # Retrieve
        context, retrieve_log = self.db_operator.execute(
            reformulated.refined_text,
            reformulated.keywords
        )
        step_logs.append(retrieve_log)
        logger.log_step(retrieve_log)
        
        # # Check completion
        # completion_score, check_log = self.completion_checker.execute(query, context)
        # logger.log_step(check_log)
        # step_logs.append(check_log)

        # if completion_score < self.completion_threshold:
        #     return {"status": "ERROR", "status_code": 400, "detail": "INSUFFICIENT_CONTEXT"}, step_logs
        
        # Generate answer
        response, generate_log = self.answer_generator.execute(query, context)
        logger.log_step(generate_log)

        step_logs.append(generate_log)
        return response, step_logs
    
    
    def _process_document(self, file: Any):
        """Accept and process files of type PDF, DOCX, TXT, HTML."""
        filename = file.filename
        try:
            document_blocks = self.extractor.execute(file, filename)
            cleaned_blocks = self.noise_remover.execute(document_blocks, min_words = 4)
            chunks = self.chunker.execute(cleaned_blocks)

            response = self.db_operator.add_documents(chunks)

            return {"status": "SUCCESS", "response": response}

        except Exception as e:
            return {"status": "EXCEPTION", "status_code": 400, "detail": str(e)}
        