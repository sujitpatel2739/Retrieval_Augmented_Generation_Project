from typing import Optional, Tuple, List, Dict, Any
import sys
from ..models import RAGResponse, QueryIntent
from ..components import (
    BaseRequestRouter,
    BaseQueryReformulator,
    BaseRetriever,
    BaseCompletionChecker,
    BaseAnswerGenerator
)
from .base import BaseWorkflow
from ..logger.base import StepLog
from ..logger.json_logger import JsonLogger
# Configure logging to write to stdout
logger = JsonLogger()

class RAGWorkflow(BaseWorkflow):
    def __init__(
        self,
        router: BaseRequestRouter,
        reformulator: BaseQueryReformulator,
        retriever: BaseRetriever,
        completion_checker: BaseCompletionChecker,
        answer_generator: BaseAnswerGenerator,
        completion_threshold: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="rag_workflow", metadata=metadata)
        self.router = router
        self.reformulator = reformulator
        self.retriever = retriever
        self.completion_checker = completion_checker
        self.answer_generator = answer_generator
        self.completion_threshold = completion_threshold
    
    def _execute(self, query: str) -> Tuple[Optional[RAGResponse], List[StepLog]]:
        step_logs: List[StepLog] = []
        
        # Route
        intent, route_log = self.router.execute(query)
        logger.log_step(route_log)

        step_logs.append(route_log)
        if intent != QueryIntent.ANSWER:
            return None, step_logs
        
        # Reformulate
        reformulated, reform_log = self.reformulator.execute(query)
        logger.log_step(reform_log)
        step_logs.append(reform_log)
        
        # Retrieve
        context, retrieve_log = self.retriever.execute(
            reformulated.refined_text,
            reformulated.keywords
        )
        step_logs.append(retrieve_log)
        logger.log_step(retrieve_log)
        # Check completion
        completion_score, check_log = self.completion_checker.execute(query, context)
        logger.log_step(check_log)
        step_logs.append(check_log)

        if completion_score < self.completion_threshold:
            return None, step_logs
        
        # Generate answer
        response, generate_log = self.answer_generator.execute(query, context)
        logger.log_step(generate_log)

        step_logs.append(generate_log)
        return response, step_logs 