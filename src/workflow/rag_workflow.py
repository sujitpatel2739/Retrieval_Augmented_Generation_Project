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
        collection_name = str(Settings().weaviate_primary_class),
        OPENAI_API_KEY = None,
        tokenizer_model_name = "bert-base-uncased",
        similarity_threshold = 0.80
    ):
        super().__init__(name="rag_workflow_primary")
        # Initialize components
        settings = Settings()
        self.extractor = UniversalExtractor()
        self.noise_remover = NoiseRemover()
        self.chunker = SmartAdaptiveChunker(tokenizer_model_name, similarity_threshold)
        self.router = LLMRequestRouter(model=settings.router_model, api_key=OPENAI_API_KEY)
        self.reformulator = LLMQueryReformulator(model=settings.reformulator_model, api_key=OPENAI_API_KEY)
        self.completion_checker = LLMCompletionChecker(model=settings.completion_model, api_key=OPENAI_API_KEY)
        self.db_operator = DBOperator(collection_name=collection_name, embedding_model_name = "all-MiniLM-L6-v2")
        self.answer_generator = LLMAnswerGenerator(model=settings.answer_model, api_key=OPENAI_API_KEY)
        self.completion_threshold = completion_threshold
    
    def _execute(self, query: str, top_k: int = 10) -> Tuple[Optional[RAGResponse], List[StepLog]]:
        step_logs: List[StepLog] = []
        
        # Reformulate
        reformulated, reform_log = self.reformulator.execute(query)
        print(reformulated)
        logger.log_step(reform_log)
        step_logs.append(reform_log)
        
        # Retrieve
        results, retrieve_log = self.db_operator.execute(
            reformulated.refined_text,
            top_k = top_k
        )
        print(results)
        step_logs.append(retrieve_log)
        logger.log_step(retrieve_log)
        
        # Generate answer
        context = "".join([result.text + "." for result in results])
        print("context:", context)
        
        response, generate_log = self.answer_generator.execute(query, context)
        logger.log_step(generate_log)

        step_logs.append(generate_log)
        return response, step_logs
    
    
    def process_document(self, file_bytes: bytes, extension, max_token_len: int = 256, min_token_len: int = 4):
        """Accept and process files of type PDF, DOCX, TXT, HTML."""
        print('process_document called')
        # try:
        document_blocks = self.extractor.execute(file_bytes, extension)
        cleaned_blocks = self.noise_remover.execute(document_blocks, min_words = min_token_len)
        chunks, embeddings = self.chunker.execute(cleaned_blocks, max_token_len, min_token_len)
        
        print(chunks)
        print(len(chunks))
        print(embeddings)
        self.db_operator.add_documents(chunks, embeddings)
        return {"status": "SUCCESS", "response": f"Inserted {len(chunks)} chunks"}
    
    def shutdown(self):
        self.db_operator.close_connection()
        print("DB connection closed!")
        
        
        