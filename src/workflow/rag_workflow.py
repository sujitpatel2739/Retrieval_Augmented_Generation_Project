from typing import Optional, Tuple, List, Dict, Any
from ..models import RAGResponse, QueryIntent
from ..components import (
    UniversalExtractor,
    NoiseRemover,
    SmartAdaptiveChunker,
    LLMRequestRouter,
    LLMQueryReformulator,
    VecOperator,
    LLMAnswerGenerator
)
from .base import BaseWorkflow
from ..logger.base import StepLog
from ..logger.json_logger import JsonLogger
from ..config import Settings
import uuid
import os

# Configure logging to write to stdout
logger = JsonLogger()

class Workflow(BaseWorkflow):
    # Class-level attributes
    extractor = None
    noise_remover = None
    chunker = None
    router = None
    reformulator = None
    vec_operator = None
    answer_generator = None
    completion_threshold = None

    @classmethod
    def init(
        cls,
        completion_threshold: float = 0.7,
        OPENAI_API_KEY: str = None,
        tokenizer_model_name: str = "bert-base-uncased",
        similarity_threshold: float = 0.80
    ):
        """Initialize the workflow pipeline (call once at startup)."""
        settings = Settings()
        cls.extractor = UniversalExtractor()
        cls.noise_remover = NoiseRemover()
        cls.chunker = SmartAdaptiveChunker(tokenizer_model_name, similarity_threshold)
        cls.router = LLMRequestRouter(model=settings.router_model, api_key=OPENAI_API_KEY)
        cls.reformulator = LLMQueryReformulator(model=settings.reformulator_model, api_key=OPENAI_API_KEY)
        cls.vec_operator = VecOperator(embedding_model_name="all-MiniLM-L6-v2"
        )
        cls.answer_generator = LLMAnswerGenerator(model=settings.answer_model, api_key=OPENAI_API_KEY)
        cls.completion_threshold = completion_threshold
        print("[Workflow] Initialized successfully")
    
    @classmethod
    def execute(cls, query: str, top_k: int = 10) -> Tuple[Optional[RAGResponse], List[StepLog]]:
        step_logs: List[StepLog] = []
    
        # Step 1: Reformulate
        reformulated, reform_log = cls.reformulator.execute(query)
        logger.log_step(reform_log)
        step_logs.append(reform_log)

        # Step 2: Retrieve
        results, retrieve_log = cls.vec_operator.execute(reformulated.refined_text, top_k=top_k)
        logger.log_step(retrieve_log)
        step_logs.append(retrieve_log)

        # Step 3: Generate Answer
        context = " ".join([result.text for result in results])
        response, generate_log = cls.answer_generator.execute(query, context)
        logger.log_step(generate_log)
        step_logs.append(generate_log)

        return response, step_logs

    @classmethod
    def create_collection(title: str, user_id: Any):
        VecOperator.create_collection(user_id)
    
    @classmethod
    def process_document(cls, collection_name: str, file: Any, max_token_len: int = 256, min_token_len: int = 4) -> Dict:
        """Process and store document chunks in vector DB."""
        print("[Workflow] process_document called")
        
        filename = file.filename
        extension = os.path.splitext(filename)[1].lower()

        if extension not in [".pdf", ".docx", ".txt", ".html", ".htm"]:
            return {"status": "ERROR", "status_code": 400, "detail": f"Unsupported file type: {extension}"}

        try:
            file_bytes = file.read()
        except Exception as e:
            return {"status": "EXCEPTION", "status_code": 400, "detail": f"Cannot read Bytes: {e}"}

        # Step 1: Extract
        document_blocks = cls.extractor.execute(file_bytes, extension)
        # Step 2: Clean
        cleaned_blocks = cls.noise_remover.execute(document_blocks, min_words=min_token_len)
        # Step 3: Chunk
        chunks, embeddings = cls.chunker.execute(cleaned_blocks, max_token_len, min_token_len)
        # Step 4: Store in vector DB
        cls.vec_operator.add_documents(collection_name, chunks, embeddings)

        return {"status": "SUCCESS", "response": f"Inserted {len(chunks)} chunks"}

    @classmethod
    def delete_collection(cls, collection_name: str) -> Dict:
        """Delete a collection from the vector DB."""
        if cls.vec_operator:
            return cls.vec_operator.delete_collection(collection_name)
        else:
            return {"status": "ERROR", "detail": "Vector operator not initialized."}

    @classmethod
    def shutdown(cls):
        """Graceful shutdown."""
        if cls.vec_operator:
            cls.vec_operator.close_connection()
        print("[Workflow] Vector DB connection closed!")
        
        