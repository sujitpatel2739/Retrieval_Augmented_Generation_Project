# Legit-RAG

A modular Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant, and OpenAI.

## System Components

- [Components](src/components/README.md) - Individual RAG components
- [Workflow Components](src/workflow/README.md) - RAG workflow implementation
- [Logging System](src/logging/README.md) - Event logging and visualization

## Workflow Components

The system follows a 5-step RAG workflow:

1. **Query Routing** (`router.py`)
   - Determines if a query can be answered (ANSWER), needs clarification (CLARIFY), or should be rejected (REJECT)
   - Uses LLM to make intelligent routing decisions
   - Extensible through `BaseRequestRouter` interface

2. **Query Reformulation** (`reformulator.py`)
   - Refines the original query for better retrieval
   - Extracts keywords for hybrid search
   - Implements `BaseQueryReformulator` for custom reformulation strategies

3. **Context Retrieval** (`retriever.py`)
   - Performs hybrid search combining:
     - Semantic search using embeddings
     - Keyword-based search
   - Currently uses Qdrant for vector storage
   - Extensible through `BaseRetriever` interface

4. **Completion Check** (`completion_checker.py`)
   - Evaluates if retrieved context is sufficient to answer the query
   - Returns confidence score
   - Customizable threshold through configuration
   - Implements `BaseCompletionChecker` interface

5. **Answer Generation** (`answer_generator.py`)
   - Generates final response using retrieved context
   - Includes relevant citations
   - Provides confidence scoring
   - Extensible through `BaseAnswerGenerator` interface

## Extensibility

The system is designed for easy extension and modification:

1. **LLM Providers**
   - Currently uses OpenAI
   - Can be extended to support other providers (Anthropic, Bedrock, etc.)
   - Each component uses abstract base classes for provider independence

2. **Vector Databases**
   - Currently implements Qdrant
   - Can be extended to support other vector DBs (Pinecone, Weaviate, etc.)
   - Abstract `BaseRetriever` interface for new implementations

3. **Document Management**
   - Flexible document model with metadata support
   - Extensible for different document types and sources

4. **Search Strategies**
   - Hybrid search combining semantic and keyword approaches
   - Customizable result merging strategies
   - Extensible for additional search methods

## Setup and Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key

### Setup Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/legit-rag.git
cd legit-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-key-here
```

## Running the System

Start our API server and the Qdrant vector database:
```bash
docker-compose up -d
```

- The API will be available at [`http://localhost:8000`](http://localhost:8000)
- The Qdrant db will be available at [`http://localhost:6333`](http://localhost:6333)

To run the API server directly (i.e. in a debugger), note that&mdash;after stopping it in Docker&mdash;it may be run with:

```bash
python -m src.api
```

## API Endpoints

### Add Documents
```bash
POST /documents
{
    "documents": [
        {
            "text": "Your document text here",
            "metadata": {"source": "wiki", "topic": "example"}
        }
    ]
}
```

### Query
```bash
POST /query
{
    "query": "Your question here"
}
```

## Example Usage

```python
import requests

# Add documents
docs = {
    "documents": [
        {
            "text": "Example document text",
            "metadata": {"source": "example"}
        }
    ]
}
response = requests.post("http://localhost:8000/documents", json=docs)

# Query
query = {
    "query": "What does the document say?"
}
response = requests.post("http://localhost:8000/query", json=query)
print(response.json())
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Key configuration options in `config.py`:
- LLM models for each component
- Vector DB settings
- Completion threshold
- API endpoints and ports

## Future Enhancements

1. Provider-agnostic LLM interface
2. Support for streaming responses
3. Additional vector database implementations
4. Enhanced document preprocessing
5. Caching layer for frequent queries
6. Batch document processing
7. Advanced result ranking strategies

