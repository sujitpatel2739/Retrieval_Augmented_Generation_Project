from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime

from src.db.session import get_db
from src.db.crud import messages as crud_messages
from src.db.crud import collections as crud_collections
from src.db.models import User
from src.core.security import get_current_user
from src.workflow import Workflow

router = APIRouter(prefix="/query", tags=["query"])

# Pydantic schemas
class Query(BaseModel):
    collection_id: uuid.UUID
    collection_name: str
    query: str
    top_k: int = 10
    temperature: float = 0.7
    maxTokens: int = 150
    includeMetadata: bool = True
    enableReranking: bool = False

    class Config:
        from_attributes = True
        
class RetryRequest(BaseModel):
    collection_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    query: str
    collection_name: str
    top_k: int = 10
    temperature: float = 0.7
    maxTokens: int = 150
    includeMetadata: bool = True
    enableReranking: bool = False

    class Config:
        from_attributes = True

@router.post("/")
def create_message(
    new_query: Query,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a query to a collection and get AI response. Only persist to Postgres when authenticated."""
    # Get AI response from Weaviate
    rag_response = Workflow.get_rag_response(
        query=new_query.query,
        collection_name=new_query.collection_name,
        top_k=new_query.top_k,
        temperature=new_query.temperature,
        maxTokens=new_query.maxTokens,
        includeMetadata=new_query.includeMetadata,
        enableReranking=new_query.enableReranking,
    )
    if not rag_response or not rag_response.answer:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating response from RAG service")
    # Ensure confidence_score is float or None, never empty string
    cs = rag_response.confidence_score
    if cs == "" or cs is None:
        cs = None
    kw = rag_response.keywords
    if kw == "" or kw is None:
        kw = []
    # If authenticated, store messages in Postgres
    if current_user:
        # Verify collection exists in PostgreSQL and belongs to user
        db_collection = crud_collections.get_collection_by_id(db, id=new_query.collection_id)
        if not db_collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        if getattr(db_collection, 'user_id', None) != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to add messages to this collection")

        # Store user query
        user_msg = crud_messages.create_message(
            db=db,
            collection_id=new_query.collection_id,
            role='user',
            content=new_query.query,
        )
        # Store AI response
        ai_msg = crud_messages.create_message(
            db=db,
            collection_id=new_query.collection_id,
            role="assistant",
            content=rag_response.answer,
            confidence_score=cs,
            keywords=kw,
        )

    return {
        "collection_id": new_query.collection_id,
        "collection_name": new_query.collection_name,
        "user_message": {
            "id": user_msg.id,
            "content": new_query.query,
            "created_at": user_msg.created_at.isoformat() if hasattr(user_msg, 'created_at') and user_msg.created_at else datetime.utcnow().isoformat(),
        },
        "assistant_message": {
            "id": ai_msg.id,
            "content": rag_response.answer,
            "confidence_score": cs,
            "keywords": kw,
            "created_at": ai_msg.created_at.isoformat() if hasattr(ai_msg, 'created_at') and ai_msg.created_at else datetime.utcnow().isoformat(),
        },
    }
    
@router.post("/retry")
def retry_message(
    retry_req: RetryRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retry a message pair: get new AI response, update both user and assistant messages in DB."""
    # Get new AI response
    rag_response = Workflow.get_rag_response(
        query=retry_req.query,
        collection_name=retry_req.collection_name,
        top_k=retry_req.top_k,
        temperature=retry_req.temperature,
        maxTokens=retry_req.maxTokens,
        includeMetadata=retry_req.includeMetadata,
        enableReranking=retry_req.enableReranking,
    )
    if not rag_response or not rag_response.answer:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating response from RAG service")

    # Ensure confidence_score is float or None, never empty string
    cs = rag_response.confidence_score
    if cs == "" or cs is None:
        cs = None
    kw = rag_response.keywords
    if kw == "" or kw is None:
        kw = []
    # Authenticated: update messages in Postgres
    if current_user:
        db_collection = crud_collections.get_collection_by_id(db, id=retry_req.collection_id)
        if not db_collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        if getattr(db_collection, 'user_id', None) != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update messages in this collection")

        # Update user message (content)
        updated_user_msg = crud_messages.update_message(db, message_id=retry_req.user_message_id, content=retry_req.query,
                                                        confidence_score=None,
                                                        keywords=None)
        if not updated_user_msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User message not found")

        # Update assistant message (content, confidence_score, keywords)
        updated_ai_msg = crud_messages.update_message(db, message_id=retry_req.assistant_message_id,
                                        content=rag_response.answer,
                                        confidence_score=cs,
                                        keywords = kw)
        if not updated_ai_msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant message not found")

    return {
        "collection_id": retry_req.collection_id,
        "collection_name": retry_req.collection_name,
        "user_message": {
            "id": retry_req.user_message_id,
            "content": retry_req.query,
            "created_at": updated_user_msg.created_at.isoformat() if hasattr(updated_user_msg, 'created_at') and updated_user_msg.created_at else datetime.utcnow().isoformat(),
        },
        "assistant_message": {
            "id": retry_req.assistant_message_id,
            "content": rag_response.answer,
            "confidence_score": cs,
            "keywords": kw,
            "created_at": updated_ai_msg.created_at.isoformat() if hasattr(updated_ai_msg, 'created_at') and updated_ai_msg.created_at else datetime.utcnow().isoformat(),
        },
    }
        

@router.get("/{collection_id}/")
def get_messages(
    collection_id: uuid.UUID,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all messages in a collection. Only available from Postgres for authenticated users."""
    if not current_user:
        # Guest users do not have messages in Postgres
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to list messages")

    db_collection = crud_collections.get_collection_by_id(db, id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    if getattr(db_collection, 'user_id', None) != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view messages for this collection")

    messages = crud_messages.get_messages_by_collection_id(db, collection_id=collection_id)
    # Always return valid ISO timestamps
    def serialize_message(msg):
        return {
            "id": msg.id,
            "collection_id": msg.collection_id,
            "role": msg.role,
            "content": msg.content,
            "confidence_score": msg.confidence_score,
            "keywords": msg.keywords,
            "created_at": msg.created_at.isoformat() if hasattr(msg, 'created_at') and msg.created_at else datetime.utcnow().isoformat(),
        }
    return [serialize_message(m) for m in messages]


@router.delete("/{collection_id}/{user_message_id}/{assistant_message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(
    collection_id: uuid.UUID,
    user_message_id: uuid.UUID,
    assistant_message_id: uuid.UUID,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a pair of messages (user + assistant). Requires authentication and ownership."""
    # Verify authentication provided by dependency
    if not current_user or not db:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Authentication failed!")
    # Attempt to delete both messages
    user_msg = crud_messages.get_message_by_id(db, message_id=user_message_id)
    ai_msg = crud_messages.get_message_by_id(db, message_id=assistant_message_id)
    if str(collection_id) != str(user_msg.collection_id) or str(collection_id) != str(ai_msg.collection_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Message does not belong to the specified collection")
    
    user_msg_del = crud_messages.delete_message(db, message_id=user_message_id)
    ai_msg_del = crud_messages.delete_message(db, message_id=assistant_message_id)
    
    if not user_msg_del or not ai_msg_del:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complete message pair not found!")
    return 