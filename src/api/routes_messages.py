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
    role: str= "user"
    query: str
    top_k: int = 10
    
    class Config:
        from_attributes = True


@router.post("/")
def create_message(
    new_query: Query
):
    """Add a query to a collection and get AI response. Only persist to Postgres when authenticated."""
    current_user: Optional[User] = Depends(get_current_user)
    db: Session = Depends(get_db)
    # Get AI response from Weaviate
    rag_response = Workflow.get_rag_response(
        query=new_query.query,
        collection_name=new_query.collection_name,
        top_k=new_query.top_k,
    )
    if not rag_response or "content" not in rag_response:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating response from RAG service")
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
            role=new_query.role,
            content=new_query.query,
        )

        # Store AI response
        ai_msg = crud_messages.create_message(
            db=db,
            collection_id=new_query.collection_id,
            role="assistant",
            content=rag_response.get("content", ""),
            confidence_score=rag_response.get("confidence_score", 0.49),
            keywords=rag_response.get("keywords", []),
        )

        return {"user_message": user_msg, "assistant_message": ai_msg}

    # Guest: return only generated response (not persisted)
    return {
        "user_message": {
            "id": str(uuid.uuid4()),
            "collection_id": new_query.collection_id,
            "collection_name": new_query.collection_name,
            "content": new_query.query,
        },
        "assistant_message": {
            "id": str(uuid.uuid4()),
            "collection_id": new_query.collection_id,
            "collection_name": new_query.collection_name,
            "content": rag_response.get("content", "Unable to get response!"),
            "confidence_score": rag_response.get("confidence_score", 0.49),
            "keywords": rag_response.get("keywords", []),
        },
    }
        

@router.get("/{collection_id}/")
def get_messages(
    collection_id: uuid.UUID
):
    """List all messages in a collection. Only available from Postgres for authenticated users."""
    current_user: Optional[User] = Depends(get_current_user)
    db: Session = Depends(get_db)
    if not current_user:
        # Guest users do not have messages in Postgres
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to list messages")

    db_collection = crud_collections.get_collection_by_id(db, id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    if getattr(db_collection, 'user_id', None) != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view messages for this collection")

    return crud_messages.get_messages_by_collection_id(db, collection_id=collection_id)


@router.delete("/{collection_id}/{user_message_id}/{assistant_message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(
    user_message_id: uuid.UUID,
    assistant_message_id: uuid.UUID,
    collection_id: uuid.UUID,
):
    """Delete a pair of messages (user + assistant). Requires authentication and ownership."""
    current_user: Optional[User] = Depends(get_current_user)
    db: Session = Depends(get_db)
    # Verify authentication provided by dependency
    # Attempt to delete both messages
    user_msg = crud_messages.get_message_by_id(db, message_id=user_message_id)
    ai_msg = crud_messages.get_message_by_id(db, message_id=assistant_message_id)
    if str(collection_id) != str(user_msg.collection_id) or str(collection_id) != str(ai_msg.collection_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Message does not belong to the specified collection")
    
    user_msg_del = crud_messages.delete_message(db, message_id=user_message_id)
    ai_msg_del = crud_messages.delete_message(db, message_id=assistant_message_id)
    
    if not user_msg_del or not ai_msg_del:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complete message pair not found!")
    return None