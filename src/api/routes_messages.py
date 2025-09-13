from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from ..db.session import get_db
from ..db.crud import messages as crud_messages
from ..db.crud import collections as crud_collections
from ..db.crud import users as crud_users
from ..components.db_operator import db_operator

router = APIRouter(prefix="/messages", tags=["messages"])

# Pydantic schemas
class MessageCreate(BaseModel):
    collection_id: uuid.UUID
    sender: str = Field(..., regex="^(user|assistant)$")
    message: str
    confidence_score: Optional[float] = None
    keywords: Optional[List[str]] = None
    sources: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    sender: str
    message: str
    confidence_score: Optional[float] = None
    keywords: Optional[List[str]] = None
    sources: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

def is_user_authenticated(user_id: Optional[uuid.UUID], db: Session) -> bool:
    """Check if user is authenticated (simplified - replace with proper auth logic)"""
    if not user_id:
        return False
    return crud_users.get_user_by_id(db, user_id=user_id) is not None

@router.post("/")
def create_message(message: MessageCreate, user_id: Optional[uuid.UUID] = None, db: Session = Depends(get_db)):
    """Add a message to a collection and get AI response"""
    # Get AI response from Weaviate
    ai_response = db_operator.get_message_response(
        collection_id=message.collection_id,
        user_message=message.message
    )
    
    # Store user message in PostgreSQL if authenticated
    if user_id and is_user_authenticated(user_id, db):
        # Verify collection exists in PostgreSQL
        db_collection = crud_collections.get_collection_by_id(db, collection_id=message.collection_id)
        if not db_collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        
        # Store user message
        user_msg = crud_messages.create_message(
            db=db,
            collection_id=message.collection_id,
            sender=message.sender,
            message=message.message,
            confidence_score=message.confidence_score,
            keywords=message.keywords,
            sources=message.sources
        )
        
        # Store AI response
        ai_msg = crud_messages.create_message(
            db=db,
            collection_id=message.collection_id,
            sender="assistant",
            message=ai_response.get("response", "I couldn't generate a response."),
            confidence_score=ai_response.get("confidence_score"),
            keywords=ai_response.get("keywords"),
            sources=ai_response.get("sources")
        )
        
        return {
            "user_message": user_msg,
            "assistant_message": ai_msg,
            "weaviate_response": ai_response
        }
    else:
        # Return only Weaviate response for non-authenticated users
        return {
            "user_message": {
                "collection_id": message.collection_id,
                "sender": message.sender,
                "message": message.message,
                "status": "weaviate_only"
            },
            "assistant_message": ai_response,
            "status": "non_authenticated_response"
        }

@router.get("/{collection_id}/")
def get_messages(collection_id: uuid.UUID, user_id: Optional[uuid.UUID] = None, db: Session = Depends(get_db)):
    """List all messages in a collection"""
    # Try PostgreSQL first if user is authenticated
    if user_id and is_user_authenticated(user_id, db):
        db_collection = crud_collections.get_collection_by_id(db, collection_id=collection_id)
        if db_collection:
            return crud_messages.get_messages_by_collection_id(db, collection_id=collection_id)
    
    # Get from Weaviate for non-authenticated users or if not in PostgreSQL
    weaviate_messages = db_operator.get_messages_from_collection(collection_id)
    if weaviate_messages:
        return weaviate_messages
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Collection not found or no messages available"
    )

@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(message_id: uuid.UUID, user_id: Optional[uuid.UUID] = None, db: Session = Depends(get_db)):
    """Delete a single message"""
    # Only delete from PostgreSQL if user is authenticated
    if user_id and is_user_authenticated(user_id, db):
        success = crud_messages.delete_message(db, message_id=message_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )
    else:
        # For non-authenticated users, we can't delete individual messages
        # as Weaviate handles collections, not individual messages
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required to delete messages"
        )