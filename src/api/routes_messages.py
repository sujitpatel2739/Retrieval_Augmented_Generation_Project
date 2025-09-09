from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from ..db.session import get_db
from ..db.crud import messages as crud_messages
from ..db.crud import collections as crud_collections

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

@router.post("/", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def create_message(message: MessageCreate, db: Session = Depends(get_db)):
    """Add a message to a collection"""
    # Verify collection exists
    db_collection = crud_collections.get_collection_by_id(db, collection_id=message.collection_id)
    if not db_collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    return crud_messages.create_message(
        db=db,
        collection_id=message.collection_id,
        sender=message.sender,
        message=message.message,
        confidence_score=message.confidence_score,
        keywords=message.keywords,
        sources=message.sources
    )

@router.get("/{collection_id}/", response_model=List[MessageResponse])
def get_messages(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    """List all messages in a collection"""
    # Verify collection exists
    db_collection = crud_collections.get_collection_by_id(db, collection_id=collection_id)
    if not db_collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    
    return crud_messages.get_messages_by_collection_id(db, collection_id=collection_id)

@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(message_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a single message"""
    success = crud_messages.delete_message(db, message_id=message_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )