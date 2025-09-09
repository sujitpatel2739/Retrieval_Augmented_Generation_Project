from sqlalchemy.orm import Session
from ..models import Message
from typing import Optional, List, Dict, Any
import uuid

def create_message(
    db: Session,
    collection_id: uuid.UUID,
    sender: str,
    message: str,
    confidence_score: Optional[float] = None,
    keywords: Optional[List[str]] = None,
    sources: Optional[Dict[str, Any]] = None
) -> Message:
    """Create a new message"""
    db_message = Message(
        collection_id=collection_id,
        sender=sender,
        message=message,
        confidence_score=confidence_score,
        keywords=keywords,
        sources=sources
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_messages_by_collection_id(db: Session, collection_id: uuid.UUID) -> List[Message]:
    """Get all messages in a collection"""
    return db.query(Message).filter(Message.collection_id == collection_id).order_by(Message.created_at).all()

def get_message_by_id(db: Session, message_id: uuid.UUID) -> Optional[Message]:
    """Get message by ID"""
    return db.query(Message).filter(Message.id == message_id).first()

def delete_message(db: Session, message_id: uuid.UUID) -> bool:
    """Delete a single message"""
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if not db_message:
        return False
    
    db.delete(db_message)
    db.commit()
    return True