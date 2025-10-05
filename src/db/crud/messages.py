from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from ..models import Message
from typing import Optional, List, Dict, Any
import uuid

def create_message(
    db: Session,
    collection_id: uuid.UUID,
    role: str,
    content: str,
    confidence_score: Optional[float] = None,
    keywords: Optional[List[str]] = None,
) -> Message:
    """Create a new message"""
    db_message = Message(
        collection_id=collection_id,
        role=role,
        content=content,
        confidence_score=confidence_score,
        keywords=keywords,
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def update_message(db: Session, message_id: uuid.UUID, content: str, confidence_score: Optional[float], keywords: Optional[List[str]]) -> Message:
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        return None
    message.content = content
    message.confidence_score = confidence_score if confidence_score else message.confidence_score
    message.keywords = keywords if keywords else message.keywords
    message.created_at = func.now()
    
    db.commit()
    db.refresh(message)
    return message

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

def delete_messages_by_collection(db: Session, collection_id: uuid.UUID) -> int:
    """Delete all messages linked to a particular collection.
    Returns the number of messages deleted.
    """
    deleted_count = db.query(Message).filter(Message.collection_id == collection_id).delete(synchronize_session=False)
    db.commit()
    return deleted_count
    