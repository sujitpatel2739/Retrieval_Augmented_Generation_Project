from sqlalchemy.orm import Session
from ..models import Collection
from typing import Optional, List
import uuid
from sqlalchemy.sql import func

def create_collection(db: Session, user_id: uuid.UUID, name: str) -> Collection:
    """Create a new collection"""
    db_collection = Collection(
        user_id=user_id,
        name=name
    )
    db.add(db_collection)
    db.commit()
    db.refresh(db_collection)
    return db_collection

def get_collection_by_id(db: Session, collection_id: uuid.UUID) -> Optional[Collection]:
    """Get collection by ID"""
    return db.query(Collection).filter(Collection.id == collection_id).first()

def get_collections_by_user_id(db: Session, user_id: uuid.UUID) -> List[Collection]:
    """Get all collections for a specific user"""
    return db.query(Collection).filter(Collection.user_id == user_id).all()

def update_collection(
    db: Session, 
    collection_id: uuid.UUID, 
    name: str = None,
    active: bool = None,
    doc_count: int = None
) -> Optional[Collection]:
    """Update collection metadata"""
    db_collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not db_collection:
        return None
    
    if name is not None:
        db_collection.name = name
    if active is not None:
        db_collection.active = active
    if doc_count is not None:
        db_collection.doc_count = doc_count
    
    # Update last_updated timestamp
    db_collection.last_updated = func.now()
    
    db.commit()
    db.refresh(db_collection)
    return db_collection

def delete_collection(db: Session, collection_id: uuid.UUID) -> bool:
    """Delete a collection (cascade deletes messages)"""
    db_collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not db_collection:
        return False
    
    db.delete(db_collection)
    db.commit()
    return True