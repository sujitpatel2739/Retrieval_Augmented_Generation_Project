from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

from ..db.session import get_db
from ..db.crud import collections as crud_collections
from ..db.crud import users as crud_users

router = APIRouter(prefix="/collections", tags=["collections"])

# Pydantic schemas
class CollectionCreate(BaseModel):
    user_id: uuid.UUID
    name: str

class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    active: Optional[bool] = None
    doc_count: Optional[int] = None

class CollectionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    init_date: datetime
    last_updated: datetime
    doc_count: int
    active: bool
    
    class Config:
        from_attributes = True

@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
def create_collection(collection: CollectionCreate, db: Session = Depends(get_db)):
    """Create a new collection"""
    # Verify user exists
    db_user = crud_users.get_user_by_id(db, user_id=collection.user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return crud_collections.create_collection(
        db=db,
        user_id=collection.user_id,
        name=collection.name
    )

@router.get("/{collection_id}", response_model=CollectionResponse)
def get_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get collection details"""
    db_collection = crud_collections.get_collection_by_id(db, collection_id=collection_id)
    if db_collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    return db_collection

@router.get("/users/{user_id}/collections/", response_model=List[CollectionResponse])
def get_user_collections(user_id: uuid.UUID, db: Session = Depends(get_db)):
    """List collections for a user"""
    # Verify user exists
    db_user = crud_users.get_user_by_id(db, user_id=user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return crud_collections.get_collections_by_user_id(db, user_id=user_id)

@router.patch("/{collection_id}", response_model=CollectionResponse)
def update_collection(
    collection_id: uuid.UUID, 
    collection_update: CollectionUpdate, 
    db: Session = Depends(get_db)
):
    """Update collection metadata"""
    db_collection = crud_collections.update_collection(
        db=db,
        collection_id=collection_id,
        name=collection_update.name,
        active=collection_update.active,
        doc_count=collection_update.doc_count
    )
    if db_collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    return db_collection

@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete collection (cascade deletes messages)"""
    success = crud_collections.delete_collection(db, collection_id=collection_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )