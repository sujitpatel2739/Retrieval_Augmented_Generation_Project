from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from ..db.session import get_db
from ..db.crud import collections as crud_collections
from ..db.crud import users as crud_users
from ..workflow import Workflow

router = APIRouter(prefix="/collections", tags=["collections"])

# Pydantic schemas
class DocumentUpload(BaseModel):
    id: uuid.UUID
    title: str
    name: str
    collection_id: uuid.UUID
    file: UploadFile = File(...)
    user_id: Optional[uuid.UUID] = None
    db: Session = Depends(get_db)
    
class CollectionUpdate(BaseModel):
    title: Optional[str] = None
    doc_count: Optional[int] = None
    archived: Optional[bool] = None

class CollectionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    name: str
    init_date: datetime
    last_updated: datetime
    doc_count: int
    archived: bool
    
    class Config:
        from_attributes = True

def is_user_authenticated(user_id: Optional[uuid.UUID], db: Session) -> bool:
    """Check if user is authenticated (simplified - replace with proper auth logic)"""
    if not user_id:
        return False
    return crud_users.get_user_by_id(db, user_id=user_id) is not None

@router.post("/")
def create_collection(user_id: uuid.UUID, title: str, db: Session = Depends(get_db)):
    """Create a new collection"""
    
    # Always create collection in Weaviate
    wf_response = Workflow.create_collection(
        user_id=user_id
    )
    
    # Only create in PostgreSQL if user is authenticated
    if user_id and is_user_authenticated(user_id, db):
        # Verify user exists
        db_user = crud_users.get_user_by_id(db, user_id=user_id)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        db_collection = crud_collections.create_collection(
            db=db,
            id = wf_response['id'],
            user_id=user_id,
            title=title,
            name = wf_response['name']
        )
        return db_collection
    else:
        # Return Weaviate collection info for non-authenticated users
        return {
            "id": wf_response['id'],
            "title": title,
            'name': wf_response['name'],
            "doc_count": 0,
            "archived": False
        }

@router.get("/{collection_id}")
def get_collection(collection_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get collection details"""
    # Try to get from PostgreSQL first
    db_collection = crud_collections.get_collection_by_id(db, collection_id=collection_id)
    if db_collection:
        return db_collection
    
    # If not in PostgreSQL, get from Weaviate
    weaviate_stats = Workflow.get_collection_stats(collection_id)
    if weaviate_stats:
        return {
            "id": collection_id,
            "name": "Weaviate Collection",
            "doc_count": weaviate_stats.get("doc_count", 0),
            "status": weaviate_stats.get("status", "archived")
        }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Collection not found"
    )

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

@router.patch("/{collection_id}")
def update_collection(
    collection_id: uuid.UUID, 
    collection_update: CollectionUpdate, 
    db: Session = Depends(get_db)
):
    """Update collection metadata"""
    # Only update PostgreSQL if collection exists there
    updated_collection = crud_collections.update_collection(
            db=db,
            collection_id=collection_id,
            name=collection_update.name,
            archived=collection_update.archived,
            doc_count=collection_update.doc_count
        )
    if not updated_collection:
        return {"status": "ERROR", "detail": "Collection not found in PostgreSQL"}
    return updated_collection

@router.post("/{collection_id}/upload")
def upload_documents(new_upload: DocumentUpload):
    """Upload documents to a collection"""
    if new_upload.id == None:
        wf_response = create_collection(new_upload.user_id, new_upload.title)
    # Upload to Weaviate
    upload_response = Workflow.process_document(new_upload.name, new_upload.file)
    
    # Update PostgreSQL if user is authenticated
    if new_upload.user_id and is_user_authenticated(new_upload.user_id, new_upload.db):
        db_collection = crud_collections.get_collection_by_id(Depends(get_db), collection_id=new_upload.id)
        if db_collection:
            # Update document count
            crud_collections.update_collection(
                db=Depends(get_db),
                collection_id=id,
                doc_count = upload_response.get("doc_count", 0) + db_collection.doc_count
            )
            update_response = crud_collections.update_collection(
                collection_id=wf_response['id'],
                doc_count=upload_response.get("doc_count", 0)
            )
        else:
            crud_collections.create_collection(Depends(get_db), id=wf_response['id'], user_id=new_upload.user_id, title=new_upload.title, name=wf_response['name'])
    return {
        "collection_id": wf_response['id'],
        'name': wf_response['name'],
        "doc_count": upload_response['doc_count'],
        "status": upload_response.get("status", "uploaded")
    }

@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: uuid.UUID, collection_name: str, user_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete collection (cascade deletes messages)"""
    # Delete from Weaviate
    wf_response = Workflow.delete_collection(collection_name)
    
    # Delete from PostgreSQL
    if user_id:
        pg_success = crud_collections.delete_collection(db, collection_id=collection_id)
    
    # Consider successful if deleted from either database
    if not pg_success and not wf_response:
        return {"status": "ERROR", "detail": "Collection not found in either database!"}