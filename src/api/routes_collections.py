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
    id: uuid.UUID
    name: str
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
def get_collection(collection_id: uuid.UUID, user_id: uuid_UUID, db: Session = Depends(get_db)):
    """Get collection details"""
    if not is_user_authenticated(user_id, db):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
    db_collection = crud_collections.get_collection_by_id(db, collection_id=collection_id)
    if db_collection:
        return db_collection
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Collection not found"
    )

@router.get("/users/{user_id}/collections/", response_model=List[CollectionResponse])
def get_user_collections(user_id: uuid.UUID, db: Session = Depends(get_db)):
    """List collections for a user"""
    if not is_user_authenticated(user_id, db):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
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
    collection_update: CollectionUpdate, 
    db: Session = Depends(get_db)
):
    """Update collection metadata"""
    # Only update PostgreSQL if collection exists there
    updated_collection = crud_collections.update_collection(
            db=db,
            collection_id=id,
            name=collection_update.name,
            archived=collection_update.archived,
            doc_count=collection_update.doc_count
        )
    if updated_collection:
        return updated_collection
    return {"status": "ERROR", "detail": "Collection not found in PostgreSQL"}


@router.post("/{collection_id}/upload")
def upload_documents(new_upload: DocumentUpload):
    """Upload documents to a collection"""
    id = new_upload.id
    coll_name = new_upload.name
    if new_upload.id == None:
        createcoll_response = create_collection(new_upload.user_id, new_upload.title)
        id = createcoll_response['id']
        coll_name = createcoll_response['name']
        
    # Upload to Weaviate
    upload_response = Workflow.process_document(new_upload.name, new_upload.file)
    if upload_response['status'] != "ERROR":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=upload_response['detail']
        )
    
    # Update PostgreSQL if user is authenticated
    if new_upload.user_id and is_user_authenticated(new_upload.user_id, new_upload.db):
        db_collection = crud_collections.get_collection_by_id(Depends(get_db), collection_id=new_upload.id)
        if db_collection:
            # Update document count
            update_response=crud_collections.update_collection(
                db=Depends(get_db),
                id=createcoll_response['id'],
                title=createcoll_response['title'],
                name=createcoll_response['name'],
                doc_count = upload_response.get("doc_count", 0) + db_collection.doc_count,
                archived=new_upload.archived
            )

    return {
        "id": id,
        'title': new_upload.title,
        'name': coll_name,
        "doc_count": upload_response['doc_count'],
        'last_updated': upload_response['last_updated']
    }

@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: uuid.UUID, collection_name: str, user_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete collection (cascade deletes messages)"""

    # Delete from Weaviate
    wf_response = Workflow.delete_collection(collection_name)
    
    # Delete from PostgreSQL
    if user_id and not is_user_authenticated(user_id, db):
        pg_success = crud_collections.delete_collection(db, collection_id=collection_id)
        return {"status": "SUCCESS", "detail": "Collection deleted."}
    
    # Consider successful if deleted from either database
    if not pg_success and not wf_response:
        return {"status": "ERROR", "detail": "Collection not found in either database!"}