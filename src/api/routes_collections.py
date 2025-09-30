from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from src.db.session import get_db
from src.db.crud import collections as crud_collections
from src.db.crud import users as crud_users
from src.db.models import User
from src.core.security import get_current_user
from src.workflow import Workflow

router = APIRouter(prefix="/collections", tags=["collections"])

# Pydantic schemas
class CollectionUpdate(BaseModel):
    id: uuid.UUID
    name: str
    title: Optional[str] = None
    doc_count: Optional[int] = None
    archived: Optional[bool] = None

class CollectionResponse(BaseModel):
    id: uuid.UUID
    title: str = 'Untitled chat'
    name: str
    init_date: datetime
    last_updated: datetime
    doc_count: int
    archived: bool
    
    class Config:
        from_attributes = True


def create_collection_in_weaviate(user_id: Optional[uuid.UUID] = None, title: str = 'Untitled chat'):
    """Helper: always create collection in Weaviate and return the weaviate response dict.
    This does NOT create a Postgres record. Postgres creation is performed during the first successful upload
    when the user is authenticated.
    """
    wf_response = Workflow.create_collection(user_id=user_id)
    return wf_response


@router.get("/{collection_id}", response_model=CollectionResponse)
def get_collection(collection_id: uuid.UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or not db:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail= "Authentication required ")

    """Get collection details. Requires authentication and ownership."""
    db_collection = crud_collections.get_collection_by_id(db, id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")

    # Ensure current user owns the collection
    if getattr(db_collection, 'user_id', None) != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this collection")
    


    return db_collection

@router.get("/users/{user_id}/collections/", response_model=List[CollectionResponse])
def get_user_collections(user_id: uuid.UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or not db:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail= "Authentication required ")

    """List collections for a user. Authenticated; only owner may list their collections."""
    # Ensure caller is requesting their own collections
    if str(user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view these collections")

    db_user = crud_users.get_user_by_id(db, user_id=current_user.id)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return crud_collections.get_collections_by_user_id(db, user_id=current_user.id)


@router.patch("/{collection_id}")
def update_collection(
    collection_id: uuid.UUID,
    collection_update: CollectionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update collection metadata. Requires authentication and ownership."""
    if not current_user or not db:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail= "Authentication required ")

    # Validate path/body match
    if str(collection_update.id) != str(collection_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path collection_id and body id must match")

    db_collection = crud_collections.get_collection_by_id(db, id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")

    # Ownership check
    if getattr(db_collection, 'user_id', None) != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this collection")

    updated_collection = crud_collections.update_collection(
        db=db,
        id=collection_id,
        title=collection_update.title,
        archived=collection_update.archived,
        doc_count=collection_update.doc_count,
    )
    if updated_collection:
        return updated_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update collection")


@router.post("/{collection_id}/upload")
def upload_documents(
    collection_id: Optional[uuid.UUID] = None,
    file: UploadFile = File(...),
    title: str = Form('Untitled chat'),
    collection_name: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a file. If the target collection does not exist, create a Weaviate collection.
    Only create a Postgres collection if the user is authenticated. Guest users create Weaviate-only collections which are not saved to Postgres.
    """
    # Validating File Upload
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
    extension = file.filename.split('.')[-1].lower()
    if extension not in ['pdf', 'docx', 'txt', 'md', 'html', 'htm']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type: {extension}")
    # if file.content_type not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown', 'text/html']:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file content type: {file.content_type}")
    if file.spool_max_size and file.spool_max_size > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File too large")
    

    # If collection_id provided, try to get Postgres collection
    db_collection = None
    if collection_id:
        db_collection = crud_collections.get_collection_by_id(db, id=collection_id)
    else:
        # If no collection (guest or new), create Weaviate collection
        # Create in Weaviate (user_id if authenticated else None)
        user_id_for_weaviate = current_user.id if current_user else None
        wf_resp = create_collection_in_weaviate(user_id=user_id_for_weaviate, title=title)
        weav_id = wf_resp.get('id')
        weav_name = wf_resp.get('name')

        # If authenticated, create Postgres record now (on first document upload)
        if current_user:
            db_collection = crud_collections.create_collection(
                db=db,
                id=weav_id,
                user_id=current_user.id,
                title=title,
                name=weav_name,
            )

    # Upload document to Weaviate
    upload_response = Workflow.process_document(collection_name, file, extension=extension)
    
    if upload_response.get('status') == "ERROR":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=upload_response.get('detail', 'Upload error'))

    # If authenticated and db_collection exists, update Postgres doc count
    if current_user and db_collection:
        new_doc_count = upload_response.get('doc_count', 0) + getattr(db_collection, 'doc_count', 0)
        crud_collections.update_collection(
            db=db,
            id=db_collection.id,
            title=title,
            archived=getattr(db_collection, 'archived', False),
            doc_count=new_doc_count,
        )
    else:
        new_doc_count = upload_response.get('doc_count', 0)

    return {
        "id": str(db_collection.id) if db_collection else str(weav_id),
        "title": title,
        'collection_name': db_collection.name if db_collection else weav_name,
        "doc_count": new_doc_count,
        'last_updated': upload_response.get('last_updated')
    }

@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: uuid.UUID, collection_name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete collection. If the collection is Postgres-backed, require authentication and ownership.
    If it's a Weaviate-only (guest) collection, allow deletion without authentication.
    """

    # Delete from Weaviate first
    wf_response = Workflow.delete_collection(collection_name)

    # Must be authenticated and owner
    if str(collection_id).startswith("temp"):
        return None
    if not current_user or not db:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to delete this collection")
    
    # Check Postgres
    db_collection = crud_collections.get_collection_by_id(db, id=collection_id)
    if db_collection:
        if getattr(db_collection, 'user_id', None) != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this collection")
        pg_success = crud_collections.delete_collection(db, id=collection_id)
    else:
        # No Postgres record: guest/temporary collection — deletion of Weaviate is sufficient
        pg_success = False

    if not pg_success and not wf_response:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found in either database")

    return None