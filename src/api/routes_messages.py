from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import uuid
from datetime import datetime

from ..db.session import get_db
from ..db.crud import messages as crud_messages
from ..db.crud import collections as crud_collections
from ..db.crud import users as crud_users
from ..workflow import Workflow

router = APIRouter(prefix="/query", tags=["query"])

# Pydantic schemas
class Query(BaseModel):
    collection_id: uuid.UUID
    collection_name: str
    sender: str= "user"
    query: str
    top_k: int = 10
    
    class Config:
        from_attributes = True

def is_user_authenticated(user_id: Optional[uuid.UUID], db: Session) -> bool:
    """Check if user is authenticated (simplified - replace with proper auth logic)"""
    if not user_id:
        return False
    return crud_users.get_user_by_id(db, user_id=user_id) is not None

@router.post("/")
def create_message(new_query: Query, user_id: Optional[uuid.UUID] = None, db: Session = Depends(get_db)):
    """Add a query to a collection and get AI response"""
    # Get AI response from Weaviate
    rag_response = Workflow.get_rag_response(
        query=new_query.query,
        collection_name=new_query.collection_name,
        top_k=new_query.top_k
    )
    
    # Store user query in PostgreSQL if authenticated
    if user_id and is_user_authenticated(user_id, db):
        # Verify collection exists in PostgreSQL
        db_collection = crud_collections.get_collection_by_id(db, collection_id=new_query.collection_id)
        if not db_collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        
        # Store user query
        user_msg = crud_messages.create_message(
            db=db,
            collection_id=new_query.collection_id,
            sender= new_query.sender,
            sender=new_query.sender,
            message=new_query.query,
        )
        
        # Store AI response
        ai_msg = crud_messages.create_message(
            db=db,
            collection_id=new_query.collection_id,
            sender="assistant",
            message=rag_response.get("answer", "Error generating response!"),
            confidence_score=rag_response.get("confidence_score", 0.49),
            keywords=rag_response.get("keywords", []),
        )
        
        return {
            "user_message": user_msg,
            "assistant_message": ai_msg,
        }
    else:
        # Return only Weaviate response for non-authenticated users
        return {
            "user_message": {
                'id': str(uuid.uuid4()),
                "collection_id": new_query.collection_id,
                'collection_name': new_query.collection_name,
                "query": new_query.query,
            },
            "assistant_message": {
                'id': str(uuid.uuid4()),
                "collection_id": new_query.collection_id,
                'collection_name': new_query.collection_name,
                'message': rag_response.get("answer", "Error generating response!"),
                "confidence_score": rag_response.get("confidence_score", 0.49),
                "keywords": rag_response.get("keywords", []),
            }
        }
        

@router.get("/{collection_id}/")
def get_messages(collection_id: uuid.UUID, user_id: Optional[uuid.UUID] = None, db: Session = Depends(get_db)):
    """List all messages in a collection"""
    # Try PostgreSQL first if user is authenticated
    if user_id and is_user_authenticated(user_id, db):
        db_collection = crud_collections.get_collection_by_id(db, collection_id=collection_id)
        if db_collection:
            return crud_messages.get_messages_by_collection_id(db, collection_id=collection_id)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Collection not found or no messages available"
    )


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(message_id: Tuple, user_id: Optional[uuid.UUID] = None, db: Session = Depends(get_db)):
    """Delete a single message by ID"""
    # Only delete from PostgreSQL if user is authenticated
    if user_id and is_user_authenticated(user_id, db):
        user_msg_del = crud_messages.delete_message(db, message_id=message_id[0])
        ai_msg_del = crud_messages.delete_message(db, message_id=message_id[1])
        if not user_msg_del or not ai_msg_del:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complete message pair not found!"
            )
    else:
        # For non-authenticated users, we can't delete individual messages
        # as Weaviate handles collections, not individual messages
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required to delete messages"
        )