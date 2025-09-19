import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from src.db.models import User
from src.core.security import hash_password, verify_password


def create_user(db: Session, name: str, email: str, password: str) -> User:
    """Create a new user (password is hashed before saving)"""
    db_user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Get a user by ID"""
    return db.query(User).filter(User.id == user_id).first()
 

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get all users with pagination"""
    return db.query(User).offset(skip).limit(limit).all()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate a user by email + password.
    Returns the user if valid, otherwise None.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def delete_user(db: Session, user_id: uuid.UUID) -> bool:
    """Delete a user by ID"""
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True
