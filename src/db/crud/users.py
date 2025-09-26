from sqlalchemy.orm import Session
from ..models import User
from ...core.security import hash_password, verify_password
import uuid

def create_user(db: Session, name: str, email: str, password: str) -> User:
    db_user = User(
        name=name,
        email=email,
        password_hash=hash_password(password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_id(db: Session, user_id):
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email=email)
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
