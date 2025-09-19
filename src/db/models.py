from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, ARRAY, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from .session import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    collections = relationship("Collection", back_populates="user", cascade="all, delete-orphan")

class Collection(Base):
    __tablename__ = "collections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    init_date = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    doc_count = Column(Integer, default=0)
    archived = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="collections")
    messages = relationship("Message", back_populates="collection", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    role = Column(Text, nullable=False)
    message = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=True)  # Only for assistant messages
    keywords = Column(ARRAY(Text), nullable=True)  # Only for assistant messages
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    
    # Check constraint for sender
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="check_role_type"),
    )
    
    # Relationships
    collection = relationship("Collection", back_populates="messages")