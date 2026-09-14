from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="quality_engineer")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    result = Column(String(50), nullable=True)
    defect_type = Column(String(100), nullable=True)
    confidence = Column(String(50), nullable=True)
    severity = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())