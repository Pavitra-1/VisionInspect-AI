from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "quality_engineer"


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class InspectionResponse(BaseModel):
    id: int
    image_name: str
    result: Optional[str] = None
    defect_type: Optional[str] = None
    confidence: Optional[str] = None
    severity: Optional[str] = None

    class Config:
        from_attributes = True