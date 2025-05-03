from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

from app.models.models import UserType

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None

class UserResponse(UserBase):
    id: int
    user_type: str
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserAdminUpdate(BaseModel):
    user_type: Optional[str] = None
    is_active: Optional[bool] = None 