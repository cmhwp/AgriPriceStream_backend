from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime

from app.models.models import User, UserType
from app.utils.auth import get_password_hash, verify_password
from app.schemas.user import UserCreate, UserUpdate, UserAdminUpdate

def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100, username: str = None, user_type: str = None, is_active: bool = None) -> List[User]:
    query = db.query(User)
    
    # 如果提供了用户名，添加过滤条件
    if username:
        query = query.filter(User.username.like(f"%{username}%"))
    
    # 如果提供了用户类型，添加过滤条件
    if user_type:
        query = query.filter(User.user_type == user_type)
    
    # 如果提供了用户状态，添加过滤条件
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    return query.offset(skip).limit(limit).all()

def count_users(db: Session, username: str = None, user_type: str = None, is_active: bool = None) -> int:
    """获取用户总数"""
    query = db.query(func.count(User.id))
    
    # 如果提供了用户名，添加过滤条件
    if username:
        query = query.filter(User.username.like(f"%{username}%"))
    
    # 如果提供了用户类型，添加过滤条件
    if user_type:
        query = query.filter(User.user_type == user_type)
    
    # 如果提供了用户状态，添加过滤条件
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    return query.scalar()

def create_user(db: Session, user: UserCreate) -> User:
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        hashed_password=hashed_password,
        user_type=UserType.USER,
        created_at=datetime.now()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_admin_user(db: Session, user: UserCreate) -> User:
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        hashed_password=hashed_password,
        user_type=UserType.ADMIN,
        created_at=datetime.now()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    update_data = user_update.dict(exclude_unset=True)
    
    # 如果更新包含密码，则哈希处理
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

def admin_update_user(db: Session, user_id: int, user_update: UserAdminUpdate) -> Optional[User]:
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    update_data = user_update.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int) -> bool:
    db_user = get_user(db, user_id)
    if not db_user:
        return False
    
    db.delete(db_user)
    db.commit()
    return True

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user 

def toggle_user_status(db: Session, user_id: int, is_active: bool) -> Optional[User]:
    """封禁或解封用户"""
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    db_user.is_active = is_active
    db.commit()
    db.refresh(db_user)
    return db_user