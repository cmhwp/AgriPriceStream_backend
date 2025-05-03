from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import List
from sqlalchemy.orm import Session
from datetime import timedelta

from app.db.database import get_db
from app.models.models import User, UserType
from app.schemas.user import UserCreate, UserResponse, UserUpdate, Token
from app.schemas.response import ResponseModel, paginate
from app.crud import users as users_crud
from app.utils.auth import get_current_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.utils.response import response_success, response_error

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.post("/register", response_model=ResponseModel)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = users_crud.get_user_by_username(db, username=user.username)
    if db_user:
        return response_error(msg="用户名已被注册", code=400)
    new_user = users_crud.create_user(db=db, user=user)
    return response_success(data=new_user, msg="用户注册成功")

@router.post("/token", response_model=ResponseModel)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = users_crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        return response_error(
            msg="用户名或密码错误",
            code=status.HTTP_401_UNAUTHORIZED
        )
    
    # 检查用户是否被封禁
    if not user.is_active:
        return response_error(
            msg="账号已被封禁，请联系管理员",
            code=status.HTTP_403_FORBIDDEN
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return response_success(data={"access_token": access_token, "token_type": "bearer"}, msg="登录成功")

@router.get("/me", response_model=ResponseModel)
def read_users_me(current_user: User = Depends(get_current_user)):
    return response_success(data=current_user)

@router.put("/me", response_model=ResponseModel)
def update_user_me(user_update: UserUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    updated_user = users_crud.update_user(db, current_user.id, user_update)
    return response_success(data=updated_user, msg="用户信息更新成功")

@router.get("/", response_model=ResponseModel)
def read_users(
    page: int = 1, 
    size: int = 10, 
    username: str = None,
    user_type: str = None,
    is_active: bool = None,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # 只有管理员可以查看所有用户
    if current_user.user_type != UserType.ADMIN:
        return response_error(msg="需要管理员权限", code=403)
    
    # 获取总用户数 (考虑所有过滤条件)
    total = users_crud.count_users(db, username=username, user_type=user_type, is_active=is_active)
    
    # 计算跳过的记录数
    skip = (page - 1) * size
    
    # 获取分页的用户数据 (包含所有过滤条件)
    users = users_crud.get_users(db, skip=skip, limit=size, username=username, user_type=user_type, is_active=is_active)
    
    # 使用分页响应工厂函数创建响应
    paginated_data = paginate(items=users, page=page, size=size, total=total)
    
    return response_success(data=paginated_data)

@router.get("/{user_id}", response_model=ResponseModel)
def read_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 用户只能查看自己的信息，管理员可以查看所有用户
    if current_user.id != user_id and current_user.user_type != UserType.ADMIN:
        return response_error(msg="权限不足", code=403)
    
    db_user = users_crud.get_user(db, user_id=user_id)
    if db_user is None:
        return response_error(msg="用户不存在", code=404)
    return response_success(data=db_user)

@router.put("/{user_id}/status", response_model=ResponseModel)
def toggle_user_active_status(
    user_id: int, 
    is_active: bool, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # 只有管理员可以封禁/解封用户
    if current_user.user_type != UserType.ADMIN:
        return response_error(msg="需要管理员权限", code=403)
    
    # 不能封禁自己
    if current_user.id == user_id:
        return response_error(msg="不能操作自己的账号", code=400)
    
    # 封禁/解封用户
    updated_user = users_crud.toggle_user_status(db, user_id, is_active)
    if not updated_user:
        return response_error(msg="用户不存在", code=404)
    
    status_msg = "封禁" if not is_active else "解封"
    return response_success(data=updated_user, msg=f"用户已{status_msg}") 