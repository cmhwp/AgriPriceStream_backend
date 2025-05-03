from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import User, UserType
from app.schemas.dashboard import UserDashboard
from app.schemas.response import ResponseModel
from app.crud import dashboard as dashboard_crud
from app.utils.auth import get_current_user
from app.utils.response import response_success, response_error

router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=ResponseModel)
def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户仪表盘数据，根据用户角色返回不同数据"""
    # 根据用户角色返回不同的仪表盘数据
    if current_user.user_type == UserType.ADMIN:
        dashboard_data = dashboard_crud.get_admin_dashboard_data(db, current_user.id)
    else:
        dashboard_data = dashboard_crud.get_user_dashboard_data(db, current_user.id)
        
    if "error" in dashboard_data:
        return response_error(msg=dashboard_data["error"], code=404)
    
    return response_success(data=dashboard_data) 