from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from sqlalchemy import func

from app.db.database import get_db
from app.models.models import User, UserType, CrawlerActivity
from app.schemas.user import UserResponse, UserAdminUpdate
from app.schemas.dashboard import AdminDashboard
from app.schemas.response import ResponseModel
from app.crud import users as users_crud
from app.crud import dashboard as dashboard_crud
from app.utils.auth import get_current_admin
from app.utils.crawler import crawl_historical_data
from app.utils.scheduler import run_crawler_once
from app.utils.response import response_success, response_error
from app.crud.dashboard import get_recent_crawler_activities as get_crawler_activities

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)

@router.get("/dashboard", response_model=ResponseModel)
def get_admin_dashboard(
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """获取管理员仪表盘数据"""
    data = dashboard_crud.get_admin_dashboard_data(db, current_user.id)
    return response_success(data=data)

@router.post("/users/{user_id}", response_model=ResponseModel)
def update_user(
    user_id: int, 
    user_update: UserAdminUpdate,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """管理员更新用户信息"""
    # 不允许更改自己的用户类型
    if user_id == current_user.id and user_update.user_type == UserType.USER:
        return response_error(msg="管理员不能降级自己的权限", code=400)
    
    updated_user = users_crud.admin_update_user(db, user_id, user_update)
    if not updated_user:
        return response_error(msg="用户不存在", code=404)
    
    return response_success(data=updated_user, msg="用户更新成功")

@router.delete("/users/{user_id}", response_model=ResponseModel)
def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """管理员删除用户"""
    # 不允许删除自己
    if user_id == current_user.id:
        return response_error(msg="管理员不能删除自己的账号", code=400)
    
    success = users_crud.delete_user(db, user_id)
    if not success:
        return response_error(msg="用户不存在", code=404)
    
    return response_success(msg="用户删除成功")

@router.post("/crawl/run-once", response_model=ResponseModel)
async def admin_run_crawler_once(
    current_user: User = Depends(get_current_admin),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """立即执行一次爬虫任务"""
    # 始终在后台运行爬虫任务，避免请求超时和异步取消错误
    background_tasks.add_task(run_crawler_once)
    return response_success(msg="爬虫任务已在后台启动")

@router.post("/crawl/historical", response_model=ResponseModel)
def admin_crawl_historical_data(
    request_data: dict,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """爬取历史数据"""
    try:
        # 从请求体中获取日期
        start_date = request_data.get("start_date")
        end_date = request_data.get("end_date")
        
        if not start_date:
            return response_error(msg="开始日期是必须的", code=400)
        
        # 解析日期
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.now().date()
        
        # 检查日期范围
        if start > end:
            return response_error(msg="开始日期不能晚于结束日期", code=400)
        
        # 限制日期范围（最多一年）
        if (end - start).days > 365:
            return response_error(msg="日期范围不能超过一年", code=400)
        
        # 检查日期是否在未来
        today = datetime.now().date()
        if start > today:
            return response_error(msg="开始日期不能是未来日期", code=400)
        if end > today:
            return response_error(msg="结束日期不能是未来日期", code=400)
        
        # 计算需要爬取的天数
        delta = end - start
        days_to_crawl = delta.days + 1
        
        # 记录爬虫活动状态为"处理中"
        from app.crud.dashboard import create_crawler_activity
        title = f"历史数据爬取 ({start} 至 {end})"
        description = f"正在爬取{days_to_crawl}天的历史数据..."
        activity = create_crawler_activity(
            db=db,
            title=title,
            description=description,
            status="processing",
            records_count=0,
            duration=0
        )
        
        # 获取活动ID
        processing_activity_id = activity.id if hasattr(activity, 'id') else None
        
        # 在后台运行爬虫任务
        if background_tasks:
            background_tasks.add_task(crawl_historical_data, start, end, processing_activity_id)
            return response_success(msg=f"历史数据爬取任务已在后台启动，时间范围：{start} 至 {end}")
        else:
            # 直接运行（会阻塞请求）
            crawl_historical_data(start, end, processing_activity_id)
            return response_success(msg=f"历史数据爬取任务已完成，时间范围：{start} 至 {end}")
    
    except ValueError:
        return response_error(msg="日期格式无效，请使用YYYY-MM-DD格式", code=400)
    except KeyError:
        return response_error(msg="请求数据格式不正确", code=400)

@router.post("/users/create-admin", response_model=ResponseModel)
def create_admin_user(
    user_create: UserAdminUpdate,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """创建管理员账号（超级管理员专用）"""
    # 仅允许用户名为"admin"的超级管理员创建其他管理员
    if current_user.username != "admin":
        return response_error(msg="只有超级管理员可以创建其他管理员账号", code=403)
    
    # 检查用户名是否已存在
    existing_user = users_crud.get_user_by_username(db, user_create.username)
    if existing_user:
        return response_error(msg="用户名已被注册", code=400)
    
    # 创建管理员用户
    from app.schemas.user import UserCreate
    user_data = UserCreate(username=user_create.username, password=user_create.password)
    new_admin = users_crud.create_admin_user(db, user_data)
    
    return response_success(data=new_admin, msg=f"管理员用户 '{new_admin.username}' 创建成功")

@router.get("/crawler/activities", response_model=ResponseModel)
def get_recent_crawler_activities(
    limit: int = Query(10, ge=1, le=100, description="返回记录数量"),
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """获取最近的爬虫活动记录"""
    
    activities = get_crawler_activities(db, limit=limit)
    
    # 转换为前端需要的格式
    formatted_activities = []
    for activity in activities:
        status_map = {
            "success": "success",
            "error": "error",
            "warning": "warning",
            "processing": "processing"
        }
        
        # 支持字典或对象访问
        if isinstance(activity, dict):
            title = activity.get("title")
            description = activity.get("description")
            time_val = activity.get("time")
            status = activity.get("status", "success")
        else:
            title = activity.title
            description = activity.description
            time_val = activity.time
            status = activity.status
        
        formatted_activities.append({
            "title": title,
            "description": description,
            "time": time_val.strftime("%Y-%m-%d %H:%M:%S") if time_val else "",
            "status": status_map.get(status, "success")
        })
    
    return response_success(data=formatted_activities)

@router.get("/crawler/status", response_model=ResponseModel)
def get_crawler_status(
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """获取爬虫状态"""
    
    # 检查是否有处于处理中状态的爬虫活动
    processing_activity = db.query(CrawlerActivity).filter(
        CrawlerActivity.status == "processing"
    ).order_by(
        CrawlerActivity.time.desc()
    ).first()
    
    # 获取最近的爬虫活动
    latest_activity = db.query(CrawlerActivity).order_by(
        CrawlerActivity.time.desc()
    ).first()
    
    # 计算今日爬取的数据量
    today = datetime.now().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())
    
    today_data_count = db.query(func.sum(CrawlerActivity.records_count)).filter(
        CrawlerActivity.time.between(today_start, today_end)
    ).scalar() or 0
    
    status_data = {
        "status": "running" if processing_activity else "idle",
        "last_run_time": latest_activity.time.strftime("%Y-%m-%d %H:%M:%S") if latest_activity else None,
        "daily_data_count": today_data_count,
        "latest_activity": {
            "title": latest_activity.title if latest_activity else None,
            "description": latest_activity.description if latest_activity else None,
            "time": latest_activity.time.strftime("%Y-%m-%d %H:%M:%S") if latest_activity else None,
            "status": latest_activity.status if latest_activity else None,
            "records_count": latest_activity.records_count if latest_activity else 0,
            "duration": latest_activity.duration if latest_activity else None
        } if latest_activity else None
    }
    
    return response_success(data=status_data) 