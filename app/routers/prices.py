from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.models import User, UserType
from app.schemas.price import PriceRecordResponse, PriceRecordCreate, PriceRecordUpdate
from app.schemas.price import VegetablePriceHistory, ChartData, RealTimePriceResponse
from app.schemas.response import ResponseModel, paginate
from app.crud import prices as prices_crud
from app.crud import vegetables as vegetables_crud
from app.utils.auth import get_current_user, get_current_admin
from app.utils.response import response_success, response_error

router = APIRouter(
    prefix="/prices",
    tags=["prices"],
    responses={404: {"description": "Not found"}},
)

@router.get("/records", response_model=ResponseModel)
def read_price_records(
    vegetable_id: Optional[int] = None,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页记录数"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取价格记录列表，支持分页"""
    # 计算总记录数
    total_records = prices_crud.get_price_records_count(db, vegetable_id=vegetable_id)
    
    # 计算分页偏移量
    offset = (page - 1) * page_size
    
    # 获取当前页数据
    records = prices_crud.get_price_records(
        db, 
        vegetable_id=vegetable_id,
        skip=offset,
        limit=page_size
    )
    
    # 封装分页响应
    pagination_data = paginate(
        items=records,
        page=page,
        size=page_size,
        total=total_records
    )
    
    return response_success(data=pagination_data)

@router.get("/records/{record_id}", response_model=ResponseModel)
def read_price_record(
    record_id: int, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取特定价格记录"""
    record = prices_crud.get_price_record(db, record_id=record_id)
    if record is None:
        return response_error(msg="价格记录不存在", code=404)
    return response_success(data=record)

@router.post("/records", response_model=ResponseModel, dependencies=[Depends(get_current_admin)])
def create_price_record(
    price_record: PriceRecordCreate,
    db: Session = Depends(get_db)
):
    """创建价格记录（仅管理员）"""
    # 检查蔬菜是否存在
    vegetable = vegetables_crud.get_vegetable(db, vegetable_id=price_record.vegetable_id)
    if not vegetable:
        return response_error(msg="蔬菜不存在", code=404)
    
    new_record = prices_crud.create_price_record(db=db, price_record=price_record)
    return response_success(data=new_record, msg="价格记录创建成功")

@router.put("/records/{record_id}", response_model=ResponseModel, dependencies=[Depends(get_current_admin)])
def update_price_record(
    record_id: int,
    price_update: PriceRecordUpdate,
    db: Session = Depends(get_db)
):
    """更新价格记录（仅管理员）"""
    db_record = prices_crud.update_price_record(db, record_id, price_update)
    if db_record is None:
        return response_error(msg="价格记录不存在", code=404)
    return response_success(data=db_record, msg="价格记录更新成功")

@router.delete("/records/{record_id}", response_model=ResponseModel, dependencies=[Depends(get_current_admin)])
def delete_price_record(record_id: int, db: Session = Depends(get_db)):
    """删除价格记录（仅管理员）"""
    success = prices_crud.delete_price_record(db, record_id)
    if not success:
        return response_error(msg="价格记录不存在", code=404)
    return response_success(msg="价格记录已删除")

@router.get("/history", response_model=ResponseModel)
def get_price_history(
    vegetable_id: int,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取蔬菜历史价格"""
    # 检查蔬菜是否存在
    vegetable = vegetables_crud.get_vegetable(db, vegetable_id=vegetable_id)
    if not vegetable:
        return response_error(msg="蔬菜不存在", code=404)
    
    # 获取历史价格记录
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    records = prices_crud.get_historical_prices(db, vegetable_id, start_date, end_date)
    
    # 准备历史数据
    history = []
    for record in records:
        history.append({
            "date": record.price_date.strftime("%Y-%m-%d"),
            "price": record.price,
            "top_price": record.top_price,
            "minimum_price": record.minimum_price
        })
    
    # 获取图表数据
    chart_data = prices_crud.get_chart_data(db, vegetable_id, days)
    
    result = {
        "vegetable_id": vegetable_id,
        "vegetable_name": vegetable.product_name,
        "provenance_name": vegetable.provenance_name,
        "history": history,
        "chart_data": chart_data
    }
    
    return response_success(data=result)

@router.get("/chart-data", response_model=ResponseModel)
def get_chart_data(
    vegetable_id: int,
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取蔬菜价格图表数据"""
    # 检查蔬菜是否存在
    vegetable = vegetables_crud.get_vegetable(db, vegetable_id=vegetable_id)
    if not vegetable:
        return response_error(msg="蔬菜不存在", code=404)
    
    chart_data = prices_crud.get_chart_data(db, vegetable_id, days)
    return response_success(data=chart_data)

@router.get("/real-time", response_model=ResponseModel)
def get_real_time_price(
    vegetable_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取蔬菜实时价格"""
    # 检查蔬菜是否存在
    vegetable = vegetables_crud.get_vegetable(db, vegetable_id=vegetable_id)
    if not vegetable:
        return response_error(msg="蔬菜不存在", code=404)
    
    # 获取价格趋势
    trend = prices_crud.get_price_trend(db, vegetable_id)
    
    # 获取三天价格对比
    three_day_comparison = prices_crud.get_three_day_comparison(db, vegetable_id)
    
    result = {
        "vegetable_id": vegetable_id,
        "vegetable_name": vegetable.product_name,
        "current_price": vegetable.average_price,
        "last_updated": vegetable.price_date,
        "trend": {
            "trend": trend["trend"],
            "percentage": trend["percentage"]
        },
        "three_day_comparison": three_day_comparison
    }
    
    return response_success(data=result)

@router.get("/seasonality", response_model=ResponseModel)
def analyze_price_seasonality(
    vegetable_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """分析蔬菜价格季节性变化"""
    # 检查蔬菜是否存在
    vegetable = vegetables_crud.get_vegetable(db, vegetable_id=vegetable_id)
    if not vegetable:
        return response_error(msg="蔬菜不存在", code=404)
    
    seasonality_data = prices_crud.analyze_price_seasonality(db, vegetable_id)
    return response_success(data=seasonality_data) 