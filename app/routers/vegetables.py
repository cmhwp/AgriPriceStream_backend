from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.models import User, UserType, Vegetable
from app.schemas.vegetable import VegetableCreate, VegetableUpdate, VegetableResponse, VegetableDetail
from app.schemas.response import ResponseModel, paginate
from app.crud import vegetables as vegetables_crud
from app.utils.auth import get_current_user, get_current_admin
from app.utils.response import response_success, response_error

router = APIRouter(
    prefix="/vegetables",
    tags=["vegetables"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=ResponseModel)
def read_vegetables(
    skip: int = 0, 
    limit: int = 20,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页记录数"), 
    name: Optional[str] = None,
    kind: Optional[str] = None,
    sort_by: Optional[str] = Query("date_desc", description="排序方式"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取蔬菜列表，支持过滤和排序"""
    # 计算总记录数
    total = vegetables_crud.get_vegetables_count(db, name_filter=name, kind_filter=kind)
    
    # 计算分页偏移量
    offset = (page - 1) * page_size
    
    # 获取当前页数据
    vegetables = vegetables_crud.get_vegetables(
        db, 
        skip=offset, 
        limit=page_size, 
        name_filter=name,
        kind_filter=kind,
        sort_by=sort_by
    )
    
    # 封装分页响应
    pagination_data = paginate(
        items=vegetables,
        page=page,
        size=page_size,
        total=total
    )
    
    return response_success(data=pagination_data)

@router.get("/kinds", response_model=ResponseModel)
def read_vegetable_kinds(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取所有蔬菜种类"""
    kinds = vegetables_crud.get_vegetable_kinds(db)
    return response_success(data=kinds)

@router.get("/provenances", response_model=ResponseModel)
def read_vegetable_provenances(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取所有蔬菜产地"""
    provenances = vegetables_crud.get_vegetable_provenances(db)
    return response_success(data=provenances)

@router.get("/recent", response_model=ResponseModel)
def read_recent_vegetables(
    days: int = Query(7, ge=1, le=30),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页记录数"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取最近更新的蔬菜"""
    # 计算总记录数
    total = vegetables_crud.get_recent_updated_vegetables_count(db, days=days)
    
    # 计算分页偏移量
    offset = (page - 1) * page_size
    
    # 获取当前页数据
    recent_vegetables = vegetables_crud.get_recent_updated_vegetables(
        db, 
        days=days, 
        skip=offset, 
        limit=page_size
    )
    
    # 封装分页响应
    pagination_data = paginate(
        items=recent_vegetables,
        page=page,
        size=page_size,
        total=total
    )
    
    return response_success(data=pagination_data)

@router.get("/{vegetable_id}", response_model=ResponseModel)
def read_vegetable(vegetable_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取特定蔬菜的详细信息"""
    db_vegetable = vegetables_crud.get_vegetable(db, vegetable_id=vegetable_id)
    if db_vegetable is None:
        return response_error(msg="蔬菜不存在", code=404)
    return response_success(data=db_vegetable)

@router.post("/", response_model=ResponseModel, dependencies=[Depends(get_current_admin)])
def create_vegetable(vegetable: VegetableCreate, db: Session = Depends(get_db)):
    """创建新蔬菜（仅管理员）"""
    # 检查蔬菜名称是否已存在
    db_vegetable = vegetables_crud.get_vegetable_by_name(db, name=vegetable.product_name)
    if db_vegetable:
        return response_error(msg="该蔬菜名称已存在", code=400)
    new_vegetable = vegetables_crud.create_vegetable(db=db, vegetable=vegetable)
    return response_success(data=new_vegetable, msg="蔬菜创建成功")

@router.put("/{vegetable_id}", response_model=ResponseModel, dependencies=[Depends(get_current_admin)])
def update_vegetable(vegetable_id: int, vegetable_update: VegetableUpdate, db: Session = Depends(get_db)):
    """更新蔬菜信息（仅管理员）"""
    db_vegetable = vegetables_crud.update_vegetable(db, vegetable_id, vegetable_update)
    if db_vegetable is None:
        return response_error(msg="蔬菜不存在", code=404)
    return response_success(data=db_vegetable, msg="蔬菜更新成功")

@router.delete("/{vegetable_id}", response_model=ResponseModel, dependencies=[Depends(get_current_admin)])
def delete_vegetable(vegetable_id: int, db: Session = Depends(get_db)):
    """删除蔬菜（仅管理员）"""
    success = vegetables_crud.delete_vegetable(db, vegetable_id)
    if not success:
        return response_error(msg="蔬菜不存在", code=404)
    return response_success(data=None, msg="蔬菜已删除")

@router.get("/price-range/", response_model=ResponseModel)
def read_vegetables_by_price_range(
    min_price: Optional[float] = Query(None, description="最低价格"),
    max_price: Optional[float] = Query(None, description="最高价格"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页记录数"),
    sort_by: Optional[str] = Query("date_desc", description="排序方式"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """根据价格范围查询蔬菜"""
    # 计算总记录数
    total = vegetables_crud.get_vegetables_by_price_range_count(
        db, min_price=min_price, max_price=max_price
    )
    
    # 计算分页偏移量
    offset = (page - 1) * page_size
    
    # 获取当前页数据
    vegetables = vegetables_crud.get_vegetables_by_price_range(
        db, 
        min_price=min_price,
        max_price=max_price,
        skip=offset, 
        limit=page_size
    )
    
    # 封装分页响应
    pagination_data = paginate(
        items=vegetables,
        page=page,
        size=page_size,
        total=total
    )
    
    return response_success(data=pagination_data)

@router.get("/{vegetable_id}/name", response_model=ResponseModel)
def get_vegetable_name(
    vegetable_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """根据蔬菜ID获取蔬菜名称"""
    name = vegetables_crud.get_vegetable_name_by_id(db, vegetable_id=vegetable_id)
    if name is None:
        return response_error(msg="蔬菜不存在", code=404)
    return response_success(data={"id": vegetable_id, "name": name}) 

@router.get("/options/past-year", response_model=ResponseModel)
def get_vegetable_options(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取过去一年内的蔬菜选项（不重复）"""
    options = vegetables_crud.get_vegetable_options_past_year(db)
    return response_success(data=options) 