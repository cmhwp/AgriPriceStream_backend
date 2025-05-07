from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.models.models import Vegetable, PriceRecord
from app.schemas.vegetable import VegetableCreate, VegetableUpdate

def get_vegetable(db: Session, vegetable_id: int) -> Optional[Vegetable]:
    return db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()

def get_vegetable_by_name(db: Session, name: str) -> Optional[Vegetable]:
    return db.query(Vegetable).filter(Vegetable.product_name == name).first()

def get_vegetables_count(db: Session, name_filter: Optional[str] = None, kind_filter: Optional[str] = None) -> int:
    """获取蔬菜总数"""
    query = db.query(func.count(Vegetable.id))
    
    if name_filter:
        query = query.filter(Vegetable.product_name.like(f"%{name_filter}%"))
    
    if kind_filter:
        query = query.filter(Vegetable.kind == kind_filter)
    
    return query.scalar()

def get_vegetables(
    db: Session, 
    skip: int = 0, 
    limit: int = 100, 
    name_filter: Optional[str] = None,
    kind_filter: Optional[str] = None,
    sort_by: Optional[str] = None
) -> List[Vegetable]:
    """获取蔬菜列表，支持分页、过滤和排序"""
    query = db.query(Vegetable)
    
    # 应用筛选条件
    if name_filter:
        query = query.filter(Vegetable.product_name.like(f"%{name_filter}%"))
        
    if kind_filter:
        query = query.filter(Vegetable.kind == kind_filter)
    
    # 应用排序
    if sort_by:
        if sort_by == "price_asc":
            query = query.order_by(Vegetable.average_price.asc())
        elif sort_by == "price_desc":
            query = query.order_by(Vegetable.average_price.desc())
        elif sort_by == "name_asc":
            query = query.order_by(Vegetable.product_name.asc())
        elif sort_by == "name_desc":
            query = query.order_by(Vegetable.product_name.desc())
        elif sort_by == "date_asc":
            query = query.order_by(Vegetable.price_date.asc())
        elif sort_by == "date_desc":
            query = query.order_by(Vegetable.price_date.desc())
    else:
        # 默认按更新日期降序排序
        query = query.order_by(Vegetable.price_date.desc())
    
    # 应用分页
    return query.offset(skip).limit(limit).all()

def get_recent_updated_vegetables_count(db: Session, days: int = 7) -> int:
    """获取最近更新的蔬菜总数"""
    date_threshold = datetime.now() - timedelta(days=days)
    return db.query(func.count(Vegetable.id)).filter(
        Vegetable.price_date >= date_threshold
    ).scalar()

def get_recent_updated_vegetables(db: Session, days: int = 7, skip: int = 0, limit: int = 100) -> List[Vegetable]:
    """获取最近更新的蔬菜"""
    date_threshold = datetime.now() - timedelta(days=days)
    return db.query(Vegetable).filter(
        Vegetable.price_date >= date_threshold
    ).order_by(
        Vegetable.price_date.desc()
    ).offset(skip).limit(limit).all()

def create_vegetable(db: Session, vegetable: VegetableCreate) -> Vegetable:
    """创建新蔬菜"""
    db_vegetable = Vegetable(
        product_name=vegetable.product_name,
        description=vegetable.description,
        provenance_name=vegetable.provenance_name,
        top_price=vegetable.top_price,
        minimum_price=vegetable.minimum_price,
        average_price=vegetable.average_price,
        standard=vegetable.standard,
        kind=vegetable.kind,
        weight=vegetable.weight,
        price_date=datetime.now()
    )
    db.add(db_vegetable)
    db.commit()
    db.refresh(db_vegetable)
    return db_vegetable

def update_vegetable(db: Session, vegetable_id: int, vegetable_update: VegetableUpdate) -> Optional[Vegetable]:
    """更新蔬菜信息"""
    db_vegetable = get_vegetable(db, vegetable_id)
    if not db_vegetable:
        return None
    
    # 更新非空字段
    update_data = vegetable_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_vegetable, key, value)
    
    db.commit()
    db.refresh(db_vegetable)
    return db_vegetable

def delete_vegetable(db: Session, vegetable_id: int) -> bool:
    """删除蔬菜"""
    db_vegetable = get_vegetable(db, vegetable_id)
    if not db_vegetable:
        return False
    
    db.delete(db_vegetable)
    db.commit()
    return True

def get_vegetable_count_by_kind(db: Session) -> List[Dict[str, Any]]:
    """获取各种类蔬菜的数量统计"""
    # 忽略kind为null的记录
    results = db.query(
        Vegetable.kind,
        func.count(Vegetable.id).label('count')
    ).filter(
        Vegetable.kind.isnot(None)
    ).group_by(
        Vegetable.kind
    ).all()
    
    return [{"kind": kind, "count": count} for kind, count in results]

def get_vegetable_count_by_provenance(db: Session) -> List[Dict[str, Any]]:
    """获取各产地蔬菜的数量统计"""
    # 忽略provenance_name为null的记录
    results = db.query(
        Vegetable.provenance_name,
        func.count(Vegetable.id).label('count')
    ).filter(
        Vegetable.provenance_name.isnot(None)
    ).group_by(
        Vegetable.provenance_name
    ).order_by(
        desc('count')
    ).all()
    
    return [{"provenance": prov, "count": count} for prov, count in results]

def get_vegetable_kinds(db: Session) -> List[str]:
    """获取所有蔬菜种类"""
    kinds = db.query(Vegetable.kind).filter(
        Vegetable.kind.isnot(None)
    ).distinct().all()
    return [k[0] for k in kinds]

def get_vegetable_provenances(db: Session) -> List[str]:
    """获取所有蔬菜产地"""
    provenances = db.query(Vegetable.provenance_name).filter(
        Vegetable.provenance_name.isnot(None)
    ).distinct().all()
    return [p[0] for p in provenances]

def get_vegetables_by_price_range(
    db: Session,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    name_filter: Optional[str] = None,
    kind_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Vegetable]:
    """根据价格范围查询蔬菜，可同时按名称和种类筛选"""
    query = db.query(Vegetable)
    
    if min_price is not None:
        query = query.filter(Vegetable.average_price >= min_price)
    
    if max_price is not None:
        query = query.filter(Vegetable.average_price <= max_price)
    
    # 应用名称筛选
    if name_filter:
        query = query.filter(Vegetable.product_name.like(f"%{name_filter}%"))
    
    # 应用种类筛选
    if kind_filter:
        query = query.filter(Vegetable.kind == kind_filter)
    
    return query.order_by(Vegetable.price_date.desc()).offset(skip).limit(limit).all()

def get_vegetables_by_price_range_count(
    db: Session,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    name_filter: Optional[str] = None,
    kind_filter: Optional[str] = None
) -> int:
    """获取价格范围内的蔬菜总数，可同时按名称和种类筛选"""
    query = db.query(func.count(Vegetable.id))
    
    if min_price is not None:
        query = query.filter(Vegetable.average_price >= min_price)
    
    if max_price is not None:
        query = query.filter(Vegetable.average_price <= max_price)
    
    # 应用名称筛选
    if name_filter:
        query = query.filter(Vegetable.product_name.like(f"%{name_filter}%"))
    
    # 应用种类筛选
    if kind_filter:
        query = query.filter(Vegetable.kind == kind_filter)
    
    return query.scalar()

def get_vegetable_name_by_id(db: Session, vegetable_id: int) -> Optional[str]:
    """根据蔬菜ID获取蔬菜名称"""
    result = db.query(Vegetable.product_name).filter(Vegetable.id == vegetable_id).first()
    return result[0] if result else None 

def get_vegetable_options_past_year(db: Session) -> List[Dict[str, Any]]:
    """获取过去一年内的蔬菜选项（不重复）"""
    # 直接查询所有蔬菜，按ID排序确保顺序稳定
    results = db.query(
        Vegetable.id,
        Vegetable.product_name
    ).order_by(
        Vegetable.id
    ).all()
    
    # 打印找到的总蔬菜数量，用于调试
    print(f"Total vegetables before deduplication: {len(results)}")
    
    # 使用字典保存不重复的蔬菜，以名称为键
    unique_vegetables = {}
    for id, name in results:
        # 确保name不为空
        if name and name.strip():
            unique_vegetables[name.strip()] = {"id": id, "name": name.strip()}
    
    # 打印去重后的数量
    options = list(unique_vegetables.values())
    print(f"Total unique vegetables: {len(options)}")
    
    return options 