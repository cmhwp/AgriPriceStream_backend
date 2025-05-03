from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from app.models.models import PriceRecord, Vegetable
from app.schemas.price import PriceRecordCreate, PriceRecordUpdate

def get_price_record(db: Session, record_id: int) -> Optional[PriceRecord]:
    return db.query(PriceRecord).filter(PriceRecord.id == record_id).first()

def get_price_records(db: Session, 
                     vegetable_id: Optional[int] = None,
                     skip: int = 0, 
                     limit: int = 100) -> List[PriceRecord]:
    query = db.query(PriceRecord)
    
    if vegetable_id:
        query = query.filter(PriceRecord.vegetable_id == vegetable_id)
    
    records = query.order_by(desc(PriceRecord.price_date)).offset(skip).limit(limit).all()
    
    # 为每条记录计算价格变化
    for record in records:
        # 添加价格变化属性
        record.price_change = calculate_price_change(db, record)
    
    return records

def get_price_records_count(db: Session, vegetable_id: Optional[int] = None) -> int:
    """获取价格记录的总数量"""
    query = db.query(func.count(PriceRecord.id))
    
    if vegetable_id:
        query = query.filter(PriceRecord.vegetable_id == vegetable_id)
    
    return query.scalar()

def create_price_record(db: Session, price_record: PriceRecordCreate) -> PriceRecord:
    db_price_record = PriceRecord(
        vegetable_id=price_record.vegetable_id,
        price=price_record.price,
        top_price=price_record.top_price,
        minimum_price=price_record.minimum_price,
        average_price=price_record.average_price or price_record.price,
        price_date=price_record.price_date,
        timestamp=datetime.now(),
        provenance_name=price_record.provenance_name,
        is_corrected=price_record.is_corrected
    )
    db.add(db_price_record)
    db.commit()
    db.refresh(db_price_record)
    
    # 更新蔬菜记录
    vegetable = db.query(Vegetable).filter(Vegetable.id == price_record.vegetable_id).first()
    if vegetable:
        vegetable.average_price = price_record.average_price or price_record.price
        vegetable.top_price = price_record.top_price
        vegetable.minimum_price = price_record.minimum_price
        vegetable.price_date = price_record.price_date
        db.commit()
    
    return db_price_record

def update_price_record(db: Session, record_id: int, price_update: PriceRecordUpdate) -> Optional[PriceRecord]:
    db_price_record = get_price_record(db, record_id)
    if not db_price_record:
        return None
    
    update_data = price_update.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(db_price_record, key, value)
    
    # 标记为已修正
    db_price_record.is_corrected = True
    
    db.commit()
    db.refresh(db_price_record)
    
    # 如果是最新记录，同时更新蔬菜表的价格
    latest_record = db.query(PriceRecord).filter(
        PriceRecord.vegetable_id == db_price_record.vegetable_id
    ).order_by(desc(PriceRecord.price_date)).first()
    
    if latest_record and latest_record.id == record_id:
        vegetable = db.query(Vegetable).filter(Vegetable.id == db_price_record.vegetable_id).first()
        if vegetable:
            vegetable.average_price = db_price_record.average_price or db_price_record.price
            vegetable.top_price = db_price_record.top_price
            vegetable.minimum_price = db_price_record.minimum_price
            db.commit()
    
    return db_price_record

def delete_price_record(db: Session, record_id: int) -> bool:
    db_price_record = get_price_record(db, record_id)
    if not db_price_record:
        return False
    
    db.delete(db_price_record)
    db.commit()
    return True

def get_historical_prices(db: Session, 
                         vegetable_id: int, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         limit: int = 30) -> List[PriceRecord]:
    """获取指定蔬菜的历史价格"""
    query = db.query(PriceRecord).filter(PriceRecord.vegetable_id == vegetable_id)
    
    if start_date:
        query = query.filter(PriceRecord.price_date >= start_date)
    if end_date:
        query = query.filter(PriceRecord.price_date <= end_date)
    
    return query.order_by(PriceRecord.price_date).limit(limit).all()

def get_price_trend(db: Session, vegetable_id: int) -> Dict[str, Any]:
    """计算价格趋势"""
    # 获取最近30天的价格记录
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    records = get_historical_prices(db, vegetable_id, start_date, end_date)
    
    if len(records) < 2:
        return {"trend": "stable", "percentage": 0.0}
    
    # 计算初始价格和最终价格
    first_price = records[0].price
    last_price = records[-1].price
    
    # 计算变化百分比
    price_change = last_price - first_price
    percentage = (price_change / first_price * 100) if first_price else 0
    
    # 确定趋势
    if percentage > 2.0:  # 上涨超过2%
        trend = "up"
    elif percentage < -2.0:  # 下跌超过2%
        trend = "down"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "percentage": round(percentage, 2)
    }

def get_three_day_comparison(db: Session, vegetable_id: int) -> Dict[str, float]:
    """获取最近三天的价格对比"""
    today = datetime.now().date()
    
    # 最近三天的日期
    days = [today - timedelta(days=i) for i in range(3)]
    
    result = {}
    for day in days:
        # 查找该日期的价格记录
        record = db.query(PriceRecord).filter(
            and_(
                PriceRecord.vegetable_id == vegetable_id,
                func.date(PriceRecord.price_date) == day
            )
        ).order_by(desc(PriceRecord.timestamp)).first()
        
        if record:
            result[day.strftime("%Y-%m-%d")] = record.price
    
    return result

def get_chart_data(db: Session, vegetable_id: int, days: int = 7) -> Dict[str, Any]:
    """生成图表数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    records = get_historical_prices(db, vegetable_id, start_date, end_date)
    
    labels = []
    prices = []
    top_prices = []
    minimum_prices = []
    
    for record in records:
        labels.append(record.price_date.strftime("%m-%d"))
        prices.append(record.price)
        top_prices.append(record.top_price if record.top_price else record.price)
        minimum_prices.append(record.minimum_price if record.minimum_price else record.price)
    
    return {
        "labels": labels,
        "prices": prices,
        "top_prices": top_prices,
        "minimum_prices": minimum_prices
    }

def analyze_price_seasonality(db: Session, vegetable_id: int) -> Dict[str, Any]:
    """分析价格的季节性变化"""
    # 获取所有历史数据
    records = db.query(PriceRecord).filter(
        PriceRecord.vegetable_id == vegetable_id
    ).order_by(PriceRecord.price_date).all()
    
    if len(records) < 30:  # 至少需要30条记录
        return {"success": False, "error": "数据不足，无法分析季节性"}
    
    # 转换为DataFrame
    data = []
    for record in records:
        data.append({
            'date': record.price_date,
            'price': record.price,
            'month': record.price_date.month,
            'day': record.price_date.day
        })
    
    df = pd.DataFrame(data)
    
    # 按月计算平均价格
    monthly_avg = df.groupby('month')['price'].mean().to_dict()
    
    # 找出价格最低的月份
    lowest_month = min(monthly_avg, key=monthly_avg.get)
    
    # 找出价格最高的月份
    highest_month = max(monthly_avg, key=monthly_avg.get)
    
    # 计算月份之间的价格波动
    months = list(range(1, 13))
    month_prices = [monthly_avg.get(m, np.nan) for m in months]
    month_prices = [p for p in month_prices if not np.isnan(p)]
    
    price_volatility = np.std(month_prices) / np.mean(month_prices) if month_prices else 0
    
    return {
        "success": True,
        "monthly_average_prices": monthly_avg,
        "lowest_price_month": lowest_month,
        "highest_price_month": highest_month,
        "price_volatility": round(price_volatility * 100, 2),  # 百分比形式
        "has_seasonality": price_volatility > 0.15  # 波动超过15%认为有季节性
    }

def calculate_price_change(db: Session, price_record: PriceRecord) -> float:
    """计算价格变化百分比（与前一天比较）"""
    # 获取当前蔬菜信息
    vegetable = db.query(Vegetable).filter(Vegetable.id == price_record.vegetable_id).first()
    if not vegetable:
        return 0.0
    
    # 获取前一天相同蔬菜名称和产地的价格记录
    previous_record = db.query(PriceRecord).join(
        Vegetable, PriceRecord.vegetable_id == Vegetable.id
    ).filter(
        Vegetable.product_name == vegetable.product_name,
        Vegetable.provenance_name == vegetable.provenance_name,
        PriceRecord.price_date < price_record.price_date
    ).order_by(desc(PriceRecord.price_date)).first()
    
    # 如果没有前一天的记录，返回0
    if not previous_record:
        return 0.0
    
    # 使用平均价格计算价格变化百分比
    current_price = price_record.average_price if price_record.average_price else price_record.price
    prev_price = previous_record.average_price if previous_record.average_price else previous_record.price
    
    # 防止除以零
    if prev_price == 0:
        return 0.0
    
    price_change = (current_price - prev_price) / prev_price * 100
    return round(price_change, 2) 