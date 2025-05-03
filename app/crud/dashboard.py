from sqlalchemy.orm import Session
from sqlalchemy import func, desc, distinct
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.models.models import User, Vegetable, PriceRecord, Prediction, ModelTraining
from app.models.models import UserType
from app.crud import vegetables as vegetables_crud
from app.crud import prices as prices_crud

def get_weekly_vegetables(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    """获取一周内最常出现的蔬菜名称和价格情况"""
    one_week_ago = datetime.now() - timedelta(days=7)
    
    weekly_vegetables = db.query(
        Vegetable.product_name,
        func.count(PriceRecord.id).label('record_count'),
        func.avg(PriceRecord.price).label('avg_price')
    ).join(
        PriceRecord,
        Vegetable.id == PriceRecord.vegetable_id
    ).filter(
        PriceRecord.timestamp >= one_week_ago
    ).group_by(
        Vegetable.product_name
    ).order_by(
        desc('record_count')
    ).limit(limit).all()
    
    result = []
    for name, count, avg_price in weekly_vegetables:
        result.append({
            "product_name": name,
            "count": count,
            "average_price": round(avg_price, 2) if avg_price else 0
        })
    
    return result

def get_user_dashboard_data(db: Session, user_id: int) -> Dict[str, Any]:
    """获取用户仪表盘数据"""
    # 获取用户基本信息
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": "未找到用户"}
    
    # 1. 用户通知数量 - 已移除通知相关功能
    
    # 2. 用户订阅的蔬菜 - 已移除通知相关功能
    subscribed_vegetables = []
    
    # 3. 蔬菜种类统计
    vegetable_counts = vegetables_crud.get_vegetable_count_by_kind(db)
    
    # 4. 获取一周内蔬菜名称情况
    weekly_vegetables = get_weekly_vegetables(db)
    
    # 5. 最近的价格更新
    recent_price_updates = []
    recent_records = db.query(
        PriceRecord,
        Vegetable.product_name
    ).join(
        Vegetable,
        PriceRecord.vegetable_id == Vegetable.id
    ).order_by(
        desc(PriceRecord.timestamp)
    ).limit(5).all()
    
    for record, name in recent_records:
        recent_price_updates.append({
            "vegetable_id": record.vegetable_id,
            "name": name,
            "price": record.price,
            "timestamp": record.timestamp,
            "is_corrected": record.is_corrected
        })
    
    # 6. 产地统计 - 获取所有产地用于词云展示
    provenance_counts = vegetables_crud.get_vegetable_count_by_provenance(db)
    
    # 为词云展示优化数据结构
    provenances = []
    
    # 找出最大计数以计算相对权重
    max_count = max([p["count"] for p in provenance_counts]) if provenance_counts else 1
    
    # 为每个产地添加相对权重
    for p in provenance_counts:
        # 权重计算为相对于最大值的百分比（1-100范围）
        weight = int((p["count"] / max_count) * 100)
        provenances.append({
            "text": p["provenance"],     # 词云文本
            "value": p["count"],         # 原始计数值
            "weight": weight,            # 相对权重（1-100）
            "count": p["count"]          # 保留原有count字段以兼容旧代码
        })
    
    # 7. 价格分析
    price_analytics = None
    if recent_records:
        current_date = datetime.now().date()  # 获取当天日期

        # --------------------------
        # 基础统计分析（使用Vegetable表）
        # --------------------------
        # 单次查询获取所有基础统计
        veg_stats = db.query(
            func.avg(Vegetable.average_price).label('avg_price'),
            func.max(Vegetable.average_price).label('max_price'),
            func.min(Vegetable.average_price).label('min_price'),
            func.avg(Vegetable.top_price - Vegetable.minimum_price).label('avg_spread')
        ).filter(
            Vegetable.price_date == current_date
        ).first()

        # --------------------------
        # 价格波动分析（使用Vegetable表）
        # --------------------------
        volatile_products = db.query(
            Vegetable.product_name,
            (Vegetable.top_price - Vegetable.minimum_price).label('price_spread')
        ).filter(
            Vegetable.price_date == current_date
        ).order_by(
            desc('price_spread')
        ).limit(5).all()

        # --------------------------
        # 产地溢价分析（使用PriceRecord表）
        # --------------------------
        origin_stats = db.query(
            PriceRecord.provenance_name,
            func.avg(PriceRecord.average_price).label('avg_price'),
            func.count(PriceRecord.id).label('record_count')
        ).filter(
            PriceRecord.price_date == current_date,
            PriceRecord.provenance_name.isnot(None)
        ).group_by(
            PriceRecord.provenance_name
        ).having(
            func.count(PriceRecord.id) > 1  # 过滤有效产地数据
        ).order_by(
            desc('avg_price')
        ).limit(5).all()

        # --------------------------
        # 价格趋势分析（混合使用历史数据）
        # --------------------------
        # 昨日趋势分析
        yesterday = current_date - timedelta(days=1)
        yesterday_avg = db.query(
            func.avg(Vegetable.average_price)
        ).filter(
            Vegetable.price_date == yesterday
        ).scalar() or 0

        # 周同比趋势分析
        last_week = current_date - timedelta(weeks=1)
        last_week_avg = db.query(
            func.avg(Vegetable.average_price)
        ).filter(
            Vegetable.price_date.between(last_week, current_date)
        ).scalar() or 0

        # 趋势计算
        daily_trend = "stable"
        weekly_trend = "stable"
        
        if veg_stats.avg_price:
            # 日环比
            if yesterday_avg > 0:
                daily_change = ((veg_stats.avg_price - yesterday_avg) / yesterday_avg) * 100
                daily_trend = "up" if daily_change > 2 else "down" if daily_change < -2 else "stable"
            
            # 周同比
            if last_week_avg > 0:
                weekly_change = ((veg_stats.avg_price - last_week_avg) / last_week_avg) * 100
                weekly_trend = "up" if weekly_change > 2 else "down" if weekly_change < -2 else "stable"

        # --------------------------
        # 价格异常检测
        # --------------------------
        price_anomalies = get_price_anomalies(db, current_date)

        # --------------------------
        # 构建分析结果
        # --------------------------
        price_analytics = {
            # 基础指标
            "date": current_date.isoformat(),
            "total_products": db.query(Vegetable).filter(Vegetable.price_date == current_date).count(),
            "average_price": round(veg_stats.avg_price, 2) if veg_stats.avg_price else 0,
            "highest_price": round(veg_stats.max_price, 2) if veg_stats.max_price else 0,
            "lowest_price": round(veg_stats.min_price, 2) if veg_stats.min_price else 0,
            "avg_daily_spread": round(veg_stats.avg_spread, 2) if veg_stats.avg_spread else 0,
            
            # 趋势分析
            "daily_trend": daily_trend,
            "weekly_trend": weekly_trend,
            "daily_change": round(daily_change, 2) if 'daily_change' in locals() else 0,
            "weekly_change": round(weekly_change, 2) if 'weekly_change' in locals() else 0,
            
            # 明细分析
            "price_volatility": [
                {"product": p.product_name, "spread": round(p.price_spread, 2)}
                for p in volatile_products
            ],
            "premium_origins": [
                {"origin": o.provenance_name, 
                 "avg_price": round(o.avg_price, 2),
                 "data_points": o.record_count}
                for o in origin_stats
            ],
            
            # 价格异常检测
            "price_anomalies": price_anomalies,
            
            # 兼容旧代码，保留price_trend字段
            "price_trend": daily_trend,
            "price_change_percentage": round(daily_change, 2) if 'daily_change' in locals() else 0
        }
    
    # 8. 获取当天价格最低的蔬菜
    lowest_priced_vegetable = get_lowest_priced_vegetable(db)
    
    return {
        "username": user.username,
        "subscribed_vegetables": subscribed_vegetables,
        "vegetable_count_by_type": vegetable_counts,
        "weekly_vegetables": weekly_vegetables,
        "recent_price_updates": recent_price_updates,
        "provenances": provenances,
        "price_analytics": price_analytics,
        "lowest_priced_vegetable": lowest_priced_vegetable
    }

def get_admin_dashboard_data(db: Session, user_id: int) -> Dict[str, Any]:
    """获取管理员仪表盘数据"""
    # 首先获取用户仪表盘数据作为基础
    user_dashboard = get_user_dashboard_data(db, user_id)
    
    # 检查是否出错
    if "error" in user_dashboard:
        return user_dashboard
    
    # 1. 总用户数
    total_users = db.query(func.count(User.id)).scalar()
    
    # 2. 总蔬菜数
    total_vegetables = db.query(func.count(Vegetable.id)).scalar()
    
    # 3. 总价格记录数
    total_price_records = db.query(func.count(PriceRecord.id)).scalar()
    
    # 4. 最近的爬虫活动
    recent_crawler_activities = []
    # 这里假设我们根据时间戳推断爬虫活动
    recent_batches = db.query(  
        func.date(PriceRecord.timestamp).label('batch_date'),
        func.count(PriceRecord.id).label('record_count')
    ).group_by(
        'batch_date'
    ).order_by(
        desc('batch_date')
    ).limit(5).all()
    
    for batch_date, record_count in recent_batches:
        recent_crawler_activities.append({
            "timestamp": datetime.combine(batch_date, datetime.min.time()),
            "activity_type": "数据爬取",
            "description": f"爬取了 {record_count} 条价格记录"
        })
    
    # 5. 模型训练状态
    model_training_status = []
    trainings = db.query(ModelTraining).order_by(desc(ModelTraining.start_time)).limit(5).all()
    
    for training in trainings:
        model_training_status.append({
            "id": training.id,
            "algorithm": training.algorithm,
            "start_time": training.start_time,
            "end_time": training.end_time,
            "status": training.status,
            "log": training.log
        })
    
    # 6. 数据更新频率
    last_30_days = datetime.now() - timedelta(days=30)
    daily_counts = db.query(
        func.date(PriceRecord.timestamp).label('date'),
        func.count(PriceRecord.id).label('count')
    ).filter(
        PriceRecord.timestamp >= last_30_days
    ).group_by(
        'date'
    ).all()
    
    data_update_frequency = {
        "total_days": 30,
        "active_days": len(daily_counts),
        "updates_per_day": round(sum(count for _, count in daily_counts) / 30, 2),
        "latest_update": max((date for date, _ in daily_counts), default=None) if daily_counts else None
    }
    
    # 合并所有数据
    admin_dashboard = {
        **user_dashboard,
        "total_users": total_users,
        "total_vegetables": total_vegetables,
        "total_price_records": total_price_records,
        "recent_crawler_activities": recent_crawler_activities,
        "model_training_status": model_training_status,
        "data_update_frequency": data_update_frequency
    }
    
    return admin_dashboard

def get_recent_crawler_activities(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    """获取最近的爬虫活动记录"""
    from app.models.models import CrawlerActivity
    
    # 首先尝试从专用的爬虫活动表中获取
    activities = db.query(CrawlerActivity).order_by(desc(CrawlerActivity.time)).limit(limit).all()
    
    if activities:
        return activities
    else:
        # 如果没有记录，则从价格记录表中推断爬虫活动（向后兼容）
        recent_activities = []
        recent_batches = db.query(
            func.date(PriceRecord.timestamp).label('batch_date'),
            func.count(PriceRecord.id).label('record_count')
        ).group_by(
            'batch_date'
        ).order_by(
            desc('batch_date')
        ).limit(limit).all()
        
        for batch_date, record_count in recent_batches:
            activity_time = datetime.combine(batch_date, datetime.min.time())
            
            # 确定状态
            status = "success"
            if record_count < 10:  # 如果记录很少，可能是异常
                status = "warning"
            
            recent_activities.append({
                "id": 0,  # 临时ID
                "title": "每日常规爬取",
                "description": f"成功获取了{record_count}条新数据",
                "time": activity_time,
                "status": status,
                "records_count": record_count,
                "duration": None
            })
        
        return recent_activities

def create_crawler_activity(
    db: Session, 
    title: str,
    description: str,
    status: str = "success",
    records_count: int = 0,
    duration: Optional[int] = None
) -> Dict[str, Any]:
    """创建爬虫活动记录"""
    from app.models.models import CrawlerActivity
    
    activity = CrawlerActivity(
        title=title,
        description=description,
        time=datetime.now(),
        status=status,
        records_count=records_count,
        duration=duration
    )
    
    db.add(activity)
    db.commit()
    db.refresh(activity)
    
    return activity

def get_lowest_priced_vegetable(db: Session) -> Dict[str, Any]:
    """获取当天价格最低的蔬菜信息"""
    # 获取当天日期
    today = datetime.now().date()
    
    # 查询当天更新的蔬菜中价格最低的
    lowest_priced_vegetable = db.query(
        Vegetable
    ).filter(
        func.date(Vegetable.price_date) == today,
        Vegetable.average_price > 0  # 确保价格有效
    ).order_by(
        Vegetable.average_price
    ).first()
    
    # 如果当天没有数据，则查找最近的数据
    if not lowest_priced_vegetable:
        # 查找最近的价格记录日期
        latest_date = db.query(
            func.date(Vegetable.price_date)
        ).filter(
            Vegetable.average_price > 0
        ).order_by(
            desc(Vegetable.price_date)
        ).first()
        
        if latest_date:
            latest_date = latest_date[0]
            # 使用最近日期查询价格最低的蔬菜
            lowest_priced_vegetable = db.query(
                Vegetable
            ).filter(
                func.date(Vegetable.price_date) == latest_date,
                Vegetable.average_price > 0
            ).order_by(
                Vegetable.average_price
            ).first()
    
    if lowest_priced_vegetable:
        return {
            "vegetable_id": lowest_priced_vegetable.id,
            "product_name": lowest_priced_vegetable.product_name,
            "price": lowest_priced_vegetable.average_price,
            "price_date": lowest_priced_vegetable.price_date,
            "provenance": lowest_priced_vegetable.provenance_name,
            "kind": lowest_priced_vegetable.kind
        }
    else:
        return {
            "vegetable_id": None,
            "product_name": None,
            "price": None,
            "price_date": None,
            "provenance": None,
            "kind": None
        }

def get_price_anomalies(db: Session, current_date) -> List[Dict[str, Any]]:
    """
    检测价格异常 - 查找价格显著偏离历史均值的蔬菜
    """
    # 获取当天的蔬菜价格
    today_prices = db.query(
        Vegetable.id,
        Vegetable.product_name,
        Vegetable.average_price
    ).filter(
        Vegetable.price_date == current_date
    ).all()
    
    anomalies = []
    
    # 对每种蔬菜，计算其历史价格和标准差
    for veg_id, product_name, today_price in today_prices:
        # 查询过去30天的价格（不包括今天）
        past_prices = db.query(
            func.avg(Vegetable.average_price).label('avg_price'),
            func.stddev(Vegetable.average_price).label('std_price')
        ).filter(
            Vegetable.product_name == product_name,
            Vegetable.price_date < current_date,
            Vegetable.price_date >= current_date - timedelta(days=30)
        ).first()
        
        if past_prices.avg_price is not None and past_prices.std_price is not None:
            # 计算Z-score
            z_score = (today_price - past_prices.avg_price) / past_prices.std_price if past_prices.std_price > 0 else 0
            
            # 如果Z-score超过阈值，认为是异常
            if abs(z_score) > 2:  # 2个标准差以外
                # 计算偏差百分比
                deviation_percent = ((today_price - past_prices.avg_price) / past_prices.avg_price) * 100
                
                anomalies.append({
                    "product": product_name,
                    "expected_price": round(past_prices.avg_price, 2),
                    "actual_price": round(today_price, 2), 
                    "deviation_percent": round(deviation_percent, 2)
                })
    
    # 只返回偏差最大的前5个异常
    return sorted(anomalies, key=lambda x: abs(x["deviation_percent"]), reverse=True)[:5] 