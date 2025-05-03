from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, date

class VegetableCountByType(BaseModel):
    kind: str
    count: int
    
class WeeklyVegetable(BaseModel):
    product_name: str
    count: int
    average_price: float

class VolatileProduct(BaseModel):
    product: str
    spread: float

class PremiumOrigin(BaseModel):
    origin: str
    avg_price: float
    data_points: int
    
class PriceAnomaly(BaseModel):
    product: str
    expected_price: float
    actual_price: float
    deviation_percent: float
    
class PriceAnalytics(BaseModel):
    # 基础指标
    date: str
    total_products: int
    average_price: float
    highest_price: float
    lowest_price: float
    avg_daily_spread: float
    
    # 趋势分析
    daily_trend: str  # up, down, stable
    weekly_trend: str  # up, down, stable
    daily_change: float
    weekly_change: float
    
    # 兼容旧代码
    price_trend: str  # up, down, stable
    price_change_percentage: float
    
    # 明细分析
    price_volatility: List[VolatileProduct]
    premium_origins: List[PremiumOrigin]
    price_anomalies: List[PriceAnomaly]
    
class ProvenanceCount(BaseModel):
    """产地数据模型，支持词云展示"""
    text: str         # 产地名称，用于词云展示文本
    value: int        # 产品数量，用于词云大小计算
    weight: int       # 相对权重（1-100范围）
    count: int        # 保留原有count字段以兼容旧代码
    
class RecentActivity(BaseModel):
    timestamp: datetime
    activity_type: str
    description: str
    
class UserDashboard(BaseModel):
    username: str
    notifications_count: int
    subscribed_vegetables: List[Dict[str, Any]]
    vegetable_count_by_type: List[Dict[str, Any]]
    weekly_vegetables: List[Dict[str, Any]]
    recent_price_updates: List[Dict[str, Any]]
    provenances: List[Dict[str, Any]]
    price_analytics: Optional[Dict[str, Any]] = None
    lowest_priced_vegetable: Optional[Dict[str, Any]] = None
    
class AdminDashboard(UserDashboard):
    total_users: int
    total_vegetables: int
    total_price_records: int
    recent_crawler_activities: List[Dict[str, Any]]
    model_training_status: List[Dict[str, Any]]
    data_update_frequency: Dict[str, Any]

class CrawlerActivityResponse(BaseModel):
    id: int
    title: str
    description: str
    time: datetime
    status: str
    records_count: int
    duration: Optional[int] = None
    
    class Config:
        from_attributes = True 