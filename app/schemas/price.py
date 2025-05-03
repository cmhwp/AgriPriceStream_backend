from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date

class PriceRecordBase(BaseModel):
    vegetable_id: int
    price: float
    
class PriceRecordCreate(PriceRecordBase):
    price_date: datetime
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    average_price: Optional[float] = None
    provenance_name: Optional[str] = None
    is_corrected: Optional[bool] = False
    
class PriceRecordUpdate(BaseModel):
    price: Optional[float] = None
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    average_price: Optional[float] = None
    is_corrected: Optional[bool] = None

class PriceRecordResponse(PriceRecordBase):
    id: int
    timestamp: datetime
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    average_price: Optional[float] = None
    price_date: Optional[datetime] = None
    provenance_name: Optional[str] = None
    is_corrected: bool
    price_change: Optional[float] = None
    
    class Config:
        from_attributes = True

class HistoricalPrice(BaseModel):
    date: str
    price: float
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    
class ChartData(BaseModel):
    labels: List[str]
    prices: List[float]
    top_prices: Optional[List[float]] = None
    minimum_prices: Optional[List[float]] = None
    
class VegetablePriceHistory(BaseModel):
    vegetable_id: int
    vegetable_name: str
    provenance_name: Optional[str] = None
    history: List[HistoricalPrice]
    chart_data: ChartData
    
class PriceTrend(BaseModel):
    trend: str  # "up", "down", "stable"
    percentage: float
    
class RealTimePriceResponse(BaseModel):
    vegetable_id: int
    vegetable_name: str
    current_price: float
    last_updated: datetime
    trend: Optional[PriceTrend] = None
    three_day_comparison: Optional[Dict[str, float]] = None 