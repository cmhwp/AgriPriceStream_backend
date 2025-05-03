from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.schemas.price import PriceRecordResponse

class VegetableBase(BaseModel):
    product_name: str
    description: Optional[str] = None
    provenance_name: Optional[str] = None
    
class VegetableCreate(VegetableBase):
    product_name: str
    provenance_name: str
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    average_price: Optional[float] = None
    standard: Optional[str] = None
    kind: Optional[str] = None
    weight: Optional[int] = None
    
class VegetableUpdate(BaseModel):
    product_name: Optional[str] = None
    description: Optional[str] = None
    provenance_name: Optional[str] = None
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    average_price: Optional[float] = None
    standard: Optional[str] = None
    kind: Optional[str] = None
    weight: Optional[int] = None

class VegetableResponse(VegetableBase):
    id: int
    top_price: Optional[float] = None
    minimum_price: Optional[float] = None
    average_price: Optional[float] = None
    standard: Optional[str] = None
    kind: Optional[str] = None
    weight: Optional[int] = None
    source_type: Optional[str] = None
    price_date: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class VegetableDetail(VegetableResponse):
    price_records: List["PriceRecordResponse"] = []
    
    class Config:
        from_attributes = True 