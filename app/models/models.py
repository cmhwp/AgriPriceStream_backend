from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, Text
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

from app.db.database import Base

class UserType(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(100))
    user_type = Column(String(10), default=UserType.USER)
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)

class Vegetable(Base):
    __tablename__ = "vegetables"

    id = Column(Integer, primary_key=True, index=True) # 蔬菜ID
    description = Column(String(200), nullable=True)  # 描述
    # 江南市场API数据字段
    product_name = Column(String(50), index=True)  # 蔬菜名称
    top_price = Column(Float, nullable=True)  # 最高价
    minimum_price = Column(Float, nullable=True)  # 最低价
    average_price = Column(Float, nullable=True)  # 平均价
    provenance_name = Column(String(50), index=True) # 产地
    weight = Column(Integer,nullable=True) # 单位
    standard = Column(String(50), nullable=True)  # 规格
    kind = Column(String(30), nullable=True)  # 种类
    source_type = Column(String(30), nullable=True)  # 来源类型
    price_date = Column(DateTime, index=True)  # 定价日期
    
    price_records = relationship("PriceRecord", back_populates="vegetable")
    predictions = relationship("Prediction", back_populates="vegetable")

class PriceRecord(Base):
    __tablename__ = "price_records"

    id = Column(Integer, primary_key=True, index=True)
    vegetable_id = Column(Integer, ForeignKey("vegetables.id"))
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    is_corrected = Column(Boolean, default=False)
    
    # 额外价格信息
    top_price = Column(Float, nullable=True)  # 最高价
    minimum_price = Column(Float, nullable=True)  # 最低价
    average_price = Column(Float, nullable=True)  # 平均价
    price_date = Column(DateTime, index=True)  # 价格日期
    provenance_name = Column(String(50), nullable=True)  # 产地
    
    # 定义price_change为非持久化属性
    _price_change = None
    
    @property
    def price_change(self):
        return self._price_change
    
    @price_change.setter
    def price_change(self, value):
        self._price_change = value
    
    vegetable = relationship("Vegetable", back_populates="price_records")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    vegetable_id = Column(Integer, ForeignKey("vegetables.id"))
    predicted_date = Column(DateTime, index=True)
    predicted_price = Column(Float)
    algorithm = Column(String(30))
    created_at = Column(DateTime, default=datetime.now)
    
    vegetable = relationship("Vegetable", back_populates="predictions")

class ModelTraining(Base):
    __tablename__ = "model_trainings"
    
    id = Column(Integer, primary_key=True, index=True)
    algorithm = Column(String(30))
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")  # running, completed, failed
    log = Column(Text, nullable=True)
    vegetable_id = Column(Integer, ForeignKey("vegetables.id"), nullable=True)
    product_name = Column(String(50), index=True, nullable=True)  # 添加蔬菜名称字段
    
    # 新增训练参数字段
    history_days = Column(Integer, default=30)
    prediction_days = Column(Integer, default=7)
    smoothing = Column(Boolean, default=True)
    seasonality = Column(Boolean, default=True)
    sequence_length = Column(Integer, default=7)
    
    # 关联蔬菜
    vegetable = relationship("Vegetable")

class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model_trainings.id"))
    algorithm = Column(String(30))
    mean_absolute_error = Column(Float)
    mean_squared_error = Column(Float)
    r_squared = Column(Float)
    prediction_accuracy = Column(Float)
    evaluation_date = Column(DateTime, default=datetime.now)
    vegetable_id = Column(Integer, ForeignKey("vegetables.id"))
    product_name = Column(String(50), index=True, nullable=True)  # 添加蔬菜名称字段
    
    # 关联
    model = relationship("ModelTraining")
    vegetable = relationship("Vegetable")

class CrawlerActivity(Base):
    __tablename__ = "crawler_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100))  # 活动标题
    description = Column(String(500))  # 活动描述
    time = Column(DateTime, default=datetime.now, index=True)  # 活动时间
    status = Column(String(20), default="success")  # success, error, warning, processing
    records_count = Column(Integer, default=0)  # 记录数量
    duration = Column(Integer, nullable=True)  # 爬取持续时间（秒） 