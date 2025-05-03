from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class PredictionBase(BaseModel):
    vegetable_id: int

class PredictionCreate(PredictionBase):
    predicted_date: datetime
    predicted_price: float
    algorithm: str

class PredictionResponse(PredictionBase):
    id: int
    predicted_date: datetime
    predicted_price: float
    algorithm: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class PredictionDay(BaseModel):
    date: str
    price: float

class PredictionRequest(BaseModel):
    vegetable_id: int
    days: Optional[int] = 7
    algorithm: Optional[str] = "LSTM"

class PredictionResult(BaseModel):
    vegetable_id: int
    vegetable_name: str
    provenance_name: Optional[str] = None
    current_price: Optional[float] = None
    predictions: List[PredictionDay]
    algorithm: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None

class ModelTrainingBase(BaseModel):
    algorithm: str
    status: str

class ModelTrainingCreate(ModelTrainingBase):
    start_time: Optional[datetime] = None
    log: Optional[str] = None
    vegetable_id: Optional[int] = None
    product_name: Optional[str] = None
    history_days: Optional[int] = None
    prediction_days: Optional[int] = None
    smoothing: Optional[bool] = None
    seasonality: Optional[bool] = None
    sequence_length: Optional[int] = None

class ModelTrainingResponse(ModelTrainingBase):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    log: Optional[str] = None
    vegetable_id: Optional[int] = None
    product_name: Optional[str] = None
    history_days: Optional[int] = None
    prediction_days: Optional[int] = None
    smoothing: Optional[bool] = None
    seasonality: Optional[bool] = None
    sequence_length: Optional[int] = None
    
    class Config:
        orm_mode = True

class BestPurchaseDay(BaseModel):
    vegetable_id: int
    vegetable_name: str
    best_purchase_date: str
    predicted_price: float
    current_price: Optional[float] = None
    savings: Optional[float] = None
    savings_percent: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None

# Model training config
class TrainingConfig(BaseModel):
    algorithm: str
    history_days: Optional[int] = 30
    prediction_days: Optional[int] = 7
    smoothing: Optional[bool] = True
    seasonality: Optional[bool] = False
    auto_optimize: Optional[bool] = False
    sequence_length: Optional[int] = 7
    custom_params: Optional[Dict[str, Any]] = None

# Model evaluation result
class ModelEvaluation(BaseModel):
    model_id: int
    algorithm: str
    mean_absolute_error: float
    mean_squared_error: float
    r_squared: float
    prediction_accuracy: float
    evaluation_date: datetime
    vegetable_id: int
    product_name: Optional[str] = None
    
    class Config:
        orm_mode = True

# Algorithm metrics
class AlgorithmMetrics(BaseModel):
    mae: float
    mse: float
    r2: float
    accuracy: float
    count: int
    evaluations: List[ModelEvaluation]

# Model comparison result
class ModelComparison(BaseModel):
    vegetable_id: int
    vegetable_name: str
    algorithm_metrics: Dict[str, AlgorithmMetrics]
    best_algorithm: str
    best_algorithm_metrics: AlgorithmMetrics
    success: bool
    error: Optional[str] = None

class ModelEvaluationBase(BaseModel):
    algorithm: str
    mean_absolute_error: float
    mean_squared_error: float
    r_squared: float
    prediction_accuracy: float
    vegetable_id: Optional[int] = None
    product_name: Optional[str] = None

class ModelEvaluationCreate(ModelEvaluationBase):
    model_id: int
    
class ModelEvaluationResponse(ModelEvaluationBase):
    id: int
    model_id: int
    evaluation_date: datetime
    
    class Config:
        from_attributes = True

class ModelEvaluationFilter(BaseModel):
    algorithm: Optional[str] = None
    vegetable_id: Optional[int] = None
    min_accuracy: Optional[float] = None
    max_error: Optional[float] = None
    
