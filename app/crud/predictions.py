from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from app.models.models import Prediction, Vegetable, PriceRecord, ModelTraining, ModelEvaluation
from app.schemas.prediction import PredictionCreate, ModelTrainingCreate, ModelEvaluationCreate

def create_prediction(db: Session, prediction: PredictionCreate) -> Prediction:
    """创建新的预测记录"""
    db_prediction = Prediction(
        vegetable_id=prediction.vegetable_id,
        predicted_date=prediction.predicted_date,
        predicted_price=prediction.predicted_price,
        algorithm=prediction.algorithm,
        created_at=datetime.now()
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction(db: Session, prediction_id: int) -> Optional[Prediction]:
    """获取特定ID的预测记录"""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()

def get_predictions_by_vegetable(
    db: Session, 
    vegetable_id: int, 
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    algorithm: Optional[str] = None
) -> List[Prediction]:
    """获取特定蔬菜的预测记录，可选择性地按日期范围过滤"""
    query = db.query(Prediction).filter(Prediction.vegetable_id == vegetable_id)
    
    if start_date:
        query = query.filter(Prediction.predicted_date >= start_date)
    
    if end_date:
        query = query.filter(Prediction.predicted_date <= end_date)
    
    if algorithm:
        query = query.filter(Prediction.algorithm == algorithm)
    
    return query.order_by(Prediction.predicted_date).all()

def get_latest_predictions(
    db: Session, 
    vegetable_id: int, 
    days: int = 7,
    algorithm: Optional[str] = None
) -> List[Prediction]:
    """获取特定蔬菜的最新预测"""
    # 计算起始和结束日期
    start_date = datetime.now()
    end_date = start_date + timedelta(days=days)
    
    query = db.query(Prediction).filter(
        Prediction.vegetable_id == vegetable_id,
        Prediction.predicted_date >= start_date,
        Prediction.predicted_date <= end_date
    )
    
    if algorithm:
        query = query.filter(Prediction.algorithm == algorithm)
    
    # 按创建时间降序排序，确保获取最新创建的预测
    return query.order_by(Prediction.predicted_date, desc(Prediction.created_at)).all()

def delete_old_predictions(db: Session, days: int = 30) -> int:
    """删除指定天数之前的预测记录"""
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = db.query(Prediction).filter(Prediction.created_at < cutoff_date).delete()
    db.commit()
    return deleted_count

def create_model_training(db: Session, training: ModelTrainingCreate) -> ModelTraining:
    """创建新的模型训练记录"""
    db_training = ModelTraining(
        algorithm=training.algorithm,
        status=training.status,
        start_time=training.start_time or datetime.now(),
        log=training.log,
        vegetable_id=training.vegetable_id,
        product_name=training.product_name,
        history_days=training.history_days,
        prediction_days=training.prediction_days,
        smoothing=training.smoothing,
        seasonality=training.seasonality,
        sequence_length=training.sequence_length
    )
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    return db_training

def update_model_training(db: Session, training_id: int, **kwargs) -> Optional[ModelTraining]:
    """更新模型训练记录"""
    db_training = db.query(ModelTraining).filter(ModelTraining.id == training_id).first()
    if not db_training:
        return None
    
    for key, value in kwargs.items():
        setattr(db_training, key, value)
    
    db.commit()
    db.refresh(db_training)
    return db_training

def get_model_training(db: Session, training_id: int) -> Optional[ModelTraining]:
    """获取特定ID的模型训练记录"""
    return db.query(ModelTraining).filter(ModelTraining.id == training_id).first()

def get_model_trainings(
    db: Session, 
    status: Optional[str] = None,
    algorithm: Optional[str] = None,
    vegetable_id: Optional[int] = None,
    limit: int = 100
) -> List[ModelTraining]:
    """获取模型训练记录列表"""
    query = db.query(ModelTraining)
    
    if status:
        query = query.filter(ModelTraining.status == status)
    
    if algorithm:
        query = query.filter(ModelTraining.algorithm == algorithm)
    
    if vegetable_id:
        query = query.filter(ModelTraining.vegetable_id == vegetable_id)
    
    return query.order_by(desc(ModelTraining.start_time)).limit(limit).all()

def create_model_evaluation(db: Session, evaluation: ModelEvaluationCreate) -> ModelEvaluation:
    """创建模型评估记录"""
    db_evaluation = ModelEvaluation(
        model_id=evaluation.model_id,
        algorithm=evaluation.algorithm,
        mean_absolute_error=evaluation.mean_absolute_error,
        mean_squared_error=evaluation.mean_squared_error,
        r_squared=evaluation.r_squared,
        prediction_accuracy=evaluation.prediction_accuracy,
        vegetable_id=evaluation.vegetable_id,
        product_name=evaluation.product_name
    )
    
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    
    return db_evaluation

def get_model_evaluation(db: Session, evaluation_id: int) -> Optional[ModelEvaluation]:
    """获取单个模型评估记录"""
    return db.query(ModelEvaluation).filter(ModelEvaluation.id == evaluation_id).first()

def delete_model_evaluation(db: Session, evaluation_id: int) -> bool:
    """删除模型评估记录"""
    evaluation = db.query(ModelEvaluation).filter(ModelEvaluation.id == evaluation_id).first()
    if evaluation:
        db.delete(evaluation)
        db.commit()
        return True
    return False

def get_model_evaluations_by_vegetable(
    db: Session, 
    vegetable_id: int,
    algorithm: Optional[str] = None
) -> List[ModelEvaluation]:
    """获取特定蔬菜的所有模型评估记录"""
    query = db.query(ModelEvaluation).filter(ModelEvaluation.vegetable_id == vegetable_id)
    
    if algorithm:
        query = query.filter(ModelEvaluation.algorithm == algorithm)
    
    return query.all()

def get_model_evaluations_by_combination(
    db: Session, 
    product_name: str,
    algorithm: Optional[str] = None
) -> List[ModelEvaluation]:
    """根据蔬菜名称获取模型评估记录
    
    Args:
        db: 数据库会话
        product_name: 蔬菜名称
        algorithm: 算法名称，可选
        
    Returns:
        评估记录列表
    """
    query = db.query(ModelEvaluation).filter(ModelEvaluation.product_name == product_name)
    
    if algorithm:
        query = query.filter(ModelEvaluation.algorithm == algorithm)
    
    return query.all()

def get_best_model_for_vegetable(db: Session, vegetable_id: int) -> Optional[Dict[str, Any]]:
    """获取特定蔬菜的最佳预测模型"""
    # 查询该蔬菜的所有模型评估，按预测准确率降序排序
    best_evaluation = db.query(ModelEvaluation).filter(
        ModelEvaluation.vegetable_id == vegetable_id
    ).order_by(desc(ModelEvaluation.prediction_accuracy)).first()
    
    if not best_evaluation:
        return None
    
    # 获取对应的训练模型
    training_model = db.query(ModelTraining).filter(
        ModelTraining.id == best_evaluation.model_id
    ).first()
    
    if not training_model:
        return None
    
    return {
        "evaluation": best_evaluation,
        "training": training_model
    }

def get_training_statistics(db: Session) -> Dict[str, int]:
    """获取模型训练统计数据"""
    total_count = db.query(func.count(ModelTraining.id)).scalar()
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(func.count(ModelTraining.id)).filter(
        ModelTraining.start_time >= today_start
    ).scalar()
    
    running_count = db.query(func.count(ModelTraining.id)).filter(
        ModelTraining.status.in_(["running", "pending"])
    ).scalar()
    
    completed_count = db.query(func.count(ModelTraining.id)).filter(
        ModelTraining.status == "completed"
    ).scalar()
    
    failed_count = db.query(func.count(ModelTraining.id)).filter(
        ModelTraining.status == "failed"
    ).scalar()
    
    return {
        "total_count": total_count,
        "today_count": today_count,
        "running_count": running_count,
        "completed_count": completed_count,
        "failed_count": failed_count
    }

def get_model_trainings_by_combination(
    db: Session, 
    product_name: str,
    status: Optional[str] = None,
    algorithm: Optional[str] = None,
    limit: int = 100
) -> List[ModelTraining]:
    """根据蔬菜名称获取模型训练记录
    
    Args:
        db: 数据库会话
        product_name: 蔬菜名称
        status: 训练状态，可选
        algorithm: 算法名称，可选
        limit: 结果数量限制
        
    Returns:
        训练记录列表
    """
    # 直接使用product_name字段进行查询
    query = db.query(ModelTraining).filter(ModelTraining.product_name == product_name)
    
    if status:
        query = query.filter(ModelTraining.status == status)
    
    if algorithm:
        query = query.filter(ModelTraining.algorithm == algorithm)
    
    return query.order_by(desc(ModelTraining.start_time)).limit(limit).all()

def get_latest_predictions_by_combination(
    db: Session, 
    product_name: str,
    days: int = 7,
    algorithm: Optional[str] = None
) -> List[Prediction]:
    """根据蔬菜名称获取最新预测
    
    Args:
        db: 数据库会话
        product_name: 蔬菜名称
        days: 预测天数
        algorithm: 算法名称，可选
        
    Returns:
        预测记录列表
    """
    # 计算起始和结束日期
    start_date = datetime.now()
    end_date = start_date + timedelta(days=days)
    
    # 找到匹配的蔬菜IDs
    vegetable_query = db.query(Vegetable.id).filter(
        Vegetable.product_name == product_name
    )
    
    vegetable_ids = [v[0] for v in vegetable_query.all()]
    if not vegetable_ids:
        return []
    
    # 查询这些蔬菜ID的预测记录
    query = db.query(Prediction).filter(
        Prediction.vegetable_id.in_(vegetable_ids),
        Prediction.predicted_date >= start_date,
        Prediction.predicted_date <= end_date
    )
    
    if algorithm:
        query = query.filter(Prediction.algorithm == algorithm)
    
    # 按日期和创建时间排序，确保获取最新数据
    return query.order_by(Prediction.predicted_date, desc(Prediction.created_at)).all()

def get_model_evaluations(
    db: Session, 
    skip: int = 0, 
    limit: int = 10, 
    algorithm: Optional[str] = None,
    vegetable_id: Optional[int] = None,
    product_name: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    max_error: Optional[float] = None
) -> Tuple[List[ModelEvaluation], int]:
    """
    获取模型评估记录，支持分页和过滤
    返回评估列表及总数
    """
    query = db.query(ModelEvaluation)
    
    # 应用过滤条件
    if algorithm:
        query = query.filter(ModelEvaluation.algorithm == algorithm)
        
    if vegetable_id:
        query = query.filter(ModelEvaluation.vegetable_id == vegetable_id)
        
    if product_name:
        # 使用 ilike 进行大小写不敏感的模糊匹配
        query = query.filter(ModelEvaluation.product_name.ilike(f"%{product_name}%"))
        
    if min_accuracy is not None:
        query = query.filter(ModelEvaluation.prediction_accuracy >= min_accuracy)
        
    if max_error is not None:
        query = query.filter(ModelEvaluation.mean_absolute_error <= max_error)
    
    # 获取总记录数
    total = query.count()
    
    # 应用分页并获取记录
    evaluations = query.order_by(desc(ModelEvaluation.evaluation_date)).offset(skip).limit(limit).all()
    
    return evaluations, total
