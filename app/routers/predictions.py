from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import re
import logging

from app.db.database import get_db, get_db_instance
from app.utils import prediction as prediction_utils
from app.crud import predictions as predictions_crud
from app.models.models import Vegetable, ModelTraining
from app.schemas.prediction import (
    PredictionResponse, 
    PredictionResult,
    ModelTrainingResponse,
    ModelEvaluation,
    ModelEvaluationResponse,
    ModelEvaluationFilter,
    TrainingConfig,
    BestPurchaseDay,
    ModelComparison,
    PredictionRequest,
    ModelTrainingCreate
)
from app.schemas.response import ResponseModel, paginate

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)

# 配置日志记录器
logger = logging.getLogger("vegetable_price_prediction")

@router.post("/train-all-models", response_model=ResponseModel)
async def train_all_vegetable_models(
    background_tasks: BackgroundTasks,
    config: Optional[TrainingConfig] = None,
    db: Session = Depends(get_db)
):
    """
    在后台任务中训练所有蔬菜的LSTM价格预测模型
    """
    if config is None:
        # 默认使用LSTM算法，其他参数使用默认值
        config = TrainingConfig(algorithm="LSTM")
    
    # 使用后台任务来运行自动训练，避免阻塞API响应
    def train_all_models_background(db_session):
        try:
            prediction_utils.auto_train_models(db_session)
        except Exception as e:
            logger.error(f"训练所有模型时出错: {str(e)}")
    
    # 启动后台任务
    background_tasks.add_task(train_all_models_background, db)
    
    return ResponseModel(
        data={
            "status": "pending",
            "message": "所有蔬菜的模型训练已启动",
            "config": {
                "algorithm": config.algorithm,
                "history_days": config.history_days or 365,
                "prediction_days": config.prediction_days or 7,
                "sequence_length": config.sequence_length or 30
            }
        },
        code=0,
        msg="所有蔬菜的模型训练已启动"
    )

@router.post("/train-model", response_model=ResponseModel)
async def train_model(
    background_tasks: BackgroundTasks,
    vegetable_id: int,
    config: Optional[TrainingConfig] = None,
    db: Session = Depends(get_db)
):
    """
    在后台任务中训练指定蔬菜的价格预测模型
    """
    if config is None:
        # 默认使用LSTM算法，其他参数使用默认值
        config = TrainingConfig(algorithm="LSTM")
    
    # 获取蔬菜信息，以获取蔬菜名称
    vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
    if not vegetable:
        return ResponseModel(
            data=None,
            code=404,
            msg="未找到指定蔬菜"
        )
    
    # 检查是否已有正在训练的该蔬菜模型
    existing_training = db.query(ModelTraining).filter(
        ModelTraining.vegetable_id == vegetable_id,
        ModelTraining.status.in_(["pending", "running"])
    ).first()
    
    if existing_training:
        return ResponseModel(
            data={
                "training_id": existing_training.id,
                "status": existing_training.status,
                "message": "该蔬菜已有正在进行的训练任务"
            },
            code=400,
            msg=f"该蔬菜已有正在进行的训练任务(ID: {existing_training.id})"
        )
    
    # 创建训练记录
    training_data = ModelTrainingCreate(
        algorithm=config.algorithm,
        status="pending",
        vegetable_id=vegetable_id,
        product_name=vegetable.product_name,
        history_days=config.history_days,
        prediction_days=config.prediction_days,
        smoothing=config.smoothing,
        seasonality=config.seasonality,
        sequence_length=config.sequence_length,
        start_time=datetime.now(),
        log="训练任务已创建，等待启动..."
    )
    
    # 创建训练记录
    training_record = predictions_crud.create_model_training(db, training_data)
    
    # 定义训练任务包装函数，确保异常被捕获和记录
    async def train_model_task(vegetable_id: int, db: Session, config: TrainingConfig, training_id: int):
        try:
            # 在新的数据库会话中训练模型
            async_db = get_db_instance()
            
            # 使用锁或标记确保任务唯一性
            # 更新状态为准备中
            predictions_crud.update_model_training(
                async_db, training_id, 
                status="pending",
                log="准备启动训练..."
            )
            
            # 调用训练函数
            result = prediction_utils.train_lstm_model(
                vegetable_id=vegetable_id,
                db=async_db,
                sequence_length=config.sequence_length or 60,
                history_days=config.history_days or 365,
                prediction_days=config.prediction_days or 7,
                training_id=training_id
            )
            
            # 检查训练结果
            if not result.get("success", False):
                error_msg = result.get("error", "未知错误")
                logger.error(f"训练失败: {error_msg}")
                
                # 更新训练记录失败状态
                predictions_crud.update_model_training(
                    async_db, training_id, 
                    status="failed",
                    log=f"{predictions_crud.get_model_training(async_db, training_id).log}\n\n训练失败: {error_msg}",
                    end_time=datetime.now()
                )
        except Exception as e:
            logger.exception(f"训练任务执行异常: {str(e)}")
            # 确保会话有效
            try:
                async_db = get_db_instance()
                # 更新训练记录为失败状态
                predictions_crud.update_model_training(
                    async_db, training_id, 
                    status="failed",
                    log=f"{predictions_crud.get_model_training(async_db, training_id).log}\n\n训练执行异常: {str(e)}",
                    end_time=datetime.now()
                )
            except Exception as inner_e:
                logger.critical(f"无法更新训练记录状态: {str(inner_e)}")
        finally:
            # 确保会话关闭
            if 'async_db' in locals():
                async_db.close()
    
    # 在后台任务中训练模型
    background_tasks.add_task(
        train_model_task,
        vegetable_id=vegetable_id,
        db=db,
        config=config,
        training_id=training_record.id
    )
    
    return ResponseModel(
        data={
            "training_id": training_record.id,
            "status": "pending",
            "message": "模型训练已启动"
        },
        code=0,
        msg="模型训练已启动"
    )

@router.post("/predict", response_model=ResponseModel)
async def predict_vegetable_price(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    预测指定蔬菜未来价格
    """
    result = prediction_utils.predict_vegetable_price(
        vegetable_id=request.vegetable_id,
        db=db,
        days=request.days,
        use_saved_model=True
    )
    
    if not result["success"]:
        return ResponseModel(
            data=None,
            code=400,
            msg=result["error"]
        )
    
    return ResponseModel(
        data=result,
        code=0,
        msg="预测成功"
    )

@router.get("/vegetable/{vegetable_id}", response_model=ResponseModel)
async def get_vegetable_predictions(
    vegetable_id: int,
    days: Optional[int] = 7,
    algorithm: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    获取指定蔬菜的预测记录
    """
    # 检查是否已经有预测数据
    predictions = predictions_crud.get_latest_predictions(db, vegetable_id, days, algorithm)
    
    # 如果没有预测数据，则生成新的预测
    if not predictions:
        result = prediction_utils.predict_vegetable_price(vegetable_id, db, days)
        
        if not result["success"]:
            return ResponseModel(
                data=None,
                code=400,
                msg=result["error"]
            )
        
        # 重新获取预测
        predictions = predictions_crud.get_latest_predictions(db, vegetable_id, days, algorithm)
    
    # 格式化结果
    vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
    if not vegetable:
        raise HTTPException(status_code=404, detail="蔬菜不存在")
    
    prediction_days = [
        {"date": p.predicted_date.isoformat(), "price": p.predicted_price}
        for p in predictions
    ]
    
    result = {
        "vegetable_id": vegetable_id,
        "vegetable_name": vegetable.product_name,
        "provenance_name": vegetable.provenance_name,
        "current_price": vegetable.average_price,
        "predictions": prediction_days,
        "algorithm": predictions[0].algorithm if predictions else None
    }
    
    return ResponseModel(
        data=result,
        code=0,
        msg="获取预测数据成功"
    )

@router.get("/best-purchase-day/{vegetable_id}", response_model=ResponseModel)
async def get_best_day_to_purchase(
    vegetable_id: int,
    db: Session = Depends(get_db)
):
    """
    获取最佳购买日建议
    """
    result = prediction_utils.get_best_purchase_day(vegetable_id, db)
    
    if not result["success"]:
        return ResponseModel(
            data=None,
            code=400,
            msg=result["error"]
        )
    
    return ResponseModel(
        data=result,
        code=0,
        msg="获取最佳购买日成功"
    )

@router.get("/training/history", response_model=ResponseModel)
async def get_training_history(
    status: Optional[str] = None,
    algorithm: Optional[str] = None,
    vegetable_id: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    获取模型训练历史记录
    """
    trainings = predictions_crud.get_model_trainings(
        db, status=status, algorithm=algorithm, vegetable_id=vegetable_id, limit=limit
    )
    
    # 将SQLAlchemy模型转换为Pydantic模式
    training_responses = [
        ModelTrainingResponse(
            id=t.id,
            algorithm=t.algorithm,
            status=t.status,
            start_time=t.start_time,
            end_time=t.end_time,
            log=t.log,
            vegetable_id=t.vegetable_id,
            product_name=t.product_name,
            history_days=t.history_days,
            prediction_days=t.prediction_days,
            smoothing=t.smoothing,
            seasonality=t.seasonality,
            sequence_length=t.sequence_length
        ) for t in trainings
    ]
    
    return ResponseModel(
        data=training_responses,
        code=0,
        msg="获取训练历史成功"
    )

@router.get("/training/{training_id}", response_model=ResponseModel)
async def get_training_details(
    training_id: int,
    db: Session = Depends(get_db)
):
    """
    获取特定训练任务的详细信息
    """
    training = predictions_crud.get_model_training(db, training_id)
    if not training:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    
    # 将SQLAlchemy模型转换为Pydantic模式
    training_response = ModelTrainingResponse(
        id=training.id,
        algorithm=training.algorithm,
        status=training.status,
        start_time=training.start_time,
        end_time=training.end_time,
        log=training.log,
        vegetable_id=training.vegetable_id,
        product_name=training.product_name,
        history_days=training.history_days,
        prediction_days=training.prediction_days,
        smoothing=training.smoothing,
        seasonality=training.seasonality,
        sequence_length=training.sequence_length
    )
    
    return ResponseModel(
        data=training_response,
        code=0,
        msg="获取训练详情成功"
    )

@router.get("/training/{training_id}/status", response_model=ResponseModel)
async def get_training_status(
    training_id: int,
    db: Session = Depends(get_db)
):
    """
    获取特定训练任务的状态信息（轻量级API，用于前端轮询）
    """
    training = predictions_crud.get_model_training(db, training_id)
    if not training:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    
    # 提取进度信息，从日志中解析当前epoch
    current_epoch = 0
    total_epochs = 0
    last_loss = None
    
    if training.log:
        # 从日志中提取进度信息
        epoch_matches = re.findall(r"Epoch \[(\d+)/(\d+)\]", training.log)
        if epoch_matches:
            current_epoch, total_epochs = map(int, epoch_matches[-1])
        
        # 提取最新的损失值
        loss_matches = re.findall(r"Loss: (\d+\.\d+)", training.log)
        if loss_matches:
            last_loss = float(loss_matches[-1])
    
    # 计算进度百分比
    progress_percent = 0
    if total_epochs > 0:
        progress_percent = min(round((current_epoch / total_epochs) * 100), 100)
    
    # 构建轻量级响应
    status_response = {
        "id": training.id,
        "status": training.status,
        "progress_percent": progress_percent,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "last_loss": last_loss,
        "start_time": training.start_time,
        "end_time": training.end_time,
        "running_time": (datetime.now() - training.start_time).total_seconds() if training.start_time else 0,
        "vegetable_id": training.vegetable_id,
        "product_name": training.product_name
    }
    
    return ResponseModel(
        data=status_response,
        code=0,
        msg="获取训练状态成功"
    )

@router.get("/evaluations", response_model=ResponseModel)
async def get_model_evaluations(
    page: int = Query(1, ge=1, description="页码，从1开始"),
    limit: int = Query(10, ge=1, le=100, description="每页数量，最大100"),
    algorithm: Optional[str] = None,
    vegetable_id: Optional[int] = None,
    product_name: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    max_error: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    获取模型评估记录，支持分页和过滤条件
    
    - **page**: 页码（从1开始）
    - **limit**: 每页显示数量
    - **algorithm**: 筛选特定算法
    - **vegetable_id**: 筛选特定蔬菜ID
    - **product_name**: 按蔬菜名称搜索（支持模糊匹配）
    - **min_accuracy**: 最小预测准确率过滤
    - **max_error**: 最大平均绝对误差过滤
    """
    # 计算跳过的记录数
    skip = (page - 1) * limit
    
    # 获取评估记录和总数
    evaluations, total = predictions_crud.get_model_evaluations(
        db=db,
        skip=skip,
        limit=limit,
        algorithm=algorithm,
        vegetable_id=vegetable_id,
        product_name=product_name,
        min_accuracy=min_accuracy,
        max_error=max_error
    )
    
    # 使用分页工厂生成分页响应
    pagination = paginate(evaluations, page, limit, total)
    
    return ResponseModel(
        data=pagination,
        code=0,
        msg="获取模型评估记录成功"
    )

@router.get("/evaluations/{evaluation_id}", response_model=ResponseModel)
async def get_evaluation_details(
    evaluation_id: int,
    db: Session = Depends(get_db)
):
    """
    获取单个模型评估的详细信息
    """
    evaluation = predictions_crud.get_model_evaluation(db, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="未找到指定的模型评估记录")
    
    # 获取相关蔬菜信息
    vegetable = db.query(Vegetable).filter(Vegetable.id == evaluation.vegetable_id).first()
    vegetable_name = vegetable.product_name if vegetable else None
    
    # 获取模型训练信息
    training = predictions_crud.get_model_training(db, evaluation.model_id)
    
    # 构建扩展的评估数据
    evaluation_data = {
        "id": evaluation.id,
        "model_id": evaluation.model_id,
        "algorithm": evaluation.algorithm,
        "mean_absolute_error": evaluation.mean_absolute_error,
        "mean_squared_error": evaluation.mean_squared_error,
        "r_squared": evaluation.r_squared,
        "prediction_accuracy": evaluation.prediction_accuracy,
        "evaluation_date": evaluation.evaluation_date,
        "vegetable_id": evaluation.vegetable_id,
        "product_name": evaluation.product_name or vegetable_name,
        "training_status": training.status if training else None,
        "training_date": training.start_time if training else None
    }
    
    return ResponseModel(
        data=evaluation_data,
        code=0,
        msg="获取模型评估详情成功"
    )