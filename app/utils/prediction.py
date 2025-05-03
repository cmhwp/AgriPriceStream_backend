import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import func
import traceback

from app.models.models import PriceRecord, Vegetable, Prediction, ModelTraining, ModelEvaluation
from app.crud import predictions as predictions_crud
from app.schemas.prediction import ModelTrainingCreate, PredictionCreate

# 配置日志
logger = logging.getLogger("vegetable_price_prediction")

# 模型保存路径
MODEL_DIR = "app/models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

# 为时间序列数据创建自定义Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        # 使用零初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        # 初始化单元状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor形状 (batch_size, seq_length, hidden_size)
        
        # 获取最后一个时间步的输出
        out = self.linear(out[:, -1, :])
        return out

def fetch_price_data(vegetable_id: int, db: Session, days: int = 365, from_2022: bool = True) -> pd.DataFrame:
    """获取特定蔬菜的历史价格数据
    
    Args:
        vegetable_id: 蔬菜ID，用于获取蔬菜名称
        db: 数据库会话
        days: 历史数据天数，如果from_2022为True，则此参数被忽略
        from_2022: 是否获取从2022年1月1日开始的所有数据
        
    Returns:
        包含价格历史数据的DataFrame
    """
    # 计算起始日期
    if from_2022:
        start_date = datetime(2022, 1, 1).date()
        logger.info(f"获取从2022年1月1日开始的所有历史数据")
    else:
        start_date = datetime.now() - timedelta(days=days)
    
    # 首先获取蔬菜信息以确定product_name
    vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
    if not vegetable:
        logger.error(f"未找到ID为{vegetable_id}的蔬菜")
        return pd.DataFrame()  # 返回空DataFrame
    
    # 获取相同名称的所有价格记录并按时间排序
    price_records = db.query(PriceRecord).join(
        Vegetable, PriceRecord.vegetable_id == Vegetable.id
    ).filter(
        Vegetable.product_name == vegetable.product_name,
        PriceRecord.price_date >= start_date
    ).all()
    
    # 创建数据框
    data = []
    for record in price_records:
        data.append({
            'price_date': record.price_date,
            'average_price': record.average_price,
            'top_price': record.top_price,
            'minimum_price': record.minimum_price,
            'provenance_name': record.provenance_name,
            'product_name': vegetable.product_name
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 如果没有数据，返回空DataFrame
    if len(df) == 0:
        return df
    
    # 对同一天的数据进行平均处理（如果一种蔬菜有多个产地，取平均价格）
    df = df.groupby('price_date').agg({
        'average_price': 'mean',
        'top_price': 'mean',
        'minimum_price': 'mean',
        'product_name': 'first'
    }).reset_index()
    
    # 按日期排序
    df = df.sort_values('price_date')
    
    # 记录数据集大小
    if len(df) > 0:
        logger.info(f"获取到蔬菜'{vegetable.product_name}'的数据 {len(df)} 条，日期范围: {df['price_date'].min()} 到 {df['price_date'].max()}")
    
    return df

def prepare_lstm_data(df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """准备LSTM训练数据

    Args:
        df: 价格数据DataFrame
        sequence_length: 序列长度 (用于预测的历史天数)
        
    Returns:
        训练数据X, 标签y, 价格缩放器
    """
    # 使用平均价格作为目标
    data = df[['average_price']].values
    
    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    # 构建序列数据
    X, y = [], []
    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:i+sequence_length])
        y.append(data_normalized[i+sequence_length])
    
    return np.array(X), np.array(y), scaler

def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> torch.utils.data.DataLoader:
    """创建用于PyTorch的DataLoader"""
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # 创建DataLoader
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_lstm_model(
    vegetable_id: int, 
    db: Session,
    sequence_length: int = 30, 
    history_days: int = 365, 
    prediction_days: int = 7, 
    training_id: Optional[int] = None,
    min_data_points: int = 30,
    from_2022: bool = True
) -> Dict[str, Any]:
    """训练LSTM模型预测蔬菜价格
    
    Args:
        vegetable_id: 蔬菜ID
        db: 数据库会话
        sequence_length: LSTM序列长度 (历史天数)，如果为None则自动设置
        history_days: 使用多少天的历史数据，如果from_2022为True则被忽略
        prediction_days: 预测未来多少天的价格
        training_id: 训练记录ID，如果为None则创建新记录
        min_data_points: 最小需要的数据点数量
        from_2022: 是否使用从2022年开始的所有数据
    """
    training_record = None
    try:
        # 记录训练开始
        if training_id:
            training_record = predictions_crud.get_model_training(db, training_id)
            if training_record:
                predictions_crud.update_model_training(db, training_id, status="running")
            else:
                logger.error(f"未找到ID为{training_id}的训练记录")
                return {"success": False, "error": f"未找到ID为{training_id}的训练记录"}
        else:
            # 创建新的训练记录
            vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
            if not vegetable:
                logger.error(f"未找到ID为{vegetable_id}的蔬菜")
                predictions_crud.update_model_training(
                    db, training_id, 
                    status="failed", 
                    log=f"未找到ID为{vegetable_id}的蔬菜",
                    end_time=datetime.now()
                )
                return {"success": False, "error": "未找到指定蔬菜"}
            
            training_data = ModelTrainingCreate(
            algorithm="LSTM",
                status="running",
                vegetable_id=vegetable_id,
                product_name=vegetable.product_name,
                history_days=history_days,
                prediction_days=prediction_days,
                sequence_length=sequence_length
            )
            training_record = predictions_crud.create_model_training(db, training_data)
            training_id = training_record.id
        
        # 获取蔬菜信息
        vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
        if not vegetable:
            logger.error(f"未找到ID为{vegetable_id}的蔬菜")
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                log=f"未找到ID为{vegetable_id}的蔬菜",
                end_time=datetime.now()
            )
            return {"success": False, "error": "未找到指定蔬菜"}
        
        # 获取价格数据 - 使用从2022年开始的数据
        df = fetch_price_data(vegetable_id, db, days=history_days, from_2022=from_2022)
        
        # 检查数据是否足够
        if len(df) < min_data_points:
            logger.warning(f"蔬菜 '{vegetable.product_name}' 数据不足，仅有 {len(df)} 条记录，需要至少 {min_data_points} 条")
            
            # 更新训练记录
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                end_time=datetime.now()
            )
            safe_log_update(db, training_id, f"\n数据不足，仅有 {len(df)} 条记录，需要至少 {min_data_points} 条")
            return {"success": False, "error": f"历史数据不足，需要至少 {min_data_points} 天的数据"}
        
        # 动态设置序列长度，最少7天，最多30天，或者数据长度的一半
        optimal_sequence_length = min(30, max(7, len(df) // 2))
        if sequence_length is None or sequence_length > len(df) - 1:
            sequence_length = optimal_sequence_length
            logger.info(f"自动设置序列长度为 {sequence_length}")
        
        # 更新训练记录
        data_period = f"从 {df['price_date'].min()} 到 {df['price_date'].max()}"
        safe_log_update(db, training_id, f"开始训练模型，数据点数: {len(df)}，序列长度: {sequence_length}，数据期间: {data_period}")
        
        # 准备训练数据
        X, y, scaler = prepare_lstm_data(df, sequence_length)
        
        # 创建DataLoader
        batch_size = min(32, len(X))
        dataloader = create_data_loader(X, y, batch_size=batch_size)
        
        # 定义模型参数
        input_size = 1
        hidden_layer_size = 50
        output_size = 1
        num_layers = 2
        dropout = 0.2
        
        # 创建模型实例
        model = LSTMModel(
            input_size=input_size, 
            hidden_layer_size=hidden_layer_size, 
            output_size=output_size, 
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        epochs = 100
        
        # 记录训练进度
        train_log = f"开始训练LSTM模型，数据点数: {len(X)}，序列长度: {sequence_length}，轮次: {epochs}，数据期间: {data_period}"
        safe_log_update(db, training_id, train_log)
        
        # 训练循环
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for X_batch, y_batch in dataloader:
                # 将数据转换为模型所需的形状 [batch_size, seq_length, feature_dimension]
                X_batch = X_batch.view(X_batch.shape[0], X_batch.shape[1], 1)
                
                # 反向传播和优化
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
                # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
                
            # 计算平均损失
            epoch_loss = total_loss / len(X)
            
            # 早停策略
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch+1) % 10 == 0 or patience_counter >= patience:
                log_message = f"\nEpoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}"
                safe_log_update(db, training_id, log_message)
                
            if patience_counter >= patience:
                log_message = f"\n提前停止训练，轮次 {epoch+1}/{epochs}"
                safe_log_update(db, training_id, log_message)
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 保存模型
        sanitized_name = vegetable.product_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        model_filename = f"lstm_vegetable_{vegetable_id}_{sanitized_name}.pth"
        model_path = os.path.join(MODEL_DIR, model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_layer_size': hidden_layer_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'sequence_length': sequence_length,
            'scaler': scaler,
            'last_trained': datetime.now().isoformat(),
            'data_start_date': df['price_date'].min().strftime('%Y-%m-%d'),
            'data_end_date': df['price_date'].max().strftime('%Y-%m-%d'),
            'data_points': len(df),
            'vegetable_name': vegetable.product_name
        }, model_path)
        
        # 预测未来价格
        future_predictions = predict_future_prices(
            model=model,
            last_sequence=X[-1],
            scaler=scaler,
            days=prediction_days
        )
        
        # 将预测结果保存到数据库
        start_date = df['price_date'].iloc[-1] + timedelta(days=1)
        for i, price in enumerate(future_predictions):
            pred_date = start_date + timedelta(days=i)
            
            # 准备预测记录
            prediction_data = PredictionCreate(
                vegetable_id=vegetable_id,
                predicted_date=pred_date,
                predicted_price=float(price),
                algorithm="LSTM"
            )
            
            # 创建预测记录
            predictions_crud.create_prediction(db, prediction_data)
        
        # 评估模型性能（如果有足够的数据进行验证）
        if len(X) >= 10:  # 至少需要10个数据点进行简单评估
            # 简单的训练集评估
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).view(X.shape[0], X.shape[1], 1)
                y_pred = model(X_tensor).numpy()
                y_true = y
                
                # 转换回原始价格
                y_pred_orig = scaler.inverse_transform(y_pred)
                y_true_orig = scaler.inverse_transform(y_true)
                
                # 计算评估指标
                mae = np.mean(np.abs(y_pred_orig - y_true_orig))
                mse = np.mean((y_pred_orig - y_true_orig) ** 2)
                rmse = np.sqrt(mse)
                
                # 计算R²: 1 - (误差平方和 / 总变异)
                ss_tot = np.sum((y_true_orig - np.mean(y_true_orig)) ** 2)
                ss_res = np.sum((y_true_orig - y_pred_orig) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # 计算预测准确率（基于相对误差）
                relative_errors = np.abs(y_pred_orig - y_true_orig) / y_true_orig
                accuracy = np.mean(relative_errors < 0.1)  # 相对误差<10%视为准确
                
                # 创建评估记录
                predictions_crud.create_model_evaluation(
                    db=db,
                    model_id=training_id,
                    algorithm="LSTM",
                        vegetable_id=vegetable_id,
                    product_name=vegetable.product_name,
                    mean_absolute_error=float(mae),
                    mean_squared_error=float(mse),
                    r_squared=float(r2),
                    prediction_accuracy=float(accuracy)
                )
                
                # 更新训练记录日志
                eval_log = f"\n模型评估结果：\nMAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}\nR²: {r2:.4f}, 准确率: {accuracy*100:.2f}%"
                safe_log_update(db, training_id, eval_log)
            
            # 更新训练记录
        final_log = f"\n模型训练完成并预测未来{prediction_days}天价格"
        safe_log_update(db, training_id, final_log)
        predictions_crud.update_model_training(
            db, training_id, 
            status="completed", 
            end_time=datetime.now()
        )
        
        # 返回结果
        return {
            "success": True,
            "training_id": training_id,
            "vegetable_name": vegetable.product_name,
            "model_path": model_path,
            "data_points": len(df),
            "date_range": f"{df['price_date'].min()} - {df['price_date'].max()}",
            "predictions": [{"date": (start_date + timedelta(days=i)).isoformat(), "price": float(price)} 
                               for i, price in enumerate(future_predictions)]
                               }
            
    except Exception as e:
        logger.error(f"训练模型时出错: {str(e)}")
        if training_id:
            error_msg = f"训练失败: {str(e)}"
            safe_log_update(db, training_id, f"\n{error_msg}")
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                end_time=datetime.now()
            )
        return {"success": False, "error": str(e)}

def predict_future_prices(model: LSTMModel, last_sequence: np.ndarray, scaler: MinMaxScaler, days: int = 7) -> List[float]:
    """使用训练好的模型预测未来价格
    
    Args:
        model: 训练好的LSTM模型
        last_sequence: 最后的价格序列 (用于预测的起点)
        scaler: 用于归一化/反归一化的缩放器
        days: 要预测的天数
        
    Returns:
        未来天数的价格预测 (原始比例)
    """
    model.eval()
    
    # 转换为PyTorch张量并调整形状
    current_batch = torch.FloatTensor(last_sequence.reshape(1, -1, 1))
    
    # 初始化预测列表
    predictions = []
    
    # 预测未来价格
    with torch.no_grad():
        for i in range(days):
            # 获取下一个预测值
            next_pred = model(current_batch)
            
            # 添加到预测列表
            predictions.append(next_pred.item())
            
            # 更新批次以进行下一次预测（移除第一个元素，添加预测）
            # 这种滚动预测方式能够考虑之前的预测对后续预测的影响
            current_batch = torch.cat([
                current_batch[:, 1:, :], 
                next_pred.view(1, 1, 1)
            ], dim=1)
    
    # 反归一化预测结果
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_array).flatten()
    
    return predictions_original

def predict_vegetable_price(vegetable_id: int, db: Session, days: int = 7, use_saved_model: bool = True, from_2022: bool = True) -> Dict[str, Any]:
    """为指定蔬菜预测未来价格
    
    Args:
        vegetable_id: 蔬菜ID
        db: 数据库会话
        days: 预测天数
        use_saved_model: 是否使用保存的模型
        from_2022: 是否使用从2022年开始的所有数据
        
    Returns:
        包含预测结果的字典
    """
    try:
        # 获取蔬菜信息
        vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
        if not vegetable:
            return {"success": False, "error": "未找到指定蔬菜"}
        
        # 获取历史数据 - 使用修改后的fetch_price_data函数，使用从2022年开始的数据
        df = fetch_price_data(vegetable_id, db, from_2022=from_2022)
        if len(df) == 0:
            return {"success": False, "error": "无历史数据"}
        
        # 检查数据是否足够
        min_data_points = 30
        if len(df) < min_data_points:
            return {"success": False, "error": f"历史数据不足，需要至少 {min_data_points} 天的数据"}
        
        # 动态设置序列长度，最少7天，最多30天，或者数据长度的一半
        optimal_sequence_length = min(30, max(7, len(df) // 2))
        
        # 检查是否有保存的模型
        sanitized_name = vegetable.product_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        model_filename = f"lstm_vegetable_{vegetable_id}_{sanitized_name}.pth"
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        # 如果没有找到精确匹配的模型文件，尝试查找同名蔬菜的模型
        if use_saved_model and not os.path.exists(model_path):
            logger.info(f"没有找到精确匹配的模型文件 {model_filename}，尝试查找同名蔬菜的模型")
            # 列出models/saved目录下的所有文件
            all_model_files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
            # 查找包含当前蔬菜名称的模型文件
            for model_file in all_model_files:
                if model_file.endswith('.pth') and sanitized_name in model_file:
                    model_path = os.path.join(MODEL_DIR, model_file)
                    logger.info(f"找到同名蔬菜的模型文件 {model_file}")
                    break
            
            # 如果依然没找到，检查数据库中是否有同名蔬菜的训练记录
            if not os.path.exists(model_path):
                logger.info(f"在文件系统中未找到同名蔬菜的模型文件，尝试查找数据库中的训练记录")
                # 查找最近完成的同名蔬菜的训练记录
                trainings = predictions_crud.get_model_trainings_by_combination(
                    db,
                    product_name=vegetable.product_name,
                    status="completed",
                    algorithm="LSTM",
                    limit=1
                )
                if trainings and len(trainings) > 0:
                    # 找到了训练记录，使用对应的蔬菜ID来构建模型文件名
                    training = trainings[0]
                    if training.vegetable_id and training.vegetable_id != vegetable_id:
                        alt_model_filename = f"lstm_vegetable_{training.vegetable_id}_{sanitized_name}.pth"
                        alt_model_path = os.path.join(MODEL_DIR, alt_model_filename)
                        if os.path.exists(alt_model_path):
                            model_path = alt_model_path
                            logger.info(f"找到同名蔬菜的训练记录对应的模型文件 {alt_model_filename}")
        
        if use_saved_model and os.path.exists(model_path):
            # 加载保存的模型
            checkpoint = torch.load(model_path)
            
            # 获取模型参数
            input_size = checkpoint.get('input_size', 1)
            hidden_layer_size = checkpoint.get('hidden_layer_size', 50)
            output_size = checkpoint.get('output_size', 1)
            num_layers = checkpoint.get('num_layers', 2)
            dropout = checkpoint.get('dropout', 0.2)
            sequence_length = checkpoint.get('sequence_length', optimal_sequence_length)
            
            # 记录模型训练数据信息
            data_start_date = checkpoint.get('data_start_date', 'unknown')
            data_end_date = checkpoint.get('data_end_date', 'unknown')
            data_points = checkpoint.get('data_points', 'unknown')
            
            logger.info(f"使用已训练的模型，序列长度={sequence_length}，训练数据点数={data_points}，日期范围={data_start_date} 到 {data_end_date}")
            
            # 重建模型
            model = LSTMModel(
                input_size=input_size, 
                hidden_layer_size=hidden_layer_size, 
                output_size=output_size, 
                num_layers=num_layers,
                dropout=dropout
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 获取缩放器
            scaler = checkpoint['scaler']
            
            # 准备预测数据
            if len(df) < sequence_length:
                logger.warning(f"数据点数不足 {len(df)}/{sequence_length}，使用可用的最大序列长度")
                sequence_length = max(7, len(df) - 1)
            
            # 获取最新的价格序列
            recent_prices = df['average_price'].values[-sequence_length:].reshape(-1, 1)
            recent_prices_scaled = scaler.transform(recent_prices).flatten()
            
            # 预测未来价格
            future_predictions = predict_future_prices(
                model=model,
                last_sequence=recent_prices_scaled,
                scaler=scaler,
                days=days
            )
        else:
            # 如果没有保存的模型或不使用保存的模型，则训练新模型
            result = train_lstm_model(
                vegetable_id=vegetable_id,
                db=db,
                sequence_length=optimal_sequence_length,  # 使用动态设置的序列长度
                history_days=365,
                prediction_days=days,
                min_data_points=min_data_points,
                from_2022=from_2022
            )
            
            if not result["success"]:
                return result
            
            # 从训练结果获取预测
            future_predictions = [item['price'] for item in result['predictions']]
        
        # 将预测结果保存到数据库
        start_date = df['price_date'].iloc[-1] + timedelta(days=1)
        
        # 清除该蔬菜的旧预测，避免冗余
        # 从vegetable表中查找相同名称的所有蔬菜ID
        similar_vegetable_ids = [v.id for v in db.query(Vegetable).filter(
            Vegetable.product_name == vegetable.product_name
        ).all()]
        
        # 删除这些蔬菜ID的预测记录
        for similar_id in similar_vegetable_ids:
            # 删除未来日期的预测
            db.query(Prediction).filter(
                Prediction.vegetable_id == similar_id,
                Prediction.predicted_date >= start_date
            ).delete()
        
        # 为当前蔬菜ID保存新的预测
        for i, price in enumerate(future_predictions):
            pred_date = start_date + timedelta(days=i)
                
            # 准备预测记录
            prediction_data = PredictionCreate(
                        vegetable_id=vegetable_id,
                        predicted_date=pred_date,
                predicted_price=float(price),
                algorithm="LSTM"
            )
            
            # 创建预测记录
            predictions_crud.create_prediction(db, prediction_data)
        
        # 获取当前价格
        current_price = df['average_price'].values[-1] if len(df) > 0 else None
        
        # 返回预测结果
        return {
                "success": True,
            "vegetable_id": vegetable_id,
                "vegetable_name": vegetable.product_name,
            "provenance_name": vegetable.provenance_name,
            "current_price": current_price,
            "data_points": len(df),
            "date_range": f"{df['price_date'].min()} 到 {df['price_date'].max()}",
            "predictions": [{"date": (start_date + timedelta(days=i)).isoformat(), "price": float(price)} 
                           for i, price in enumerate(future_predictions)],
            "algorithm": "LSTM"
            }
    
    except Exception as e:
        logger.error(f"预测价格时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def get_best_purchase_day(vegetable_id: int, db: Session) -> Dict[str, Any]:
    """分析预测结果，推荐最佳购买日期
    
    Args:
        vegetable_id: 蔬菜ID
        db: 数据库会话
        
    Returns:
        包含最佳购买日建议的字典
    """
    try:
        # 获取蔬菜信息
        vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
        if not vegetable:
            return {"success": False, "error": "未找到指定蔬菜"}
        
        # 首先尝试获取该特定蔬菜ID的预测
        predictions = predictions_crud.get_latest_predictions(db, vegetable_id, days=7)
        
        # 如果没有预测数据，使用蔬菜名称查询
        if not predictions:
            predictions = predictions_crud.get_latest_predictions_by_combination(
                db,
                product_name=vegetable.product_name,
                days=7
            )
        
        # 如果仍然没有预测数据，尝试生成新的预测
        if not predictions:
            result = predict_vegetable_price(vegetable_id, db, days=7)
            if not result["success"]:
                return {"success": False, "error": "无法生成预测数据"}
            
            # 重新获取预测数据
            predictions = predictions_crud.get_latest_predictions(db, vegetable_id, days=7)
        
        if not predictions:
            return {"success": False, "error": "无法获取预测数据"}
        
        # 找出价格最低的日期
        lowest_price = float('inf')
        best_day = None
        
        for pred in predictions:
            if pred.predicted_price < lowest_price:
                lowest_price = pred.predicted_price
                best_day = pred.predicted_date
        
        # 获取当前价格
        current_price = vegetable.average_price if vegetable.average_price else None
        
        # 计算价格降幅百分比
        if current_price and lowest_price < current_price:
            savings = current_price - lowest_price
            savings_percent = (savings / current_price) * 100
        else:
            savings = None
            savings_percent = None
        
        return {
            "success": True,
            "vegetable_id": vegetable_id,
            "vegetable_name": vegetable.product_name,
            "provenance_name": vegetable.provenance_name,
            "best_purchase_date": best_day.isoformat(),
            "predicted_price": lowest_price,
            "current_price": current_price,
            "savings": savings,
            "savings_percent": savings_percent
        }
    
    except Exception as e:
        logger.error(f"获取最佳购买日时出错: {str(e)}")
        return {"success": False, "error": str(e)}

def compare_models(vegetable_id: int, db: Session) -> Dict[str, Any]:
    """比较不同算法对特定蔬菜的预测效果
    
    Args:
        vegetable_id: 蔬菜ID
        db: 数据库会话
        
    Returns:
        包含模型比较结果的字典
    """
    try:
        # 获取蔬菜信息
        vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
        if not vegetable:
            return {"success": False, "error": "未找到指定蔬菜"}
        
        # 获取所有评估记录 - 首先基于vegetable_id查找
        evaluations = predictions_crud.get_model_evaluations_by_vegetable(db, vegetable_id)
        
        # 如果基于ID没有找到记录，则尝试使用产品名称查找
        if not evaluations or len(evaluations) == 0:
            evaluations = predictions_crud.get_model_evaluations_by_combination(db, vegetable.product_name)
            
        if not evaluations:
            return {"success": False, "error": "尚无模型评估数据"}
        
        # 按算法分组统计指标
        algorithm_metrics = {}
        
        for eval in evaluations:
            alg = eval.algorithm
            
            if alg not in algorithm_metrics:
                algorithm_metrics[alg] = {
                    "mae_sum": 0,
                    "mse_sum": 0,
                    "r2_sum": 0,
                    "accuracy_sum": 0,
                    "count": 0,
                    "evaluations": []
                }
            
            # 累加指标
            algorithm_metrics[alg]["mae_sum"] += eval.mean_absolute_error
            algorithm_metrics[alg]["mse_sum"] += eval.mean_squared_error
            algorithm_metrics[alg]["r2_sum"] += eval.r_squared
            algorithm_metrics[alg]["accuracy_sum"] += eval.prediction_accuracy
            algorithm_metrics[alg]["count"] += 1
            algorithm_metrics[alg]["evaluations"].append(eval)
        
        # 计算每个算法的平均指标
        results = {}
        for alg, metrics in algorithm_metrics.items():
            count = metrics["count"]
            results[alg] = {
                "mae": metrics["mae_sum"] / count,
                "mse": metrics["mse_sum"] / count,
                "r2": metrics["r2_sum"] / count,
                "accuracy": metrics["accuracy_sum"] / count,
                "count": count,
                "evaluations": metrics["evaluations"]
            }
        
        # 找出最佳算法 (基于准确率)
        best_algorithm = max(results.items(), key=lambda x: x[1]["accuracy"])
        
        return {
            "success": True,
            "vegetable_id": vegetable_id,
            "vegetable_name": vegetable.product_name,
            "algorithm_metrics": results,
            "best_algorithm": best_algorithm[0],
            "best_algorithm_metrics": best_algorithm[1]
        }
    
    except Exception as e:
        logger.error(f"比较模型时出错: {str(e)}")
        return {"success": False, "error": str(e)}

def get_unique_vegetable_combinations(db: Session) -> List[Dict[str, Any]]:
    """获取所有唯一的蔬菜名称
    
    Args:
        db: 数据库会话
        
    Returns:
        包含唯一蔬菜名称的列表
    """
    # 使用sqlalchemy的distinct和group_by来获取唯一蔬菜名称
    unique_combinations = db.query(
        Vegetable.product_name,
        func.count(Vegetable.id).label('count'),
        func.min(Vegetable.id).label('representative_id')  # 使用每组的最小ID作为代表
    ).group_by(
        Vegetable.product_name
    ).all()
    
    return [
        {
            "product_name": combo.product_name,
            "count": combo.count,
            "vegetable_id": combo.representative_id
        }
        for combo in unique_combinations
    ]

def auto_train_models(db: Session) -> None:
    """
    自动训练所有蔬菜的LSTM价格预测模型，基于蔬菜名称
    
    Args:
        db: 数据库会话
    """
    try:
        # 获取所有唯一的蔬菜名称
        vegetable_combinations = get_unique_vegetable_combinations(db)
        if not vegetable_combinations:
            logger.warning("没有找到要训练的蔬菜")
            return
        
        logger.info(f"找到 {len(vegetable_combinations)} 种蔬菜，开始自动训练LSTM模型")
        
        # 统计成功和失败的数量
        success_count = 0
        skipped_count = 0
        failed_count = 0
        insufficient_data_count = 0
        
        for combo in vegetable_combinations:
            vegetable_id = combo["vegetable_id"]
            product_name = combo["product_name"]
            
            # 组合名称用于日志记录
            combo_name = product_name
            
            # 检查是否已经有已完成的训练记录 - The function has been updated to only consider product_name
            existing_models = predictions_crud.get_model_trainings_by_combination(
                db, 
                product_name=product_name,
                status="completed", 
                algorithm="LSTM",
                limit=1
            )
            
            # 判断是否需要训练新模型
            # 如果已有训练好的模型，且训练时间在30天内，则跳过
            should_train = True
            if existing_models and len(existing_models) > 0:
                last_trained = existing_models[0].end_time
                if last_trained and (datetime.now() - last_trained).days < 30:
                    logger.info(f"蔬菜 '{combo_name}' 已有训练好的模型，跳过训练")
                    skipped_count += 1
                    should_train = False
            
            # 创建训练记录并启动训练
            if should_train:
                logger.info(f"开始训练蔬菜 '{combo_name}' 的LSTM模型")
                
                # 首先获取数据，以便动态设置参数
                df = fetch_price_data(vegetable_id, db, days=365)
                
                # 检查数据是否足够
                min_data_points = 30
                if len(df) < min_data_points:
                    logger.warning(f"蔬菜 '{combo_name}' 数据不足，仅有 {len(df)} 条记录，需要至少 {min_data_points} 条，跳过训练")
                    insufficient_data_count += 1
                    continue
                
                # 显示详细信息，帮助调试
                logger.info(f"蔬菜 '{combo_name}' 有 {len(df)} 条数据，日期范围: {df['price_date'].min()} 到 {df['price_date'].max()}")
                
                # 动态设置序列长度
                sequence_length = min(30, max(7, len(df) // 2))
                history_days = 365    # 使用一年的历史数据
                prediction_days = 7   # 预测未来7天
                
                # 创建训练记录
                training_data = ModelTrainingCreate(
                    algorithm="LSTM",
                    status="pending",
                    vegetable_id=vegetable_id,
                    product_name=combo["product_name"],  # 使用combo中的product_name
                    history_days=history_days,
                    prediction_days=prediction_days,
                    sequence_length=sequence_length
                )
                
                training_record = predictions_crud.create_model_training(db, training_data)
                
                # 异步训练模型
                # 注意：这里不使用后台任务，而是直接调用训练函数
                # 在实际部署中，应该使用后台任务来避免阻塞启动过程
                result = train_lstm_model(
                    vegetable_id=vegetable_id,
                    db=db,
                    sequence_length=sequence_length,
                    history_days=history_days,
                    prediction_days=prediction_days,
                    training_id=training_record.id,
                    min_data_points=min_data_points
                )
                
                if result["success"]:
                    logger.info(f"成功训练蔬菜 '{combo_name}' 的LSTM模型")
                    success_count += 1
                else:
                    logger.warning(f"训练蔬菜 '{combo_name}' 的LSTM模型失败: {result['error']}")
                    failed_count += 1
        
        # 输出汇总信息
        logger.info(f"模型训练完成汇总: 总共 {len(vegetable_combinations)} 种蔬菜，成功 {success_count}，跳过 {skipped_count}，失败 {failed_count}，数据不足 {insufficient_data_count}")
    
    except Exception as e:
        logger.error(f"自动训练模型时出错: {str(e)}")
        logger.error(traceback.format_exc())

# 日志处理函数，确保日志不会过长，如果需要可以截断
def safe_log_update(db: Session, training_id: int, log_message: str, max_length: int = 65000) -> None:
    """安全更新训练日志，确保不超过数据库字段大小限制
    
    Args:
        db: 数据库会话
        training_id: 训练记录ID
        log_message: 要添加的日志消息
        max_length: 日志最大长度，默认65000字符（MySQL TEXT类型通常限制为65,535字节）
    """
    try:
        # 获取当前日志
        training_record = db.query(ModelTraining).filter(ModelTraining.id == training_id).first()
        if not training_record:
            logger.error(f"未找到ID为{training_id}的训练记录")
            return
            
        current_log = training_record.log or ""
        
        # 添加新日志
        updated_log = current_log + log_message
        
        # 如果日志过长，截断并添加提示
        if len(updated_log) > max_length:
            # 保留前30%和后70%的日志
            first_part_length = int(max_length * 0.3)
            second_part_length = max_length - first_part_length - 50  # 为截断消息预留空间
            
            truncated_log = (
                current_log[:first_part_length] + 
                "\n... [日志太长，中间部分已省略] ...\n" + 
                updated_log[-second_part_length:]
            )
            
            logger.warning(f"训练ID {training_id} 的日志过长，已截断")
            
            # 更新数据库
            training_record.log = truncated_log
        else:
            # 直接更新
            training_record.log = updated_log
        
        db.commit()
    except Exception as e:
        logger.error(f"更新训练日志时出错: {str(e)}")
        logger.error(traceback.format_exc())
        # 回滚事务
        db.rollback()
