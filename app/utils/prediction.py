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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.models.models import PriceRecord, Vegetable, Prediction, ModelTraining, ModelEvaluation
from app.crud import predictions as predictions_crud
from app.schemas.prediction import ModelTrainingCreate, PredictionCreate, ModelEvaluationCreate

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
        sequence_length: 序列长度
        history_days: 历史数据天数
        prediction_days: 预测天数
        training_id: 训练记录ID
        min_data_points: 最小数据点数
        from_2022: 是否只使用2022年之后的数据
        
    Returns:
        模型训练结果
    """
    try:
        # 记录训练开始
        if training_id:
            training_record = predictions_crud.get_model_training(db, training_id)
            if training_record:
                predictions_crud.update_model_training(db, training_id, status="running", log="初始化训练环境...")
            else:
                logger.error(f"未找到ID为{training_id}的训练记录")
                return {"success": False, "error": f"未找到ID为{training_id}的训练记录"}
        else:
            # 创建新的训练记录
            vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
            if not vegetable:
                logger.error(f"未找到ID为{vegetable_id}的蔬菜")
                if training_id:
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
                sequence_length=sequence_length,
                log="初始化训练环境..."
            )
            training_record = predictions_crud.create_model_training(db, training_data)
            training_id = training_record.id
        
        # 获取蔬菜信息
        vegetable = db.query(Vegetable).filter(Vegetable.id == vegetable_id).first()
        if not vegetable:
            error_msg = f"未找到ID为{vegetable_id}的蔬菜"
            logger.error(error_msg)
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                log=error_msg,
                end_time=datetime.now()
            )
            return {"success": False, "error": error_msg}
        
        # 更新日志
        safe_log_update(db, training_id, f"正在获取蔬菜 '{vegetable.product_name}' 的历史价格数据...")
        
        # 获取历史价格数据 - 基于蔬菜名称而不仅是ID
        # 首先获取所有具有相同名称的蔬菜ID
        same_name_vegetables = db.query(Vegetable).filter(
            Vegetable.product_name == vegetable.product_name
        ).all()
        
        vegetable_ids = [v.id for v in same_name_vegetables]
        
        if not vegetable_ids:
            vegetable_ids = [vegetable_id]  # 如果没有找到相同名称的蔬菜，则使用当前ID
        
        # 基于所有相同名称的蔬菜ID获取价格记录
        safe_log_update(db, training_id, f"找到 {len(vegetable_ids)} 个相同名称的蔬菜记录，ID: {vegetable_ids}")
        
        # 构建查询
        history_prices_query = db.query(PriceRecord).filter(
            PriceRecord.vegetable_id.in_(vegetable_ids)
        )
        
        if from_2022:
            history_prices_query = history_prices_query.filter(PriceRecord.price_date >= datetime(2022, 1, 1))
        else:
            # 仅获取指定天数的历史数据
            if history_days > 0:
                start_date = datetime.now() - timedelta(days=history_days)
                history_prices_query = history_prices_query.filter(PriceRecord.price_date >= start_date)
        
        # 获取所有价格记录并按日期排序
        history_prices = history_prices_query.order_by(
            PriceRecord.price_date.asc()
        ).all()
        
        if len(history_prices) < min_data_points:
            error_msg = f"数据点数量不足，当前仅有{len(history_prices)}个，最小需要{min_data_points}个"
            logger.warning(error_msg)
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                log=error_msg,
                end_time=datetime.now()
            )
            return {"success": False, "error": error_msg}
        
        safe_log_update(db, training_id, f"成功获取{len(history_prices)}条历史价格数据，开始预处理...")
        
        # 准备数据集 - 需要处理可能的重复日期问题
        # 按日期分组并计算每天的平均价格
        price_data = {}
        for p in history_prices:
            date_key = p.price_date.strftime('%Y-%m-%d')
            if date_key not in price_data:
                price_data[date_key] = {'date': p.price_date, 'prices': []}
            price_data[date_key]['prices'].append(p.price)
        
        # 计算每天的平均价格
        dates = []
        prices = []
        for date_key, data in sorted(price_data.items()):
            dates.append(data['date'])
            # 计算该日期的平均价格
            avg_price = sum(data['prices']) / len(data['prices'])
            prices.append(avg_price)
        
        safe_log_update(db, training_id, f"数据预处理：按日期合并后共有{len(dates)}个唯一数据点")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        # 设置日期为索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 数据规范化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['price']])
        
        # 创建序列数据
        X = []
        y = []
        
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length])
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        
        # 确保数据量充足
        if len(X) < 10:  # 小于10个样本时无法有效训练
            error_msg = f"处理后的数据点数量不足以进行训练，当前仅有{len(X)}个样本"
            logger.warning(error_msg)
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                log=error_msg,
                end_time=datetime.now()
            )
            return {"success": False, "error": error_msg}
        
        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        safe_log_update(db, training_id, f"数据预处理完成，训练集样本数: {len(X_train)}，测试集样本数: {len(X_test)}")
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        batch_size = min(32, len(X_train))
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 使用已定义的LSTMModel类
        # 设置模型参数
        input_size = 1  # 输入特征维度
        hidden_layer_size = 64  # 隐藏层维度
        output_size = 1  # 输出维度
        num_layers = 2  # LSTM层数
        dropout = 0.2  # Dropout率
        
        # 初始化模型
        model = LSTMModel(
            input_size=input_size,
            hidden_layer_size=hidden_layer_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        try:
            # 检查是否有CUDA可用
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            safe_log_update(db, training_id, f"使用设备: {device}")
        except Exception as e:
            # 如果CUDA初始化失败，回退到CPU
            device = torch.device('cpu')
            model = model.to(device)
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            safe_log_update(db, training_id, f"CUDA初始化失败，使用CPU: {str(e)}")
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 设置早停参数
        patience = 20  # 如果20个epoch内验证损失没有改善，则停止训练
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 设置训练参数
        epochs = 200  # 最大训练轮次
        
        safe_log_update(db, training_id, f"开始训练LSTM模型，最大轮次: {epochs}，早停耐心值: {patience}")
        safe_log_update(db, training_id, f"模型结构: 输入维度={input_size}, 隐藏层维度={hidden_layer_size}, LSTM层数={num_layers}, 输出维度={output_size}")
        
        # 训练模型
        model.train()
        
        try:
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
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
                epoch_loss = total_loss / len(X_train)
                
                # 计算验证损失
                model.eval()
                with torch.no_grad():
                    X_test_reshaped = X_test.view(X_test.shape[0], X_test.shape[1], 1)
                    test_outputs = model(X_test_reshaped)
                    val_loss = criterion(test_outputs, y_test).item()
                
                # 早停策略
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                
                # 记录日志
                if (epoch+1) % 5 == 0 or patience_counter >= patience or epoch == 0 or epoch == epochs-1:
                    log_message = f"\nEpoch [{epoch+1}/{epochs}], 训练损失: {epoch_loss:.6f}, 验证损失: {val_loss:.6f}"
                    safe_log_update(db, training_id, log_message)
                
                # 检查是否需要提前停止
                if patience_counter >= patience:
                    log_message = f"\n提前停止训练: 验证损失{patience}轮未改善，当前轮次 {epoch+1}/{epochs}"
                    safe_log_update(db, training_id, log_message)
                    break
                
                # 每10轮检查一次训练记录状态，确保任务未被取消
                if (epoch+1) % 10 == 0:
                    # 检查训练记录状态
                    updated_record = predictions_crud.get_model_training(db, training_id)
                    if updated_record.status != "running":
                        log_message = f"\n训练被用户中断，状态: {updated_record.status}"
                        safe_log_update(db, training_id, log_message)
                        return {"success": False, "error": "训练被中断"}
            
            # 加载最佳模型
            model.load_state_dict(best_model_state)
            
            # 保存模型 - 确保文件名表明这是基于产品名称的模型
            sanitized_name = vegetable.product_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            model_filename = f"lstm_product_{sanitized_name}.pth"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            # 移动模型回CPU再保存
            model = model.to(torch.device('cpu'))
            
            # 检查状态字典中的键名是否需要重命名（比如将'linear.'重命名为'fc.'以保持一致性）
            state_dict = model.state_dict()
            standardized_state_dict = state_dict
            
            # 保存模型，包含更多关于训练数据的信息
            torch.save({
                'model_state_dict': standardized_state_dict,
                'input_size': input_size,  # 使用与加载部分一致的变量名
                'hidden_layer_size': hidden_layer_size,  # 使用与加载部分一致的变量名
                'output_size': output_size,  # 使用与加载部分一致的变量名
                'num_layers': num_layers,
                'scaler': scaler,
                'product_name': vegetable.product_name,
                'vegetable_id': vegetable_id,
                'included_vegetable_ids': vegetable_ids,
                'training_date': datetime.now().isoformat(),
                'data_points': len(history_prices),
                'unique_dates': len(dates),
                'date_range': f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
            }, model_path)
            
            # 模型评估
            model.eval()
            with torch.no_grad():
                # 所有测试数据的预测
                X_test_cpu = X_test.to(torch.device('cpu'))
                X_test_reshaped = X_test_cpu.view(X_test_cpu.shape[0], X_test_cpu.shape[1], 1)
                test_predictions = model(X_test_reshaped).numpy()
                
                # 反归一化
                test_predictions = scaler.inverse_transform(test_predictions)
                y_test_actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
                
                # 计算评估指标
                mse = mean_squared_error(y_test_actual, test_predictions)
                mae = mean_absolute_error(y_test_actual, test_predictions)
                r2 = r2_score(y_test_actual, test_predictions)
                
                # 计算预测准确率 (预测价格与实际价格的百分比偏差)
                accuracy_threshold = 0.15  # 15%偏差内认为是准确的
                accurate_predictions = 0
                
                for i in range(len(test_predictions)):
                    actual = y_test_actual[i][0]
                    predicted = test_predictions[i][0]
                    percent_error = abs(actual - predicted) / actual if actual != 0 else 1
                    
                    if percent_error <= accuracy_threshold:
                        accurate_predictions += 1
                
                prediction_accuracy = accurate_predictions / len(test_predictions) if len(test_predictions) > 0 else 0
                
                # 创建并保存评估记录
                evaluation_data = ModelEvaluationCreate(
                    model_id=training_id,
                    algorithm="LSTM",
                    mean_absolute_error=float(mae),
                    mean_squared_error=float(mse),
                    r_squared=float(r2),
                    prediction_accuracy=float(prediction_accuracy),
                    vegetable_id=vegetable_id,
                    product_name=vegetable.product_name
                )
                
                evaluation = predictions_crud.create_model_evaluation(db, evaluation_data)
                
                # 更新训练记录为完成状态
                predictions_crud.update_model_training(
                    db, training_id, 
                    status="completed", 
                    end_time=datetime.now()
                )
                
                # 记录最终评估结果
                eval_log = f"\n训练完成，模型评估结果:\n"
                eval_log += f"均方误差 (MSE): {mse:.4f}\n"
                eval_log += f"平均绝对误差 (MAE): {mae:.4f}\n"
                eval_log += f"决定系数 (R²): {r2:.4f}\n"
                eval_log += f"预测准确率: {prediction_accuracy*100:.2f}%\n"
                eval_log += f"模型保存路径: {model_path}"
                
                safe_log_update(db, training_id, eval_log)
                
                # 返回结果
                return {
                    "success": True,
                    "model_path": model_path,
                    "evaluation": {
                        "id": evaluation.id,
                        "mse": float(mse),
                        "mae": float(mae),
                        "r2": float(r2),
                        "accuracy": float(prediction_accuracy)
                    }
                }
        except Exception as e:
            error_msg = f"训练过程中出错: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                log=f"{training_record.log}\n{error_msg}",
                end_time=datetime.now()
            )
            return {"success": False, "error": str(e)}
            
    except Exception as e:
        error_msg = f"模型训练初始化失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        if training_id:
            predictions_crud.update_model_training(
                db, training_id, 
                status="failed", 
                log=error_msg,
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
    
    # 检查last_sequence是否已经是3D张量
    if len(last_sequence.shape) == 1:
        # 如果是1D数组，转换为3D: [batch_size, sequence_length, feature_dim]
        current_batch = torch.FloatTensor(last_sequence).reshape(1, -1, 1)
    elif len(last_sequence.shape) == 2:
        # 如果是2D数组，增加特征维度: [batch_size, sequence_length, feature_dim]
        current_batch = torch.FloatTensor(last_sequence).reshape(1, last_sequence.shape[0], 1)
    else:
        # 假设已经是正确的3D张量形状
        current_batch = torch.FloatTensor(last_sequence)
    
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
        model_filename = f"lstm_product_{sanitized_name}.pth"
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
            
            # 获取模型参数 - 处理新格式和旧格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新格式 - 完整的字典包含额外信息
                input_dim = checkpoint.get('input_size', 1)
                hidden_dim = checkpoint.get('hidden_layer_size', 64)
                output_dim = checkpoint.get('output_size', 1)
                num_layers = checkpoint.get('num_layers', 2)
                
                # 记录模型训练数据信息
                product_name = checkpoint.get('product_name', vegetable.product_name)
                training_date = checkpoint.get('training_date', 'unknown')
                date_range = checkpoint.get('date_range', 'unknown')
                data_points = checkpoint.get('data_points', 'unknown')
                
                logger.info(f"使用已训练的产品'{product_name}'模型，训练日期={training_date}，数据点数={data_points}，日期范围={date_range}")
                
                # 重建模型
                model = LSTMModel(
                    input_size=input_dim, 
                    hidden_layer_size=hidden_dim, 
                    output_size=output_dim, 
                    num_layers=num_layers
                )
                
                # 处理键名不匹配的情况 (fc -> linear)
                state_dict = checkpoint['model_state_dict']
                
                # 检查是否需要处理键名不匹配的问题
                # 检查state_dict是否有fc层相关的键
                has_fc_keys = any(key.startswith('fc.') for key in state_dict.keys())
                
                if has_fc_keys:
                    # 需要重命名键
                    new_state_dict = {}
                    for key in state_dict:
                        if key.startswith('fc.'):
                            new_key = key.replace('fc.', 'linear.')
                            new_state_dict[new_key] = state_dict[key]
                        else:
                            new_state_dict[key] = state_dict[key]
                    model.load_state_dict(new_state_dict)
                else:
                    # 直接加载
                    model.load_state_dict(state_dict)
                
                # 获取缩放器
                scaler = checkpoint['scaler']
            else:
                # 旧格式 - 直接是模型状态字典
                input_dim = 1
                hidden_dim = 64
                output_dim = 1
                num_layers = 2
                
                # 记录使用默认参数
                logger.info(f"使用旧格式的模型，使用默认参数")
                
                # 创建LSTM模型
                model = LSTMModel(
                    input_size=input_dim, 
                    hidden_layer_size=hidden_dim, 
                    output_size=output_dim, 
                    num_layers=num_layers
                )
                
                # 检查是否需要处理键名不匹配的问题
                # 检查checkpoint是否有fc层相关的键
                has_fc_keys = any(key.startswith('fc.') for key in checkpoint.keys())
                
                if has_fc_keys:
                    # 需要重命名键
                    new_state_dict = {}
                    for key in checkpoint:
                        if key.startswith('fc.'):
                            new_key = key.replace('fc.', 'linear.')
                            new_state_dict[new_key] = checkpoint[key]
                        else:
                            new_state_dict[key] = checkpoint[key]
                    model.load_state_dict(new_state_dict)
                else:
                    # 直接加载
                    model.load_state_dict(checkpoint)
                
                # 创建一个新的缩放器
                scaler = MinMaxScaler(feature_range=(0, 1))
                # 使用历史数据拟合缩放器
                if len(df) > 0:
                    scaler.fit(df[['price']].values)
            
            model.eval()
            
            # 准备预测数据的序列长度
            sequence_length = checkpoint.get('sequence_length', optimal_sequence_length)
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
