from typing import Any, Dict, List, Optional, Union
from app.schemas.response import ResponseModel
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import scoped_session
from pydantic import BaseModel
import json
from datetime import datetime, date

def serialize_datetime(obj):
    """序列化日期时间对象"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def model_to_dict(obj, exclude_attrs=None):
    """
    将SQLAlchemy模型对象转换为字典
    
    Args:
        obj: SQLAlchemy模型对象或对象列表
        exclude_attrs: 排除的属性列表
        
    Returns:
        可序列化的字典或列表
    """
    if exclude_attrs is None:
        exclude_attrs = []
    
    # 处理列表
    if isinstance(obj, list):
        return [model_to_dict(item, exclude_attrs) for item in obj]
    
    # 处理None
    if obj is None:
        return None
    
    # 处理已经是字典的情况
    if isinstance(obj, dict):
        return obj
    
    # 处理基本类型
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 处理日期时间对象
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # 处理SQLAlchemy模型对象
    if hasattr(obj, "__table__"):
        result = {}
        # 添加列属性
        for column in obj.__table__.columns:
            if column.name not in exclude_attrs:
                value = getattr(obj, column.name)
                if isinstance(value, (datetime, date)):
                    result[column.name] = value.isoformat()
                else:
                    result[column.name] = value
        
        # 尝试添加关系属性(避免循环引用)
        for key in dir(obj):
            if key.startswith('_') or key in exclude_attrs or key in result:
                continue
            
            value = getattr(obj, key)
            if callable(value):
                continue
                
            try:
                # 尝试转换为JSON以检查是否可序列化
                json.dumps(value, default=serialize_datetime)
                result[key] = value
            except (TypeError, OverflowError):
                # 如果不可序列化，尝试递归转换
                try:
                    result[key] = model_to_dict(value, exclude_attrs + [key])
                except:
                    # 如果失败，则跳过该属性
                    pass
        
        return result
    
    # 处理Pydantic模型
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    
    # 其他情况
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return str(obj)

def response_success(data: Optional[Union[Dict[str, Any], List[Any], str, Any]] = None, 
                    msg: str = "success") -> ResponseModel:
    """
    创建一个成功的标准响应
    
    Args:
        data: 响应数据
        msg: 响应消息
        
    Returns:
        标准响应对象
    """
    # 处理SQLAlchemy模型对象
    if data is not None:
        if hasattr(data, "__table__") or (isinstance(data, list) and len(data) > 0 and hasattr(data[0], "__table__")):
            data = model_to_dict(data)
        elif isinstance(data, BaseModel):
            data = data.model_dump()
    
    return ResponseModel(
        code=200,
        data=data,
        msg=msg
    )

def response_error(msg: str = "error", code: int = 400) -> ResponseModel:
    """
    创建一个错误的标准响应
    
    Args:
        msg: 错误消息
        code: 错误代码
        
    Returns:
        标准响应对象
    """
    return ResponseModel(
        code=code,
        data=None,
        msg=msg
    ) 