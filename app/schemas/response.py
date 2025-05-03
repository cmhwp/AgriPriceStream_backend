from pydantic import BaseModel
from typing import Union, Optional, List, Dict, Any, Generic, TypeVar
import json
from datetime import datetime, date

T = TypeVar('T')

# Common response format
class ResponseModel(BaseModel):
    code: int = 200
    data: Optional[Union[Dict[str, Any], List[Any], str, None]] = None
    msg: str = "success"

class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    items: List[T]
    total: int
    page: int
    size: int
    pages: int
    
    @property
    def has_next(self) -> bool:
        return self.page < self.pages
    
    @property
    def has_prev(self) -> bool:
        return self.page > 1

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
                    msg: str = "success") -> 'ResponseModel':
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
    
    from app.schemas.response import ResponseModel
    return ResponseModel(
        code=200,
        data=data,
        msg=msg
    )

def response_error(msg: str = "error", code: int = 400) -> 'ResponseModel':
    """
    创建一个错误的标准响应
    
    Args:
        msg: 错误消息
        code: 错误代码
        
    Returns:
        标准响应对象
    """
    from app.schemas.response import ResponseModel
    return ResponseModel(
        code=code,
        data=None,
        msg=msg
    )

# 添加分页响应工厂函数
def paginate(items: list, page: int, size: int, total: int = None) -> Dict[str, Any]:
    """
    创建分页响应数据
    
    Args:
        items: 当前页的项目列表
        page: 当前页码
        size: 每页大小
        total: 总记录数(不提供时使用items长度)
        
    Returns:
        分页响应数据
    """
    if total is None:
        total = len(items)
    
    pages = (total + size - 1) // size if size > 0 else 0
    
    # 处理SQLAlchemy模型对象列表
    if items and len(items) > 0 and hasattr(items[0], "__table__"):
        items = model_to_dict(items)
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    } 