from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from app.utils.response import response_error
import logging

logger = logging.getLogger("exception_handlers")

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理请求验证异常
    """
    errors = exc.errors()
    error_messages = []
    
    for error in errors:
        error_messages.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_error(
            msg="输入数据验证失败",
            code=status.HTTP_422_UNPROCESSABLE_ENTITY
        ).dict()
    )

async def http_exception_handler(request: Request, exc):
    """
    处理HTTP异常
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=response_error(
            msg=exc.detail,
            code=exc.status_code
        ).dict()
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    处理未捕获的异常
    """
    logger.exception(f"未捕获的异常: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_error(
            msg="服务器内部错误",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ).dict()
    ) 