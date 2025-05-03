from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn
from contextlib import asynccontextmanager
from app.routers import users, vegetables, prices, predictions, admin, dashboard
from app.utils.scheduler import start_scheduler
from app.db.database import engine, SessionLocal
from app.models import models
from app.schemas.response import ResponseModel
from app.utils.response import response_success
from app.utils.exception_handlers import validation_exception_handler, http_exception_handler, unhandled_exception_handler
from app.utils.crawler import crawl_from_2022_to_now
import logging
import threading
import os

# 配置日志
logger = logging.getLogger("main")

# Create database tables
models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动定时爬虫任务
    start_scheduler()
    
    # 检查是否需要初始化历史数据
    db = SessionLocal()
    try:
        # 检查是否有价格记录，如果没有则是首次启动
        record_count = db.query(models.PriceRecord).count()
        if record_count == 0:
            logger.info("首次启动系统，开始爬取历史数据...")
            # 在后台线程中爬取数据，避免阻塞API启动
            thread = threading.Thread(target=crawl_from_2022_to_now)
            thread.daemon = True
            thread.start()
            logger.info("历史数据爬取任务已在后台启动")
    except Exception as e:
        logger.error(f"检查数据初始化状态时出错: {str(e)}")
    finally:
        db.close()
    
    yield
    # 清理资源可以在这里完成

app = FastAPI(
    title="AgriPriceStream API",
    description="蔬菜价格数据采集与分析系统API",
    version="1.0.0",
    lifespan=lifespan
)

# 注册异常处理器
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含所有路由
app.include_router(users.router)
app.include_router(vegetables.router)
app.include_router(prices.router)
app.include_router(predictions.router)
app.include_router(admin.router)
app.include_router(dashboard.router)

@app.get("/", response_model=ResponseModel)
async def root():
    return response_success(
        data={"message": "欢迎使用蔬菜价格数据采集与分析系统API"}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
