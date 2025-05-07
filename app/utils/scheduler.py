import logging
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.sql import func

from app.db.database import SessionLocal
from app.models.models import PriceRecord, Vegetable, CrawlerActivity
from app.utils.crawler import fetch_data, save_vegetable_data, crawl_historical_data
from app.crud.dashboard import create_crawler_activity

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("自动爬虫调度器")

# 创建调度器
scheduler = AsyncIOScheduler()

async def crawl_latest_data():
    """
    爬取最新的数据（从数据库中最新记录日期到当前日期的数据）
    """
    logger.info("开始执行自动爬取任务...")
    
    # 创建数据库会话
    db = SessionLocal()
    try:
        # 获取数据库中最新的价格记录日期
        latest_record = db.query(
            func.max(PriceRecord.price_date).label('latest_date')
        ).first()
        
        latest_date = latest_record.latest_date if latest_record and latest_record.latest_date else None
        current_date = datetime.now().date()
        
        logger.info(f"数据库中最新记录日期: {latest_date}, 当前日期: {current_date}")
        
        # 如果没有记录或最新记录不是今天，则爬取最新数据
        if not latest_date or latest_date.date() < current_date:
            # 确定爬取的开始日期
            start_date = latest_date.date() + timedelta(days=1) if latest_date else current_date - timedelta(days=30)
            
            # 使用crawl_historical_data函数爬取从最新记录日期到当前日期的所有数据
            logger.info(f"开始爬取从 {start_date} 到 {current_date} 的数据")
            
            # 关闭当前数据库会话，因为crawl_historical_data会创建自己的会话
            db.close()
            
            # 调用crawl_historical_data爬取数据
            crawl_historical_data(start_date, current_date)
            
            logger.info(f"成功爬取了从 {start_date} 到 {current_date} 的数据")
        else:
            logger.info("数据库已包含最新日期的数据，无需爬取")
            db.close()
    
    except Exception as e:
        logger.error(f"自动爬取过程中出错: {str(e)}")
        if db:
            db.close()
    finally:
        logger.info("自动爬取任务完成")

def start_scheduler():
    """
    启动定时任务调度器
    """
    if scheduler.running:
        logger.warning("调度器已经在运行中")
        return
    
    # 每天上午10:30执行一次爬虫任务（江南市场每日价格通常在上午更新）
    scheduler.add_job(
        crawl_latest_data,
        CronTrigger(hour=10, minute=30),
        id="daily_crawler",
        replace_existing=True
    )
    
    # 添加额外的下午3点的爬取任务（以防上午的数据还未更新）
    scheduler.add_job(
        crawl_latest_data,
        CronTrigger(hour=15, minute=0),
        id="afternoon_crawler",
        replace_existing=True
    )
    
    # 启动调度器
    scheduler.start()
    logger.info("数据爬取定时任务已启动")
    
def stop_scheduler():
    """
    停止调度器
    """
    if scheduler.running:
        scheduler.shutdown()
        logger.info("调度器已停止")
    else:
        logger.warning("调度器未运行")

# 手动执行一次爬取（用于测试）
async def run_crawler_once():
    """运行一次爬虫任务，爬取从最新日期到当前日期的数据"""
    logger.info("开始运行爬虫任务")
    
    # 记录爬虫启动活动
    db = SessionLocal()
    processing_activity_id = None
    
    try:
        # 创建处理中状态的活动记录
        activity = create_crawler_activity(
            db=db,
            title="手动触发爬虫",
            description="爬虫任务正在运行中...",
            status="processing",
            records_count=0
        )
        processing_activity_id = activity.id if hasattr(activity, 'id') else None
        
        # 关闭会话，避免长时间占用连接
        db.close()
    except Exception as e:
        logger.error(f"记录爬虫启动活动失败: {str(e)}")
        if db:
            db.close()
    
    try:
        # 调用crawl_latest_data爬取从数据库中最新日期到当前日期的数据
        await crawl_latest_data()
        logger.info("爬虫任务完成")
        
        # 更新活动状态为成功
        try:
            db = SessionLocal()
            if processing_activity_id:
                activity = db.query(CrawlerActivity).filter(CrawlerActivity.id == processing_activity_id).first()
                if activity:
                    # 获取今天爬取的数据量
                    today = datetime.now().date()
                    today_start = datetime.combine(today, datetime.min.time())
                    today_end = datetime.combine(today, datetime.max.time())
                    
                    records_count = db.query(func.count(PriceRecord.id)).filter(
                        PriceRecord.timestamp.between(today_start, today_end)
                    ).scalar() or 0
                    
                    activity.status = "success"
                    activity.description = f"成功获取了{records_count}条新数据"
                    activity.records_count = records_count
                    db.commit()
            db.close()
        except Exception as ex:
            logger.error(f"更新爬虫活动状态失败: {str(ex)}")
            if db:
                db.close()
    except Exception as e:
        logger.error(f"爬虫任务失败: {str(e)}")
        
        # 如果有处理中的活动记录，更新为错误状态
        if processing_activity_id:
            try:
                db = SessionLocal()
                
                activity = db.query(CrawlerActivity).filter(CrawlerActivity.id == processing_activity_id).first()
                if activity:
                    activity.status = "error"
                    activity.description = f"爬虫任务失败: {str(e)}"
                    db.commit()
                
                db.close()
            except Exception as ex:
                logger.error(f"更新爬虫活动状态失败: {str(ex)}")
                if db:
                    db.close()

# 定时任务函数
async def schedule_crawler():
    """定时调度爬虫任务"""
    logger.info("爬虫调度器启动")
    
    while True:
        # 获取当前时间
        now = datetime.now()
        
        # 计算下一个早上8点的时间
        if now.hour >= 8:
            next_run = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
        else:
            next_run = now.replace(hour=8, minute=0, second=0, microsecond=0)
        
        # 计算等待时间
        wait_seconds = (next_run - now).total_seconds()
        
        logger.info(f"下次爬虫定时任务将在 {next_run.strftime('%Y-%m-%d %H:%M:%S')} 运行，等待 {wait_seconds} 秒")
        
        # 等待到下一次运行时间
        await asyncio.sleep(wait_seconds)
        
        # 运行爬虫任务
        try:
            await run_crawler_once()
        except Exception as e:
            logger.error(f"运行爬虫任务时出错: {str(e)}") 