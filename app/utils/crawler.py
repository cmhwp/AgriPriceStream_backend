import requests
import logging
import time
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from dateutil.parser import parse

from app.models.models import Vegetable, PriceRecord, CrawlerActivity
from app.db.database import SessionLocal

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("江南市场爬虫")

# API基础URL
BASE_URL = "https://jnmarket.net/api/dailypricelist"

def fetch_data(page_num=1, page_size=2000, kind=1):
    """从江南市场API获取数据"""
    try:
        url = f"{BASE_URL}?pageNum={page_num}&pageSize={page_size}&kind={kind}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200:
                return data
            else:
                logger.error(f"API返回错误: {data.get('msg')}")
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
    except Exception as e:
        logger.error(f"获取数据时出错: {str(e)}")
    
    return None

def save_vegetable_data(db: Session, data_item):
    """保存蔬菜数据到数据库"""
    try:
        # 解析日期
        price_date = parse(data_item.get("priceDate"))
        
        # 先根据产品名称和日期查找是否已存在记录
        existing_vegetable = db.query(Vegetable).filter(
            Vegetable.product_name == data_item.get("productName"),
            Vegetable.price_date == price_date
        ).first()
        
        if existing_vegetable:
            # 如果存在，则更新记录
            existing_vegetable.top_price = float(data_item.get("topPrice", 0))
            existing_vegetable.minimum_price = float(data_item.get("minimumPrice", 0))
            existing_vegetable.average_price = float(data_item.get("averagePrice", 0))
            existing_vegetable.product_name = data_item.get('productName')
            existing_vegetable.price_date = price_date
            existing_vegetable.weight = data_item.get("weight")
            existing_vegetable.provenance_name = data_item.get("provenanceName")
            existing_vegetable.standard = data_item.get("standard")
            existing_vegetable.kind = data_item.get("kind")
            
            # 保存价格记录
            price_record = PriceRecord(
                vegetable_id=existing_vegetable.id,
                price=float(data_item.get("averagePrice", 0)),
                top_price=float(data_item.get("topPrice", 0)),
                minimum_price=float(data_item.get("minimumPrice", 0)),
                average_price=float(data_item.get("averagePrice", 0)),
                price_date=price_date,
                timestamp=datetime.now(),
                provenance_name=data_item.get("provenanceName")
            )
            db.add(price_record)
            
        else:
            # 如果不存在，则创建新记录
            new_vegetable = Vegetable(
                description=f"{data_item.get('productName')}，来自{data_item.get('provenanceName')}",
                product_name=data_item.get("productName"),
                provenance_name=data_item.get("provenanceName"),
                top_price=float(data_item.get("topPrice", 0)),
                minimum_price=float(data_item.get("minimumPrice", 0)),
                average_price=float(data_item.get("averagePrice", 0)),
                standard=data_item.get("standard"),
                kind=data_item.get("kind"),
                source_type=data_item.get("sourceType"),
                price_date=price_date,
                weight=data_item.get("weight")
            )
            db.add(new_vegetable)
            db.flush()  # 立即获取id
            
            # 保存价格记录
            price_record = PriceRecord(
                vegetable_id=new_vegetable.id,
                price=float(data_item.get("averagePrice", 0)),
                top_price=float(data_item.get("topPrice", 0)),
                minimum_price=float(data_item.get("minimumPrice", 0)),
                average_price=float(data_item.get("averagePrice", 0)),
                price_date=price_date,
                timestamp=datetime.now(),
                provenance_name=data_item.get("provenanceName")
            )
            db.add(price_record)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"保存数据时出错: {str(e)}")
        return False

def crawl_historical_data(start_date, end_date=None, processing_activity_id=None):
    """爬取指定日期范围内的历史数据"""
    if end_date is None:
        end_date = datetime.now().date()
    
    # 创建数据库会话
    db = SessionLocal()
    start_time = time.time()
    total_processed = 0
    
    try:
        # 计算需要爬取的天数
        delta = end_date - start_date
        days_to_crawl = delta.days + 1
        
        logger.info(f"开始爬取从 {start_date} 到 {end_date} 的数据，共 {days_to_crawl} 天")
        
        total_pages = 0
        page_size = 2000  # 每页获取2000条数据
        
        # 从第一页开始，获取总页数
        first_page_data = fetch_data(page_num=1, page_size=page_size, kind=1)
        if first_page_data:
            total_records = first_page_data.get("total", 0)
            total_pages = (total_records + page_size - 1) // page_size
            
            logger.info(f"共有 {total_records} 条记录，{total_pages} 页")
            
            # 处理第一页数据
            for item in first_page_data.get("rows", []):
                try:
                    price_date = parse(item.get("priceDate")).date()
                    if start_date <= price_date <= end_date:
                        if save_vegetable_data(db, item):
                            total_processed += 1
                except Exception as e:
                    logger.error(f"处理数据项时出错: {str(e)}")
            
            # 处理剩余页
            for page in range(2, total_pages + 1):
                logger.info(f"正在处理第 {page}/{total_pages} 页")
                page_data = fetch_data(page_num=page, page_size=page_size, kind=1)
                
                if page_data:
                    for item in page_data.get("rows", []):
                        try:
                            price_date = parse(item.get("priceDate")).date()
                            if start_date <= price_date <= end_date:
                                if save_vegetable_data(db, item):
                                    total_processed += 1
                        except Exception as e:
                            logger.error(f"处理数据项时出错: {str(e)}")
                
                # 避免请求过于频繁
                time.sleep(1)
        
        logger.info(f"爬取完成，共处理 {total_processed} 条记录")
        
        # 记录爬虫活动
        end_time = time.time()
        duration = int(end_time - start_time)
        status = "success" if total_processed > 0 else "warning"
        
        try:
            from app.crud.dashboard import create_crawler_activity
            
            # 如果传入了处理中活动ID，则更新该活动
            if processing_activity_id:
                activity = db.query(CrawlerActivity).filter(CrawlerActivity.id == processing_activity_id).first()
                if activity:
                    activity.status = status
                    activity.records_count = total_processed
                    activity.duration = duration
                    
                    # 确定标题和描述
                    if (end_date - start_date).days > 1:
                        activity.description = f"爬取了{days_to_crawl}天的历史数据，成功获取了{total_processed}条记录"
                    else:
                        activity.description = f"成功获取了{total_processed}条新数据"
                    
                    db.commit()
                    logger.info(f"更新爬虫活动记录，ID: {processing_activity_id}")
                else:
                    # 如果找不到处理中的活动，则创建新活动
                    if (end_date - start_date).days > 1:
                        title = f"历史数据爬取 ({start_date} 至 {end_date})"
                        description = f"爬取了{days_to_crawl}天的历史数据，成功获取了{total_processed}条记录"
                    else:
                        title = "每日常规爬取"
                        description = f"成功获取了{total_processed}条新数据"
                    
                    create_crawler_activity(
                        db=db,
                        title=title,
                        description=description,
                        status=status,
                        records_count=total_processed,
                        duration=duration
                    )
            else:
                # 没有处理中活动ID，创建新活动
                # 确定标题和描述
                if (end_date - start_date).days > 1:
                    title = f"历史数据爬取 ({start_date} 至 {end_date})"
                    description = f"爬取了{days_to_crawl}天的历史数据，成功获取了{total_processed}条记录"
                else:
                    title = "每日常规爬取"
                    description = f"成功获取了{total_processed}条新数据"
                
                create_crawler_activity(
                    db=db,
                    title=title,
                    description=description,
                    status=status,
                    records_count=total_processed,
                    duration=duration
                )
        except Exception as e:
            logger.error(f"记录爬虫活动时出错: {str(e)}")
        
    except Exception as e:
        logger.error(f"爬取历史数据时出错: {str(e)}")
        
        # 记录错误活动
        try:
            from app.crud.dashboard import create_crawler_activity
            
            end_time = time.time()
            duration = int(end_time - start_time)
            
            # 如果传入了处理中活动ID，则更新该活动为错误状态
            if processing_activity_id:
                activity = db.query(CrawlerActivity).filter(CrawlerActivity.id == processing_activity_id).first()
                if activity:
                    activity.status = "error"
                    activity.description = f"爬取数据时发生错误: {str(e)}"
                    activity.records_count = total_processed
                    activity.duration = duration
                    db.commit()
                    logger.info(f"更新爬虫活动为错误状态，ID: {processing_activity_id}")
                else:
                    # 如果找不到处理中的活动，则创建新活动
                    create_crawler_activity(
                        db=db,
                        title="爬虫错误",
                        description=f"爬取数据时发生错误: {str(e)}",
                        status="error",
                        records_count=total_processed,
                        duration=duration
                    )
            else:
                # 没有处理中活动ID，创建新错误活动
                create_crawler_activity(
                    db=db,
                    title="爬虫错误",
                    description=f"爬取数据时发生错误: {str(e)}",
                    status="error",
                    records_count=total_processed,
                    duration=duration
                )
        except Exception as ex:
            logger.error(f"记录爬虫错误活动时出错: {str(ex)}")
    finally:
        db.close()

def crawl_from_2022_to_now():
    """爬取从2022年1月1日到现在的数据"""
    start_date = datetime(2022, 1, 1).date()
    end_date = datetime.now().date()
    
    crawl_historical_data(start_date, end_date)

if __name__ == "__main__":
    # 如果直接运行此脚本，则爬取从2022年到现在的数据
    crawl_from_2022_to_now() 