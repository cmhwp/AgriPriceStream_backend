from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL数据库连接配置
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/agriconnect"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 依赖函数，用于获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_instance():
    """
    获取数据库实例（非依赖注入版本，用于后台任务）
    
    与get_db不同，这个函数直接返回数据库会话实例，而不是yield。
    适用于后台任务或异步上下文中创建新的数据库会话。
    
    Returns:
        数据库会话实例
    """
    return SessionLocal() 