from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# 从环境变量获取数据库连接串，如果没有则使用默认值
# 格式: mysql+pymysql://user:password@host:port/db_name
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@localhost:3306/graduation_project")

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 自动重连
    pool_recycle=3600,   # 连接回收时间
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()

def get_db():
    """
    Dependency helper for FastAPI to get a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
