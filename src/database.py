"""数据库连接模块

负责配置和管理 SQLAlchemy 数据库连接，提供会话工厂和依赖注入函数。

包含的主要组件：
- SQLALCHEMY_DATABASE_URL: 数据库连接字符串
- engine: SQLAlchemy 数据库引擎
- SessionLocal: 会话工厂
- Base: SQLAlchemy 声明式基类
- get_db(): FastAPI 依赖注入函数，获取数据库会话
- init_db(): 初始化数据库表
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    raise RuntimeError("DATABASE_URL 未配置，请在 .env 中设置数据库连接串。")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI 依赖注入函数，获取数据库会话。
    
    使用 yield 模式提供数据库会话，请求结束后自动关闭会话。
    
    Yields:
        Session: SQLAlchemy 数据库会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库表。
    
    创建所有继承自 Base 的模型对应的数据库表。
    仅在应用启动时调用一次。
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")