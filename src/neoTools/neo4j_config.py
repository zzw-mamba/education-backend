"""
Neo4j 图数据库配置文件
用于连接和管理Neo4j数据库
"""

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j 连接配置
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 创建Neo4j驱动
driver = None


class Neo4jConnectionError(RuntimeError):
    """Neo4j 连接不可用时抛出的统一异常。"""


def init_neo4j_driver(force: bool = False):
    """初始化Neo4j驱动"""
    global driver
    if driver is not None and not force:
        return driver

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            encrypted=False  # 本地开发可以关闭加密
        )
        # 验证连接
        with driver.session() as session:
            session.run("RETURN 1")
        print("✓ Neo4j 连接成功")
        return driver
    except (ServiceUnavailable, Neo4jError, OSError) as e:
        driver = None
        print(f"✗ 无法连接到Neo4j: {e}")
        raise Neo4jConnectionError(f"无法连接到 Neo4j: {e}") from e


def close_neo4j_driver():
    """关闭Neo4j驱动"""
    global driver
    if driver:
        driver.close()
        driver = None
        print("✓ Neo4j 连接已关闭")

