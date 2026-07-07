"""Neo4j 图数据库配置模块

提供 Neo4j 数据库的全局连接管理，包括驱动初始化和关闭。

主要功能：
- 从环境变量读取 Neo4j 连接配置
- 全局共享的 Neo4j 驱动实例
- 连接异常处理和统一异常类型
- 驱动生命周期管理

注意：此模块提供全局单例驱动，供 GraphRAG 模块内部使用。
"""

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j 连接配置
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 创建Neo4j驱动
driver = None


class Neo4jConnectionError(RuntimeError):
    """Neo4j 连接不可用时抛出的统一异常。"""


def init_neo4j_driver(force: bool = False) -> GraphDatabase.driver:
    """初始化 Neo4j 驱动。

    创建全局共享的 Neo4j 驱动实例，并验证连接可用性。
    如果驱动已存在且 force 为 False，则直接返回现有驱动。

    Args:
        force: 是否强制重新创建驱动，默认为 False

    Returns:
        Neo4j 驱动实例

    Raises:
        Neo4jConnectionError: 当密码未配置或连接失败时
    """
    global driver
    if driver is not None and not force:
        return driver
    if not NEO4J_PASSWORD:
        raise Neo4jConnectionError("NEO4J_PASSWORD 未配置，请在 .env 中设置 Neo4j 密码。")

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


def close_neo4j_driver() -> None:
    """关闭 Neo4j 驱动。

    释放全局驱动资源，关闭所有连接。
    """
    global driver
    if driver:
        driver.close()
        driver = None
        print("✓ Neo4j 连接已关闭")
