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


def ensure_neo4j_driver():
    """返回已初始化的 Neo4j driver；如果尚未初始化则自动初始化。"""
    if driver is None:
        return init_neo4j_driver()
    return driver

def close_neo4j_driver():
    """关闭Neo4j驱动"""
    global driver
    if driver:
        driver.close()
        driver = None
        print("✓ Neo4j 连接已关闭")

def get_neo4j_session():
    """获取Neo4j会话"""
    return ensure_neo4j_driver().session()

class Neo4jService:
    """Neo4j 服务类，包含常用操作"""
    
    def __init__(self, driver=None):
        self.driver = driver or get_neo4j_session()
    
    @staticmethod
    def execute_query(query, params=None):
        """
        执行Neo4j查询
        
        Args:
            query: Cypher查询语句
            params: 查询参数字典
        
        Returns:
            查询结果列表
        """
        with get_neo4j_session() as session:
            result = session.run(query, params or {})
            return [record for record in result]
    
    @staticmethod
    def create_node(label, properties):
        """
        创建节点
        
        Args:
            label: 节点标签名
            properties: 节点属性字典
        
        Returns:
            创建的节点信息
        """
        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"CREATE (n:{label} {{{prop_str}}}) RETURN n"
        
        with get_neo4j_session() as session:
            result = session.run(query, properties)
            return result.single()
    
    @staticmethod
    def create_relationship(start_label, start_id_prop, start_id_val, 
                           rel_type, rel_props, 
                           end_label, end_id_prop, end_id_val):
        """
        创建关系
        
        Args:
            start_label: 起始节点标签
            start_id_prop: 起始节点ID属性名
            start_id_val: 起始节点ID值
            rel_type: 关系类型
            rel_props: 关系属性字典
            end_label: 结束节点标签
            end_id_prop: 结束节点ID属性名
            end_id_val: 结束节点ID值
        
        Returns:
            创建的关系信息
        """
        rel_str = ""
        if rel_props:
            rel_str = "{" + ", ".join([f"{k}: ${k}" for k in rel_props.keys()]) + "}"
        
        query = f"""
        MATCH (start:{start_label} {{{start_id_prop}: $start_val}})
        MATCH (end:{end_label} {{{end_id_prop}: $end_val}})
        CREATE (start)-[r:{rel_type} {rel_str}]->(end)
        RETURN r
        """
        
        params = {"start_val": start_id_val, "end_val": end_id_val}
        if rel_props:
            params.update(rel_props)
        
        with get_neo4j_session() as session:
            result = session.run(query, params)
            return result.single()
    
    @staticmethod
    def find_node(label, properties):
        """
        查找节点
        
        Args:
            label: 节点标签
            properties: 查询属性字典
        
        Returns:
            找到的节点列表
        """
        where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n"
        
        with get_neo4j_session() as session:
            result = session.run(query, properties)
            return [record for record in result]
    
    @staticmethod
    def find_path(start_label, start_prop, start_val, 
                  end_label, end_prop, end_val, max_depth=5):
        """
        查找两个节点之间的路径
        
        Args:
            start_label: 起始节点标签
            start_prop: 起始节点属性名
            start_val: 起始节点属性值
            end_label: 结束节点标签
            end_prop: 结束节点属性名
            end_val: 结束节点属性值
            max_depth: 最大路径深度
        
        Returns:
            路径信息列表
        """
        query = f"""
        MATCH path = shortestPath(
            (start:{start_label} {{{start_prop}: $start_val}})-[*1..{max_depth}]-(end:{end_label} {{{end_prop}: $end_val}})
        )
        RETURN path
        """
        
        params = {"start_val": start_val, "end_val": end_val}
        
        with get_neo4j_session() as session:
            result = session.run(query, params)
            return [record for record in result]
    
    @staticmethod
    def get_related_nodes(label, prop, val, rel_type=None, depth=1):
        """
        获取相关节点
        
        Args:
            label: 起始节点标签
            prop: 起始节点属性名
            val: 起始节点属性值
            rel_type: 关系类型（可选）
            depth: 查询深度
        
        Returns:
            相关节点列表
        """
        rel_pattern = f"[:{rel_type}]" if rel_type else ""
        query = f"""
        MATCH (start:{label} {{{prop}: $val}})-{rel_pattern}*1..{depth}-(related)
        RETURN DISTINCT related
        """
        
        with get_neo4j_session() as session:
            result = session.run(query, {"val": val})
            return [record for record in result]
    
    @staticmethod
    def delete_node(label, properties):
        """
        删除节点（包括关系）
        
        Args:
            label: 节点标签
            properties: 节点属性字典
        
        Returns:
            删除的节点数
        """
        where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"""
        MATCH (n:{label}) WHERE {where_clause}
        DETACH DELETE n
        RETURN COUNT(n) as count
        """
        
        with get_neo4j_session() as session:
            result = session.run(query, properties)
            record = result.single()
            return record["count"] if record else 0
