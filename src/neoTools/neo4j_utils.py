"""
Neo4j 实用工具模块
包含知识图谱构建、查询、分析等相关功能
"""

from .neo4j_config import Neo4jService
from typing import Dict, List, Tuple, Optional
import json


class KnowledgeGraph:
    """知识图谱操作类"""
    
    @staticmethod
    def build_knowledge_graph_from_kb(kb_id: int, kb_title: str, kb_content: str, 
                                      category: str = None, tags: List[str] = None):
        """
        从知识库项目构建知识图谱
        
        Args:
            kb_id: 知识库ID
            kb_title: 标题
            kb_content: 内容
            category: 分类
            tags: 标签列表
        """
        # 创建知识库节点
        kb_props = {
            "kb_id": kb_id,
            "title": kb_title,
            "content": kb_content[:1000] if kb_content else "",  # 限制内容长度
            "category": category or "Uncategorized"
        }
        
        try:
            Neo4jService.create_node("KnowledgeBase", kb_props)
            
            # 创建标签关系
            if tags:
                for tag in tags:
                    # 先创建标签节点（如果不存在）
                    tag_props = {"tag_id": hash(tag), "name": tag}
                    try:
                        Neo4jService.create_node("Tag", tag_props)
                    except:
                        pass  # 标签可能已存在
                    
                    # 创建关系
                    Neo4jService.create_relationship(
                        "KnowledgeBase", "kb_id", kb_id,
                        "HAS_TAG", {},
                        "Tag", "name", tag
                    )
            
            return True
        except Exception as e:
            print(f"构建知识图谱出错: {e}")
            return False
    
class GraphAnalysis:
    """图分析工具类"""
    
    @staticmethod
    def get_centrality_nodes(label: str = "KnowledgeBase", 
                            top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取中心性最高的节点（最多关系的节点）
        
        Args:
            label: 节点标签
            top_n: 返回前N个
        
        Returns:
            节点及其关系数列表
        """
        query = f"""
        MATCH (n:{label})
        WITH n, SIZE([(n)-[*1..2]-() | 1]) as degree
        RETURN n.title as name, degree
        ORDER BY degree DESC
        LIMIT {top_n}
        """
        
        try:
            results = Neo4jService.execute_query(query, {})
            return [(record["name"], record["degree"]) for record in results]
        except Exception as e:
            print(f"获取中心性节点出错: {e}")
            return []
    
    @staticmethod
    def get_connected_components(label: str = "KnowledgeBase") -> Dict[int, int]:
        """
        获取连通分量统计
        
        Args:
            label: 节点标签
        
        Returns:
            连通分量大小分布
        """
        query = f"""
        MATCH (n:{label})
        WITH n, SIZE([(n)-[*0..5]-() | 1]) as component_size
        RETURN component_size, COUNT(*) as count
        """
        
        try:
            results = Neo4jService.execute_query(query, {})
            return {record["component_size"]: record["count"] for record in results}
        except Exception as e:
            print(f"获取连通分量出错: {e}")
            return {}
    
    @staticmethod
    def find_shortest_path(start_title: str, end_title: str, 
                          label: str = "KnowledgeBase") -> Optional[List[str]]:
        """
        查找两个节点间最短路径
        
        Args:
            start_title: 起始节点标题
            end_title: 结束节点标题
            label: 节点标签
        
        Returns:
            路径中节点标题列表
        """
        query = f"""
        MATCH (start:{label} {{title: $start}})
        MATCH (end:{label} {{title: $end}})
        MATCH path = shortestPath((start)-[*1..10]-(end))
        WITH nodes(path) as path_nodes
        RETURN [n.title for n in path_nodes] as path
        """
        
        try:
            result = Neo4jService.execute_query(
                query, 
                {"start": start_title, "end": end_title}
            )
            if result:
                return result[0]["path"]
        except Exception as e:
            print(f"查找最短路径出错: {e}")
        
        return None
    
    @staticmethod
    def get_graph_statistics() -> Dict:
        """
        获取图的统计信息
        
        Returns:
            包含节点数、关系数等的统计字典
        """
        queries = {
            "total_nodes": "MATCH (n) RETURN COUNT(n) as count",
            "total_relationships": "MATCH ()-[r]-() RETURN COUNT(r) as count",
            "kb_count": "MATCH (n:KnowledgeBase) RETURN COUNT(n) as count",
            "tag_count": "MATCH (n:Tag) RETURN COUNT(n) as count",
            "user_count": "MATCH (n:User) RETURN COUNT(n) as count",
        }
        
        stats = {}
        try:
            for key, query in queries.items():
                result = Neo4jService.execute_query(query, {})
                if result:
                    stats[key] = result[0]["count"]
        except Exception as e:
            print(f"获取图统计信息出错: {e}")
        
        return stats


class UserBehaviorGraph:
    """用户行为图谱"""
    
    @staticmethod
    def record_search_query(user_id: int, query_text: str, 
                           kb_ids: List[int] = None):
        """
        记录用户搜索查询
        
        Args:
            user_id: 用户ID
            query_text: 查询文本
            kb_ids: 找到的知识库ID列表
        """
        query = """
        MATCH (user:User {user_id: $user_id})
        CREATE (sq:SearchQuery {query_text: $query_text, timestamp: timestamp()})
        CREATE (user)-[:SEARCHED]->(sq)
        """
        
        try:
            Neo4jService.execute_query(
                query,
                {"user_id": user_id, "query_text": query_text}
            )
            
            # 创建搜索结果关系
            if kb_ids:
                for kb_id in kb_ids:
                    result_query = """
                    MATCH (sq:SearchQuery {query_text: $query_text})
                    MATCH (kb:KnowledgeBase {kb_id: $kb_id})
                    CREATE (sq)-[:FOUND]->(kb)
                    """
                    Neo4jService.execute_query(
                        result_query,
                        {"query_text": query_text, "kb_id": kb_id}
                    )
            
            return True
        except Exception as e:
            print(f"记录搜索查询出错: {e}")
            return False
    
    @staticmethod
    def get_user_search_history(user_id: int, limit: int = 10) -> List[Dict]:
        """
        获取用户搜索历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
        
        Returns:
            搜索历史列表
        """
        query = """
        MATCH (user:User {user_id: $user_id})-[:SEARCHED]->(sq:SearchQuery)
        RETURN sq.query_text as query, sq.timestamp as timestamp
        ORDER BY sq.timestamp DESC
        LIMIT $limit
        """
        
        try:
            results = Neo4jService.execute_query(
                query,
                {"user_id": user_id, "limit": limit}
            )
            return [dict(record) for record in results]
        except Exception as e:
            print(f"获取搜索历史出错: {e}")
            return []
    
    @staticmethod
    def get_popular_searches(limit: int = 10) -> List[Tuple[str, int]]:
        """
        获取热门搜索
        
        Args:
            limit: 返回数量限制
        
        Returns:
            [(搜索词, 频率), ...]
        """
        query = """
        MATCH (user:User)-[:SEARCHED]->(sq:SearchQuery)
        WITH sq.query_text as query, COUNT(*) as frequency
        RETURN query, frequency
        ORDER BY frequency DESC
        LIMIT $limit
        """
        
        try:
            results = Neo4jService.execute_query(query, {"limit": limit})
            return [(record["query"], record["frequency"]) for record in results]
        except Exception as e:
            print(f"获取热门搜索出错: {e}")
            return []
