"""
Neo4j 相关的API路由
展示如何在FastAPI中集成Neo4j
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from neo4j.neo4j_utils import KnowledgeGraph, GraphAnalysis, UserBehaviorGraph
import json

router = APIRouter(prefix="/api/graph", tags=["knowledge-graph"])


# ============ Pydantic 模型 ============

class KBNode(BaseModel):
    kb_id: int
    title: str
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class SearchQuery(BaseModel):
    query_text: str
    limit: Optional[int] = 10


class PathRequest(BaseModel):
    start_title: str
    end_title: str


# ============ 知识图谱操作接口 ============

@router.post("/build-graph")
async def build_knowledge_graph(kb: KBNode):
    """
    构建知识图谱
    
    Args:
        kb: 知识库数据
    
    Returns:
        创建结果
    """
    success = KnowledgeGraph.build_knowledge_graph_from_kb(
        kb_id=kb.kb_id,
        kb_title=kb.title,
        kb_content=kb.content or "",
        category=kb.category,
        tags=kb.tags or []
    )
    
    if success:
        return {"success": True, "message": "知识图谱构建成功"}
    else:
        raise HTTPException(status_code=500, detail="构建知识图谱失败")


@router.get("/related-documents/{kb_id}")
async def get_related_documents(
    kb_id: int,
    depth: int = Query(2, ge=1, le=5),
    limit: int = Query(10, ge=1, le=50)
):
    """
    获取相关文档
    
    Args:
        kb_id: 知识库ID
        depth: 查询深度
        limit: 返回数量
    
    Returns:
        相关文档列表
    """
    results = KnowledgeGraph.find_related_documents(kb_id, depth)
    return {"related_documents": results[:limit]}


@router.get("/documents-by-tags")
async def get_documents_by_tags(
    tags: List[str] = Query(..., description="标签列表"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    按标签查找文档
    
    Args:
        tags: 标签列表
        limit: 返回数量
    
    Returns:
        文档列表
    """
    results = KnowledgeGraph.find_documents_by_tags(tags, limit)
    return {"documents": results}


# ============ 图分析接口 ============

@router.get("/centrality")
async def get_centrality_analysis(
    top_n: int = Query(10, ge=1, le=50)
):
    """
    获取中心性最高的知识库节点
    
    Args:
        top_n: 返回前N个
    
    Returns:
        节点及其关系数
    """
    results = GraphAnalysis.get_centrality_nodes("KnowledgeBase", top_n)
    return {
        "centrality_nodes": [
            {"name": name, "degree": degree} 
            for name, degree in results
        ]
    }


@router.get("/statistics")
async def get_graph_statistics():
    """
    获取知识图谱统计信息
    
    Returns:
        图的统计数据
    """
    stats = GraphAnalysis.get_graph_statistics()
    return {"statistics": stats}


@router.post("/shortest-path")
async def find_shortest_path(request: PathRequest):
    """
    查找两个节点间的最短路径
    
    Args:
        request: 包含起始和结束节点标题
    
    Returns:
        路径信息
    """
    path = GraphAnalysis.find_shortest_path(
        request.start_title,
        request.end_title
    )
    
    if path:
        return {
            "path": path,
            "length": len(path) - 1,
            "message": f"找到路径，长度为 {len(path) - 1}"
        }
    else:
        raise HTTPException(status_code=404, detail="未找到路径")


# ============ 用户行为图接口 ============

@router.post("/record-search")
async def record_search(user_id: int, query_text: str, kb_ids: Optional[List[int]] = None):
    """
    记录用户搜索查询
    
    Args:
        user_id: 用户ID
        query_text: 查询文本
        kb_ids: 搜索结果的知识库ID列表
    
    Returns:
        记录结果
    """
    success = UserBehaviorGraph.record_search_query(user_id, query_text, kb_ids)
    
    if success:
        return {"success": True, "message": "搜索记录已保存"}
    else:
        raise HTTPException(status_code=500, detail="保存搜索记录失败")


@router.get("/user-search-history/{user_id}")
async def get_user_search_history(
    user_id: int,
    limit: int = Query(10, ge=1, le=50)
):
    """
    获取用户搜索历史
    
    Args:
        user_id: 用户ID
        limit: 返回数量
    
    Returns:
        搜索历史列表
    """
    history = UserBehaviorGraph.get_user_search_history(user_id, limit)
    return {"search_history": history}


@router.get("/popular-searches")
async def get_popular_searches(
    limit: int = Query(10, ge=1, le=50)
):
    """
    获取热门搜索
    
    Args:
        limit: 返回数量
    
    Returns:
        热门搜索列表
    """
    results = UserBehaviorGraph.get_popular_searches(limit)
    return {
        "popular_searches": [
            {"query": query, "frequency": frequency}
            for query, frequency in results
        ]
    }


# ============ 健康检查接口 ============

@router.get("/health")
async def health_check():
    """检查Neo4j连接状态"""
    try:
        stats = GraphAnalysis.get_graph_statistics()
        return {
            "status": "healthy",
            "graph_stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j服务不可用: {str(e)}"
        )
