from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from graphrag.graphrag_service import get_graphrag_service, GRAPHRAG_IMPORT_ERROR


router = APIRouter(prefix="/api/graphrag", tags=["graphrag"])


class EntityInput(BaseModel):
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")


class ChunkInput(BaseModel):
    paper_id: int = Field(..., description="对应 MySQL KnowledgeBase.id")
    title: str = Field(default="", description="文献标题")
    year: Optional[int] = Field(default=None, description="年份")
    chunk_id: str = Field(..., description="Chunk唯一ID，如 paperId_index")
    text: str = Field(..., description="Chunk文本")
    index: int = Field(default=0, description="Chunk顺序")
    entities: Optional[List[EntityInput]] = Field(default_factory=list, description="实体列表")


class UpsertChunksRequest(BaseModel):
    chunks: List[ChunkInput]
    create_index: bool = True


class SearchRequest(BaseModel):
    query_text: str
    top_k: int = Field(default=5, ge=1, le=30)
    paper_ids: Optional[List[int]] = Field(default=None, description="限定检索范围的论文 ID 列表")


class CreateIndexRequest(BaseModel):
    force_recreate: bool = False


class SyncFromMySQLRequest(BaseModel):
    paper_ids: Optional[List[int]] = None
    limit: int = Field(default=100, ge=1, le=20000)
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    auto_extract_entities: bool = Field(default=True, description="是否自动抽取实体并构建 MENTIONS 关系")


class SetupLocalDBRequest(BaseModel):
    create_vector_index: bool = True
    force_recreate_index: bool = False
    ontology_file_path: str = Field(default="src/CSO.3.5.nt", description="本体 N-Triples 文件路径")
    sync_from_mysql: bool = False
    paper_ids: Optional[List[int]] = None
    limit: int = Field(default=100, ge=1, le=20000)
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    auto_extract_entities: bool = Field(default=True, description="同步时是否自动抽取实体")


class PaperSummaryRequest(BaseModel):
    paper_id: int
    top_entities: int = Field(default=10, ge=1, le=30)
    snippets_per_entity: int = Field(default=2, ge=1, le=5)
    neighbor_limit: int = Field(default=5, ge=0, le=20)
    recursive_group_size: int = Field(default=4, ge=1, le=10)
    section_aware: bool = Field(default=True, description="是否启用章节感知摘要")


def _service_or_500():
    """
    获取 GraphRAG 服务实例或抛出 HTTP 500 错误。
    
    Returns:
        GraphRAGService: GraphRAG 服务实例
        
    Raises:
        HTTPException: 当 neo4j-graphrag 导入失败或服务初始化失败时
    """
    if GRAPHRAG_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"neo4j-graphrag 导入失败: {GRAPHRAG_IMPORT_ERROR}",
        )
    try:
        return get_graphrag_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def graphrag_health():
    """
    GraphRAG 服务健康检查接口。
    
    Returns:
        Dict[str, Any]: 包含服务状态、索引名称、节点标签、模型配置等信息的字典
    """
    service = _service_or_500()
    return {
        "status": "ok",
        "index_name": service.settings.vector_index_name,
        "node_label": service.settings.vector_node_label,
        "embedding_model": service.settings.embedding_model,
        "llm_model": service.settings.llm_model,
    }


@router.get("/diagnose")
async def diagnose():
    """
    诊断 Neo4j Desktop 连接状态和 APOC 库可用性。
    
    Returns:
        Dict[str, Any]: 诊断结果，包含连接状态、APOC 可用性、向量索引状态、约束信息和提示
    """
    service = _service_or_500()
    diagnostics = {
        "neo4j_uri": service.settings.neo4j_uri,
        "neo4j_user": service.settings.neo4j_user,
        "connection": "failed",
        "apoc_available": False,
        "vector_index_exists": False,
        "constraints": [],
        "tips": [
            "✓ Neo4j Desktop 已连接" if service.driver else "✗ Neo4j Desktop 未连接",
            "✓ 确保 Database 已启动（绿色指示灯）",
            "✓ 确保已安装 APOC 库（Plugins 中搜索）",
        ],
    }
    
    try:
        with service.driver.session() as session:
            session.run("RETURN 1")
            diagnostics["connection"] = "ok"
            
            # 检查 APOC
            try:
                result = session.run("RETURN apoc.version()").single()
                if result:
                    diagnostics["apoc_available"] = True
                    diagnostics["apoc_version"] = str(result[0])
            except Exception:
                diagnostics["apoc_available"] = False
            
            # 检查向量索引
            try:
                result = list(session.run(
                    "SHOW VECTOR INDEXES YIELD name WHERE name = $idx RETURN count(*) AS cnt",
                    {"idx": service.settings.vector_index_name}
                ))
                if result and result[0]["cnt"] > 0:
                    diagnostics["vector_index_exists"] = True
            except Exception:
                pass
            
            # 检查约束
            try:
                result = list(session.run("SHOW CONSTRAINTS YIELD name RETURN name"))
                diagnostics["constraints"] = [r["name"] for r in result]
            except Exception:
                pass
    except Exception as e:
        diagnostics["connection"] = f"failed: {str(e)}"
        diagnostics["tips"].append(f"✗ 连接错误: {e}")
    
    return diagnostics


@router.post("/init")
async def init_graphrag():
    """
    初始化 GraphRAG 服务。
    
    Returns:
        Dict[str, Any]: 包含成功状态、消息和索引名称的字典
    """
    service = _service_or_500()
    return {
        "success": True,
        "message": "GraphRAG 初始化成功",
        "index_name": service.settings.vector_index_name,
    }


@router.post("/setup-local-db")
async def setup_local_db(request: SetupLocalDBRequest):
    """
    设置本地 Neo4j 数据库，包括创建约束、索引和可选的数据同步。
    
    Args:
        request (SetupLocalDBRequest): 设置请求，包含索引创建、数据同步等配置
        
    Returns:
        Dict[str, Any]: 包含设置结果和同步结果的字典
        
    Raises:
        HTTPException: 当设置过程发生错误时
    """
    service = _service_or_500()
    try:
        setup_result = service.setup_local_database(
            create_vector_index=request.create_vector_index,
            force_recreate_index=request.force_recreate_index,
            ontology_file_path=request.ontology_file_path,
        )

        sync_result = None
        if request.sync_from_mysql:
            sync_result = service.sync_from_mysql_knowledge_base(
                paper_ids=request.paper_ids,
                limit=request.limit,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                auto_extract_entities=request.auto_extract_entities,
            )

        return {
            "success": True,
            "message": "本地 Neo4j 数据库初始化完成",
            "setup": setup_result,
            "sync": sync_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/create-index")
async def create_index(request: CreateIndexRequest):
    """
    创建或更新向量索引。
    
    Args:
        request (CreateIndexRequest): 索引创建请求，包含是否强制重建的配置
        
    Returns:
        Dict[str, Any]: 包含成功状态和索引创建结果的字典
        
    Raises:
        HTTPException: 当索引创建失败时
    """
    service = _service_or_500()
    try:
        service.ensure_graph_schema()
        result = service.ensure_vector_index(force_recreate=request.force_recreate)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/upsert-chunks")
async def upsert_chunks(request: UpsertChunksRequest):
    """
    插入或更新论文 chunks 到 Neo4j 数据库。
    
    Args:
        request (UpsertChunksRequest): 包含 chunks 列表和索引创建配置
        
    Returns:
        Dict[str, Any]: 包含成功状态和 upsert 结果的字典
        
    Raises:
        HTTPException: 当 upsert 操作失败时
    """
    service = _service_or_500()
    try:
        if request.create_index:
            service.ensure_vector_index(force_recreate=False)
        result = service.upsert_paper_chunks([chunk.model_dump() for chunk in request.chunks])
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sync-from-mysql")
async def sync_from_mysql(request: SyncFromMySQLRequest):
    """
    从 MySQL KnowledgeBase 表同步数据到 Neo4j 图数据库。
    
    Args:
        request (SyncFromMySQLRequest): 同步请求，包含论文 ID、切片参数、实体抽取配置等
        
    Returns:
        Dict[str, Any]: 包含成功状态和同步结果的字典
        
    Raises:
        HTTPException: 当同步过程发生错误时
    """
    service = _service_or_500()
    try:
        result = service.sync_from_mysql_knowledge_base(
            paper_ids=request.paper_ids,
            limit=request.limit,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            auto_extract_entities=request.auto_extract_entities,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/graph-refined-context")
async def graph_refined_context(request: PaperSummaryRequest):
    """
    获取基于图结构精炼的论文上下文信息。
    
    Args:
        request (PaperSummaryRequest): 包含论文 ID、实体数量、片段数量、邻居限制等参数
        
    Returns:
        Dict[str, Any]: 包含成功状态和图精炼上下文结果的字典
        
    Raises:
        HTTPException: 当获取上下文失败时
    """
    service = _service_or_500()
    try:
        result = service.get_graph_refined_context(
            paper_id=request.paper_id,
            top_entities=request.top_entities,
            snippets_per_entity=request.snippets_per_entity,
            neighbor_limit=request.neighbor_limit,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/paper-summary")
async def paper_summary(request: PaperSummaryRequest):
    """
    生成论文摘要，支持章节感知的递归摘要生成。
    
    Args:
        request (PaperSummaryRequest): 包含论文 ID、实体数量、递归分组大小、章节感知配置等
        
    Returns:
        Dict[str, Any]: 包含成功状态和生成的摘要结果的字典
        
    Raises:
        HTTPException: 当摘要生成失败时
    """
    service = _service_or_500()
    try:
        result = service.generate_paper_summary(
            paper_id=request.paper_id,
            top_entities=request.top_entities,
            snippets_per_entity=request.snippets_per_entity,
            neighbor_limit=request.neighbor_limit,
            recursive_group_size=request.recursive_group_size,
            section_aware=request.section_aware,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/similarity-search")
async def similarity_search(request: SearchRequest):
    """
    基于向量相似度的语义搜索。
    
    Args:
        request (SearchRequest): 包含查询文本和返回结果数量的请求
        
    Returns:
        Dict[str, Any]: 相似度搜索结果，包含匹配的 chunks 和相似度分数
        
    Raises:
        HTTPException: 当搜索失败时
    """
    service = _service_or_500()
    try:
        return service.similarity_search(
            request.query_text,
            top_k=request.top_k,
            paper_ids=request.paper_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/related-papers/{paper_id}")
async def related_papers(
    paper_id: int,
    top_k: int = Query(default=10, ge=1, le=50, description="返回相关文章数量"),
    per_chunk_k: int = Query(default=8, ge=1, le=30, description="每个源切片向量召回数量"),
    source_chunk_limit: int = Query(default=8, ge=1, le=50, description="参与对比的源论文切片数量"),
    evidence_limit: int = Query(default=3, ge=1, le=10, description="每篇候选论文返回的证据条数"),
    concept_weight: float = Query(default=0.65, ge=0.0, le=1.0, description="概念对齐分权重"),
    vector_weight: float = Query(default=0.35, ge=0.0, le=1.0, description="向量证据分权重"),
    min_shared_concepts: int = Query(default=1, ge=0, le=20, description="最少共享概念数过滤阈值"),
):
    """给定论文 ID，返回基于 Concept 对齐+向量证据的相关文章。"""
    service = _service_or_500()
    try:
        result = service.related_papers_by_id(
            paper_id=paper_id,
            top_k=top_k,
            per_chunk_k=per_chunk_k,
            source_chunk_limit=source_chunk_limit,
            evidence_limit=evidence_limit,
            concept_weight=concept_weight,
            vector_weight=vector_weight,
            min_shared_concepts=min_shared_concepts,
        )
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search")
async def rag_search(request: SearchRequest):
    """
    检索增强生成 (RAG) 搜索接口。
    
    Args:
        request (SearchRequest): 包含查询文本和返回结果数量的请求
        
    Returns:
        Dict[str, Any]: RAG 搜索结果，包含检索到的上下文和生成的回答
        
    Raises:
        HTTPException: 当 RAG 搜索失败时
    """
    service = _service_or_500()
    try:
        return service.rag_search(
            request.query_text,
            top_k=request.top_k,
            paper_ids=request.paper_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e