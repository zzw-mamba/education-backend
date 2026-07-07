"""GraphRAG module lifecycle hooks for the FastAPI application."""

from .neo4j_config import Neo4jConnectionError, close_neo4j_driver, init_neo4j_driver
from .service import get_graphrag_service


def graphrag_startup(auto_setup_local_neo4j: bool = False) -> None:
    """Initialize GraphRAG dependencies used by the host application."""
    try:
        init_neo4j_driver()
    except Neo4jConnectionError as e:
        print(f"[Neo4j] ✗ 启动时初始化失败，将在请求时重试: {e}")

    if auto_setup_local_neo4j:
        try:
            graphrag_service = get_graphrag_service()
            graphrag_service.setup_local_database(create_vector_index=True, force_recreate_index=False)
            print("[GraphRAG] ✓ 本地 Neo4j schema/index 初始化完成")
        except Exception as e:
            print(f"[GraphRAG] ✗ 本地 Neo4j 自动初始化失败: {e}")


def graphrag_shutdown() -> None:
    """Release GraphRAG resources used by the host application."""
    close_neo4j_driver()
