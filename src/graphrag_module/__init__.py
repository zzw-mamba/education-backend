"""GraphRAG module public API."""

from .routes import router
from .lifecycle import graphrag_shutdown, graphrag_startup
from .service import GRAPHRAG_IMPORT_ERROR, GraphRAGService, get_graphrag_service

__all__ = [
    "GRAPHRAG_IMPORT_ERROR",
    "GraphRAGService",
    "graphrag_shutdown",
    "graphrag_startup",
    "get_graphrag_service",
    "router",
]
