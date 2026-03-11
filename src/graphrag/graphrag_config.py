from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass
class GraphRAGSettings:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    vector_index_name: str
    vector_node_label: str
    vector_embedding_property: str
    vector_text_property: str
    embedding_model: str
    embedding_dimensions: int
    similarity_fn: str
    llm_model: str
    local_embedding_base_url: str
    local_embedding_api_path: str
    local_embedding_timeout: float

    @classmethod
    def from_env(cls) -> "GraphRAGSettings":
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            vector_index_name=os.getenv("GRAPHRAG_VECTOR_INDEX_NAME", "chunk_vector_index"),
            vector_node_label=os.getenv("GRAPHRAG_VECTOR_NODE_LABEL", "Chunk"),
            vector_embedding_property=os.getenv("GRAPHRAG_VECTOR_EMBEDDING_PROPERTY", "embedding"),
            vector_text_property=os.getenv("GRAPHRAG_VECTOR_TEXT_PROPERTY", "text"),
            embedding_model=os.getenv("GRAPHRAG_EMBEDDING_MODEL", "Qwen3-Embedding-8B"),
            embedding_dimensions=int(os.getenv("GRAPHRAG_EMBEDDING_DIMENSIONS", "1536")),
            similarity_fn=os.getenv("GRAPHRAG_SIMILARITY_FN", "cosine"),
            llm_model=os.getenv("GRAPHRAG_LLM_MODEL", "/mnt/chenbaiming/chenbaiming/Qwen3-30B-A3B-Instruct-2507"),
            local_embedding_base_url=os.getenv("LOCAL_EMBEDDING_BASE_URL", "http://localhost:9090"),
            local_embedding_api_path=os.getenv("LOCAL_EMBEDDING_API_PATH", "/v1/embeddings"),
            local_embedding_timeout=float(os.getenv("LOCAL_EMBEDDING_TIMEOUT", "30")),
        )

    def validate(self) -> None:
        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise RuntimeError(
                "Neo4j 连接信息不完整。检查 .env 中的："
                "NEO4J_URI(默认 bolt://localhost:7687), "
                "NEO4J_USER(neo4j), NEO4J_PASSWORD"
            )
    
    def debug_info(self) -> str:
        return (
            f"Neo4j URI: {self.neo4j_uri}\n"
            f"Neo4j User: {self.neo4j_user}\n"
            f"Embedding Model: {self.embedding_model}\n"
            f"Local Embedding Base URL: {self.local_embedding_base_url}\n"
            f"Vector Index: {self.vector_index_name}\n"
            f"LLM Model: {self.llm_model}\n"
            f"\nNeo4j Desktop 使用提示:\n"
            f"  1. 确保 Database 已启动（指示灯为绿色）\n"
            f"  2. 检查 Bolt 端口（默认 7687，可在 Settings 中查看）\n"
            f"  3. 已安装 APOC 库（Plugins 中搜索并下载）\n"
            f"  4. 重启 Database 后刷新连接"
        )


def get_graphrag_settings() -> GraphRAGSettings:
    settings = GraphRAGSettings.from_env()
    settings.validate()
    return settings