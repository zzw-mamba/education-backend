import os
from typing import Any, Dict, List, Optional
import requests
import re
import json

from neo4j import GraphDatabase

from graphrag.graphrag_config import GraphRAGSettings, get_graphrag_settings
from database import SessionLocal
from models import KnowledgeBase
from prompt import (
    GRAPHRAG_ENTITY_EXTRACT_SYSTEM_PROMPT,
    GRAPHRAG_ENTITY_EXTRACT_USER_PROMPT_TEMPLATE,
    GRAPHRAG_RAG_QA_SYSTEM_PROMPT,
    GRAPHRAG_RAG_QA_USER_PROMPT_TEMPLATE,
    GRAPHRAG_SUMMARY_BLOCK_USER_PROMPT_TEMPLATE,
    GRAPHRAG_SUMMARY_FINAL_USER_PROMPT_TEMPLATE,
    GRAPHRAG_SUMMARY_SECTION_USER_PROMPT_TEMPLATE,
    GRAPHRAG_SUMMARY_SYSTEM_PROMPT,
)
from utils.model import ask_messages, LLMError

try:
    from neo4j_graphrag.indexes import create_vector_index
    from neo4j_graphrag.retrievers import VectorRetriever
    GRAPHRAG_IMPORT_ERROR = None
except Exception as e:
    create_vector_index = None
    VectorRetriever = None
    GRAPHRAG_IMPORT_ERROR = e


class LocalEmbeddings:
    """本地 Embedding 服务包装类，兼容 neo4j-graphrag 的 Embedder 接口。"""

    def __init__(self, base_url: str, api_path: str, model: str, timeout: int = 30):
        """初始化本地 Embedding 客户端配置。

        Args:
            base_url: Embedding 服务基础地址。
            api_path: Embedding API 路径。
            model: 使用的 embedding 模型名。
            timeout: HTTP 请求超时时间（秒）。
        """
        self.url = base_url.rstrip("/") + ("/" + api_path.lstrip("/") if api_path else "/v1/embeddings")
        self.model = model
        self.timeout = timeout

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条文本，返回向量。"""
        payload = {"input": text, "model": self.model}
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data["data"] and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        if "embedding" in data:
            return data["embedding"]
        raise RuntimeError(f"本地 embedding 返回格式不支持: {data}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入。"""
        return [self.embed_query(t) for t in texts]


def _normalize_section_name(raw_title: str) -> str:
    """将章节标题标准化为统一的章节名称。"""
    title = (raw_title or "").strip().lower()
    if not title:
        return "Body"
    if "abstract" in title or "摘要" in title:
        return "Abstract"
    if "introduction" in title or "引言" in title or "前言" in title:
        return "Introduction"
    if "method" in title or "approach" in title or "方法" in title or "模型" in title:
        return "Methodology"
    if "experiment" in title or "evaluation" in title or "实验" in title or "结果" in title:
        return "Experiments"
    if "conclusion" in title or "总结" in title or "结论" in title:
        return "Conclusion"
    return raw_title.strip() if raw_title.strip() else "Body"


def _split_by_markdown_sections(text: str) -> List[Dict[str, str]]:
    """按 Markdown 标题切分正文，返回带章节名的内容段。"""
    source = (text or "").strip()
    if not source:
        return []

    lines = source.splitlines()
    sections: List[Dict[str, str]] = []
    current_title = "Body"
    buffer: List[str] = []

    header_pattern = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*$")
    for line in lines:
        match = header_pattern.match(line)
        if match:
            content = "\n".join(buffer).strip()
            if content:
                sections.append({
                    "section_name": _normalize_section_name(current_title),
                    "content": content,
                })
            current_title = match.group(1)
            buffer = []
        else:
            buffer.append(line)

    tail = "\n".join(buffer).strip()
    if tail:
        sections.append({
            "section_name": _normalize_section_name(current_title),
            "content": tail,
        })

    if not sections:
        return [{"section_name": "Body", "content": source}]

    return sections


def _semantic_window_chunks(text: str, chunk_size: int = 600, chunk_overlap: int = 90) -> List[str]:
    """按语义句窗切片，保留相邻块的可控重叠。"""
    source = (text or "").strip()
    if not source:
        return []

    if chunk_size <= 0:
        return [source]

    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, int(chunk_size * 0.15))

    sentence_parts = [s.strip() for s in re.split(r"(?<=[。！？!?\.])\s+|\n+", source) if s.strip()]
    if not sentence_parts:
        sentence_parts = [source]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentence_parts:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current).strip())

            overlap_keeper: List[str] = []
            overlap_len = 0
            for prev in reversed(current):
                prev_len = len(prev)
                if overlap_len + prev_len <= chunk_overlap:
                    overlap_keeper.insert(0, prev)
                    overlap_len += prev_len
                else:
                    break

            current = overlap_keeper.copy()
            current_len = sum(len(x) for x in current)

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def _hierarchical_semantic_chunking(text: str, chunk_size: int = 600, chunk_overlap: int = 90) -> List[Dict[str, str]]:
    """先按章节再按语义窗口切片，输出带章节与全局序号的切片。"""
    sections = _split_by_markdown_sections(text)
    if not sections:
        return []

    output: List[Dict[str, str]] = []
    global_index = 0

    # 优化：过滤无用章节 
    # 定义需要被丢弃的章节关键词（不区分大小写）
    ignore_keywords = [
        "reference",       # 参考文献 
        "acknowledgment",  # 致谢
        "conflict of interest", # 利益冲突
        "data availability",    # 数据可用性声明
        "appendix",        # 附录
        "参考文献", "致谢", "利益冲突"
    ]

    for section in sections:
        section_name = section["section_name"]
        
        # 拦截逻辑：一旦章节名字包含了上述词汇，这章就完全不要了
        if any(kw in section_name.lower() for kw in ignore_keywords):
            print(f"[DEBUG - 🚀 文本切片优化] 成功拦截并抛弃无用垃圾章节: {section_name}")
            continue

        content = section["content"]
        sub_chunks = _semantic_window_chunks(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        print(f"\n[DEBUG - 文本切片] 章节: {section_name}")
        print(f"[DEBUG - 文本切片] 原始文本长度: {len(content)} 字符")
        print(f"[DEBUG - 文本切片] 切成了 {len(sub_chunks)} 块")
        
        for i, chunk_text in enumerate(sub_chunks):
            if i < 2: # 只打印前两块预览避免刷屏
                print(f"   [块 {i}] 长度={len(chunk_text)}, 预览: {chunk_text[:50]}...")
                
            output.append(
                {
                    "section_name": section_name,
                    "chunk_text": chunk_text,
                    "chunk_index": global_index,
                }
            )
            global_index += 1

    return output


def _extract_key_entities_from_text(text: str, llm_model: Optional[str] = None) -> List[str]:
    """
    从文本中提取关键实体。
    优先使用 LLM 提取（更准确），回落到正则表达式（更快速）。
    """
    # 快速回落：基于大写字母、括号提示的简单启发式方法
    def _heuristic_extract(source: str) -> List[str]:
        """使用轻量正则规则从文本中提取候选实体。"""
        entities = []
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[\[\(][A-Za-z\-]+[\]\)]'
        matches = re.findall(pattern, source)
        entities.extend([m.strip() for m in matches if m.strip()])
        print(f"[DEBUG - 实体提取 - 正则后备] 提取结果: {list(set(entities))[:10]}")
        return list(set(entities))[:10]

    try:
        print(f"\n[DEBUG - 实体提取 - 输入文本片段] 准备提取实体的文本 (前200字): {text[:200]}...")
        response = ask_messages(
            model=llm_model,
            temperature=0,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个学术论文NLP专家。从给定文本中提取最重要的3-8个领域关键实体。\n"
                        "【严格要求】：\n"
                        "1. 实体必须是专业术语、方法名、模型名(如Transformer)或核心概念。\n"
                        "2. 绝对不可以提取：时间/年份(如2010)、常见宽泛词汇(如Education, Research, Study, We)。\n"
                        "3. 请以纯 JSON 字符串数组形式返回，例如: [\"Transformer\", \"Machine Learning\"]。只返回数组，不要包含其他废话。"
                    ),
                },
                {
                    "role": "user",
                    "content": GRAPHRAG_ENTITY_EXTRACT_USER_PROMPT_TEMPLATE.format(text=text[:500]),
                }
            ]
        )
        content = response.content.strip()
        print(f"[DEBUG - 实体提取 - LLM回复内容]: {content}")
        # 尝试从 JSON 中解析
        if content.startswith('['):
            entities = json.loads(content)
            return [str(e).strip() for e in entities if e][:10]
    except (LLMError, json.JSONDecodeError, Exception) as e:
        print(f"[GraphRAG] ⚠ LLM 实体提取失败，使用启发式方法: {e}")

    return _heuristic_extract(text)


class GraphRAGService:
    def __init__(self, settings: Optional[GraphRAGSettings] = None):
        """初始化 GraphRAG 服务实例与运行时状态。"""
        self.settings = settings or get_graphrag_settings()
        self.driver = None
        self.embedder = None
        self.retriever = None
        # 章节权重：不同章节对摘要的重要性权重
        self.section_weights = {
            "Abstract": 0.0,      # 摘要本身不编入权重计算
            "Introduction": 0.15,  # 背景和问题陈述
            "Methodology": 0.35,   # 方法论最重要
            "Experiments": 0.35,   # 实验结果同样重要
            "Conclusion": 0.15,    # 结论和展望
            "Body": 0.10           # 其他内容权重较低
        }

    def _has_local_embedding(self) -> bool:
        """检查是否配置了本地 embedding 服务地址。"""
        return bool(self.settings.local_embedding_base_url.strip())

    def _local_embedding_url(self) -> str:
        """拼接并返回本地 embedding 服务完整 URL。"""
        base_url = self.settings.local_embedding_base_url.rstrip("/")
        api_path = self.settings.local_embedding_api_path
        if not api_path.startswith("/"):
            api_path = "/" + api_path
        return base_url + api_path

    def _embed_with_local_service(self, text: str) -> List[float]:
        """直接调用本地 embedding HTTP 接口生成向量。"""
        payload = {
            "input": text,
            "model": self.settings.embedding_model,
        }
        response = requests.post(
            self._local_embedding_url(),
            json=payload,
            timeout=self.settings.local_embedding_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data and data["data"] and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        if "embedding" in data:
            return data["embedding"]
        raise RuntimeError(f"本地 embedding 服务返回格式不支持: {data}")

    def _embed_text(self, text: str) -> List[float]:
        """使用本地 embedding 服务生成向量。"""
        if self.embedder is None:
            raise RuntimeError("Embedding 服务未初始化，请检查 LOCAL_EMBEDDING_BASE_URL 配置")
        return self.embedder.embed_query(text)

    def initialize(self) -> None:
        """初始化 Neo4j 连接、Embedding 客户端与向量检索器。"""
        if GRAPHRAG_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"neo4j-graphrag 未安装或导入失败: {GRAPHRAG_IMPORT_ERROR}。请执行 pip install neo4j-graphrag"
            )

        print(f"[GraphRAG] Initializing with settings:\n{self.settings.debug_info()}")
        
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )
        print("[GraphRAG] Connecting to Neo4j...")
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("[GraphRAG] ✓ Neo4j connection successful")
        except Exception as e:
            self.driver.close()
            raise RuntimeError(
                f"无法连接到 Neo4j。请检查:\n"
                f"  1. Neo4j Desktop Database 是否已启动（绿色指示灯）\n"
                f"  2. 地址: {self.settings.neo4j_uri}\n"
                f"  3. 用户名/密码: {self.settings.neo4j_user}\n"
                f"  4. 若路径无效，更新 NEO4J_URI 后重启应用\n"
                f"原错误: {e}"
            ) from e

        # ---- 使用本地 embedding 服务（localhost:9090）----
        if not self._has_local_embedding():
            raise RuntimeError(
                "本地 embedding 服务未配置，请在 .env 中设置:\n"
                "  LOCAL_EMBEDDING_BASE_URL=http://localhost:9090\n"
                "  LOCAL_EMBEDDING_API_PATH=/v1/embeddings\n"
                "  GRAPHRAG_EMBEDDING_MODEL=<model_name>"
            )

        self.embedder = LocalEmbeddings(
            base_url=self.settings.local_embedding_base_url,
            api_path=self.settings.local_embedding_api_path,
            model=self.settings.embedding_model,
            timeout=self.settings.local_embedding_timeout,
        )
        print(f"[GraphRAG] ✓ 使用本地 embedding: {self._local_embedding_url()}")

        self.retriever = VectorRetriever(
            self.driver,
            self.settings.vector_index_name,
            self.embedder,
        )

    def close(self) -> None:
        """关闭 Neo4j 驱动连接。"""
        if self.driver:
            self.driver.close()

    def _ensure_initialized(self) -> None:
        """确保服务已初始化；未初始化时自动初始化。"""
        if self.driver is None:
            self.initialize()

    def _ensure_ai_ready(self) -> None:
        """确保问答能力可用（初始化完成且检索器就绪）。"""
        self._ensure_initialized()
        if self.retriever is None:
            raise RuntimeError("Retriever 未就绪，请检查本地 Embedding 和 Neo4j 配置。")

    def _ensure_embedding_ready(self) -> None:
        """确保 embedding 客户端可用。"""
        self._ensure_initialized()
        if self.embedder is None:
            raise RuntimeError("Embedding 服务未初始化，请检查 LOCAL_EMBEDDING_BASE_URL 配置")

    def _ensure_summary_ready(self) -> None:
        """确保摘要链路依赖已初始化。"""
        self._ensure_initialized()

    def _summarize_with_llm(self, context: str, stage: str = "final", section_hint: Optional[str] = None) -> str:
        """基于阶段化提示词调用 LLM 生成局部或最终摘要。"""
        self._ensure_summary_ready()
        system_prompt = GRAPHRAG_SUMMARY_SYSTEM_PROMPT

        if stage == "block":
            section_hint_text = f"当前处理的章节重点：{section_hint}。" if section_hint else ""
            user_prompt = GRAPHRAG_SUMMARY_BLOCK_USER_PROMPT_TEMPLATE.format(
                section_hint_text=section_hint_text,
                context=context,
            )
        elif stage == "section":
            user_prompt = GRAPHRAG_SUMMARY_SECTION_USER_PROMPT_TEMPLATE.format(
                section_hint=section_hint,
                context=context,
            )
        else:  # final
            user_prompt = GRAPHRAG_SUMMARY_FINAL_USER_PROMPT_TEMPLATE.format(context=context)

        response = ask_messages(
            model=self.settings.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.content.strip()

    def get_graph_refined_context(
        self,
        paper_id: int,
        top_entities: int = 10,
        snippets_per_entity: int = 2,
        neighbor_limit: int = 5,
        section_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        从知识图谱中提取关键实体及其上下文。
        支持按章节过滤，以获得更聚焦的摘要内容。
        
        Args:
            section_filter: 章节列表，如 ["Methodology", "Conclusion"]，为空则不过滤
        """
        self._ensure_initialized()

        section_filter_clause = ""
        if section_filter and len(section_filter) > 0:
            quoted_sections = ','.join([f"'{s}'" for s in section_filter])
            section_filter_clause = f"AND c.section_name IN [{quoted_sections}]"

        topology_query = f"""
        MATCH (p:Paper {{paper_id: $paper_id}})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
        {section_filter_clause}
        WITH e, count(DISTINCT c) AS importance, collect(DISTINCT c)[0..$snippets_per_entity] AS chunks
        ORDER BY importance DESC
        LIMIT $top_entities
        RETURN
          e.name AS entity_name,
          e.type AS entity_type,
          importance,
          [x IN chunks | x.text] AS texts,
          [x IN chunks | x.section_name] AS sections
        """

        community_query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(core:Entity {name: $entity_name, type: $entity_type})
        MATCH (c)-[:MENTIONS]->(nbr:Entity)
        WHERE nbr <> core
        WITH nbr, count(DISTINCT c) AS co_occurrence
        ORDER BY co_occurrence DESC
        LIMIT $neighbor_limit
        RETURN collect({name: nbr.name, type: nbr.type, score: co_occurrence}) AS neighbors
        """

        with self.driver.session() as session:
            records = list(
                session.run(
                    topology_query,
                    {
                        "paper_id": int(paper_id),
                        "top_entities": int(top_entities),
                        "snippets_per_entity": int(snippets_per_entity),
                    },
                )
            )

            refined = []
            for row in records:
                neighbors_row = session.run(
                    community_query,
                    {
                        "paper_id": int(paper_id),
                        "entity_name": row["entity_name"],
                        "entity_type": row["entity_type"],
                        "neighbor_limit": int(neighbor_limit),
                    },
                ).single()
                refined.append(
                    {
                        "entity_name": row["entity_name"],
                        "entity_type": row["entity_type"],
                        "importance": row["importance"],
                        "texts": row["texts"],
                        "sections": row.get("sections", []),
                        "neighbors": neighbors_row["neighbors"] if neighbors_row else [],
                    }
                )

        context_blocks = []
        for item in refined:
            entity_line = f"【实体】{item['entity_name']} ({item['entity_type']})"
            score_line = f"【重要度】{item['importance']}"
            sections = item.get("sections", [])
            section_info = f"【章节】{', '.join(set(sections))}" if sections else "【章节】Body"
            neighbors = item.get("neighbors", [])
            neighbor_line = "【关联】" + ", ".join([f"{n['name']}({n['type']})" for n in neighbors]) if neighbors else "【关联】无"
            snippets = item.get("texts", [])[:snippets_per_entity]
            snippet_line = "【上下文】" + "；".join(snippets) if snippets else "【上下文】无"
            context_blocks.append("\n".join([entity_line, score_line, section_info, neighbor_line, snippet_line]))

        return {
            "paper_id": int(paper_id),
            "context_blocks": context_blocks,
            "entities": refined,
        }

    def generate_paper_summary(
        self,
        paper_id: int,
        top_entities: int = 10,
        snippets_per_entity: int = 2,
        neighbor_limit: int = 5,
        recursive_group_size: int = 4,
        section_aware: bool = True,
    ) -> Dict[str, Any]:
        """
        生成论文摘要，支持章节感知的递归聚合。
        
        Args:
            section_aware: 是否按章节分别生成摘要后合成（更精准但更慢）
        """
        # 如果启用章节感知，按不同章节重要度分别提取和摘要
        if section_aware:
            section_summaries = {}
            section_order = ["Introduction", "Methodology", "Experiments", "Conclusion"]
            
            for section in section_order:
                context_info = self.get_graph_refined_context(
                    paper_id=paper_id,
                    top_entities=max(3, top_entities // 3),
                    snippets_per_entity=snippets_per_entity,
                    neighbor_limit=neighbor_limit,
                    section_filter=[section],
                )
                
                blocks = context_info.get("context_blocks", [])
                if blocks:
                    combined_context = "\n\n".join(blocks)
                    section_summaries[section] = self._summarize_with_llm(
                        combined_context,
                        stage="section",
                        section_hint=section
                    )
                else:
                    section_summaries[section] = ""
            
            # 整合各章节摘要为最终摘要
            final_components = [
                s for s in [
                    section_summaries.get("Introduction", ""),
                    section_summaries.get("Methodology", ""),
                    section_summaries.get("Experiments", ""),
                    section_summaries.get("Conclusion", ""),
                ] if s
            ]
            
            combined_section_summaries = "\n\n".join(final_components)
            final_summary = self._summarize_with_llm(combined_section_summaries, stage="final")
            
            return {
                "paper_id": int(paper_id),
                "summary": final_summary,
                "section_summaries": section_summaries,
                "intermediate_summaries": final_components,
                "summary_method": "section_aware",
            }
        
        # 原有逻辑：非章节感知的聚合
        context_info = self.get_graph_refined_context(
            paper_id=paper_id,
            top_entities=top_entities,
            snippets_per_entity=snippets_per_entity,
            neighbor_limit=neighbor_limit,
        )

        blocks = context_info["context_blocks"]
        if not blocks:
            return {
                "paper_id": int(paper_id),
                "summary": "未找到可用于摘要的图上下文，请先完成切片与实体关系构建。",
                "intermediate_summaries": [],
                "context_blocks": [],
                "summary_method": "none",
            }

        group_size = max(1, int(recursive_group_size))
        grouped = [blocks[i:i + group_size] for i in range(0, len(blocks), group_size)]

        intermediate_summaries = []
        for group in grouped:
            block_context = "\n\n".join(group)
            intermediate_summaries.append(self._summarize_with_llm(block_context, stage="block"))

        final_context = "\n\n".join(intermediate_summaries)
        final_summary = self._summarize_with_llm(final_context, stage="final")

        return {
            "paper_id": int(paper_id),
            "summary": final_summary,
            "intermediate_summaries": intermediate_summaries,
            "context_blocks": blocks,
            "summary_method": "recursive_aggregation",
        }

    def setup_local_database(self, create_vector_index: bool = True, force_recreate_index: bool = False) -> Dict[str, Any]:
        """准备本地图数据库环境并返回当前统计信息。"""
        self._ensure_initialized()
        schema_result = self.ensure_graph_schema()
        index_result = {"created": False, "index_name": self.settings.vector_index_name}
        if create_vector_index:
            index_result = self.ensure_vector_index(force_recreate=force_recreate_index)

        with self.driver.session() as session:
            paper_count = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]

        return {
            "schema_ready": schema_result.get("schema_ready", False),
            "vector_index": index_result,
            "counts": {
                "paper": paper_count,
                "chunk": chunk_count,
                "entity": entity_count,
            },
        }

    def ensure_graph_schema(self) -> Dict[str, Any]:
        """创建图谱约束与索引（幂等）。"""
        self._ensure_initialized()

        schema_queries = [
            "CREATE CONSTRAINT paper_paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year)",
        ]

        with self.driver.session() as session:
            for query in schema_queries:
                session.run(query)

        return {"schema_ready": True}

    def ensure_vector_index(self, force_recreate: bool = False) -> Dict[str, Any]:
        """确保向量索引存在，可按需强制重建。"""
        self._ensure_initialized()
        self.ensure_graph_schema()

        with self.driver.session() as session:
            rows = list(
                session.run(
                    "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN name, options",
                    {"name": self.settings.vector_index_name},
                )
            )

            exists = len(rows) > 0
            existing_dimension = None
            if exists:
                options = rows[0].get("options") or {}
                index_config = options.get("indexConfig") or {}
                existing_dimension = index_config.get("vector.dimensions")

            if exists and force_recreate:
                session.run(f"DROP INDEX {self.settings.vector_index_name} IF EXISTS")
                exists = False
            elif exists and existing_dimension not in (None, self.settings.embedding_dimensions):
                raise RuntimeError(
                    f"向量索引 {self.settings.vector_index_name} 的维度为 {existing_dimension}，"
                    f"但当前配置 GRAPHRAG_EMBEDDING_DIMENSIONS={self.settings.embedding_dimensions}。"
                    "请调用 /api/graphrag/create-index 并传 force_recreate=true，"
                    "然后重新执行 /api/graphrag/sync-from-mysql 以重建向量索引和 embedding 数据。"
                )

        if not exists:
            create_vector_index(
                self.driver,
                self.settings.vector_index_name,
                label="Chunk",
                embedding_property="embedding",
                dimensions=self.settings.embedding_dimensions,
                similarity_fn=self.settings.similarity_fn,
            )
            return {"created": True, "index_name": self.settings.vector_index_name}

        return {"created": False, "index_name": self.settings.vector_index_name}

    def _get_vector_index_dimension(self) -> Optional[int]:
        """读取当前向量索引的配置维度。"""
        self._ensure_initialized()

        with self.driver.session() as session:
            row = session.run(
                "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options",
                {"name": self.settings.vector_index_name},
            ).single()

        if not row:
            return None

        options = row.get("options") or {}
        index_config = options.get("indexConfig") or {}
        return index_config.get("vector.dimensions")

    def _validate_query_vector_dimension(self, vector: List[float]) -> int:
        """校验查询向量维度与向量索引维度一致。"""
        actual_dimension = len(vector)
        index_dimension = self._get_vector_index_dimension()

        if index_dimension is None:
            raise RuntimeError(
                f"向量索引 {self.settings.vector_index_name} 不存在，请先调用 /api/graphrag/create-index 创建索引。"
            )

        if actual_dimension != index_dimension:
            raise RuntimeError(
                f"向量维度不匹配：索引 {self.settings.vector_index_name} 维度为 {index_dimension}，"
                f"当前向量维度为 {actual_dimension}。"
                "这通常表示 embedding 模型已切换但索引未重建。"
                "请调用 /api/graphrag/create-index 并传 force_recreate=true，"
                "然后重新执行 /api/graphrag/sync-from-mysql。"
            )

        return index_dimension

    def upsert_paper_chunks(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """写入论文切片、实体关系与结构化关联边。"""
        self._ensure_embedding_ready()

        if not rows:
            return {"upserted": 0}

        normalized_rows = []
        for row in rows:
            paper_id = row.get("paper_id")
            title = str(row.get("title") or "").strip()
            year = row.get("year")
            chunk_id = str(row.get("chunk_id") or "").strip()
            text = str(row.get("text") or "").strip()
            index = row.get("index")
            section_name = str(row.get("section_name") or "Body")
            key_entities = row.get("key_entities") or []
            entities = row.get("entities") or []

            if paper_id is None or not chunk_id or not text:
                continue

            # 如果未提供实体，尝试从文本中自动提取
            if not entities:
                extracted_entity_names = _extract_key_entities_from_text(text, self.settings.llm_model)
                entities = [{"name": name, "type": "Concept"} for name in extracted_entity_names]

            normalized_entities = []
            for entity in entities:
                entity_name = str(entity.get("name") or "").strip()
                if not entity_name:
                    continue
                normalized_entities.append(
                    {
                        "name": entity_name,
                        "type": str(entity.get("type") or "Unknown").strip() or "Unknown",
                    }
                )

            # 更新 key_entities 列表
            if normalized_entities and not key_entities:
                key_entities = [e["name"] for e in normalized_entities]

            normalized_rows.append(
                {
                    "paper_id": int(paper_id),
                    "title": title,
                    "year": int(year) if year is not None else None,
                    "chunk_id": chunk_id,
                    "text": text,
                    "index": int(index) if index is not None else 0,
                    "section_name": section_name,
                    "key_entities": key_entities,
                    "entities": normalized_entities,
                }
            )

        if not normalized_rows:
            return {"upserted": 0}

        for row in normalized_rows:
            row["embedding"] = self._embed_text(row["text"])

        query = """
        UNWIND $rows AS row
        MERGE (p:Paper {paper_id: row.paper_id})
        SET p.title = row.title,
            p.year = row.year,
            p.updated_at = datetime()

        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.text = row.text,
            c.embedding = row.embedding,
            c.index = row.index,
            c.chunk_index = row.index,
            c.section_name = row.section_name,
            c.key_entities = row.key_entities,
            c.updated_at = datetime()

        MERGE (p)-[:HAS_CHUNK]->(c)

        FOREACH (ent IN coalesce(row.entities, []) |
            MERGE (e:Entity {name: ent.name, type: ent.type})
            MERGE (c)-[:MENTIONS]->(e)
            MERGE (p)-[:HAS_ENTITY]->(e)
        )

        RETURN count(c) AS count
        """

        with self.driver.session() as session:
            record = session.run(query, {"rows": normalized_rows}).single()
            count = record["count"] if record else 0

            # 建立切片的线性链接（[:NEXT]关系）
            # 这保证了摘要时的叙事流连贯性
            next_query = """
            MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)
            WITH p, c ORDER BY p.paper_id, c.chunk_index
            WITH p, collect(c) AS chunk_list
            UNWIND range(0, size(chunk_list) - 2) AS i
            WITH chunk_list[i] AS curr, chunk_list[i + 1] AS nxt
            MERGE (curr)-[:NEXT]->(nxt)
            SET curr.has_next = true, nxt.has_prev = true
            """
            session.run(next_query)
            
            # 建立同章节内的切片关联（[:CONTEXT_NEIGHBOR]关系）
            # 这帮助相同章节的实体互相关联
            context_query = """
            MATCH (p:Paper)-[:HAS_CHUNK]->(c1:Chunk)
            MATCH (p)-[:HAS_CHUNK]->(c2:Chunk)
            WHERE c1.section_name = c2.section_name 
              AND c1.chunk_index < c2.chunk_index
              AND c2.chunk_index - c1.chunk_index <= 3
            MERGE (c1)-[:CONTEXT_NEIGHBOR]->(c2)
            """
            session.run(context_query)

        return {"upserted": count}

    def sync_from_mysql_knowledge_base(
        self,
        paper_ids: Optional[List[int]] = None,
        limit: int = 100,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        auto_extract_entities: bool = True,
    ) -> Dict[str, Any]:
        """
        从 MySQL 知识库同步论文数据到 Neo4j，支持自动实体提取。
        
        Args:
            auto_extract_entities: 是否自动从切片中提取关键实体（需要 LLM 服务）
        """
        self._ensure_embedding_ready()
        db = SessionLocal()
        try:
            query = db.query(KnowledgeBase)
            if paper_ids:
                query = query.filter(KnowledgeBase.id.in_(paper_ids))
            if limit > 0:
                query = query.limit(limit)

            papers = query.all()

            rows: List[Dict[str, Any]] = []
            for paper in papers:
                content = paper.content or ""
                hierarchical_chunks = _hierarchical_semantic_chunking(
                    content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if not hierarchical_chunks:
                    continue

                paper_entities = []
                try:
                    for tag in (paper.tags or []):
                        tag_name = (tag.name or "").strip()
                        if tag_name:
                            paper_entities.append({"name": tag_name, "type": "Tag"})
                except Exception:
                    pass

                key_entities = [entity["name"] for entity in paper_entities]

                for chunk_item in hierarchical_chunks:
                    chunk_text = chunk_item["chunk_text"]
                    
                    # 自动从切片提取实体，与标签合并
                    chunk_entities = paper_entities.copy()
                    if auto_extract_entities:
                        try:
                            extracted_names = _extract_key_entities_from_text(chunk_text, self.settings.llm_model)
                            for name in extracted_names:
                                if not any(e["name"] == name for e in chunk_entities):
                                    chunk_entities.append({"name": name, "type": "Concept"})
                        except Exception as e:
                            print(f"[GraphRAG] ⚠ 自动实体提取失败: {e}")
                    
                    rows.append(
                        {
                            "paper_id": paper.id,
                            "title": paper.title or "",
                            "year": paper.year,
                            "chunk_id": f"{paper.id}_{chunk_item['chunk_index']}",
                            "text": chunk_text,
                            "index": chunk_item["chunk_index"],
                            "section_name": chunk_item["section_name"],
                            "key_entities": key_entities,
                            "entities": chunk_entities,
                        }
                    )

            if not rows:
                return {"papers": len(papers), "chunks": 0, "upserted": 0}

            self.ensure_vector_index(force_recreate=False)
            result = self.upsert_paper_chunks(rows)
            return {
                "papers": len(papers),
                "chunks": len(rows),
                "upserted": result.get("upserted", 0),
                "auto_extract_entities": auto_extract_entities,
            }
        finally:
            db.close()


    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """基于向量索引执行相似度检索。"""
        self._ensure_embedding_ready()
        query_embedding = self._embed_text(query_text)
        self._validate_query_vector_dimension(query_embedding)

        normalized_paper_ids: Optional[List[int]] = None
        if paper_ids is not None:
            normalized_paper_ids = sorted({int(pid) for pid in paper_ids})

        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        MATCH (p:Paper)-[:HAS_CHUNK]->(node)
        WHERE $paper_ids IS NULL OR p.paper_id IN $paper_ids
        RETURN
          node.chunk_id AS chunk_id,
          node.text AS text,
          node.index AS chunk_index,
          score,
          p.paper_id AS paper_id,
          p.title AS title,
          p.year AS year
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            rows = list(
                session.run(
                    query,
                    {
                        "index_name": self.settings.vector_index_name,
                        "top_k": top_k,
                        "embedding": query_embedding,
                        "paper_ids": normalized_paper_ids,
                    },
                )
            )

        response = [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "chunk_index": r["chunk_index"],
                "score": r["score"],
                "paper_id": r["paper_id"],
                "title": r["title"],
                "year": r["year"],
            }
            for r in rows
        ]
        return {
            "query_text": query_text,
            "top_k": top_k,
            "paper_ids": normalized_paper_ids,
            "results": response,
        }

    def related_papers_by_id(
        self,
        paper_id: int,
        top_k: int = 10,
        per_chunk_k: int = 8,
        source_chunk_limit: int = 8,
        evidence_limit: int = 3,
    ) -> Dict[str, Any]:
        """给定论文 ID，基于 Chunk 向量相似度聚合召回相关文章。"""
        self._ensure_embedding_ready()
        with self.driver.session() as session:
            source_row = session.run(
                """
                MATCH (p:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN p.paper_id AS paper_id, p.title AS title, count(c) AS chunk_count
                """,
                {"paper_id": int(paper_id)},
            ).single()

            if not source_row:
                raise ValueError(f"论文 {paper_id} 尚未同步到 GraphRAG 图数据库")

            sample_chunk = session.run(
                """
                MATCH (:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN c.embedding AS embedding
                LIMIT 1
                """,
                {"paper_id": int(paper_id)},
            ).single()

            if not sample_chunk or not sample_chunk.get("embedding"):
                raise ValueError(f"论文 {paper_id} 缺少可用的 embedding，请先重新同步到 GraphRAG")

            self._validate_query_vector_dimension(sample_chunk["embedding"])

            rows = list(
                session.run(
                    """
                    MATCH (source:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(source_chunk:Chunk)
                    WHERE source_chunk.embedding IS NOT NULL
                    WITH source, collect(source_chunk)[0..$source_chunk_limit] AS source_chunks
                    UNWIND source_chunks AS source_chunk
                    CALL db.index.vector.queryNodes($index_name, $per_chunk_k, source_chunk.embedding)
                    YIELD node, score
                    MATCH (candidate:Paper)-[:HAS_CHUNK]->(node)
                    WHERE candidate.paper_id <> $paper_id
                    WITH source, candidate,
                         collect(DISTINCT {
                            source_chunk_id: source_chunk.chunk_id,
                            matched_chunk_id: node.chunk_id,
                            score: score,
                            text: substring(coalesce(node.text, ''), 0, 180)
                         }) AS evidence,
                         max(score) AS max_score,
                         avg(score) AS avg_score,
                         count(DISTINCT node) AS matched_chunks
                    RETURN
                      source.paper_id AS source_paper_id,
                      source.title AS source_title,
                      candidate.paper_id AS paper_id,
                      candidate.title AS title,
                      candidate.year AS year,
                      max_score,
                      avg_score,
                      matched_chunks,
                      evidence[0..$evidence_limit] AS evidence
                    ORDER BY avg_score DESC, matched_chunks DESC, max_score DESC
                    LIMIT $top_k
                    """,
                    {
                        "paper_id": int(paper_id),
                        "index_name": self.settings.vector_index_name,
                        "top_k": int(top_k),
                        "per_chunk_k": int(per_chunk_k),
                        "source_chunk_limit": int(source_chunk_limit),
                        "evidence_limit": int(evidence_limit),
                    },
                )
            )


        return {
            "paper_id": int(source_row["paper_id"]),
            "title": source_row["title"],
            "source_chunk_count": source_row["chunk_count"],
            "top_k": int(top_k),
            "results": [
                {
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "year": row["year"],
                    "max_score": row["max_score"],
                    "avg_score": row["avg_score"],
                    "matched_chunks": row["matched_chunks"],
                    "evidence": row["evidence"],
                }
                for row in rows
            ],
        }

    def rag_search(
        self,
        query_text: str,
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """检索相关 chunks 后调用本地 LLM 生成回答。"""
        self._ensure_ai_ready()

        # 1) 向量检索获取上下文
        search_result = self.similarity_search(query_text, top_k=top_k, paper_ids=paper_ids)
        chunks = search_result.get("results", [])
        if not chunks:
            return {
                "query_text": query_text,
                "top_k": top_k,
                "paper_ids": search_result.get("paper_ids"),
                "answer": "未找到相关内容。",
                "context_chunks": [],
            }

        # 2) 拼接上下文
        context = "\n\n".join(
            f"[来源: {c.get('title', '未知')}] {c['text']}" for c in chunks if c.get("text")
        )

        # 3) 调用本地 LLM 生成回答
        response = ask_messages(
            model=self.settings.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": GRAPHRAG_RAG_QA_SYSTEM_PROMPT},
                {"role": "user", "content": GRAPHRAG_RAG_QA_USER_PROMPT_TEMPLATE.format(context=context, query_text=query_text)},
            ],
        )
        answer = response.content.strip()

        return {
            "query_text": query_text,
            "top_k": top_k,
            "paper_ids": search_result.get("paper_ids"),
            "answer": answer,
            "context_chunks": [{"chunk_id": c.get("chunk_id"), "text": c.get("text"), "score": c.get("score")} for c in chunks],
        }

    def query_section_entities(self, paper_id: int, section: str) -> Dict[str, Any]:
        """
        查询指定章节中的所有实体及其共现频率统计。
        用于快速了解各章节的核心概念分布。
        """
        self._ensure_initialized()

        query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(c:Chunk {section_name: $section})-[:MENTIONS]->(e:Entity)
        WITH e, count(DISTINCT c) AS frequency
        ORDER BY frequency DESC
        RETURN 
          e.name AS entity_name,
          e.type AS entity_type,
          frequency
        LIMIT 20
        """

        with self.driver.session() as session:
            records = list(session.run(query, {"paper_id": int(paper_id), "section": section}))

        return {
            "paper_id": int(paper_id),
            "section": section,
            "entities": [
                {"name": r["entity_name"], "type": r["entity_type"], "frequency": r["frequency"]}
                for r in records
            ],
        }

    def query_chunk_chain(self, paper_id: int, start_chunk_id: int, depth: int = 3) -> Dict[str, Any]:
        """
        查询某个切片的前后链。用于理解切片之间的依赖关系和信息流。
        """
        self._ensure_initialized()

        query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:HAS_CHUNK]->(start:Chunk {chunk_id: $chunk_id})
        MATCH path = (start)-[:NEXT*0..{depth}]->(end:Chunk)
        RETURN [node IN nodes(path) | {
          chunk_id: node.chunk_id,
          section: node.section_name,
          text: node.text[0..100]
        }] AS chunk_chain
        """.format(depth=int(depth))

        with self.driver.session() as session:
            record = session.run(query, {"paper_id": int(paper_id), "chunk_id": start_chunk_id}).single()

        if record and record.get("chunk_chain"):
            return {
                "paper_id": int(paper_id),
                "start_chunk_id": start_chunk_id,
                "depth": depth,
                "chunk_chain": record["chunk_chain"],
            }
        
        return {
            "paper_id": int(paper_id),
            "start_chunk_id": start_chunk_id,
            "depth": depth,
            "chunk_chain": [],
        }

    def get_entity_context_snippets(self, entity_name: str, entity_type: str, paper_id: Optional[int] = None, limit: int = 5) -> Dict[str, Any]:
        """
        获取某个实体在论文中的所有上下文片段。
        支持跨论文查询，用于比较同一实体在不同论文中的角色。
        """
        self._ensure_initialized()

        paper_filter = ""
        if paper_id:
            paper_filter = f"AND p.paper_id = {int(paper_id)}"

        query = f"""
        MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity {{name: $entity_name, type: $entity_type}})
        {paper_filter}
        RETURN
          p.paper_id AS paper_id,
          p.title AS paper_title,
          c.chunk_id AS chunk_id,
          c.section_name AS section,
          c.text AS text
        LIMIT $limit
        """

        with self.driver.session() as session:
            records = list(session.run(query, {"entity_name": entity_name, "entity_type": entity_type, "limit": limit}))

        return {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "paper_id": paper_id,
            "snippets": [
                {
                    "paper_id": r["paper_id"],
                    "paper_title": r["paper_title"],
                    "chunk_id": r["chunk_id"],
                    "section": r["section"],
                    "text": r["text"],
                }
                for r in records
            ],
        }


_graphrag_service: Optional[GraphRAGService] = None


def get_graphrag_service() -> GraphRAGService:
    """获取全局单例 GraphRAGService，首次调用时自动初始化。"""
    global _graphrag_service
    if _graphrag_service is None:
        _graphrag_service = GraphRAGService()
        _graphrag_service.initialize()
    return _graphrag_service