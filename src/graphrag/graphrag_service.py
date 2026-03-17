import os
from typing import Any, Dict, List, Optional
import requests
import re
import json

from neo4j import GraphDatabase

from graphrag.graphrag_config import GraphRAGSettings, get_graphrag_settings
from database import SessionLocal
from models import KnowledgeBase
from utils.model import ask_messages, LLMError
from prompt import (
    GRAPHRAG_QUERY_ENTITY_EXPANSION_SYSTEM_PROMPT,
    GRAPHRAG_QUERY_ENTITY_EXPANSION_USER_PROMPT_TEMPLATE,
)

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


def _normalize_concept_term(raw: str) -> str:
    """将术语规范化为可匹配 key（小写+下划线）。"""
    term = (raw or "").strip().lower()
    if term.startswith("http://") or term.startswith("https://"):
        marker = "/topics/"
        if marker in term:
            term = term.split(marker, 1)[1]
        else:
            term = term.rsplit("/", 1)[-1]
    term = term.replace("-", "_").replace(" ", "_").replace("/", "_")
    term = re.sub(r"[^0-9a-z_]+", "", term)
    term = re.sub(r"_+", "_", term).strip("_")
    return term


def _label_from_topic_uri(uri: str) -> str:
    """从 topic URI 生成可读名称。"""
    value = (uri or "").strip()
    marker = "/topics/"
    if marker in value:
        slug = value.split(marker, 1)[1]
    else:
        slug = value.rsplit("/", 1)[-1]
    return re.sub(r"[_\-]+", " ", slug).strip()


def _parse_nt_uri_triple(line: str) -> Optional[tuple]:
    """解析 N-Triples 中 subject/predicate/object 全为 URI 的行。"""
    match = re.match(r'^\s*<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$', line)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


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


def _optimize_chunk_params_by_language(text: str, default_size: int = 600, default_overlap: int = 90) -> tuple:
    """根据中英文比例自适应调整切片参数。
    
    不同语言的信息密度差异极大：
    1. 中文（高信息密度）：500字已包含大量核心逻辑推导。如果采用极大的 chunk_size (如1000)，
       在进行向量相似度检索（Cosine Similarity）时，用户的微小 Query 会被同切片中大量无关内容严重稀释。
       同时，中文绝大多数学术复合长句均在 80 字以内，overlap=80 已足以保证上下文完全不切断。
       
    2. 英文（低信息密度/单词长）：英文单词长度大，长定语从句结构复杂。
       对于英文文献，1000 个字符往往才能传达一个完整的实验结论片段。
       如果 overlap 太短（<150），具有 200+ 字符的长句会因为 “不可腰斩” 的强约束机制而被无情完全舍弃，导致图谱断层。
    """
    zh_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    total_len = len(text) or 1
    zh_ratio = zh_chars / total_len
    
    # 如果中文占比小（少于 5%），认定为纯英文论文
    if zh_ratio < 0.05:
        return 1000, 150 # 英文使用大窗口，高冗余重叠
    else:
        return 500, 80   # 中文使用高密度紧凑切分


def _hierarchical_semantic_chunking(text: str, chunk_size: int = 600, chunk_overlap: int = 90) -> List[Dict[str, str]]:
    """先按章节再按语义窗口切片，输出带章节与全局序号的切片。"""
    
    # ================= 优化 2：语言自适应计算最佳切片结界 =================
    adapted_size, adapted_overlap = _optimize_chunk_params_by_language(text, chunk_size, chunk_overlap)
    print(f"\n[DEBUG - 🌍 语言自适应切片] 检测总体文本度 {len(text)} 字符 -> 动态应用 chunk_size={adapted_size}, chunk_overlap={adapted_overlap}")
    chunk_size = adapted_size
    chunk_overlap = adapted_overlap

    sections = _split_by_markdown_sections(text)
    if not sections:
        return []

    output: List[Dict[str, str]] = []
    global_index = 0

    # ================= 优化 1：过滤无用章节 =================
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


def _extract_triplets_from_text(text: str, llm_model: Optional[str] = None) -> List[Dict[str, str]]:
    # ================= 优化 3：提取实体->提取关系 =================
    """
    从文本中提取知识图谱三元组 (实体-关系-实体)。
    优先使用 LLM 提取（包含图谱结构的边），回落到正则表达式（仅提取孤立实体）。
    """
    def _heuristic_extract(source: str) -> List[Dict[str, str]]:
        """使用轻量正则后备，只提取部分独立实体，无法建立高级关系。"""
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[\[\(][A-Za-z\-]+[\]\)]'
        matches = list(set([m.strip() for m in re.findall(pattern, source) if m.strip()]))
        # 把提取到的实体首尾相连造伪关系防止断网，或干脆就不加关系
        if len(matches) >= 2:
            return [{"source": matches[0], "target": matches[1], "relation": "RELATED"}]
        return []

    try:
        print(f"\n[DEBUG - 节点关系提取 - 输入文本] 准备提取知识网 (前200字): {text[:200]}...")
        response = ask_messages(
            model=llm_model,
            temperature=0.1,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个高级知识图谱信息抽取专家。请从给定文本中提取核心的知识图谱三元组（实体-关系-实体）。\n"
                        "【严格要求】：\n"
                        "1. 实体必须是具体的专业术语、方法名、模型或核心研究概念。\n"
                        "2. 关系必须简短、明确（例如：包含、应用于、提升了、基于、对比）。\n"
                        "3. 只输出关键的 3-8 个三元组。\n"
                        "4. 请务必严格以合法的 JSON 对象数组格式返回！要求完全符合如下格式，绝对不要加任何外部注释或 markdown 代码块：\n"
                        '[{"source": "实体1", "target": "实体2", "relation": "关系描述"}]'
                    ),
                },
                {
                    "role": "user",
                    "content": f"提取下面文本的三元字典：\n{text[:1000]}",
                }
            ]
        )
        content = response.content.strip()
        # 清理可能存在的 markdown code block
        content = re.sub(r"^```[a-zA-Z]*\n|```$", "", content, flags=re.MULTILINE).strip()
        print(f"[DEBUG - 三元组提取 - LLM回复]: {content}")
        
        if content.startswith('['):
            triplets = json.loads(content)
            # 过滤掉不规范的数据
            valid_triplets = []
            for t in triplets:
                if isinstance(t, dict) and "source" in t and "target" in t and "relation" in t:
                    valid_triplets.append({
                        "source": str(t["source"]).strip(),
                        "target": str(t["target"]).strip(),
                        "relation": str(t["relation"]).strip()
                    })
            return valid_triplets[:8]
    except (LLMError, json.JSONDecodeError, Exception) as e:
        print(f"[GraphRAG] ⚠ LLM 实体边提取失败，回落到启发式方法: {e}")

    return _heuristic_extract(text)


class GraphRAGService:
    def __init__(self, settings: Optional[GraphRAGSettings] = None):
        """初始化 GraphRAG 服务实例与运行时状态。"""
        self.settings = settings or get_graphrag_settings()
        self.driver = None
        self.embedder = None
        self.retriever = None

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
        system_prompt = (
            "你是一个学术论文分析专家，擅长根据知识图谱结构化上下文生成严谨、精炼的中文摘要。"
            "注意：摘要应按逻辑顺序组织（问题→方法→结果→结论），避免重复和冗余。"
        )

        if stage == "block":
            section_hint_text = f"当前处理的章节重点：{section_hint}。" if section_hint else ""
            user_prompt = (
                f"{section_hint_text}"
                "以下是论文知识图谱中一组核心实体及上下文。"
                "请提炼该组信息的局部摘要，聚焦 Problem/Method/Contribution，控制在120字以内。\n\n"
                f"{context}"
            )
        elif stage == "section":
            user_prompt = (
                f"以下是论文【{section_hint}】章节的结构化信息。"
                "请生成该章节的摘要，控制在150字以内，突出该章节的核心贡献：\n\n"
                f"{context}"
            )
        else:  # final
            user_prompt = (
                "以下是从论文知识图谱多轮聚合得到的结构化信息。"
                "请生成最终学术摘要，要求：\n"
                "1) 明确核心问题（Problem Statement）\n"
                "2) 提炼关键方法（Methodology）\n"
                "3) 总结主要贡献/实验结论（Contribution）\n"
                "4) 语言专业、精炼、避免重复，中文输出，200-300字。\n\n"
                f"{context}"
            )

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

    def setup_local_database(
        self,
        create_vector_index: bool = True,
        force_recreate_index: bool = False,
        ontology_file_path: str = "CSO.3.5.nt",
    ) -> Dict[str, Any]:
        """准备本地图数据库环境并返回当前统计信息。"""
        self._ensure_initialized()
        schema_result = self.ensure_graph_schema()
        index_result = {"created": False, "index_name": self.settings.vector_index_name}
        if create_vector_index:
            index_result = self.ensure_vector_index(force_recreate=force_recreate_index)

        ontology_path = ontology_file_path
        if not os.path.isabs(ontology_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            ontology_path = os.path.join(project_root, os.path.normpath(ontology_path).replace("src\\", "").replace("src/", ""))
        print(f"[GraphRAG] 使用本体文件路径: {ontology_path}")
        ontology_result = self.import_ontology(
            nt_file_path=ontology_path,
            batch_size=3000,
            max_triples=0,
            link_existing_entities=True,
        )

        with self.driver.session() as session:
            paper_count = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            concept_count = session.run("MATCH (c:Concept) RETURN count(c) AS c").single()["c"]

        return {
            "schema_ready": schema_result.get("schema_ready", False),
            "vector_index": index_result,
            "ontology": ontology_result,
            "counts": {
                "paper": paper_count,
                "chunk": chunk_count,
                "entity": entity_count,
                "concept": concept_count,
            },
        }

    def link_entities_to_concept(
        self,
        entity_rows: Optional[List[Dict[str, str]]] = None,
        only_unlinked: bool = True,
    ) -> Dict[str, Any]:
        """将 Entity 节点按标准化名称关联到 Concept。"""
        self._ensure_initialized()
        self.ensure_graph_schema()

        with self.driver.session() as session:
            concept_count_row = session.run("MATCH (c:Concept) RETURN count(c) AS c").single()
            concept_count = int(concept_count_row["c"] if concept_count_row else 0)
            if concept_count == 0:
                return {"linked": 0, "concepts": 0, "matched_candidates": 0}

            rows_for_match: List[Dict[str, str]] = []
            if entity_rows is not None:
                for item in entity_rows:
                    name = str(item.get("name") or "").strip()
                    etype = str(item.get("type") or "Unknown").strip() or "Unknown"
                    if not name:
                        continue
                    normalized_name = _normalize_concept_term(name)
                    if not normalized_name:
                        continue
                    rows_for_match.append({"name": name, "type": etype, "normalized_name": normalized_name})
            else:
                if only_unlinked:
                    fetch_query = """
                    MATCH (e:Entity)
                    WHERE e.name IS NOT NULL AND NOT (e)-[:ALIGNED_TO_CONCEPT]->(:Concept)
                    RETURN e.name AS name, e.type AS type
                    """
                else:
                    fetch_query = """
                    MATCH (e:Entity)
                    WHERE e.name IS NOT NULL
                    RETURN e.name AS name, e.type AS type
                    """
                for record in session.run(fetch_query):
                    name = str(record["name"] or "").strip()
                    etype = str(record["type"] or "Unknown").strip() or "Unknown"
                    if not name:
                        continue
                    normalized_name = _normalize_concept_term(name)
                    if not normalized_name:
                        continue
                    rows_for_match.append({"name": name, "type": etype, "normalized_name": normalized_name})

            dedup: Dict[str, Dict[str, str]] = {}
            for row in rows_for_match:
                dedup[f"{row['name']}||{row['type']}"] = row
            rows = list(dedup.values())
            if not rows:
                return {"linked": 0, "concepts": concept_count, "matched_candidates": 0}

            link_query = """
            UNWIND $rows AS row
            MATCH (e:Entity {name: row.name, type: row.type})
            MATCH (c:Concept {normalized_name: row.normalized_name})
            MERGE (e)-[:ALIGNED_TO_CONCEPT {method: 'normalized_exact'}]->(c)
            RETURN count(*) AS linked
            """
            linked_row = session.run(link_query, {"rows": rows}).single()
            linked_count = int(linked_row["linked"] if linked_row else 0)

        return {
            "linked": linked_count,
            "concepts": concept_count,
            "matched_candidates": len(rows),
        }

    def import_ontology(
        self,
        nt_file_path: str,
        batch_size: int = 2000,
        max_triples: int = 0,
        link_existing_entities: bool = True,
    ) -> Dict[str, Any]:
        """从 N-Triples 文件导入领域概念及其关系。"""
        self._ensure_initialized()
        self.ensure_graph_schema()

        file_path = nt_file_path if os.path.isabs(nt_file_path) else os.path.join(os.getcwd(), nt_file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到本体文件: {file_path}")

        predicate_to_rel = {
            "http://cso.kmi.open.ac.uk/schema/cso#superTopicOf": "SUPER_TOPIC_OF",
            "http://cso.kmi.open.ac.uk/schema/cso#contributesTo": "CONTRIBUTES_TO",
            "http://cso.kmi.open.ac.uk/schema/cso#preferentialEquivalent": "PREFERENTIAL_EQUIVALENT",
            "http://cso.kmi.open.ac.uk/schema/cso#relatedEquivalent": "RELATED_EQUIVALENT",
        }

        node_upsert_query = """
        UNWIND $nodes AS node
        MERGE (c:Concept {uri: node.uri})
        SET c.name = node.name,
            c.normalized_name = node.normalized_name,
            c.updated_at = datetime()
        """

        relation_queries = {
            "SUPER_TOPIC_OF": """
            UNWIND $rows AS row
            MATCH (s:Concept {uri: row.source_uri})
            MATCH (t:Concept {uri: row.target_uri})
            MERGE (s)-[r:SUPER_TOPIC_OF]->(t)
            SET r.source = 'ontology'
            """,
            "CONTRIBUTES_TO": """
            UNWIND $rows AS row
            MATCH (s:Concept {uri: row.source_uri})
            MATCH (t:Concept {uri: row.target_uri})
            MERGE (s)-[r:CONTRIBUTES_TO]->(t)
            SET r.source = 'ontology'
            """,
            "PREFERENTIAL_EQUIVALENT": """
            UNWIND $rows AS row
            MATCH (s:Concept {uri: row.source_uri})
            MATCH (t:Concept {uri: row.target_uri})
            MERGE (s)-[r:PREFERENTIAL_EQUIVALENT]->(t)
            SET r.source = 'ontology'
            """,
            "RELATED_EQUIVALENT": """
            UNWIND $rows AS row
            MATCH (s:Concept {uri: row.source_uri})
            MATCH (t:Concept {uri: row.target_uri})
            MERGE (s)-[r:RELATED_EQUIVALENT]->(t)
            SET r.source = 'ontology'
            """,
        }

        def _flush_batches(session_obj, node_map: Dict[str, Dict[str, str]], rel_rows: List[Dict[str, str]]) -> Dict[str, int]:
            written_nodes = 0
            written_rels = 0

            if node_map:
                node_rows = list(node_map.values())
                session_obj.run(node_upsert_query, {"nodes": node_rows})
                written_nodes = len(node_rows)

            if rel_rows:
                grouped: Dict[str, List[Dict[str, str]]] = {
                    "SUPER_TOPIC_OF": [],
                    "CONTRIBUTES_TO": [],
                    "PREFERENTIAL_EQUIVALENT": [],
                    "RELATED_EQUIVALENT": [],
                }
                for rel in rel_rows:
                    grouped[rel["rel_type"]].append(rel)

                for rel_type, rows in grouped.items():
                    if not rows:
                        continue
                    session_obj.run(relation_queries[rel_type], {"rows": rows})
                    written_rels += len(rows)

            return {"nodes": written_nodes, "rels": written_rels}

        triples_scanned = 0
        triples_imported = 0
        concept_nodes_written = 0
        relations_written = 0
        per_batch_nodes: Dict[str, Dict[str, str]] = {}
        per_batch_rels: List[Dict[str, str]] = []
        batch_limit = max(200, int(batch_size or 2000))

        with self.driver.session() as session:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    triples_scanned += 1
                    parsed = _parse_nt_uri_triple(raw_line)
                    if parsed is None:
                        continue

                    subject_uri, predicate_uri, object_uri = parsed
                    rel_type = predicate_to_rel.get(predicate_uri)
                    if rel_type is None:
                        continue

                    subject_name = _label_from_topic_uri(subject_uri)
                    object_name = _label_from_topic_uri(object_uri)
                    per_batch_nodes[subject_uri] = {
                        "uri": subject_uri,
                        "name": subject_name,
                        "normalized_name": _normalize_concept_term(subject_name),
                    }
                    per_batch_nodes[object_uri] = {
                        "uri": object_uri,
                        "name": object_name,
                        "normalized_name": _normalize_concept_term(object_name),
                    }
                    per_batch_rels.append(
                        {
                            "source_uri": subject_uri,
                            "target_uri": object_uri,
                            "rel_type": rel_type,
                        }
                    )
                    triples_imported += 1

                    if len(per_batch_rels) >= batch_limit:
                        flushed = _flush_batches(session, per_batch_nodes, per_batch_rels)
                        concept_nodes_written += flushed["nodes"]
                        relations_written += flushed["rels"]
                        per_batch_nodes = {}
                        per_batch_rels = []

                    if max_triples and triples_imported >= int(max_triples):
                        break

            flushed = _flush_batches(session, per_batch_nodes, per_batch_rels)
            concept_nodes_written += flushed["nodes"]
            relations_written += flushed["rels"]

            concept_count_row = session.run("MATCH (c:Concept) RETURN count(c) AS c").single()
            concept_count = int(concept_count_row["c"] if concept_count_row else 0)

        link_result = {"linked": 0, "concepts": concept_count, "matched_candidates": 0}
        if link_existing_entities:
            link_result = self.link_entities_to_concept(entity_rows=None, only_unlinked=True)

        return {
            "file_path": file_path,
            "triples_scanned": triples_scanned,
            "triples_imported": triples_imported,
            "batch_size": batch_limit,
            "nodes_written_in_batches": concept_nodes_written,
            "relations_written_in_batches": relations_written,
            "concept_count": concept_count,
            "entity_linking": link_result,
        }

    def ensure_graph_schema(self) -> Dict[str, Any]:
        """创建图谱约束与索引（幂等）。"""
        self._ensure_initialized()

        schema_queries = [
            "CREATE CONSTRAINT paper_paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE CONSTRAINT concept_uri_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.uri IS UNIQUE",
            "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX concept_normalized_name_index IF NOT EXISTS FOR (c:Concept) ON (c.normalized_name)",
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
                    "SHOW VECTOR INDEXES YIELD name WHERE name = $name RETURN name",
                    {"name": self.settings.vector_index_name},
                )
            )

            exists = len(rows) > 0

            if exists and force_recreate:
                session.run(f"DROP INDEX {self.settings.vector_index_name} IF EXISTS")
                exists = False

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
            relations = row.get("relations") or []

            if paper_id is None or not chunk_id or not text:
                continue

            # 如果未提供实体且未提供关系，尝试从文本中自动提取三元组
            if not entities and not relations:
                extracted_triplets = _extract_triplets_from_text(text, self.settings.llm_model)
                relations = extracted_triplets
                
                # 从三元组中归纳出所有去重的实体节点
                entity_set = set()
                for trip in extracted_triplets:
                    entity_set.add(trip["source"])
                    entity_set.add(trip["target"])
                entities = [{"name": name, "type": "Concept"} for name in entity_set]

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
                    "relations": relations,
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
        
        FOREACH (rel IN coalesce(row.relations, []) |
            MERGE (e1:Entity {name: rel.source, type: 'Concept'})
            MERGE (e2:Entity {name: rel.target, type: 'Concept'})
            MERGE (e1)-[r:RELATED_TO {relation: rel.relation}]->(e2)
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

        concept_candidates: List[Dict[str, str]] = []
        for row in normalized_rows:
            for entity in row.get("entities", []):
                name = str(entity.get("name") or "").strip()
                etype = str(entity.get("type") or "Unknown").strip() or "Unknown"
                if name:
                    concept_candidates.append({"name": name, "type": etype})
            for rel in row.get("relations", []):
                source = str(rel.get("source") or "").strip()
                target = str(rel.get("target") or "").strip()
                if source:
                    concept_candidates.append({"name": source, "type": "Concept"})
                if target:
                    concept_candidates.append({"name": target, "type": "Concept"})

        link_result = self.link_entities_to_concept(entity_rows=concept_candidates, only_unlinked=False)

        return {
            "upserted": count,
            "aligned_concepts": link_result.get("linked", 0),
        }

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
                    
                    # 自动从切片提取三元组，并将其节点合并为实体标签
                    chunk_entities = paper_entities.copy()
                    chunk_relations = []
                    if auto_extract_entities:
                        try:
                            extracted_triplets = _extract_triplets_from_text(chunk_text, self.settings.llm_model)
                            chunk_relations = extracted_triplets
                            
                            # 将三元组中的所有节点也打平放入 entities 列表中
                            for t in extracted_triplets:
                                for en in [t["source"], t["target"]]:
                                    if not any(e["name"] == en for e in chunk_entities):
                                        chunk_entities.append({"name": en, "type": "Concept"})
                        except Exception as e:
                            print(f"[GraphRAG] ⚠ 自动三元组提取失败: {e}")
                    
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
                            "relations": chunk_relations,
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

    def chunk_and_store_from_mysql(
        self,
        paper_ids: Optional[List[int]] = None,
        limit: int = 100,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        auto_extract_entities: bool = True,
    ) -> Dict[str, Any]:
        """同步接口别名：分块后写入 Neo4j。"""
        return self.sync_from_mysql_knowledge_base(
            paper_ids=paper_ids,
            limit=limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            auto_extract_entities=auto_extract_entities,
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将外部传入的切片列表标准化并批量写入图数据库。"""
        self._ensure_embedding_ready()
        rows = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "").strip()
            text = str(chunk.get("text") or "").strip()
            paper_id = chunk.get("paper_id", -1)
            title = chunk.get("title", "Adhoc Paper")
            year = chunk.get("year")
            index = chunk.get("index", i)
            section_name = chunk.get("section_name", "Body")
            key_entities = chunk.get("key_entities") or []
            entities = chunk.get("entities") or []
            if chunk_id and text:
                rows.append(
                    {
                        "paper_id": int(paper_id),
                        "title": title,
                        "year": year,
                        "chunk_id": chunk_id,
                        "text": text,
                        "index": index,
                        "section_name": section_name,
                        "key_entities": key_entities,
                        "entities": entities,
                    }
                )
        self.ensure_vector_index(force_recreate=False)
        return self.upsert_paper_chunks(rows)

    def _expand_query_with_entities(self, query_text: str) -> str:
        """调用 LLM 提取 query 中的核心实体，并给出中英文名称及同义词，返回拼接后的扩展查询文本。

        若 LLM 调用失败或解析出错，降级为原始 query，保证检索链路不中断。
        """
        try:
            response = ask_messages(
                model=self.settings.llm_model,
                temperature=0.1,
                max_tokens=300,
                messages=[
                    {
                        "role": "system",
                        "content": GRAPHRAG_QUERY_ENTITY_EXPANSION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": GRAPHRAG_QUERY_ENTITY_EXPANSION_USER_PROMPT_TEMPLATE.format(query=query_text),
                    },
                ],
            )
            content = response.content.strip()
            # 清理可能的 markdown code block
            content = re.sub(r"^```[a-zA-Z]*\n?|```$", "", content, flags=re.MULTILINE).strip()
            expanded_terms = json.loads(content)
            if isinstance(expanded_terms, list) and expanded_terms:
                valid_terms = [str(t).strip() for t in expanded_terms if t and str(t).strip()]
                expanded = query_text + " " + " ".join(valid_terms)
                print(f"[GraphRAG] ✓ 查询实体扩展成功: {valid_terms}")
                return expanded
        except (LLMError, json.JSONDecodeError, Exception) as e:
            print(f"[GraphRAG] ⚠ 查询实体扩展失败，降级到原始查询: {e}")
        return query_text

    def similarity_search(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """基于向量索引执行相似度检索。检索前先通过 LLM 扩展 query 中的实体（中英文），再向量化检索。"""
        self._ensure_embedding_ready()

        # === 查询实体扩展：调用 LLM 提取关键实体（含中英文同义词），拼接为增强查询文本 ===
        expanded_query = self._expand_query_with_entities(query_text)
        query_embedding = self._embed_text(expanded_query)

        vector_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        OPTIONAL MATCH (p:Paper)-[:HAS_CHUNK]->(node)
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

        keyword_query = """
        MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)
        WHERE ($paper_ids IS NULL OR p.paper_id IN $paper_ids)
          AND (
            size($keywords) = 0
            OR any(k IN $keywords WHERE toLower(coalesce(c.text, "")) CONTAINS k OR toLower(coalesce(p.title, "")) CONTAINS k)
          )
        WITH p, c,
             reduce(s = 0.0, k IN $keywords |
                s + CASE
                    WHEN toLower(coalesce(c.text, "")) CONTAINS k THEN 1.0
                    WHEN toLower(coalesce(p.title, "")) CONTAINS k THEN 0.6
                    ELSE 0.0
                END
             ) AS lexical_score
        RETURN
          c.chunk_id AS chunk_id,
          c.text AS text,
          c.index AS chunk_index,
          lexical_score AS score,
          p.paper_id AS paper_id,
          p.title AS title,
          p.year AS year
        ORDER BY lexical_score DESC, p.year DESC
        LIMIT $limit
        """

        def _normalize_keywords(source: str) -> List[str]:
            tokens = re.findall(r"[A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", (source or "").lower())
            # 保序去重，避免关键词列表过长导致查询噪声。
            ordered_unique: List[str] = []
            for token in tokens:
                if token not in ordered_unique:
                    ordered_unique.append(token)
            return ordered_unique[:10]

        def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            best_by_chunk: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                chunk_id = row.get("chunk_id")
                if not chunk_id:
                    continue
                current = best_by_chunk.get(chunk_id)
                current_score = float(current.get("score") or 0.0) if current else float("-inf")
                new_score = float(row.get("score") or 0.0)
                if current is None or new_score > current_score:
                    best_by_chunk[chunk_id] = row
            ordered = list(best_by_chunk.values())
            ordered.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
            return ordered

        keywords = _normalize_keywords(expanded_query)
        search_mode = "vector"
        all_rows: List[Dict[str, Any]] = []

        # 修复：补充之前未定义的检索规模控制变量
        effective_top_k = max(top_k, 3)
        vector_fetch_k = min(effective_top_k * 2, 50)
        expanded_fetch_k = min(effective_top_k * 3, 100)
        normalized_paper_ids = None  # 当前接口不支持按论文过滤，设为 None

        with self.driver.session() as session:
            # 第一阶段：向量检索（保留用户给定 paper_ids 过滤）
            base_rows = list(
                session.run(
                    vector_query,
                    {
                        "index_name": self.settings.vector_index_name,
                        "top_k": vector_fetch_k,
                        "embedding": query_embedding,
                    },
                )
            )

            all_rows.extend(base_rows)

            # 第二阶段：若过滤后召回过少，放宽到全库补齐。
            if len(base_rows) < effective_top_k and normalized_paper_ids is not None:
                relaxed_vector_rows = list(
                    session.run(
                        vector_query,
                        {
                            "index_name": self.settings.vector_index_name,
                            "top_k": expanded_fetch_k,
                            "embedding": query_embedding,
                            "paper_ids": None,
                        },
                    )
                )
                if relaxed_vector_rows:
                    search_mode = "vector_relaxed_paper_filter"
                    all_rows.extend(relaxed_vector_rows)

            merged_rows = _dedupe_rows(all_rows)

            # 第三阶段：向量仍不足时，使用关键词回退检索。
            if len(merged_rows) < max(3, effective_top_k // 2):
                lexical_rows = list(
                    session.run(
                        keyword_query,
                        {
                            "paper_ids": normalized_paper_ids,
                            "keywords": keywords,
                            "limit": expanded_fetch_k,
                        },
                    )
                )
                if lexical_rows:
                    search_mode = "keyword_fallback"
                    merged_rows = _dedupe_rows(merged_rows + lexical_rows)

            if len(merged_rows) < max(3, effective_top_k // 2) and normalized_paper_ids is not None:
                relaxed_lexical_rows = list(
                    session.run(
                        keyword_query,
                        {
                            "paper_ids": None,
                            "keywords": keywords,
                            "limit": expanded_fetch_k,
                        },
                    )
                )
                if relaxed_lexical_rows:
                    search_mode = "keyword_fallback_relaxed_paper_filter"
                    merged_rows = _dedupe_rows(merged_rows + relaxed_lexical_rows)

        rows = merged_rows[:effective_top_k]

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
            "expanded_query": expanded_query,
            "top_k": top_k,
            "search_mode": search_mode,
            "results": response,
        }

    def rag_search(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """检索相关 chunks 后调用本地 LLM 生成回答。"""
        self._ensure_ai_ready()

        # 1) 向量检索获取上下文
        search_result = self.similarity_search(query_text, top_k=top_k)
        chunks = search_result.get("results", [])
        if not chunks:
            return {
                "query_text": query_text,
                "top_k": top_k,
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
                {"role": "system", "content": "你是一个学术论文助手。根据提供的论文片段回答用户问题，回答要准确、有依据。如果片段中没有相关信息，请如实说明。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query_text}"},
            ],
        )
        answer = response.content.strip()

        return {
            "query_text": query_text,
            "top_k": top_k,
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