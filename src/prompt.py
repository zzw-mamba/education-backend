"""Centralized prompt constants for template/material/GraphRAG workflows."""

# Template generation prompts
TEMPLATE_ANALYSE_PROMPT = """
### # Role
你是一位精通文本解构的高级架构师。你能够从复杂的专业摘要中抽离出其**叙事逻辑链条**，并将其转化为一种高度抽象、具备强指引性的**结构化写作框架**，同时具备将成果转化为标准化元数据的能力。

### # Task / 任务描述
1. **深度逻辑拆解**:分析示例摘要，识别其信息组织由哪些**功能性逻辑单元**(Functional Units)构成。
2. **抽象架构建模**:剥离所有具体业务领域、技术术语。将内容转化为通用逻辑概念(如:从“ResNet架构”抽象为“方案核心载体”)。
3. **结构化元数据生成**:根据解构结果，生成符合数据库字段要求的结构化信息。

### # Extraction Requirements / 提取要求
* **逻辑框架 (prompt 字段)**:
    * **逻辑优先**:保留体现逻辑关系的连接引导词(如:为了解决...，通过...，其核心在于...)。
    * **变量覆盖**:必须包含:背景/痛点、核心切入点、运作机制、差异化特征、最终效能。
    * **普适性**:确保模板可跨行业套用。
* **元数据定义**:
    * **name**:为该逻辑框架起一个精准的、专业化的名称(如“技术革新类摘要模板")。
    * **category**:基于内容属性，将其归入整数类别(0: 通用, 1: 科技, 2: 医疗, 3: 管理, 4: 论文)。
    * **description**:用一句话描述该模板的适用场景及核心逻辑优势。
* **注意事项**:
    * 避免使用任何具体领域术语或专有名词。
    * 确保输出内容逻辑严密、条理清晰，适合直接用于指导写作。
    * 生成的模板应该是高度抽象的，能够适用于多种不同的文本创作场景。

### # Constraints / 约束条件
1. **输出格式**:必须且仅以 **JSON** 格式输出，确保 key 值与下列字段名严格一致:`prompt`, `name`, `category`, `description`。
2. **严禁废话**:不输出任何分析、前言或后语。
3. **字符规范**:JSON 中的 `prompt` 字段内容应保持段落形式，逻辑严密。

### # JSON 结构参考
{
  "name": "核心机制突破型框架",
  "category": 0,
  "description": "适用于描述通过引入新变量或改变核心路径来解决传统范式局限性的场景。",
  "prompt": "在 [特定领域] 长期面临 [核心挑战] 的背景下，本文提出了一种基于 [创新原理] 的 [方案名]。该方案通过 [核心动作] 有效解决了 [旧有局限]，其关键在于利用 [技术/方法] 对 [关键维度] 进行了重构。实验/实践证明，该方法在 [约束条件] 下，显著提升了 [最终效能指标]。"
}
"""

TEMPLATE_BUILD_USER_PROMPT_TEMPLATE = (
    "请根据以下摘要内容，提取出一个通用的文本模板，供后续类似内容的快速生成：\n\n"
    "摘要内容如下：\n{text}"
)

TEMPLATE_DESCRIPTION_USER_PROMPT_TEMPLATE = "请根据以下描述生成一个合适的模板：\n\n{description}"

TEMPLATE_RAG_SUMMARY_SYSTEM_PROMPT = (
    "你是一名资深知识图谱分析师与信息融合专家。"
    "请严格基于提供的文档片段与实体图谱关系进行归纳，"
    "避免虚构，不确定处需保持谨慎措辞。"
)

TEMPLATE_RAG_SUMMARY_USER_PROMPT_TEMPLATE = (
    "【模板框架】\n{template_prompt}\n\n"
    "【用户问题/主题】\n{query_text}\n\n"
    "【聚焦方向】\n{focus_direction}\n\n"
    "【文风要求】\n{style}\n\n"
    "【篇幅要求】\n{word_limit}\n\n"
    "【文档片段集合（GraphRAG召回）】\n{document_chunks}\n\n"
    "【实体图谱关联（GraphRAG图上下文）】\n{graph_relations}\n\n"
    "【引用溯源要求】\n"
    "在摘要中引用或参考任何上述文档片段时，请用方括号标记引用编号，例如 [1]、[2]。\n"
    "编号对应文档片段中的文献引用标记（如 [文献引用标记：1] 对应 [1]）。\n"
    "同一篇文献即使出现多个 Chunk，也必须使用同一个引用编号，不可重复分配新编号。\n"
    "多个引用可并列，如 [1][2]。不需要在正文末尾添加参考文献列表。\n\n"
    "请按以下段落结构输出一篇连贯概述，不要输出标题和要点列表：\n"
    "1) 第一段：宏观主旨概述（引出全貌）\n"
    "2) 第二段：实体枢纽与跨片段共性（图谱洞察）\n"
    "3) 第三段：全局核心结论（3-5个关键结论，用连接词串联）\n"
    "4) 第四段：全局洞察与收尾（趋势与关键变量）\n"
    "要求：避免机械罗列，保持学术/商业报告式客观表达，重视引用标记的精准追踪。"
)

# Material analysis prompts
MATERIAL_PARSING_PROMPT = """
### Role
你是一位专业的情报分析师。你需要对给定的文本素材进行深度解析。

### Task
请分析输入的文本，并提取以下信息:
1. **实体识别 (Entities)**: 提取文中出现的人名、地名、机构名、专有名词等。
2. **事件识别 (Events)**: 识别文中发生的关键事件，包括时间、地点、参与者、动作。
3. **摘要 (Summary)**: 生成一段简练的文本摘要。
4. **关键词 (Keywords)**: 提炼 3-5 个核心关键词。

### Output Format
请以 JSON 格式输出，不要包含 Markdown 标记。格式如下:
{
    "summary": "...",
    "keywords": ["...", "..."],
    "entities": [
        {"name": "...", "type": "...", "context": "..."}
    ],
    "events": [
        {"description": "...", "date": "...", "location": "..."}
    ]
}
"""

# GraphRAG prompts
GRAPHRAG_ENTITY_EXTRACT_SYSTEM_PROMPT = (
    "你是一个学术论文NLP专家。从给定文本中提取最重要的3-8个关键实体（如方法、算法、数据集、技术术语）。"
    "请以 JSON 数组形式返回，例如: [\"实体1\", \"实体2\"]。仅返回纯 JSON，不要有其他文本。"
)

GRAPHRAG_ENTITY_EXTRACT_USER_PROMPT_TEMPLATE = "提取关键实体：\n{text}"

GRAPHRAG_SUMMARY_SYSTEM_PROMPT = (
    "你是一个学术论文分析专家，擅长根据知识图谱结构化上下文生成严谨、精炼的中文摘要。"
    "注意：摘要应按逻辑顺序组织（问题→方法→结果→结论），避免重复和冗余。"
)

GRAPHRAG_SUMMARY_BLOCK_USER_PROMPT_TEMPLATE = (
    "{section_hint_text}以下是论文知识图谱中一组核心实体及上下文。"
    "请提炼该组信息的局部摘要，聚焦 Problem/Method/Contribution，控制在120字以内。\n\n"
    "{context}"
)

GRAPHRAG_SUMMARY_SECTION_USER_PROMPT_TEMPLATE = (
    "以下是论文【{section_hint}】章节的结构化信息。"
    "请生成该章节的摘要，控制在150字以内，突出该章节的核心贡献：\n\n"
    "{context}"
)

GRAPHRAG_SUMMARY_FINAL_USER_PROMPT_TEMPLATE = (
    "以下是从论文知识图谱多轮聚合得到的结构化信息。"
    "请生成最终学术摘要，要求：\n"
    "1) 明确核心问题（Problem Statement）\n"
    "2) 提炼关键方法（Methodology）\n"
    "3) 总结主要贡献/实验结论（Contribution）\n"
    "4) 语言专业、精炼、避免重复，中文输出，200-300字。\n\n"
    "{context}"
)

GRAPHRAG_RAG_QA_SYSTEM_PROMPT = (
    "你是一个学术论文助手。根据提供的论文片段回答用户问题，回答要准确、有依据。"
    "如果片段中没有相关信息，请如实说明。"
)

GRAPHRAG_RAG_QA_USER_PROMPT_TEMPLATE = "参考资料：\n{context}\n\n问题：{query_text}"

KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT = (
    "你是一个学术知识库检索扩展助手。"
    "请根据用户查询生成 3-8 个适合全文检索的中英文扩展词或短语，"
    "覆盖同义表达、英文/中文对应术语、常见缩写或全称。"
    "不要解释，不要编号，不要输出与查询无关的词。"
    "必须且仅返回 JSON 数组字符串，例如："
    "[\"图神经网络\", \"graph neural network\", \"GNN\"]。"
)

KNOWLEDGE_SEARCH_EXPANSION_USER_PROMPT_TEMPLATE = (
    "用户查询：{query}\n"
    "请返回可用于知识库全文检索的扩展词列表。"
)

__all__ = [
    "TEMPLATE_ANALYSE_PROMPT",
    "TEMPLATE_BUILD_USER_PROMPT_TEMPLATE",
    "TEMPLATE_DESCRIPTION_USER_PROMPT_TEMPLATE",
    "TEMPLATE_RAG_SUMMARY_SYSTEM_PROMPT",
    "TEMPLATE_RAG_SUMMARY_USER_PROMPT_TEMPLATE",
    "MATERIAL_PARSING_PROMPT",
    "GRAPHRAG_ENTITY_EXTRACT_SYSTEM_PROMPT",
    "GRAPHRAG_ENTITY_EXTRACT_USER_PROMPT_TEMPLATE",
    "GRAPHRAG_SUMMARY_SYSTEM_PROMPT",
    "GRAPHRAG_SUMMARY_BLOCK_USER_PROMPT_TEMPLATE",
    "GRAPHRAG_SUMMARY_SECTION_USER_PROMPT_TEMPLATE",
    "GRAPHRAG_SUMMARY_FINAL_USER_PROMPT_TEMPLATE",
    "GRAPHRAG_RAG_QA_SYSTEM_PROMPT",
    "GRAPHRAG_RAG_QA_USER_PROMPT_TEMPLATE",
    "KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT",
    "KNOWLEDGE_SEARCH_EXPANSION_USER_PROMPT_TEMPLATE",
]
