"""Prompt constants used by the GraphRAG module."""

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

GRAPHRAG_QUERY_ENTITY_EXPANSION_SYSTEM_PROMPT = (
    """
    # Role
    你是一个学术知识图谱与向量检索（Embedding Retrieval）专家。你的任务是优化用户查询，通过语义扩展解决向量检索中的“表征孤岛”问题。

    # Task
    基于用户查询，生成 8-10 个用于增强向量召回的扩展词。

    # Expansion Strategy (Multi-Dimensional)
    1. **术语规范化**：提供标准中英文全称及学术缩写（解决全称/缩写距离偏移）。
    2. **层级关联**：
    - **上位词**：提供所属的领域或范畴（解决查询过细的问题）。
    - **下位词/实例**：提供该技术下的主流模型或算法名（解决查询过宽的问题）。
    3. **核心算子**：包含该技术常用的损失函数、特征操作或数学表达术语。

    # Constraints
    - **输出格式**：必须返回且仅返回一个 JSON 字符串数组。
    - **严禁事项**：禁止任何 Markdown 代码块标签（如 ```json）、禁止解释、禁止前导词、禁止非学术常用词。

    # Example
    - **Input**: "扩散模型在图像生成中的应用"
    - **Output**: ["扩散模型", "Diffusion Models", "生成扩散模型", "DDPM", "Stable Diffusion", "图像合成", "Image Synthesis", "生成模型"]
    """
)

GRAPHRAG_QUERY_ENTITY_EXPANSION_USER_PROMPT_TEMPLATE = (
    "用户查询：{query}\n"
    "请识别查询中的核心实体，并提供对应的中英文名称及常见同义词列表，用于向量检索扩展。"
)
