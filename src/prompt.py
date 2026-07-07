"""提示词常量模块

集中管理模板生成和素材分析流程中使用的提示词常量。

包含的主要提示词：
- TEMPLATE_ANALYSE_PROMPT: 模板分析提示词
- TEMPLATE_BUILD_USER_PROMPT_TEMPLATE: 模板构建用户提示词模板
- TEMPLATE_DESCRIPTION_USER_PROMPT_TEMPLATE: 模板描述用户提示词模板
- TEMPLATE_RAG_SUMMARY_SYSTEM_PROMPT: 模板RAG摘要系统提示词
- TEMPLATE_RAG_SUMMARY_USER_PROMPT_TEMPLATE: 模板RAG摘要用户提示词模板
- MATERIAL_PARSING_PROMPT: 素材分析提示词
- KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT: 知识检索扩展系统提示词
- KNOWLEDGE_SEARCH_EXPANSION_USER_PROMPT_TEMPLATE: 知识检索扩展用户提示词模板
"""

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
    """
    # Role
    你是一个学术综述与信息合成专家，擅长从异构的图谱信息中提取核心精华，生成逻辑高度致密的学术摘要。

    # Inputs
    - **核心主题**：{query_text}
    - **聚焦方向**：{focus_direction}
    - **篇幅限制**：{word_limit}
    - **数据源（片段）**：{document_chunks}
    - **数据源（关系）**：{graph_relations}

    # Task
    请基于上述召回的片段与图谱关系，合成一篇逻辑连贯的深度摘要。摘要需体现【聚焦方向】中的核心关切，并揭示不同文档片段间的共性与关联。

    # Citation Requirements (STRICT)
    - **唯一溯源标准**：引用任何观点或事实时，必须严格使用文档片段中标记的 `[文献引用标记：X]`。
    - **标注格式**：直接在引用处标注数字编号，如 `[X]`（例如：[1]）。
    - **去重逻辑**：同一来源的文献在全文中必须统一使用同一个编号，不可分配新编号。
    - **并列标注**：若多个来源支撑同一观点，请并列标注，如 `[1][2]`。

    # Synthesis Guidelines
    1. **去结构化叙述**：严禁输出标题、子标题或任何形式的要点列表（Bullet points）。通过逻辑连接词实现段落平滑过渡，确保摘要整体的连贯性。
    2. **深度信息融合**：避免简单的内容罗列。需结合【数据源（关系）】揭示实体间的联动效应，将离散的片段缝合成完整的语义整体。
    3. **精准追踪**：确保每一个关键断言都有对应的 `[文献引用标记：X]` 支持。

    # Output Format
    直接输出摘要正文。严禁任何开场白（如 "根据文档..."）、严禁解释性文字、严禁输出参考文献列表。
    """
)

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

KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT = (
    """
    # Role
    你是一个学术文献检索增强助手，擅长通过语义扩展提升论文查全率。

    # Task
    基于用户查询，生成 8-15 个高质量中英文检索词。扩展逻辑必须包含：
    1. **标准对译**：中英文学术全称。
    2. **术语变体**：常用缩写、同义表达、异名词。
    3. **关联扩展**：该领域的上位核心概念、代表性子领域、或核心算法名称。

    # Constraints
    - **格式**：必须且仅返回一个标准的 JSON 数组字符串（不带 Markdown 代码块标签）。
    - **质量**：词汇必须具有学术严谨性，中英文词汇需成对或成组对应。
    - **纯净度**：无解释、无编号、无前缀/后缀。

    # Example
    Input: 迁移学习
    Output: ["迁移学习", "Transfer Learning", "领域自适应", "Domain Adaptation", "预训练模型", "Pre-trained models", "知识迁移", "Knowledge Transfer", "Fine-tuning", "微调", "Inductive Transfer", "Multi-task learning", "多任务学习"]
    """
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
    "KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT",
    "KNOWLEDGE_SEARCH_EXPANSION_USER_PROMPT_TEMPLATE",
]