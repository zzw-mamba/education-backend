# Backend 项目说明

本项目是一个基于 FastAPI 的后端服务，核心能力包括：

- 用户认证与模板管理
- 知识库（MySQL）入库、检索与推荐
- OCR / 文档解析（UniParse）
- 基于 Neo4j 的知识图谱能力
- 基于 Neo4j + 向量索引 + 本地 LLM 的 GraphRAG 检索增强问答与摘要

## 1. 项目结构

```text
backend/
├─ package-list.txt
├─ database/                # 论文/文档文件目录（如 PDF）
├─ bibs/                    # BibTeX 元数据目录
├─ analysis_results/        # 批处理分析结果输出目录
├─ ocr_outputs/             # OCR 输出样例目录
├─ uploaded_files/          # 上传文件目录
├─ temp_uploads/            # OCR 临时文件目录
├─ src/
│  ├─ main.py
│  ├─ database.py
│  ├─ models.py
│  ├─ prompt.py
│  ├─ sync_data.py
│  ├─ routers/
│  │  ├─ ocr.py
│  │  ├─ db_routes.py
│  │  ├─ user.py
│  │  ├─ template.py
│  │  ├─ parsing.py
│  │  ├─ graphrag_routes.py
│  │  └─ neo4j_routes.py
│  ├─ graphrag/
│  │  ├─ graphrag_config.py
│  │  └─ graphrag_service.py
│  ├─ neo4j/
│  │  ├─ neo4j_config.py
│  │  └─ neo4j_utils.py
│  └─ utils/
│     ├─ model.py
│     └─ get_resources_content.py
└─ .env                     # 本地环境变量（请勿提交敏感信息）
```

## 2. 运行架构与数据流

### 2.1 总体架构

- API 框架：FastAPI（`src/main.py`）
- 关系数据库：MySQL（SQLAlchemy ORM）
- 图数据库：Neo4j（通用图 + GraphRAG 图）
- 向量检索：Neo4j Vector Index
- 大模型/Embedding：通过 HTTP 接口调用本地或远端服务

### 2.2 关键链路

1. 文献导入：`bibs/*.bib + database/*.pdf` -> `src/sync_data.py` -> MySQL `knowledge_base`
2. GraphRAG 构建：MySQL `knowledge_base` -> 语义切片/实体抽取 -> Neo4j `Paper/Chunk/Entity`
3. 问答检索：用户 query -> embedding -> Neo4j 向量召回 -> LLM 生成答案
4. 摘要生成：图上下文精炼 -> 分层摘要（章节感知/递归聚合）

## 3. 各模块与代码文件作用

### 3.1 根目录文件

- `package-list.txt`
  - Conda 导出的环境清单（Windows 平台），可用于复现实验环境。

### 3.2 `src/` 顶层

- `src/main.py`
  - FastAPI 应用入口。
  - 注册路由：OCR、数据库、用户、模板、解析、GraphRAG。
  - 应用生命周期中执行：
    - 启动 SSH 隧道（依赖 `LLM_IP`、`LOCAL_EMBEDDING_IP`）。
    - 初始化 MySQL 表。
    - 可选自动初始化本地 Neo4j GraphRAG schema/index（`AUTO_SETUP_LOCAL_NEO4J=true`）。
- `src/database.py`
  - SQLAlchemy 引擎/会话工厂初始化。
  - `get_db()` 提供 FastAPI 依赖注入。
  - `init_db()` 统一创建 ORM 表结构。
- `src/models.py`
  - ORM 模型定义：`User`、`Log`、`Template`、`KnowledgeBase`、`Tag`、`KBTagRelation`。
  - `KnowledgeBase` 内置 MySQL 全文索引（`MATCH AGAINST`）用于检索。
  - `KBService` 提供入库、全文检索、标签推荐能力。
- `src/prompt.py`
  - 统一维护提示词模板：模板抽象分析、素材信息解析。
- `src/sync_data.py`
  - 批量同步 `bibs + pdf` 到 MySQL。
  - 自动解析标题/作者/年份、提取 PDF 文本、生成关键词标签。

### 3.3 `src/routers/` API 路由层

- `src/routers/user.py`
  - 用户注册/登录/更新/改密/当前用户信息。
  - 基于 JWT + bcrypt。
- `src/routers/template.py`
  - 根据摘要调用 LLM 生成写作模板。
  - 保存模板到 MySQL（需鉴权）。
- `src/routers/db_routes.py`
  - 知识库入库、检索、推荐、附件下载接口。
  - 检索包含基于统一 LLM 接口的中英文查询扩展逻辑。
- `src/routers/ocr.py`
  - 上传 PDF/图片到 UniParse，返回 Markdown 解析结果。
- `src/routers/parsing.py`
  - 按知识库 ID 批量拉取内容并调用 LLM 做结构化解析。
  - 结果写入 `analysis_results/`，同时记录日志。
- `src/routers/graphrag_routes.py`
  - GraphRAG 专用接口：健康检查、诊断、schema/index 初始化、从 MySQL 同步、相似检索、RAG 搜索、论文摘要。
- `src/routers/neo4j_routes.py`
  - 通用 Neo4j 图能力 API（构图、图分析、最短路径、用户搜索行为图谱）。

### 3.4 `src/graphrag/` GraphRAG 核心

- `src/graphrag/graphrag_config.py`
  - GraphRAG 配置数据类（Neo4j、向量索引、Embedding、LLM）。
  - 从环境变量读取并校验。
- `src/graphrag/graphrag_service.py`
  - GraphRAG 核心服务实现：
    - 章节感知 + 语义窗口切片
    - 实体抽取（LLM + 启发式回退）
    - Neo4j schema 与向量索引创建
    - MySQL -> Neo4j 同步
    - 向量相似检索、RAG 问答
    - 图上下文精炼与论文摘要生成

### 3.5 `src/neo4j/` 通用图工具

- `src/neo4j/neo4j_config.py`
  - Neo4j 驱动初始化与基础增删改查封装。
- `src/neo4j/neo4j_utils.py`
  - 三类图工具：
    - `KnowledgeGraph`：知识图谱构建与标签查询
    - `GraphAnalysis`：中心性、连通性、路径、统计
    - `UserBehaviorGraph`：搜索行为记录与分析

### 3.6 `src/utils/` 工具层

- `src/utils/model.py`
  - 通用 LLM 客户端（OpenAI-compatible），封装 chat/completions 调用。
- `src/utils/get_resources_content.py`
  - 长文本分块、并发调用 LLM、合并结果并持久化到 JSON。

## 4. 环境准备与安装

## 4.1 前置要求

- Python：建议 `3.12`
- Conda（推荐）或 venv
- MySQL 8.0（本机安装并启动）
- Neo4j Desktop（本机安装并启动）

### 4.2 方式 A：使用 Conda + package-list.txt（推荐复现实验环境）

```bash
conda create -n graduation_backend --file package-list.txt
conda activate graduation_backend
```

说明：`package-list.txt` 体积较大，若创建失败可改用方式 B。

### 4.3 方式 B：最小依赖安装（快速启动）

```bash
conda create -n graduation_backend python=3.12 -y
conda activate graduation_backend

pip install fastapi uvicorn sqlalchemy pymysql python-dotenv python-multipart
pip install passlib[bcrypt] python-jose email-validator
pip install jieba bibtexparser pymupdf requests deep-translator nltk
pip install neo4j neo4j-graphrag
```


## 5. 配置 `.env`

请在项目根目录创建或维护 `.env`，示例：

```env
# MySQL
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/graduation_project

# UniParse
PARSER_API=your_uniparse_token

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# GraphRAG / Vector
GRAPHRAG_VECTOR_INDEX_NAME=chunk_vector_index
GRAPHRAG_EMBEDDING_MODEL=Qwen3-Embedding-8B
GRAPHRAG_EMBEDDING_DIMENSIONS=4096
LOCAL_EMBEDDING_BASE_URL=http://localhost:9090
LOCAL_EMBEDDING_API_PATH=/v1/embeddings

# LLM
LLM_API_BASE=http://127.0.0.1:8000/v1
LLM_MODEL=your_llm_model
LLM_API_KEY=

# main.py 中 SSH 隧道相关
LLM_IP=127.0.0.1
LOCAL_EMBEDDING_IP=127.0.0.1
AUTO_SETUP_LOCAL_NEO4J=false
```

## 6. 安装 Neo4j Desktop（重点）

### 6.1 下载与安装

1. 访问 Neo4j 官方下载页，下载安装 `Neo4j Desktop`（Windows）。
2. 安装完成后打开 Neo4j Desktop，登录或跳过登录进入本地管理界面。

### 6.2 创建本地数据库

1. 点击 `New` -> `Create Local DBMS`。
2. 设置数据库名称（例如 `graduation-neo4j`）。
3. 设置用户名和密码（默认用户一般是 `neo4j`）。
4. 记录 Bolt 连接地址（通常 `bolt://localhost:7687`）。
5. 点击 `Start`，确保数据库状态是绿色运行中。

### 6.3 安装 APOC 插件

1. 进入该 DBMS 的 `Plugins`。
2. 搜索并安装 `APOC`。
3. 重启数据库使插件生效。

### 6.4 与本项目对接

1. 将 `.env` 中 `NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD` 配置为 Desktop 实际值。
2. 启动后端后调用：`POST /api/graphrag/setup-local-db`。
3. 可先访问：`GET /api/graphrag/diagnose` 检查连接、APOC、索引状态。

### 6.5 常见问题排查

- 连接失败：确认 Neo4j Desktop 数据库已启动，neo4j 端口正确。
- 认证失败：检查 `.env` 密码与 Desktop 设置一致。
- APOC 不可用：确认插件已安装且重启 DBMS。
- 向量检索报错：确认已执行 `setup-local-db` 或 `create-index`。

## 7. 启动方式

### 7.1 启动 MySQL 与 Neo4j

1. 启动本机 MySQL 服务，确认账号密码可用（需与 `.env` 的 `DATABASE_URL` 一致）。
2. 打开 Neo4j Desktop，启动目标 DBMS（绿色状态），确认 Bolt 地址可连接。

### 7.2 启动后端

```bash
cd src
python main.py
```

服务启动后可访问：

- 健康检查：`GET /health`
- 文档：`http://127.0.0.1:8000/docs`

## 8. 常用接口速览

- 用户认证：`/auth/register`、`/auth/login`、`/auth/me`
- OCR：`POST /ocr`
- 知识库：`/knowledge/add`、`/knowledge/search`、`/knowledge/recommend`
- 模板：`POST /template/build`、`PUT /template/add`
- 素材分析：`POST /material/parse`
- GraphRAG：
  - `GET /api/graphrag/health`
  - `GET /api/graphrag/diagnose`
  - `POST /api/graphrag/setup-local-db`
  - `POST /api/graphrag/sync-from-mysql`
  - `POST /api/graphrag/similarity-search`
  - `POST /api/graphrag/search`
  - `POST /api/graphrag/paper-summary`

## 9. 开发建议

- `src/main.py` 里包含 SSH 隧道启动逻辑；本地无该环境时，建议先注释或通过环境变量控制。
- `src/neo4j/neo4j_utils.py` 目前使用 `from neo4j_config import ...`，运行路径变化时可能触发导入问题，建议改为包内绝对导入（如 `from neo4j.neo4j_config import ...`）。
- `.env` 和任何包含密钥、密码的配置文件不要提交到 Git。

