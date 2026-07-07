"""FastAPI 应用入口模块

ResearchGraph-RAG 后端服务的主入口，负责：
1. 应用生命周期管理（启动/关闭）
2. 路由注册
3. CORS 配置
4. 静态文件服务

主要组件：
- lifespan: 应用生命周期管理器
- app: FastAPI 应用实例
- /health: 健康检查端点
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import models
from database import init_db
import os
from routers import ocr, db_routes, user, template, parsing, log
from graphrag_module import graphrag_shutdown, graphrag_startup, router as graphrag_router


def _to_bool(value: str | None) -> bool:
    """将字符串值转换为布尔值。
    
    支持的真值: "1", "true", "yes", "on"（不区分大小写）
    
    Args:
        value: 待转换的字符串值
        
    Returns:
        bool: 转换后的布尔值
    """
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理器。
    
    在应用启动时执行初始化操作，在应用关闭时执行清理操作。
    
    启动时：
    1. 初始化数据库表
    2. 启动 GraphRAG 模块
    
    关闭时：
    1. 关闭 GraphRAG 模块
    
    Args:
        app: FastAPI 应用实例
        
    Yields:
        None
    """
    init_db()
    graphrag_startup(auto_setup_local_neo4j=_to_bool(os.getenv("AUTO_SETUP_LOCAL_NEO4J")))
    yield
    graphrag_shutdown()


app = FastAPI(title="ResearchGraph-RAG Backend", version="0.1.0", lifespan=lifespan)

analysis_results_dir = Path(__file__).resolve().parents[1] / "analysis_results"
analysis_results_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/analysis_results",
    StaticFiles(directory=str(analysis_results_dir)),
    name="analysis_results",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router)
app.include_router(db_routes.router)
app.include_router(user.router)
app.include_router(template.router)
app.include_router(parsing.router)
app.include_router(graphrag_router)
app.include_router(log.router)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """健康检查端点。
    
    返回应用的运行状态，用于监控和负载均衡探测。
    
    Returns:
        dict: 包含状态信息的字典，如 {"status": "ok"}
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=_to_bool(os.getenv("APP_RELOAD", "true")),
    )