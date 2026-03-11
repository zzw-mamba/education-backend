from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import models
from database import engine, init_db
import os
from routers import ocr, db_routes, user, template, parsing, graphrag_routes
import subprocess
import asyncio
from graphrag.graphrag_service import get_graphrag_service

LLM_IP = os.getenv("LLM_IP")
LOCAL_EMBEDDING_IP = os.getenv("LOCAL_EMBEDDING_IP")

SSH_COMMAND = [
    "ssh", "-N", 
    "-p", "23686", 
    "-o", "ServerAliveInterval=60", 
    "-o", "StrictHostKeyChecking=no",
    "-L", "8080:" + LLM_IP + ":8000",
    "-L", "9090:" + LOCAL_EMBEDDING_IP + ":8000",
    "root@cci-proxy.cn-sh-01.sensecore.cn"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    ssh_process = subprocess.Popen(SSH_COMMAND)
    init_db()
    auto_setup_local_neo4j = os.getenv("AUTO_SETUP_LOCAL_NEO4J", "false").lower() == "true"
    if auto_setup_local_neo4j:
        try:
            graphrag_service = get_graphrag_service()
            graphrag_service.setup_local_database(create_vector_index=True, force_recreate_index=False)
            print("[GraphRAG] ✓ 本地 Neo4j schema/index 初始化完成")
        except Exception as e:
            print(f"[GraphRAG] ✗ 本地 Neo4j 自动初始化失败: {e}")
    await asyncio.sleep(2) 
    yield

    if ssh_process.poll() is None:  # 检查SSH进程是否还在运行
        ssh_process.terminate()     # 终止进程
        ssh_process.wait()          # 等待进程退出
        print(f"[SSH] 进程（PID: {ssh_process.pid}）已终止")



app = FastAPI(title="Backend Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由模块
app.include_router(ocr.router)
app.include_router(db_routes.router)
app.include_router(user.router)
app.include_router(template.router)
app.include_router(parsing.router)
app.include_router(graphrag_routes.router)

class Item(BaseModel):
    name: str
    description: str | None = None


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Lightweight health probe."""
    return {"status": "ok"}


@app.get("/", status_code=status.HTTP_200_OK)
async def read_root() -> dict[str, str]:
    return {"message": "FastAPI is running"}


@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item) -> Item:
    return item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
