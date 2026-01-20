from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import models
from database import engine
import os
import torch
from routers import ocr, db_routes, user, template, parsing
import subprocess
import asyncio

SSH_COMMAND = [
    "ssh", "-N", 
    "-p", "23686", 
    "-o", "ServerAliveInterval=60", 
    "-o", "StrictHostKeyChecking=no",
    "-L", "8080:10.119.19.154:8000", 
    "root@cci-proxy.cn-sh-01.sensecore.cn"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    os.environ["INFERENCE_DEVICE_TYPE"] = "cuda"
    os.environ["DATAPYPES"] = "fp16"
    from marker.models import create_model_dict
    app.state.marker = create_model_dict()

    ssh_process = subprocess.Popen(SSH_COMMAND)

    await asyncio.sleep(2) 
    yield

    del app.state.marker
    torch.cuda.empty_cache()


app = FastAPI(title="Backend Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定具体域名，如 ["http://localhost:3000"]
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
