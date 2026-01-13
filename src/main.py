from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import models
from database import engine

# 导入路由
from routers import ocr, db_routes, user

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.Base.metadata.create_all(bind=engine)

    from paddleocr import PaddleOCRVL, PPStructureV3

    app.state.ocr = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )
    yield
    app.state.ocr = None


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
