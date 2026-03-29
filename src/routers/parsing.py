from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from database import get_db
from models import KnowledgeBase, Log
import uuid
import os
import time
import requests
import zipfile
import io
from datetime import datetime
from typing import List
from utils.get_resources_content import batch_process_to_file
from dotenv import load_dotenv

load_dotenv()
# 配置 UniParse 访问信息
UNIPARSE_BASE_URL = "https://uniparse.cn-sh-01.sensecoreapi.cn"
UNIPARSE_TOKEN = os.getenv("PARSER_API")

router = APIRouter(tags=["Material Analysis"])

def uniparse_file(file_path: str):
    """
    内部辅助函数：调用 UniParse API 解析文件并返回 Markdown 内容
    """
    headers = {
        "Authorization": f"Bearer {UNIPARSE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 1. 获取上传地址
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    resp = requests.post(
        f"{UNIPARSE_BASE_URL}/api/v1/files",
        json={"file_name": file_name, "file_size": file_size},
        headers=headers
    )
    upload_info = resp.json()
    file_id = upload_info["file_id"]
    upload_url = upload_info["upload_url"]
    
    # 2. 上传文件
    with open(file_path, "rb") as f:
        requests.put(upload_url, data=f, headers={"Content-Type": "application/octet-stream"})
    
    # 3. 创建解析任务 (根据后缀自动判断，或统一用 PIPELINE)
    task_resp = requests.post(
        f"{UNIPARSE_BASE_URL}/api/v1/parseTasks",
        json={
            "file_name": file_name,
            "file_id": file_id,
            "parse_type": "PARSE_TYPE_PIPLELINE" # 通用高精度解析
        },
        headers=headers
    )
    task_id = task_resp.json()["task_id"]
    
    # 4. 轮询等待
    while True:
        status_resp = requests.get(f"{UNIPARSE_BASE_URL}/api/v1/parseTasks/{task_id}", headers=headers)
        status = status_resp.json()
        
        if status["state"] == "TASK_STATE_SUCCEEDED":
            # 5. 下载并解压结果
            zip_url = status['result_zip_url']
            content_resp = requests.get(zip_url)
            with zipfile.ZipFile(io.BytesIO(content_resp.content)) as z:
                # 寻找解析后的内容文件，通常是 content.md 或类似名称
                # 这里假设获取压缩包内第一个 .md 文件
                md_files = [f for f in z.namelist() if f.endswith('.md')]
                if md_files:
                    return z.read(md_files[0]).decode('utf-8')
                return "Parsing successful but no markdown found."
                
        elif status["state"] in ["TASK_STATE_FAILED", "TASK_STATE_CANCELED"]:
            raise Exception(f"UniParse Task Failed: {status.get('error_message')}")
        
        time.sleep(2) # 轮询间隔


@router.post("/material/parse")
async def parse_material_batch(request_data: List[int], request: Request, db: Session = Depends(get_db)):
    kb_ids = request_data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())
    output_filename = f"analysis_{timestamp}_{unique_id}.json"

    if not kb_ids:
        raise HTTPException(status_code=400, detail="kb_ids list cannot be empty")

    kb_items = db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(kb_ids)).all()

    materials = []
    for kb_item in kb_items:
        content = None
        # 判断是否为 PDF 或 图片 (jpg, png, jpeg)
        is_file = kb_item.file_path and os.path.exists(kb_item.file_path)
        
        if is_file:
            try:
                print(f"Starting UniParse for ID: {kb_item.id}, File: {kb_item.file_path}")
                # 使用 UniParse 替换原有的 extract_with_marker
                content = uniparse_file(kb_item.file_path)
            except Exception as e:
                print(f"Error calling UniParse for KB ID {kb_item.id}: {e}")
                content = kb_item.content  # 失败则回退到原始 content
        else:
            content = kb_item.content
        
        if content:
            materials.append({
                "id": str(kb_item.id),
                "content": content,
                "title": kb_item.title
            })

    if not materials:
        raise HTTPException(status_code=400, detail="No materials with content found")

    # 后续处理逻辑保持不变
    batch_result = batch_process_to_file(materials, output_file=output_filename)

    # 记录日志等...
    user_id = getattr(request.state, "user", None)
    if hasattr(user_id, "id"): user_id = user_id.id

    log_entry = Log(
        user_id=user_id if isinstance(user_id, int) else None,
        template_id=None,
        knowledge_ids=",".join(map(str, kb_ids)),
        result_path=batch_result["file_path"],
    )
    db.add(log_entry)
    db.commit()

    return {
        "message": "Batch analysis completed",
        "file_path": batch_result["file_path"],
        "statistics": batch_result
    }