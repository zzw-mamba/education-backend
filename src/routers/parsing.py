"""素材分析路由模块

提供知识库素材的批量解析和分析功能，支持通过 UniParse API 解析 PDF 和图片文件。

主要端点：
- POST /material/parse: 批量解析知识库素材并生成分析结果
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
import uuid
import os

from database import get_db
from models import KnowledgeBase, Log
from utils.get_resources_content import batch_process_to_file
from utils.uniparse_client import parse_file_with_uniparse, UniParseError

router = APIRouter(tags=["Material Analysis"])


@router.post("/material/parse")
async def parse_material_batch(
    request_data: List[int],
    request: Request,
    db: Session = Depends(get_db),
):
    """批量解析知识库素材并生成分析结果。
    
    根据提供的知识库 ID 列表，批量获取素材内容（文件或文本），调用 UniParse API 解析
    PDF/图片文件，然后使用 LLM 进行内容分析，最后将分析结果保存到文件并记录日志。
    
    Args:
        request_data: 知识库 ID 列表
        request: HTTP 请求对象
        db: 数据库会话
        
    Returns:
        包含消息、文件路径和统计信息的字典
        
    Raises:
        HTTPException: 当知识库 ID 列表为空或没有找到有效内容时
    """
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
        is_file = kb_item.file_path and os.path.exists(kb_item.file_path)
        
        if is_file:
            try:
                print(f"Starting UniParse for ID: {kb_item.id}, File: {kb_item.file_path}")
                content = parse_file_with_uniparse(kb_item.file_path)
            except UniParseError as e:
                print(f"Error calling UniParse for KB ID {kb_item.id}: {e}")
                content = kb_item.content
            except Exception as e:
                print(f"Error processing KB ID {kb_item.id}: {e}")
                content = kb_item.content
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

    batch_result = batch_process_to_file(materials, output_file=output_filename)

    user_id = getattr(request.state, "user", None)
    if hasattr(user_id, "id"):
        user_id = user_id.id

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