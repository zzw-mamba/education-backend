from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from database import get_db
from models import KnowledgeBase, Log
from utils.get_resources_content import process_material_workflow
import json

router = APIRouter(tags=["Material Analysis"])

@router.post("/material/parse/{kb_id}")
async def parse_material(kb_id: int, request: Request, db: Session = Depends(get_db)):
    """
    对指定的知识库条目进行深度解析（实体识别、事件提取等），
    并将结果记录在 Log 中，实现转换与溯源。
    """
    # 1. 获取知识库条目 (溯源核心：基于已有KB ID)
    kb_item = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb_item:
        raise HTTPException(status_code=404, detail="Material not found")

    if not kb_item.content:
        raise HTTPException(status_code=400, detail="Material content is empty")

    # 2. 调用解析工作流
    # 构造输入数据，process_material_workflow 期望 {"id": ..., "content": ...}
    material_input = {
        "id": str(kb_item.id), # 转即字符串为了通用性
        "content": kb_item.content
    }
    
    # 这里调用我们在 utils 中实现的具体逻辑
    result = process_material_workflow(material_input)
    
    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

    analysis_data = result.get("data", {})

    # 3. 结果保存与溯源记录
    # 将解析结果存入 Log 表，关联 user_id (如果已登录) 和 kb_id
    
    # 尝试从 request.state 或 session 获取用户信息(如果有鉴权中间件)
    user_id = getattr(request.state, "user", None)
    if hasattr(user_id, "id"):
        user_id = user_id.id
    
    log_entry = Log(
        user_id=user_id if isinstance(user_id, int) else None,
        action="material_analysis",
        details=json.dumps({
            "source_kb_id": kb_id,
            "parsed_data": analysis_data
        }, ensure_ascii=False),
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log_entry)
    
    # 也可以选择更新 KnowledgeBase 的某些字段，例如自动打标签
    # 如果 analysis_data 中有 keywords，可以添加到 tags
    if isinstance(analysis_data, dict) and "keywords" in analysis_data:
        # 这里可以实现自动打标签逻辑，暂时略过
        pass

    db.commit()

    return {
        "message": "Analysis completed",
        "kb_id": kb_id,
        "title": kb_item.title,
        "analysis_result": analysis_data
    }
