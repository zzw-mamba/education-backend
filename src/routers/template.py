from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import ast
from prompt import TEMPLATE_ANALYSE_PROMPT
from sqlalchemy.orm import Session
from database import get_db
from models import Template, User
from routers.user import get_current_user
from utils.model import ask_messages, LLMError

router = APIRouter(tags=["Template"])

class TemplateResponse(BaseModel):
    content: dict


class TemplateRequest(BaseModel):
    """保存模板请求模型"""
    name: str
    prompt: str
    category: int = 0
    description: str = None
    example: str = None
    icon_path: str = None

def get_template_prompt(description: str) -> str:
    return f"请根据以下描述生成一个合适的模板：\n\n{description}"


def extract_first_brace_block(text: str) -> str:
    """Return substring from first '{' to last '}' if both exist."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return text
    return text[start:end + 1]

@router.post("/template/build", response_model=TemplateResponse)
def build_template(request: str):
    """
    接收用户描述，调用大模型生成模板
    """
    # 构造 Prompt，根据你的模型特性进行微调
    messages = [
        {"role": "system", "content": TEMPLATE_ANALYSE_PROMPT},
        {"role": "user", "content": "请根据以下摘要内容，提取出一个通用的文本模板，供后续类似内容的快速生成：\n\n摘要内容如下：\n" + request}
    ]

    try:
        llm_result = ask_messages(
            messages=messages,
            max_tokens=4096,
            temperature=0.95,
            top_p=0.6,
            extra_payload={
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = llm_result.content or ""

        content = extract_first_brace_block(content)
        try:
            content_dict = ast.literal_eval(content)
        except Exception:
            content_dict = {}
        
        return TemplateResponse(content=content_dict)
    except LLMError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到大模型服务: {exc}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"模板生成过程中发生错误: {exc}"
        )


@router.put("/template/add", status_code=200)
def add_template(information: TemplateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    将生成的模板保存到数据库
    """
    try:
        # 创建新的 Template 对象，user_id 从 token 中提取
        new_template = Template(
            user_id=current_user.id,
            name=information.name,
            prompt=information.prompt,
            category=information.category,
            description=information.description,
            example=information.example,
            icon_path=information.icon_path
        )
        
        db.add(new_template)
        db.commit()
        db.refresh(new_template)
        
        return {
            "code": 200,
            "message": "模板保存成功",
            "data": {
                "template_id": new_template.id,
                "name": new_template.name,
                "created_at": new_template.created_at.isoformat() if new_template.created_at else None
            }
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"保存模板失败: {str(e)}"
        )