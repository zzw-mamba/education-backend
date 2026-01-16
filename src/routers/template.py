from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from prompt import TEMPLATE_ANALYSE_PROMPT
from dotenv import load_dotenv
import os

router = APIRouter(tags=["Template"])

# 大模型 API 基地址
load_dotenv()
LLM_API_BASE = os.getenv("LLM_API_BASE")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 60.0))
MODEL_NAME = os.getenv("MODEL_NAME")

class TemplateResponse(BaseModel):
    content: str

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

    payload = dict(
        messages=messages,
        model=MODEL_NAME,
        max_tokens=4096,
        temperature=0.95,
        top_p=0.6,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        chat_template_kwargs={"enable_thinking": False}
    )

    try:
        response = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
            verify=False
        )
        
        if response.status_code != 200:
            print(111)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM API Error: {response.text}"
            )

        data = response.json()
        
        content = data.get("content")
        if not content and "choices" in data:
                choice = data["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content")
                else:
                    content = choice.get("text")
        
        if content is None:
            content = ""

        return TemplateResponse(content=content)

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到大模型服务: {exc}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"模板生成过程中发生错误: {exc}"
        )
