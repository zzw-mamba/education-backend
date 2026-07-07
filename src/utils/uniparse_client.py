"""UniParse 客户端模块

封装与 UniParse API 的交互逻辑，提供文件解析服务。

UniParse 是一个文档解析服务，支持 PDF、图片等格式的解析，返回 Markdown 格式内容。

主要函数：
- parse_file_with_uniparse: 解析本地文件并返回 Markdown 内容
"""

import os
import time
import zipfile
import io
import requests
from dotenv import load_dotenv

load_dotenv()

UNIPARSE_BASE_URL = "https://uniparse.cn-sh-01.sensecoreapi.cn"
UNIPARSE_TOKEN = os.getenv("PARSER_API")


class UniParseError(RuntimeError):
    """UniParse API 调用失败时抛出的异常。"""


def parse_file_with_uniparse(file_path: str, file_name: str | None = None) -> str:
    """使用 UniParse API 解析本地文件并返回 Markdown 内容。
    
    该函数完成以下步骤：
    1. 获取文件上传地址
    2. 上传文件到 UniParse 服务器
    3. 创建解析任务
    4. 轮询等待解析完成
    5. 下载并提取解析结果
    
    Args:
        file_path: 本地文件的绝对路径
        file_name: 自定义文件名，默认为文件路径中的文件名
        
    Returns:
        解析后的 Markdown 文本内容
        
    Raises:
        UniParseError: 当 UniParse API 配置缺失或调用失败时
        FileNotFoundError: 当指定的文件路径不存在时
    """
    if not UNIPARSE_TOKEN:
        raise UniParseError("UniParse token not configured in .env")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    unique_filename = file_name or os.path.basename(file_path)
    headers = {
        "Authorization": f"Bearer {UNIPARSE_TOKEN}",
        "Content-Type": "application/json",
    }
    
    file_size = os.path.getsize(file_path)
    
    try:
        resp = requests.post(
            f"{UNIPARSE_BASE_URL}/api/v1/files",
            json={"file_name": unique_filename, "file_size": file_size},
            headers=headers,
        )
        resp.raise_for_status()
        upload_info = resp.json()
        file_id = upload_info["file_id"]
        upload_url = upload_info["upload_url"]
        
        with open(file_path, "rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": "application/octet-stream"},
            )
            put_resp.raise_for_status()
        
        task_resp = requests.post(
            f"{UNIPARSE_BASE_URL}/api/v1/parseTasks",
            json={
                "file_name": unique_filename,
                "file_id": file_id,
                "parse_type": "PARSE_TYPE_PIPLELINE",
            },
            headers=headers,
        )
        task_resp.raise_for_status()
        task_id = task_resp.json()["task_id"]
        
        for _ in range(30):
            status_resp = requests.get(
                f"{UNIPARSE_BASE_URL}/api/v1/parseTasks/{task_id}",
                headers=headers,
            )
            status_resp.raise_for_status()
            status = status_resp.json()
            state = status.get("state")
            
            if state == "TASK_STATE_SUCCEEDED":
                zip_url = status.get("result_zip_url")
                if zip_url:
                    content_resp = requests.get(zip_url)
                    content_resp.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(content_resp.content)) as z:
                        md_files = [f for f in z.namelist() if f.endswith(".md")]
                        if md_files:
                            return z.read(md_files[0]).decode("utf-8")
                elif status.get("result_json"):
                    md_parts = []
                    for page in status["result_json"]:
                        for block in page.get("content", []):
                            text = block.get("text", "")
                            block_type = block.get("type", "")
                            if block_type == "heading":
                                level = block.get("heading_level", 1)
                                md_parts.append(f"{'#' * level} {text}")
                            else:
                                md_parts.append(text)
                    return "\n\n".join(md_parts)
                
                return "Parsing succeeded, but no text content found."
            
            if state in ["TASK_STATE_FAILED", "TASK_STATE_CANCELED"]:
                raise UniParseError(f"Task failed: {status.get('error_message')}")
            
            time.sleep(2)
        
        raise UniParseError("Parsing timeout")
    
    except requests.RequestException as e:
        raise UniParseError(f"UniParse API request failed: {e}") from e