import os
import time
import uuid
import zipfile
import io
import requests
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

UNIPARSE_BASE_URL = "https://uniparse.cn-sh-01.sensecoreapi.cn"
UNIPARSE_TOKEN = os.getenv("PARSER_API")

router = APIRouter(tags=["OCR"])

@router.post("/ocr", status_code=200)
async def ocr_recognize(request: Request, file: UploadFile = File(...)) -> PlainTextResponse:
    """
    使用 UniParse API 识别 PDF 或 图片并返回 Markdown 内容
    """
    if not UNIPARSE_TOKEN:
        raise HTTPException(status_code=500, detail="UniParse token not configured in .env")

    # 1. 检查后缀名 (保留原有校验逻辑)
    def _detect_ext(file: UploadFile) -> str:
        name = (file.filename or "").lower()
        known_exts = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]
        for ext in known_exts:
            if name.endswith(ext):
                return ext
        
        ct = (file.content_type or "").lower()
        ct_map = {
            "application/pdf": ".pdf",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
        }
        if ct in ct_map:
            return ct_map[ct]
        raise HTTPException(status_code=415, detail="Unsupported file type")

    ext = _detect_ext(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # 2. 临时保存上传的文件
    # 为了获取文件大小和后续上传，先存到临时目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(base_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    unique_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(temp_dir, unique_filename)
    
    with open(temp_path, "wb") as f:
        f.write(data)

    # --- UniParse 解析逻辑开始 ---
    try:
        headers = {
            "Authorization": f"Bearer {UNIPARSE_TOKEN}",
            "Content-Type": "application/json"
        }

        # Step 1: 获取上传地址
        file_size = os.path.getsize(temp_path)
        resp = requests.post(
            f"{UNIPARSE_BASE_URL}/api/v1/files",
            json={"file_name": unique_filename, "file_size": file_size},
            headers=headers
        )
        resp.raise_for_status()
        upload_info = resp.json()
        print(upload_info)
        file_id = upload_info["file_id"]
        upload_url = upload_info["upload_url"]

        # Step 2: 上传文件
        with open(temp_path, "rb") as f:
            requests.put(upload_url, data=f, headers={"Content-Type": "application/octet-stream"})

        # Step 3: 创建任务
        task_resp = requests.post(
            f"{UNIPARSE_BASE_URL}/api/v1/parseTasks",
            json={
                "file_name": unique_filename,
                "file_id": file_id,
                "parse_type": "PARSE_TYPE_PIPLELINE"  # Pipeline 模式支持布局分析和 OCR
            },
            headers=headers
        )
        task_id = task_resp.json()["task_id"]

        # Step 4: 轮询结果
        markdown_content = ""
        for i in range(30):
            status_resp = requests.get(f"{UNIPARSE_BASE_URL}/api/v1/parseTasks/{task_id}", headers=headers)
            status = status_resp.json()
            state = status.get("state")
            
            if state == "TASK_STATE_SUCCEEDED":
                zip_url = status.get('result_zip_url')
                
                # 情况 A: 如果有 ZIP 下载地址 (通常是多页或复杂文档)
                if zip_url:
                    print("Downloading result from ZIP URL...")
                    content_resp = requests.get(zip_url)
                    with zipfile.ZipFile(io.BytesIO(content_resp.content)) as z:
                        md_files = [f for f in z.namelist() if f.endswith('.md')]
                        if md_files:
                            markdown_content = z.read(md_files[0]).decode('utf-8')
                
                # 情况 B: 如果没有 ZIP，直接从 result_json 拼接 (通常是图片或单页)
                elif "result_json" in status and status["result_json"]:
                    print("Extracting result from result_json...")
                    md_parts = []
                    # 遍历每一页
                    for page in status["result_json"]:
                        # 遍历页内的每一个内容块
                        for block in page.get("content", []):
                            text = block.get("text", "")
                            block_type = block.get("type", "")
                            
                            if block_type == "heading":
                                level = block.get("heading_level", 1)
                                md_parts.append(f"{'#' * level} {text}")
                            else:
                                md_parts.append(text)
                    markdown_content = "\n\n".join(md_parts)
                
                if not markdown_content:
                    markdown_content = "Parsing succeeded, but no text content found."
                break
            
            elif state in ["TASK_STATE_FAILED", "TASK_STATE_CANCELED"]:
                raise Exception(f"Task failed: {status.get('error_message')}")
            
            time.sleep(2)
        else:
            raise Exception("Parsing timeout")

    except Exception as e:
        print(f"UniParse Error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR Service Error: {str(e)}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 返回 Markdown 内容
    return PlainTextResponse(content=markdown_content, media_type="text/markdown")