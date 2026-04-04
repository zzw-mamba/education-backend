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


def parse_file_with_uniparse(file_path: str, file_name: str | None = None) -> str:
    """使用现有 UniParse 流程解析本地文件并返回 Markdown 文本。"""
    if not UNIPARSE_TOKEN:
        raise RuntimeError("UniParse token not configured in .env")

    unique_filename = file_name or os.path.basename(file_path)
    headers = {
        "Authorization": f"Bearer {UNIPARSE_TOKEN}",
        "Content-Type": "application/json",
    }

    file_size = os.path.getsize(file_path)
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
        put_resp = requests.put(upload_url, data=f, headers={"Content-Type": "application/octet-stream"})
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

    markdown_content = ""
    for _ in range(30):
        status_resp = requests.get(f"{UNIPARSE_BASE_URL}/api/v1/parseTasks/{task_id}", headers=headers)
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
                        markdown_content = z.read(md_files[0]).decode("utf-8")
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
                markdown_content = "\n\n".join(md_parts)

            if not markdown_content:
                markdown_content = "Parsing succeeded, but no text content found."
            return markdown_content

        if state in ["TASK_STATE_FAILED", "TASK_STATE_CANCELED"]:
            raise RuntimeError(f"Task failed: {status.get('error_message')}")

        time.sleep(2)

    raise RuntimeError("Parsing timeout")

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

    # 识别结果本地保存目录
    output_dir = os.path.join(base_dir, "ocr_outputs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        markdown_content = parse_file_with_uniparse(temp_path, unique_filename)
    except Exception as e:
        print(f"UniParse Error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR Service Error: {str(e)}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 将 OCR 结果落盘到本地，便于后续排查和复用
    output_filename = f"{uuid.uuid4()}.md"
    output_rel_path = os.path.join("ocr_outputs", output_filename)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(markdown_content)

    # 返回 Markdown 内容
    response = PlainTextResponse(content=markdown_content, media_type="text/markdown")
    response.headers["X-OCR-Output-File"] = output_filename
    response.headers["X-OCR-Output-Path"] = output_rel_path.replace("\\", "/")
    return response