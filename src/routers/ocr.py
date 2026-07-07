"""OCR 识别路由模块

提供 PDF 和图片文件的 OCR 识别功能，通过 UniParse API 将文档转换为 Markdown 格式。

主要端点：
- POST /ocr: 上传文件并返回 Markdown 识别结果
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

from utils.uniparse_client import parse_file_with_uniparse, UNIPARSE_TOKEN, UniParseError

load_dotenv()

router = APIRouter(tags=["OCR"])


@router.post("/ocr", status_code=200)
async def ocr_recognize(request: Request, file: UploadFile = File(...)) -> PlainTextResponse:
    """使用 UniParse API 识别 PDF 或图片并返回 Markdown 内容。
    
    支持的文件格式：PDF、PNG、JPG、JPEG、BMP、WEBP、TIF、TIFF。
    
    处理流程：
    1. 检测文件类型并验证
    2. 将上传文件临时保存到本地
    3. 调用 UniParse API 进行解析
    4. 返回 Markdown 内容，并将结果保存到本地供后续使用
    
    Args:
        request: HTTP 请求对象
        file: 上传的文件对象
        
    Returns:
        PlainTextResponse: 包含 Markdown 内容的响应，带有 X-OCR-Output-File 和 X-OCR-Output-Path 头部
        
    Raises:
        HTTPException: 当文件类型不支持、文件为空或 UniParse 服务出错时
    """
    if not UNIPARSE_TOKEN:
        raise HTTPException(status_code=500, detail="UniParse token not configured in .env")

    def _detect_ext(file: UploadFile) -> str:
        """检测上传文件的扩展名。
        
        优先根据文件名判断，其次根据 Content-Type 判断。
        
        Args:
            file: 上传文件对象
            
        Returns:
            文件扩展名（带点）
            
        Raises:
            HTTPException: 当文件类型不支持时
        """
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

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(base_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    unique_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(temp_dir, unique_filename)
    
    with open(temp_path, "wb") as f:
        f.write(data)

    output_dir = os.path.join(base_dir, "ocr_outputs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        markdown_content = parse_file_with_uniparse(temp_path, unique_filename)
    except UniParseError as e:
        print(f"UniParse Error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR Service Error: {str(e)}")
    except Exception as e:
        print(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR Service Error: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    output_filename = f"{uuid.uuid4()}.md"
    output_rel_path = os.path.join("ocr_outputs", output_filename)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(markdown_content)

    response = PlainTextResponse(content=markdown_content, media_type="text/markdown")
    response.headers["X-OCR-Output-File"] = output_filename
    response.headers["X-OCR-Output-Path"] = output_rel_path.replace("\\", "/")
    return response