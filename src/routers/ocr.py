from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse
import os
import uuid
import glob

router = APIRouter(tags=["OCR"])

@router.post("/ocr", status_code=200)
async def ocr_recognize(request: Request, file: UploadFile = File(...)) -> PlainTextResponse:
    def _detect_ext(file: UploadFile) -> str:
        name = (file.filename or "").lower()
        ct = (file.content_type or "").lower()

        known_exts = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]
        for ext in known_exts:
            if name.endswith(ext):
                return ext

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
    
    ocr = request.app.state.ocr
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = _detect_ext(file)
    md_parts: list[str] = []

    # 使用相对于当前文件 (src/routers/ocr.py) 的相对路径 ../../
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_root = os.path.join(base_dir, "uploaded_files")  # 自定义固定目录
    output_dir = os.path.join(base_dir, "ocr_outputs")
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 生成唯一文件名（避免同名文件覆盖）
    unique_filename = f"{uuid.uuid4()}{ext}"
    upload_path = os.path.join(save_root, unique_filename)
    print(f"长期保存上传文件到：{upload_path}")

    with open(upload_path, "wb") as fh:
        fh.write(data)

    results = ocr.predict(upload_path)
    for res in results:
        res.save_to_markdown(save_path=output_dir)

    # 读取所有生成的markdown文件并合并
    markdown_files = sorted(glob.glob(os.path.join(output_dir, "*.md")))
    for md_file in markdown_files:
        with open(md_file, "r", encoding="utf-8") as f:
            md_parts.append(f.read())

    markdown_content = "\n\n".join(md_parts)
    return PlainTextResponse(content=markdown_content, media_type="text/markdown")
