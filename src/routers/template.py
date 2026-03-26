import ast
import html
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from prompt import (
    TEMPLATE_ANALYSE_PROMPT,
    TEMPLATE_BUILD_USER_PROMPT_TEMPLATE,
    TEMPLATE_DESCRIPTION_USER_PROMPT_TEMPLATE,
    TEMPLATE_RAG_SUMMARY_SYSTEM_PROMPT,
    TEMPLATE_RAG_SUMMARY_USER_PROMPT_TEMPLATE,
)
from sqlalchemy.orm import Session
from sqlalchemy import or_
from database import get_db
from models import Template, User, Log
from graphrag.graphrag_service import get_graphrag_service, GRAPHRAG_IMPORT_ERROR
from routers.user import get_current_user
from utils.model import ask_messages, LLMError

router = APIRouter(tags=["Template"])

SERVER_ROOT_DIR = Path(__file__).resolve().parents[2]
SUMMARY_EXPORT_DIR = Path(__file__).resolve().parents[2] / "analysis_results" / "summary_exports"

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


class TemplateSummaryRequest(BaseModel):
    """基于模板 + GraphRAG 生成摘要请求模型。"""
    query_text: str = Field(..., description="用户关注主题/问题")
    paper_ids: Optional[List[int]] = Field(default=None, description="可选：限定论文 ID 范围")
    top_k: int = Field(default=8, ge=1, le=30, description="向量召回数量")
    focus_direction: str = Field(default="行业发展趋势与技术演进路径", description="摘要聚焦方向")
    style: str = Field(default="客观严谨的学术/商业报告风格", description="文风要求")
    word_limit: str = Field(default="500-800字", description="篇幅要求")
    graph_top_entities: int = Field(default=8, ge=1, le=20, description="每篇论文抽取的核心实体数")
    snippets_per_entity: int = Field(default=2, ge=1, le=5, description="每个实体保留的上下文片段数")
    neighbor_limit: int = Field(default=4, ge=0, le=20, description="图谱实体邻居数量")
    max_graph_papers: int = Field(default=3, ge=1, le=10, description="最多补充图谱上下文的论文数")

def get_template_prompt(description: str) -> str:
    return TEMPLATE_DESCRIPTION_USER_PROMPT_TEMPLATE.format(description=description)


def extract_first_brace_block(text: str) -> str:
    """Return substring from first '{' to last '}' if both exist."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return text
    return text[start:end + 1]


def _format_document_chunks(chunks: List[dict], paper_citation_ids: dict) -> str:
    """将召回片段格式化为可读上下文，并按文献分配唯一引用编号。"""
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        score = chunk.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "N/A"
        paper_id = chunk.get("paper_id")
        paper_key = f"paper:{paper_id}" if paper_id is not None else f"title:{(chunk.get('title') or '').strip()}"
        citation_id = paper_citation_ids.get(paper_key)
        lines.append(
            (
                f"[Chunk {idx}] [文献引用标记：{citation_id}]"
                # f"年份={chunk.get('year')}，chunk_index={chunk.get('chunk_index')}，相似度={score_text}\n"
                f"{chunk.get('text') or ''}"
            )
        )
    print("\n".join(lines))
    return "\n\n".join(lines)


def _build_citation_mapping(chunks: List[dict]) -> tuple[dict, dict]:
    """按文献维度构建引用映射，保证一篇文献仅对应一个引用标记。"""
    citations: dict = {}
    paper_citation_ids: dict = {}

    for chunk in chunks:
        paper_id = chunk.get("paper_id")
        title = (chunk.get("title") or "未知").strip() or "未知"
        paper_key = f"paper:{paper_id}" if paper_id is not None else f"title:{title}"

        if paper_key not in paper_citation_ids:
            citation_id = len(paper_citation_ids) + 1
            paper_citation_ids[paper_key] = citation_id
            citations[citation_id] = {
                "id": citation_id,
                "paper_id": paper_id,
                "title": title,
                "year": chunk.get("year"),
                "chunk_ids": [],
                "scores": [],
                "text_excerpts": [],
            }

        citation_id = paper_citation_ids[paper_key]
        citation_item = citations[citation_id]
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in citation_item["chunk_ids"]:
            citation_item["chunk_ids"].append(chunk_id)

        score = chunk.get("score")
        if score is not None:
            citation_item["scores"].append(float(score))

        excerpt = (chunk.get("text") or "")[:200]
        if excerpt and excerpt not in citation_item["text_excerpts"] and len(citation_item["text_excerpts"]) < 3:
            citation_item["text_excerpts"].append(excerpt)

    for citation in citations.values():
        scores = citation.pop("scores", [])
        citation["max_score"] = max(scores) if scores else None
        citation["avg_score"] = (sum(scores) / len(scores)) if scores else None

    return citations, paper_citation_ids


def _format_graph_context(service, request: TemplateSummaryRequest, chunks: List[dict]) -> str:
    """基于召回论文补充图谱实体上下文，增强跨片段关系理解。"""
    paper_ids: List[int] = []
    for chunk in chunks:
        paper_id = chunk.get("paper_id")
        if isinstance(paper_id, int) and paper_id not in paper_ids:
            paper_ids.append(paper_id)

    graph_context_parts: List[str] = []
    for paper_id in paper_ids[: request.max_graph_papers]:
        try:
            refined = service.get_graph_refined_context(
                paper_id=paper_id,
                top_entities=request.graph_top_entities,
                snippets_per_entity=request.snippets_per_entity,
                neighbor_limit=request.neighbor_limit,
            )
        except Exception:
            continue

        blocks = refined.get("context_blocks", [])
        if blocks:
            graph_context_parts.append(f"[Paper {paper_id}]\n" + "\n\n".join(blocks))

    return "\n\n".join(graph_context_parts)


def _build_knowledge_ids_for_log(search_result: dict, chunks: List[dict]) -> str:
    """为日志构建 knowledge_ids 字段，优先使用检索返回的 paper_ids。"""
    raw_ids = search_result.get("paper_ids") or []
    unique_ids: List[str] = []

    for item in raw_ids:
        if isinstance(item, int):
            item_str = str(item)
            if item_str not in unique_ids:
                unique_ids.append(item_str)

    if not unique_ids:
        for chunk in chunks:
            paper_id = chunk.get("paper_id")
            if isinstance(paper_id, int):
                paper_id_str = str(paper_id)
                if paper_id_str not in unique_ids:
                    unique_ids.append(paper_id_str)

    # logs.knowledge_ids 为 VARCHAR(255)，超长时进行截断避免写库失败。
    return ",".join(unique_ids)[:255]


def _ensure_summary_export_dir() -> Path:
    SUMMARY_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return SUMMARY_EXPORT_DIR


def _export_markdown_file(markdown_text: str, file_path: Path) -> None:
    file_path.write_text(markdown_text, encoding="utf-8")


def _export_docx_file(markdown_text: str, file_path: Path) -> None:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("缺少 python-docx 依赖，无法导出 Word") from exc

    document = Document()
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            document.add_paragraph("")
            continue

        if line.startswith("### "):
            document.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            document.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            document.add_heading(line[2:].strip(), level=1)
        elif line.startswith("- "):
            document.add_paragraph(line[2:].strip(), style="List Bullet")
        else:
            document.add_paragraph(line)

    document.save(file_path)


def _export_pdf_file(markdown_text: str, file_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("缺少 reportlab 依赖，无法导出 PDF") from exc

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    doc = SimpleDocTemplate(
        str(file_path),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyCN",
        parent=styles["Normal"],
        fontName="STSong-Light",
        fontSize=11,
        leading=18,
    )
    h1_style = ParagraphStyle(
        "H1CN",
        parent=styles["Heading1"],
        fontName="STSong-Light",
        fontSize=18,
        leading=24,
    )
    h2_style = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=15,
        leading=21,
    )
    h3_style = ParagraphStyle(
        "H3CN",
        parent=styles["Heading3"],
        fontName="STSong-Light",
        fontSize=13,
        leading=18,
    )

    story = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            story.append(Spacer(1, 8))
            continue

        escaped_line = html.escape(line)
        if line.startswith("### "):
            story.append(Paragraph(html.escape(line[4:].strip()), h3_style))
        elif line.startswith("## "):
            story.append(Paragraph(html.escape(line[3:].strip()), h2_style))
        elif line.startswith("# "):
            story.append(Paragraph(html.escape(line[2:].strip()), h1_style))
        elif line.startswith("- "):
            bullet_text = html.escape(line[2:].strip())
            story.append(Paragraph(f"• {bullet_text}", body_style))
        else:
            story.append(Paragraph(escaped_line, body_style))

    if not story:
        story.append(Paragraph("(Empty Markdown)", body_style))

    doc.build(story)


def _export_summary_bundle(markdown_text: str, result_id: str) -> dict:
    export_dir = _ensure_summary_export_dir()
    md_path = export_dir / f"{result_id}.md"
    docx_path = export_dir / f"{result_id}.docx"
    pdf_path = export_dir / f"{result_id}.pdf"

    _export_markdown_file(markdown_text, md_path)
    _export_docx_file(markdown_text, docx_path)
    _export_pdf_file(markdown_text, pdf_path)

    def _to_server_relative(path: Path) -> str:
        try:
            return path.relative_to(SERVER_ROOT_DIR).as_posix()
        except ValueError:
            # 异常场景回退，避免接口直接返回 Windows 绝对盘符。
            return path.name

    return {
        "result_id": result_id,
        "output_dir": _to_server_relative(export_dir),
        "markdown_path": _to_server_relative(md_path),
        "word_path": _to_server_relative(docx_path),
        "pdf_path": _to_server_relative(pdf_path),
    }

@router.post("/template/build", response_model=TemplateResponse)
def build_template(request: str):
    """
    接收用户描述，调用大模型生成模板
    """
    # 构造 Prompt，根据你的模型特性进行微调
    messages = [
        {"role": "system", "content": TEMPLATE_ANALYSE_PROMPT},
        {
            "role": "user",
            "content": TEMPLATE_BUILD_USER_PROMPT_TEMPLATE.format(text=request),
        }
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


@router.post("/template/{template_id}/summary", status_code=200)
def generate_summary_by_template(
    template_id: int,
    request: TemplateSummaryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """根据模板 ID + GraphRAG 检索结果生成结构化全局摘要。"""
    print(request)
    template = (
        db.query(Template)
        .filter(
            Template.id == template_id,
            or_(Template.user_id == current_user.id, Template.user_id == 0),
        )
        .first()
    )
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在或无权限访问")

    if GRAPHRAG_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"neo4j-graphrag 导入失败: {GRAPHRAG_IMPORT_ERROR}",
        )

    try:
        graphrag_service = get_graphrag_service()
        search_result = graphrag_service.similarity_search(
            request.query_text,
            top_k=request.top_k,
            paper_ids=request.paper_ids,
        )
        chunks = search_result.get("results", [])
        if not chunks:
            return {
                "code": 200,
                "message": "未检索到相关片段，无法生成摘要",
                "data": {
                    "template_id": template.id,
                    "template_name": template.name,
                    "query_text": request.query_text,
                    "summary": "未检索到相关片段，请调整检索问题或放宽 paper_ids 范围。",
                    "retrieved_chunks": 0,
                    "graph_context_blocks": 0,
                },
            }

        citations, paper_citation_ids = _build_citation_mapping(chunks)
        document_chunks = _format_document_chunks(chunks, paper_citation_ids)
        graph_relations = _format_graph_context(graphrag_service, request, chunks)
        if not graph_relations.strip():
            graph_relations = "暂无可用图谱上下文。"

        prompt_text = TEMPLATE_RAG_SUMMARY_USER_PROMPT_TEMPLATE.format(
            template_prompt=template.prompt,
            query_text=request.query_text,
            focus_direction=request.focus_direction,
            style=request.style,
            word_limit=request.word_limit,
            document_chunks=document_chunks,
            graph_relations=graph_relations,
        )
        prompt_text += (
            "\n\n输出格式要求：\n"
            "1) 必须输出标准 Markdown。\n"
            "2) 结构需包含标题、分节小标题和要点列表。\n"
            "3) 引用请保留 [文献引用标记：x]，不要输出 JSON。"
        )

        llm_result = ask_messages(
            messages=[
                {"role": "system", "content": TEMPLATE_RAG_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=2200,
            temperature=0.2,
            top_p=0.8,
            extra_payload={
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        summary_markdown = (llm_result.content or "").strip()
        if not summary_markdown:
            raise HTTPException(status_code=502, detail="大模型返回为空，未生成有效 Markdown")

        result_id = str(uuid4())
        export_info = _export_summary_bundle(summary_markdown, result_id)

        graph_block_count = graph_relations.count("【实体】")
        knowledge_ids = _build_knowledge_ids_for_log(search_result, chunks)
        log_entry = Log(
            user_id=current_user.id,
            template_id=template.id,
            knowledge_ids=knowledge_ids,
            result_path=export_info["markdown_path"],
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        print({
            "code": 200,
            "message": "摘要生成成功",
            "data": {
                "template_id": template.id,
                "template_name": template.name,
                "query_text": request.query_text,
                "result_id": result_id,
                "summary": summary_markdown,
                "summary_markdown": summary_markdown,
                "retrieved_chunks": len(chunks),
                "graph_context_blocks": graph_block_count,
                "paper_ids": search_result.get("paper_ids"),
                "citations": citations,
                "log_id": log_entry.id,
                "files": export_info,
            },
        })
        return {
            "code": 200,
            "message": "摘要生成成功",
            "data": {
                "template_id": template.id,
                "template_name": template.name,
                "query_text": request.query_text,
                "result_id": result_id,
                "summary": summary_markdown,
                "summary_markdown": summary_markdown,
                "retrieved_chunks": len(chunks),
                "graph_context_blocks": graph_block_count,
                "paper_ids": search_result.get("paper_ids"),
                "citations": citations,
                "log_id": log_entry.id,
                "files": export_info,
            },
        }
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"无法连接到大模型服务: {exc}")
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"基于模板生成摘要失败: {exc}")


@router.get("/template/my", status_code=200)
def get_my_templates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    获取当前用户可见模板：
    1. 用户自己的模板（user_id = 当前用户id）
    2. 公共模板（user_id = 0）
    """
    try:
        templates = (
            db.query(Template)
            .filter(or_(Template.user_id == current_user.id, Template.user_id == 0))
            .order_by(Template.created_at.desc())
            .all()
        )

        return {
            "code": 200,
            "message": "获取模板成功",
            "data": [
                {
                    "id": template.id,
                    "user_id": template.user_id,
                    "name": template.name,
                    "prompt": template.prompt,
                    "category": template.category,
                    "description": template.description,
                    "example": template.example,
                    "icon_path": template.icon_path,
                    "labels": template.labels,
                    "created_at": template.created_at.isoformat() if template.created_at else None,
                    "updated_at": template.updated_at.isoformat() if template.updated_at else None,
                }
                for template in templates
            ],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"获取模板失败: {str(exc)}",
        )