"""知识库操作路由模块

提供知识库的增删改查、全文检索、推荐和文件下载功能。

主要端点：
- POST /knowledge/add: 添加知识条目（自动提取标签）
- GET /knowledge/search: 全文检索（支持LLM扩展和重排）
- GET /knowledge/recommend: 基于标签重合度推荐相似条目
- GET /knowledge/recommend/{kb_id}: 单篇文章推荐
- GET /knowledge/content/{kb_id}: 获取知识内容
- GET /knowledge/file/{file_id}: 下载关联文件

主要辅助函数：
- _resolve_existing_file_path: 解析文件路径
- _expand_search_terms_with_llm: 使用LLM扩展搜索词
- rerank_documents: 文档相关性重排
"""

import jieba.analyse
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import os
from pydantic import BaseModel

from database import get_db
import models
import re
from routers.user import get_current_user
from prompt import (
    KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT,
    KNOWLEDGE_SEARCH_EXPANSION_USER_PROMPT_TEMPLATE,
)
from utils.model import ask_messages, LLMError

router = APIRouter(tags=["Database"])


def _resolve_existing_file_path(raw_path: str) -> str:
    """Resolve a knowledge file path against common project roots."""
    if not raw_path:
        return ""

    normalized = raw_path.replace("\\", os.sep).replace("/", os.sep)
    candidates = []

    # Absolute path as-is.
    if os.path.isabs(normalized):
        candidates.append(normalized)

    # Resolve against several likely roots.
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_root = os.path.dirname(src_root)
    workspace_root = os.path.dirname(backend_root)

    candidates.extend(
        [
            os.path.abspath(normalized),
            os.path.join(src_root, normalized),
            os.path.join(backend_root, normalized),
            os.path.join(workspace_root, normalized),
        ]
    )

    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _fetch_tag_map_for_kb_ids(db: Session, kb_ids: list[int]) -> dict[int, set[str]]:
    if not kb_ids:
        return {}

    sql = text(
        """
        SELECT r.kb_id, t.name
        FROM kb_tag_relation r
        JOIN tags t ON r.tag_id = t.id
        WHERE r.kb_id IN :ids
        """
    )
    rows = db.execute(sql, {"ids": tuple(kb_ids)}).all()

    tag_map: dict[int, set[str]] = {}
    for row in rows:
        tag_map.setdefault(int(row.kb_id), set()).add(str(row.name))
    return tag_map


def _enrich_recommendations_with_tag_diff(
    db: Session,
    recommendation_docs: list[dict],
    seed_kb_ids: list[int],
) -> list[dict]:
    if not recommendation_docs:
        return recommendation_docs

    seed_tag_map = _fetch_tag_map_for_kb_ids(db, seed_kb_ids)
    seed_tags = set().union(*seed_tag_map.values()) if seed_tag_map else set()
    candidate_ids = [int(doc["id"]) for doc in recommendation_docs]
    candidate_tag_map = _fetch_tag_map_for_kb_ids(db, candidate_ids)

    for doc in recommendation_docs:
        candidate_tags = candidate_tag_map.get(int(doc["id"]), set())
        same_tags = sorted(seed_tags.intersection(candidate_tags))
        different_tags = sorted(candidate_tags.difference(seed_tags))

        doc["common_tags"] = len(same_tags)
        doc["same_tags"] = same_tags
        doc["different_tags"] = different_tags

    return recommendation_docs


def _extract_first_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or start >= end:
        return text
    return text[start:end + 1]


def _normalize_search_term(term: str) -> str:
    return re.sub(r"\s+", " ", term).strip().strip('"')


# 中英双向同义词兜底：避免仅靠 LLM 扩展导致某一语言检索召回不足
_BILINGUAL_SYNONYM_MAP = {
    "人工智能": ["artificial intelligence", "ai"],
    "ai": ["人工智能", "机器智能"],
    "机器学习": ["machine learning", "ml"],
    "深度学习": ["deep learning"],
    "神经网络": ["neural network", "neural networks"],
    "自然语言处理": ["natural language processing", "nlp"],
    "计算机视觉": ["computer vision"],
    "注意力机制": ["attention mechanism", "attention"],
    "自注意力": ["self-attention", "self attention"],
    "transformer": ["Transformer模型", "注意力机制"],
    "attention": ["注意力机制", "自注意力"],
}


def _expand_terms_with_bilingual_synonyms(terms: set[str]) -> set[str]:
    expanded = set(terms)

    for term in list(terms):
        normalized = _normalize_search_term(term)
        lower_term = normalized.lower()

        # 精准命中词典键
        if normalized in _BILINGUAL_SYNONYM_MAP:
            expanded.update(_BILINGUAL_SYNONYM_MAP[normalized])
        if lower_term in _BILINGUAL_SYNONYM_MAP:
            expanded.update(_BILINGUAL_SYNONYM_MAP[lower_term])

        # 子串命中（例如“人工智能技术”包含“人工智能”）
        for key, alias_list in _BILINGUAL_SYNONYM_MAP.items():
            key_norm = key.lower()
            if key in normalized or key_norm in lower_term:
                expanded.update(alias_list)

    # 全量再归一化一遍，去除空值与重复
    normalized_terms = set()
    for item in expanded:
        normalized_item = _normalize_search_term(str(item))
        if normalized_item:
            normalized_terms.add(normalized_item)
    return normalized_terms


def _expand_search_terms_with_llm(query: str) -> List[str]:
    normalized_query = query.strip()
    if not normalized_query:
        return []

    terms = {normalized_query}
    try:
        llm_result = ask_messages(
            messages=[
                {"role": "system", "content": KNOWLEDGE_SEARCH_EXPANSION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": KNOWLEDGE_SEARCH_EXPANSION_USER_PROMPT_TEMPLATE.format(query=normalized_query),
                },
            ],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
            extra_payload={
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = _extract_first_json_array((llm_result.content or "").strip())
        candidates = json.loads(content)
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, str):
                    continue
                normalized_term = _normalize_search_term(item)
                if normalized_term:
                    terms.add(normalized_term)
    except (LLMError, json.JSONDecodeError, TypeError, ValueError):
        pass

    terms = _expand_terms_with_bilingual_synonyms(terms)
    return list(terms)[:16]

def rerank_documents_with_llm(query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
    """兜底降级方案：使用现有大语言模型进行相关性重排。"""
    print(f"[LLM Re-rank] 正在使用现有大模型对 {len(docs)} 个文本进行相关性重排...")
    
    system_prompt = """
    # Role
    你是一位精通自然语义理解的搜索相关性专家（Rerank Expert）。你的任务是严谨评估文档（Doc）与用户查询（Query）之间的语义匹配程度。

    # Task
    根据用户提供的 Query 和一系列 Doc，依下列【评分准则】为每个文档打分。

    # Scoring Rubric (0-100)
    - **80-100（高相关）**: 文档完美回答了 Query，包含核心答案或高度匹配的关键信息。
    - **40-79（部分相关）**: 文档主题相关，但仅触及边缘信息，未直接回答核心问题，或存在信息缺失。
    - **0-39（不相关）**: 文档内容偏离主题、存在语义反转（如否定词）、或是毫无关联的干扰信息。

    # Constraints
    1. **语义优先**：关注意图匹配而非单纯的关键词重叠。
    2. **严苛评估**：若文档只是字面相似但逻辑相悖，必须判定为不相关（0-39）。
    3. **输出格式**：禁止输出任何推理过程、解释或开场白。必须严格输出标准的 JSON 字典格式。

    # Output Format
    {"doc_n": score, ...}
    """

    docs_text = []
    for i, d in enumerate(docs):
        preview = str(d["content"])[:400].replace('\n', ' ')
        docs_text.append(f"Doc ID: doc_{i}\nContent: {preview}")
        
    user_prompt = f"Query: {query}\n\nDocuments:\n" + "\n\n".join(docs_text)
    
    try:
        llm_result = ask_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=32768,
            temperature=0.1,
        )
        
        reply_content = llm_result.content
        import json
        import re
        
        json_match = re.search(r'\{.*\}', reply_content, re.DOTALL)
        if json_match:
            score_dict = json.loads(json_match.group())
            for i, d in enumerate(docs):
                key = f"doc_{i}"
                if key in score_dict:
                    d["rerank_score"] = float(score_dict[key])
                else:
                    d["rerank_score"] = 0.0
                    
            reranked_docs = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
            for d in reranked_docs:
                d["score"] = d["rerank_score"]
            return reranked_docs[:top_k]
        else:
            print("[LLM Re-rank] 大模型未返回合法 JSON。")

    except Exception as e:
        print(f"[LLM Re-rank] LLM 联合打分失败: {e}")
        
    print("[LLM Re-rank] 所有重排手段均失败，执行最终降级回退(返回原始排序)。")
    return docs[:top_k]


def rerank_documents(query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
    """直接使用大语言模型对候选文档进行相关性重排。"""
    if not docs:
        return docs

    return rerank_documents_with_llm(query, docs, top_k)

class AddKnowledgeRequest(BaseModel):
    title: str
    content: str
    category: str = None

# --- 2. 添加词条 (核心：入库并自动提取标签) ---
@router.post("/knowledge/add")
def add_knowledge_entry(
    req: AddKnowledgeRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        entry_kwargs = {
            "title": req.title,
            "content": req.content,
            "category": req.category,
        }
        # 兼容历史库结构：只有模型存在 user_id 字段时才写入。
        if hasattr(models.KnowledgeBase, "user_id"):
            entry_kwargs["user_id"] = current_user.id

        # 1. 创建知识库主条目
        new_entry = models.KnowledgeBase(**entry_kwargs)
        db.add(new_entry)
        db.flush()  # 获取自增 ID

        keywords = jieba.analyse.extract_tags(f"{req.title} {req.title} {req.content}", topK=5)

        # 3. 关联标签
        for kw in keywords:
            # 查找标签是否已存在，不存在则创建
            tag = db.query(models.Tag).filter(models.Tag.name == kw).first()
            if not tag:
                tag = models.Tag(name=kw)
                db.add(tag)
                db.flush()
            
            # 建立多对多关联
            if tag not in new_entry.tags:
                new_entry.tags.append(tag)
        
        db.commit()
        return {
            "status": "success", 
            "id": new_entry.id, 
            "extracted_tags": keywords
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add entry: {str(e)}")


@router.get("/knowledge/search")
def search_knowledge_robust(q: str, db: Session = Depends(get_db)):
    search_terms = _expand_search_terms_with_llm(q)
    search_payload = " ".join([f'"{term}"' for term in search_terms])
    print(f"Expanded search terms: {search_terms}, payload: {search_payload}")

    # 第 1 步：粗筛（高召回），多拿一些数据（20条），不管是不是字面上刚好撞车的
    sql = text("""
        SELECT id, title, authors, year, content,
            (
                (MATCH(title) AGAINST(:payload IN BOOLEAN MODE) * 5) + 
                (MATCH(content) AGAINST(:payload IN BOOLEAN MODE) * 1)
            ) AS score
        FROM knowledge_base
        WHERE MATCH(title, content) AGAINST(:payload IN BOOLEAN MODE)
        ORDER BY score DESC
        LIMIT 20
    """)

    result = db.execute(sql, {"payload": search_payload}).all()

    # 将查询结果转成字典列表以便后续 LLM 重排处理
    docs_to_rerank = [
        {
            "id": r.id, 
            "title": r.title, 
            "score": round(r.score, 2),
            "authors": r.authors,
            "year": r.year,
            "content": r.content
        } for r in result
    ]

    # 第 2 步：精排（高精度），用轻量级 LLM/重排模型剔除“字面相似语义相反”的内容，挑选 Top 5
    final_docs = rerank_documents(query=q, docs=docs_to_rerank, top_k=10)

    # 返回精选后的内容并裁剪预览
    for doc in final_docs:
        doc["content"] = doc["content"][:200]
        
    return final_docs


@router.get("/knowledge/recommend")
def recommend_similar_multiple(
    kb_ids: list[int] = Query(...), # 接收类似 ?kb_ids=1&kb_ids=2 的参数
    db: Session = Depends(get_db), 
    limit: int = 10
):
    """
    推荐逻辑：输入文章 ID 列表，寻找与这些文章标签重合度最高的内容
    """
    if not kb_ids:
        return []

    sql = text("""
        SELECT r2.kb_id, k.title, k.authors, k.year, COUNT(*) as common_tags_count
        FROM kb_tag_relation r1
        JOIN kb_tag_relation r2 ON r1.tag_id = r2.tag_id
        JOIN knowledge_base k ON r2.kb_id = k.id
        WHERE r1.kb_id IN :ids           -- 匹配列表中的任何一篇文章的标签
          AND r2.kb_id NOT IN :ids      -- 排除掉列表本身的文章
        GROUP BY r2.kb_id, k.title, k.authors, k.year
        ORDER BY common_tags_count DESC
        LIMIT :limit
    """)
    
    result = db.execute(sql, {"ids": tuple(kb_ids), "limit": limit}).all()

    docs = [
        {
            "id": r.kb_id, 
            "title": r.title, 
            "authors": r.authors,
            "year": r.year,
            "common_tags": r.common_tags_count
        } 
        for r in result
    ]

    return _enrich_recommendations_with_tag_diff(db, docs, kb_ids)


@router.get("/knowledge/recommend/{kb_id}")
def recommend_similar_by_tags(
    kb_id: int,
    db: Session = Depends(get_db),
    limit: int = 10,
):
    """直接暴露 models.KBService 中基于 MySQL 标签重合度的单篇推荐逻辑。"""
    result = models.KBService.recommend_similar(db, kb_id, limit)
    docs = [
        {
            "id": row.id,
            "title": row.title,
            "authors": row.authors,
            "year": row.year,
            "common_tags": row.common_tags,
        }
        for row in result
    ]
    return _enrich_recommendations_with_tag_diff(db, docs, [kb_id])


@router.get("/knowledge/content/{kb_id}")
def get_knowledge_content(kb_id: int, db: Session = Depends(get_db)):
    kb_entry = db.query(models.KnowledgeBase).filter(models.KnowledgeBase.id == kb_id).first()
    if not kb_entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")

    return {
        "id": kb_entry.id,
        "title": kb_entry.title,
        "authors": kb_entry.authors,
        "year": kb_entry.year,
        "content": kb_entry.content,
    }

@router.get("/knowledge/file/{file_id}")
def get_knowledge_file(file_id: int, db: Session = Depends(get_db)):
    # 1. 从数据库查找记录
    kb_entry = db.query(models.KnowledgeBase).filter(models.KnowledgeBase.id == file_id).first()
    if not kb_entry:
        raise HTTPException(status_code=404, detail="File not found in database")
    
    # 2. 检查是否有文件路径
    if not kb_entry.file_path:
        raise HTTPException(status_code=404, detail="No file path associated with this entry")
        
    # 3. 检查文件物理路径是否存在（兼容绝对/相对路径）
    file_path = _resolve_existing_file_path(kb_entry.file_path)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found on disk")
        
    # 4. 准备文件名
    # 优先使用数据库中的 title 加上原文件的扩展名
    original_filename = os.path.basename(file_path)
    file_ext = os.path.splitext(original_filename)[1]
    
    download_filename = original_filename
    if kb_entry.title:
        # 清理 title 中的非法文件字符
        safe_title = re.sub(r'[\\/*?:"<>|]', "", kb_entry.title)
        download_filename = f"{safe_title}{file_ext}"
        
    return FileResponse(path=file_path, filename=download_filename)
