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

    return list(terms)[:8]

def rerank_documents_with_llm(query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
    """兜底降级方案：使用已有大语言模型作为 Reranker (LLM-as-a-Judge)"""
    print(f"[Reranker - Fallback] 正在使用现有大模型(LLM)对 {len(docs)} 个文本进行深度语义重排...")
    
    system_prompt = """你是一个智能的搜索相关性重排专家。
对于用户的查询（Query），你需要评估后续提供的几个文档片段（Doc）与查询的相关性。
请给每个文档打分，分数范围在 0 到 100 之间。
相关：直接回答了问题或包含关键信息，打 80-100 分。
部分相关：有关联但没有直接回答，打 40-79 分。
不相关：字面一样但语义反转，或者是无关内容，打 0-39 分。
请你强制输出合法的 JSON 字典，键为文档ID（如 "doc_0"，"doc_1"），值为整数分数。不要输出任何除了 JSON 之外的其他分析或废话。
示例格式：{"doc_0": 85, "doc_1": 10, "doc_2": 95}"""

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
            max_tokens=600,
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
            print("[Reranker - Fallback] 大模型未返回合法 JSON。")

    except Exception as e:
        print(f"[Reranker - Fallback] LLM 联合打分失败: {e}")
        
    print("[Reranker] 所有重排手段均失败，执行最终降级回退(返回原始排序)。")
    return docs[:top_k]


def rerank_documents(query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
    # ================= 优化 5：rerank重排 =================
    """主引流入口：优先尝试专属 Reranker API，如果失败则走 LLM 降级"""
    if not docs:
        return docs
        
    api_key = os.getenv("RERANKER_API_KEY", "").strip('\"\'') 
    api_base = os.getenv("RERANKER_API_BASE", "https://api.siliconflow.cn/v1/rerank").strip('\"\'')
    model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip('\"\'')

    if api_key and api_key != "your_siliconflow_api_key_here":
        try:
            print(f"[Reranker] 尝试请求专属重排模型API进行首选打分 ({model})...")
            import requests
            texts = [str(d["content"])[:512].replace('\n', ' ') for d in docs]
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "query": query,
                "documents": texts
            }
            
            resp = requests.post(api_base, json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                results.sort(key=lambda x: x["index"])
                
                for i, r in enumerate(results):
                    docs[i]["rerank_score"] = float(r["relevance_score"])
                    
                reranked_docs = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
                for d in reranked_docs:
                    # 保留两位小数给前端
                    d["score"] = round(d["rerank_score"], 4) 
                print("[Reranker] 专属API重排成功！")
                return reranked_docs[:top_k]
            else:
                print(f"[Reranker] 专属API返回错误: HTTP {resp.status_code} - {resp.text}，准备降级。")
        except Exception as e:
            print(f"[Reranker] 专属API调用异常: {e}，准备降级。")
    else:
        print("[Reranker] 专属API未配置或者为空，跳过API重排。")
        
    # 如果API失败，或者上面的逻辑没有return掉，则进入大模型打分
    return rerank_documents_with_llm(query, docs, top_k)

# --- 1. 测试连接 (保留并增强) ---
@router.get("/db-test")
def test_db_connection(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "success", "message": "Database connection established!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

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

    # 第 1 步：粗筛（高召回），为避免 MySQL 引擎对 MATCH 浮点数进行加权运算时出现 DOUBLE out of range 越界 Bug，
    # 我们将单独获取两部分的分数，然后再在 Python 代码层进行归一化及加权运算和重新排序。
    sql = text("""
        SELECT id, title, authors, year, content,
            MATCH(title) AGAINST(:payload IN BOOLEAN MODE) AS title_score,
            MATCH(content) AGAINST(:payload IN BOOLEAN MODE) AS content_score
        FROM knowledge_base
        WHERE MATCH(title, content) AGAINST(:payload IN BOOLEAN MODE)
        ORDER BY MATCH(title, content) AGAINST(:payload IN BOOLEAN MODE) DESC
        LIMIT 40
    """)

    result = db.execute(sql, {"payload": search_payload}).all()

    # 将查询结果转成字典列表并在此处计算实际加权分数，随后进行重新排序选出前 20 条
    docs_to_rerank = []
    for r in result:
        # Title score 的权重为 5，content score 的权重为 1
        weighted_score = (r.title_score * 5.0) + (r.content_score * 1.0)
        docs_to_rerank.append({
            "id": r.id, 
            "title": r.title, 
            "score": round(weighted_score, 4),
            "authors": r.authors,
            "year": r.year,
            "content": r.content
        })
    
    # 根据归一化加权后的分数进行降序排序，取前 20 条喂给后续的精排模型
    docs_to_rerank.sort(key=lambda x: x["score"], reverse=True)
    docs_to_rerank = docs_to_rerank[:20]

    # 第 2 步：精排（高精度），用轻量级 LLM/重排模型剔除“字面相似语义相反”的内容，挑选 Top 5
    final_docs = rerank_documents(query=q, docs=docs_to_rerank, top_k=5)

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