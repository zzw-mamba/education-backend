import jieba.analyse
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import os

from database import get_db
import models  # 确保你之前的 User, KnowledgeBase, Tag 等模型都在这里
from deep_translator import GoogleTranslator
from nltk.corpus import wordnet
import re

router = APIRouter(tags=["Database"])

# --- 1. 测试连接 (保留并增强) ---
@router.get("/db-test")
def test_db_connection(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "success", "message": "Database connection established!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# --- 2. 添加词条 (核心：入库并自动提取标签) ---
@router.post("/knowledge/add")
def add_knowledge_entry(
    title: str, 
    content: str, 
    user_id: int, 
    category: str = None, 
    db: Session = Depends(get_db)
):
    try:
        # 1. 创建知识库主条目
        new_entry = models.KnowledgeBase(
            user_id=user_id,
            title=title,
            content=content,
            category=category
        )
        db.add(new_entry)
        db.flush()  # 获取自增 ID

        keywords = jieba.analyse.extract_tags(f"{title} {title} {content}", topK=5)

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
    def get_wordnet_expansions(word_en):
        synonyms = []
        for syn in wordnet.synsets(word_en)[:2]:
            for lemma in syn.lemmas():
                synonyms.append(lemma.name().replace('_', ' '))
        return list(set(synonyms))[:5]

    def expand_search_terms(q: str):
        terms = {q.strip()}
        try:
            if re.search(r'[\u4e00-\u9fa5]', q):
                translated = GoogleTranslator(source='zh-CN', target='en').translate(q)
                print(translated)
                for w in get_wordnet_expansions(translated):
                    terms.add(w)

            else:
                translated = GoogleTranslator(source='en', target='zh-CN').translate(q)
                for w in get_wordnet_expansions(q):
                    terms.add(w)

            terms.add(translated.strip())
        except:
            pass

        return list(terms)
    
    search_terms = expand_search_terms(q)
    print(search_terms)
    search_payload = " ".join([f'"{term}"' for term in search_terms])

    sql = text("""
        SELECT id, title, authors, year,
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

    return [
        {
            "id": r.id, 
            "title": r.title, 
            "score": round(r.score, 2),
            "authors": r.authors,
            "year": r.year
        } for r in result
    ]


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
    
    return [
        {
            "id": r.kb_id, 
            "title": r.title, 
            "authors": r.authors,
            "year": r.year,
            "common_tags": r.common_tags_count
        } 
        for r in result
    ]

@router.get("/knowledge/file/{file_id}")
def get_knowledge_file(file_id: int, db: Session = Depends(get_db)):
    # 1. 从数据库查找记录
    kb_entry = db.query(models.KnowledgeBase).filter(models.KnowledgeBase.id == file_id).first()
    if not kb_entry:
        raise HTTPException(status_code=404, detail="File not found in database")
    
    # 2. 检查是否有文件路径
    if not kb_entry.file_path:
        raise HTTPException(status_code=404, detail="No file path associated with this entry")
        
    # 3. 检查文件物理路径是否存在
    # 如果路径是相对路径，确保主要是相对于运行目录
    file_path = kb_entry.file_path
    if not os.path.isabs(file_path):
         # 如果是相对路径，可以尝试根据项目根目录拼接（视具体运行方式而定，暂时直接使用）
         pass

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found on disk: {file_path}")
        
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