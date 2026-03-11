import os
import re
import fitz  # PyMuPDF
import bibtexparser
import jieba.analyse
from sqlalchemy.orm import Session
from database import SessionLocal
from models import KnowledgeBase, Tag, KBTagRelation, Log  # 确保导入你的模型

def clean_bib_text(text):
    """清理 BibTeX 中的花括号"""
    if not text:
        return ""
    return text.replace('{', '').replace('}', '')

def extract_pdf_info(pdf_path):
    """
    提取 PDF 内容
    返回: (全文内容, 用于提取标签的前3页内容)
    """
    full_text = ""
    core_text = ""
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            page_text = page.get_text()
            full_text += page_text
            if i < 3:  # 仅提取前3页作为核心内容
                core_text += page_text
        doc.close()
    except Exception as e:
        print(f"读取 PDF 失败 {pdf_path}: {e}")
    return full_text, core_text

def parse_year(year_str):
    """
    从字符串中提取 4 位数字年份。
    支持格式: {2023}, 2023-10, May 2023, 23 (排除)
    """
    if not year_str:
        return None
    match = re.search(r'\d{4}', str(year_str))
    return int(match.group()) if match else None

def sync_papers(db: Session, bibs_dir: str, pdfs_dir: str):
    """
    同步 BibTeX 和 PDF 到数据库，包含 year 属性处理
    """
    if not os.path.exists(pdfs_dir):
        print(f"❌ 错误: 找不到 PDF 文件夹 {pdfs_dir}")
        return

    # 获取所有 PDF 文件
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]
    
    for pdf_file in pdf_files:
        paper_id = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        bib_path = os.path.join(bibs_dir, f"{paper_id}.bib")
        
        print(f"🔍 正在处理: {paper_id}...")

        # 1. 默认元数据
        title = paper_id
        authors = ""
        year_val = None
        
        # 2. 从 BibTeX 读取信息
        if os.path.exists(bib_path):
            try:
                with open(bib_path, encoding='utf-8') as b_file:
                    bib_db = bibtexparser.load(b_file)
                    if bib_db.entries:
                        entry = bib_db.entries[0]
                        title = clean_bib_text(entry.get('title', paper_id))
                        authors = clean_bib_text(entry.get('author', ''))
                        # 使用正则解析年份
                        year_val = parse_year(entry.get('year', ''))
            except Exception as e:
                print(f"  ⚠️ 解析 BibTeX 失败 ({paper_id}): {e}")
        else:
            print(f"  ⚠️ 未找到 Bib 文件: {paper_id}.bib")

        # 3. 检查数据库是否已存在 (避免重复导入)
        existing = db.query(KnowledgeBase).filter(KnowledgeBase.title == title).first()
        if existing:
            print(f"  ⏭️ 跳过: {title} 已存在")
            continue

        # 4. 提取 PDF 内容 (假设你已有 extract_pdf_info 函数)
        try:
            # full_text: 用于全文检索, core_text: 用于生成标签
            full_text, core_text = extract_pdf_info(pdf_path) 
        except Exception as e:
            print(f"  ❌ PDF 提取失败: {e}")
            continue

        # 5. 写入数据库
        try:
            new_entry = KnowledgeBase(
                title=title,
                content=full_text,
                authors=authors,
                year=year_val,          # 新增属性
                file_path=pdf_path,
                file_type="pdf",
                category="Paper"
            )
            db.add(new_entry)
            db.flush()  # 生成自增 ID

            # 6. 自动标签生成 (基于标题加权)
            tag_source = f"{title} {title} {core_text}"
            keywords = jieba.analyse.extract_tags(tag_source, topK=5)

            for kw in keywords:
                # 检查标签池
                tag = db.query(Tag).filter(Tag.name == kw).first()
                if not tag:
                    tag = Tag(name=kw)
                    db.add(tag)
                    db.flush()
                
                # 建立多对多关联
                if tag not in new_entry.tags:
                    new_entry.tags.append(tag)
            
            db.commit()
            print(f"  ✅ 成功入库: {title} ({year_val if year_val else '未知年份'})")

        except Exception as e:
            db.rollback()
            print(f"  ❌ 数据库写入失败: {e}")


if __name__ == "__main__":
    # 配置你的路径
    BIB_FOLDER = os.path.join("..", "bibs")
    PDF_FOLDER = os.path.join("..", "database")
    
    # 启动同步
    db_session = SessionLocal()
    try:
        sync_papers(db_session, BIB_FOLDER, PDF_FOLDER)
    finally:
        db_session.close()