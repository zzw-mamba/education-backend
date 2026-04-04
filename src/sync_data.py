import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import bibtexparser
import jieba.analyse
from sqlalchemy.orm import Session
from database import SessionLocal
from models import KnowledgeBase, Tag, KBTagRelation, Log  # 确保导入你的模型
from routers.ocr import parse_file_with_uniparse

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
    try:
        full_text = parse_file_with_uniparse(pdf_path)
        core_text = full_text[:3000]
    except Exception as e:
        print(f"OCR 解析 PDF 失败 {pdf_path}: {e}")
        full_text = ""
        core_text = ""
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


def _process_single_paper(bibs_dir: str, pdfs_dir: str, pdf_file: str):
    """处理单篇论文（用于并发任务）。"""
    db = SessionLocal()
    try:
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
                with open(bib_path, encoding="utf-8") as b_file:
                    bib_db = bibtexparser.load(b_file)
                    if bib_db.entries:
                        entry = bib_db.entries[0]
                        title = clean_bib_text(entry.get("title", paper_id))
                        authors = clean_bib_text(entry.get("author", ""))
                        year_val = parse_year(entry.get("year", ""))
            except Exception as e:
                print(f"  ⚠️ 解析 BibTeX 失败 ({paper_id}): {e}")
        else:
            print(f"  ⚠️ 未找到 Bib 文件: {paper_id}.bib")

        # 3. 检查数据库是否已存在 (避免重复导入)
        existing = db.query(KnowledgeBase).filter(KnowledgeBase.title == title).first()
        if existing:
            return f"  ⏭️ 跳过: {title} 已存在"

        # 4. 提取 PDF 内容
        try:
            full_text, core_text = extract_pdf_info(pdf_path)
        except Exception as e:
            return f"  ❌ PDF 提取失败 ({paper_id}): {e}"

        # 5. 写入数据库
        try:
            new_entry = KnowledgeBase(
                title=title,
                content=full_text,
                authors=authors,
                year=year_val,
                file_path=pdf_path,
                file_type="pdf",
                category="Paper",
            )
            db.add(new_entry)
            db.flush()

            # 6. 自动标签生成 (基于标题加权)
            tag_source = f"{title} {title} {core_text}"
            keywords = jieba.analyse.extract_tags(tag_source, topK=5)

            for kw in keywords:
                tag = db.query(Tag).filter(Tag.name == kw).first()
                if not tag:
                    tag = Tag(name=kw)
                    db.add(tag)
                    db.flush()

                if tag not in new_entry.tags:
                    new_entry.tags.append(tag)

            db.commit()
            return f"  ✅ 成功入库: {title} ({year_val if year_val else '未知年份'})"
        except Exception as e:
            db.rollback()
            return f"  ❌ 数据库写入失败 ({paper_id}): {e}"
    finally:
        db.close()


def sync_papers(db: Session, bibs_dir: str, pdfs_dir: str, max_workers: int = 10, batch_size: int = 20):
    """
    同步 BibTeX 和 PDF 到数据库，包含 year 属性处理。
    改为按批次并发处理，提高处理吞吐。
    """
    if not os.path.exists(pdfs_dir):
        print(f"❌ 错误: 找不到 PDF 文件夹 {pdfs_dir}")
        return

    # 获取所有 PDF 文件
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("ℹ️ 未发现可处理的 PDF 文件")
        return

    # 传入的 db 仅为兼容旧签名；并发时每个任务使用独立 Session。
    _ = db

    total = len(pdf_files)
    print(f"🚀 开始批处理: 共 {total} 篇, 并发 {max_workers}, 批大小 {batch_size}")

    for i in range(0, total, batch_size):
        batch = pdf_files[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"\n📦 第 {batch_no} 批: {len(batch)} 篇")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_single_paper, bibs_dir, pdfs_dir, pdf_file)
                for pdf_file in batch
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"  ❌ 并发任务异常: {e}")


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