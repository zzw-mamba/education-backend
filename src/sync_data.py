"""论文数据同步工具模块

提供从 BibTeX 文件和 PDF 目录批量导入论文数据到数据库的功能。

主要功能：
- 解析 BibTeX 文件提取论文元数据
- 使用 UniParse API 提取 PDF 内容
- 自动生成标签（基于关键词提取）
- 按批次并发处理，提高导入效率
"""

import os
import re
from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed
import bibtexparser
import jieba.analyse
from sqlalchemy.orm import Session

from database import SessionLocal
from models import KnowledgeBase, Tag
from routers.ocr import parse_file_with_uniparse

def clean_bib_text(text: str) -> str:
    """清理 BibTeX 中的花括号。

    Args:
        text: 原始 BibTeX 字段值

    Returns:
        清理后的文本
    """
    if not text:
        return ""
    return text.replace('{', '').replace('}', '')

def extract_pdf_info(pdf_path: str) -> tuple[str, str]:
    """提取 PDF 内容。

    使用 UniParse API 解析 PDF 文件，返回全文内容和用于标签提取的核心内容。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        元组 (全文内容, 用于提取标签的前3000字符内容)
    """
    try:
        full_text = parse_file_with_uniparse(pdf_path)
        core_text = full_text[:3000]
    except Exception as e:
        print(f"OCR 解析 PDF 失败 {pdf_path}: {e}")
        full_text = ""
        core_text = ""
    return full_text, core_text

def parse_year(year_str: str) -> int | None:
    """从字符串中提取 4 位数字年份。

    支持格式: {2023}, 2023-10, May 2023
    排除不足 4 位的数字（如 23）。

    Args:
        year_str: 包含年份的字符串

    Returns:
        提取的年份整数，未找到时返回 None
    """
    if not year_str:
        return None
    match = re.search(r'\d{4}', str(year_str))
    return int(match.group()) if match else None


def _process_single_paper(bibs_dir: str, pdfs_dir: str, pdf_file: str) -> str:
    """处理单篇论文（用于并发任务）。

    处理流程：
    1. 从文件名提取论文 ID
    2. 解析对应的 BibTeX 文件获取元数据
    3. 检查数据库是否已存在（避免重复导入）
    4. 使用 UniParse 提取 PDF 内容
    5. 写入数据库并自动生成标签

    Args:
        bibs_dir: BibTeX 文件所在目录
        pdfs_dir: PDF 文件所在目录
        pdf_file: PDF 文件名

    Returns:
        处理结果状态字符串
    """
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


def sync_papers(db: Session, bibs_dir: str, pdfs_dir: str, max_workers: int = 10, batch_size: int = 20) -> None:
    """同步 BibTeX 和 PDF 文件到数据库。

    按批次并发处理论文数据，提高导入效率。
    每篇论文独立使用数据库会话，避免并发冲突。

    Args:
        db: SQLAlchemy 数据库会话（仅用于兼容旧签名，实际使用独立会话）
        bibs_dir: BibTeX 文件所在目录
        pdfs_dir: PDF 文件所在目录
        max_workers: 并发处理的最大线程数，默认为 10
        batch_size: 每批次处理的论文数量，默认为 20
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