import arxiv
import os

def download_arxiv_papers(query, max_results=5, pdf_dir=None, bib_dir=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = pdf_dir or os.path.join(project_root, "database")
    bib_dir = bib_dir or os.path.join(project_root, "bibs")

    # 创建保存目录
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(bib_dir, exist_ok=True)

    # 1. 初始化查询
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate # 按发布时间排序
    )

    for idx, result in enumerate(search.results(), start=1):
        paper_id = result.get_short_id()
        
        print(f"正在处理: {result.title}")

        # 2. 下载 PDF
        pdf_filename = f"{idx}.pdf"
        result.download_pdf(dirpath=pdf_dir, filename=pdf_filename)
        print(f"  [√] PDF 已保存到: {pdf_dir}")

        # 3. 生成并保存 BibTeX
        # 注意：arXiv API不直接返回标准BibTeX，但我们可以根据返回字段手动拼接
        bib_content = f"""@article{{{paper_id},
            title={{ {result.title} }},
            author={{ {', '.join([a.name for a in result.authors])} }},
            journal={{arXiv preprint arXiv:{paper_id}}},
            year={{ {result.published.year} }},
            url={{ {result.entry_id} }}
        }}
        """
        bib_filename = os.path.join(bib_dir, f"{idx}.bib")
        with open(bib_filename, "w", encoding="utf-8") as f:
            f.write(bib_content)
        print(f"  [√] BibTeX 已保存到: {bib_dir}")

# 执行搜索：例如搜索 AI 相关的论文
download_arxiv_papers("cat:cs.AI AND ti:machine learning", max_results=200)