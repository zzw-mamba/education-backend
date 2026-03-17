# -*- coding: utf-8 -*-
import os
import sys
import fitz

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graphrag.graphrag_service import GraphRAGService, _extract_triplets_from_text

from dotenv import load_dotenv
load_dotenv()

def extract_pdf_snippet(pdf_name, start_idx, length):
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "database", pdf_name)
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc[:5]])
        doc.close()
        return text[start_idx : start_idx+length]
    except Exception as e:
        print(f"Error reading {pdf_name}: {e}")
        return ""

print("\n" + "="*60)
print("🕸️ GraphRAG 核心杀器测试：【知识图谱三元组抽取】")
print("="*60 + "\n")

# 0. 初始化 GraphRAG 服务，准备写入 Neo4j
print("\n🔧 正在启动 Neo4j 服务并校准向量索引...")
service = GraphRAGService()
service.setup_local_database(create_vector_index=True)
service.initialize()
chunks_to_upsert = []

# 1. 测试 21.pdf (中文中文环境)
zh_text = extract_pdf_snippet("21.pdf", 2500, 600)
if zh_text:
    print("\n[📄 中文文献 21.pdf 片段]")
    print(zh_text[:150] + "...\n")
    print("🧠 正在调用模型抽取关系...")
    zh_result = _extract_triplets_from_text(zh_text)
    print("\n✅ 中文抽取结果 (三元组 Array):")
    
    zh_entities = []
    for t in zh_result[:5]:
        print(f"   {t['source']} --[{t['relation']}]--> {t['target']}")
    
    for r in zh_result:
        if "source" in r and "target" in r:
            zh_entities.append({"name": r["source"], "type": "Entity"})
            zh_entities.append({"name": r["target"], "type": "Entity"})
            
    chunks_to_upsert.append({
        "paper_id": 21000,
        "title": "测试文档-21(中文)",
        "year": 2024,
        "chunk_id": "test_chunk_zh",
        "text": zh_text,
        "index": 0,
        "relations": zh_result,
        "entities": zh_entities
    })

print("\n" + "-"*60)

# 2. 测试 1.pdf (英文环境)
en_text = extract_pdf_snippet("1.pdf", 2000, 1000)
if en_text:
    print("\n[📄 英文文献 1.pdf 片段]")
    print(en_text[:150] + "...\n")
    print("🧠 正在调用模型抽取关系...")
    en_result = _extract_triplets_from_text(en_text)
    print("\n✅ 英文抽取结果 (三元组 Array):")
    
    en_entities = []
    for t in en_result[:5]:
        print(f"   {t['source']} --[{t['relation']}]--> {t['target']}")
        
    for r in en_result:
        if "source" in r and "target" in r:
            en_entities.append({"name": r["source"], "type": "Entity"})
            en_entities.append({"name": r["target"], "type": "Entity"})
            
    chunks_to_upsert.append({
        "paper_id": 10000,
        "title": "测试文档-1(英文)",
        "year": 2024,
        "chunk_id": "test_chunk_en",
        "text": en_text,
        "index": 0,
        "relations": en_result,
        "entities": en_entities
    })

print("\n" + "="*60)

if chunks_to_upsert:
    print("✨ 正在将【双语三元组关系网】注入 Neo4j 数据库...")
    res = service.upsert_paper_chunks(chunks_to_upsert)
    print(f"✅ 注入完成！共影响/创建图谱节点与文本块：{res}")
    print("🎉 请进入 Neo4j 浏览器 (http://localhost:7474) 执行：MATCH (n)-[r:RELATED_TO]->(m) RETURN n, r, m")
