import os
from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase

print("🚀 正在预创建 Neo4j 索引...")
uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "Fuchen20050420")
idx_name = os.getenv("GRAPHRAG_VECTOR_INDEX_NAME", "chunk_vector_index")
dim = int(os.getenv("GRAPHRAG_EMBEDDING_DIMENSIONS", 1536))

driver = GraphDatabase.driver(uri, auth=(user, pwd))

try:
    from neo4j_graphrag.indexes import create_vector_index
    create_vector_index(
        driver, 
        idx_name, 
        label="Chunk", 
        embedding_property="embedding", 
        dimensions=dim, 
        similarity_fn="cosine"
    )
    print("✅ 矢量索引已就绪！")
except Exception as e:
    print("索引创建异常:", e)

from graphrag.graphrag_service import GraphRAGService, _extract_triplets_from_text

print("🚀 启动 GraphRAGService...")
service = GraphRAGService()
service.initialize()

text_zh = '''干预、一次自然冲击、一个历史事件或者经济主体的一次选择行为。
基本因果模型III 则表明，这种相关性可能是因为Y 对D 的反向影响( Ｒeverse Causality) 。如果在特定的研究情境下，变量之间满足一定的假设条件，使得一个特定的因果模型没有与之竞争的、观测上等价( Observationally Equivalent) 的因果模型，则称这个特定的因果模型被识别，这样的假设被称作因果识别假设。简言之，因果识别( Causal Identification) 就是在一定的假设条件下，利用样本数据来推断出总体的因果关系或潜在因果模型的过程。
因此，任何一个因果识别策略均依赖于一定的识别假设。在利用相关性进行因果推断时，最直接的问题便是: 在给定数据以及所研究的现象和情境下，因果推断的假设能否满足？'''

print("🧠 正在提取中文关系...")
zh_relations = _extract_triplets_from_text(text_zh)
zh_entities = []
for r in zh_relations:
    if "source" in r and "target" in r:
        zh_entities.append({"name": r["source"], "type": "Entity"})
        zh_entities.append({"name": r["target"], "type": "Entity"})

row_zh = {
    "paper_id": 21000,
    "title": "测试文档-21(中文)",
    "year": 2024,
    "chunk_id": "test_chunk_zh",
    "text": text_zh,
    "index": 0,
    "relations": zh_relations,
    "entities": zh_entities
}

print("✨ 正在将【三元组关系网】注入 Neo4j 数据库...")
res = service.upsert_paper_chunks([row_zh])
print(f"✅ 注入完成！受影响记录：{res}")
print("🎉 现在，请打开 Neo4j 浏览器 (http://localhost:7474) 并执行以下查询来查看你的知识星空网：")
print("MATCH (n)-[r:RELATED_TO]->(m) RETURN n, r, m")