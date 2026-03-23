import os
import sys
import time
from dotenv import load_dotenv

load_dotenv() 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graphrag.graphrag_service import LocalEmbeddings
from graphrag.graphrag_config import get_graphrag_settings

def test_embedding_cache_and_batch_speedup():
    print("\n==================================================")
    print(" 🚀 [测试] 批量 Embedding 与本地持久化缓存 加速比对")
    print("==================================================\n")

    settings = get_graphrag_settings()
    
    # 初始化改造后的带有缓存和批处理的 Embedding 服务
    embedder = LocalEmbeddings(
        base_url=settings.local_embedding_base_url,
        api_path=settings.local_embedding_api_path,
        model=settings.embedding_model,
        timeout=settings.local_embedding_timeout
    )
    
    # 强制清理缓存，防止之前的测试数据干扰
    with sqlite3.connect(embedder.cache_db_path) as conn:
        conn.execute("DELETE FROM embeddings")

    # 制造出批量请求供测试
    test_texts = [
        f"这是用于测试 Embedding 批处理加速系统的第 {i} 句短小的科研推导文本片段。具有足够随机性防止短路。" 
        for i in range(20)
    ]
    
    print("【第一阶段：使用单条逐个请求(模拟旧版实现)】")
    t_single_start = time.time()
    for text in test_texts:
        embedder.embed_query(text)
    t_single_end = time.time()
    print(f"   -> 逐个调用 20 次 embed_query，耗时：{t_single_end - t_single_start:.4f} 秒。")

    # 再次清理缓存
    with sqlite3.connect(embedder.cache_db_path) as conn:
        conn.execute("DELETE FROM embeddings")

    print("\n【第二阶段：使用新版批量请求(Batch)】")
    t_batch_start = time.time()
    # 这一次性发送20条给服务端
    embedder.embed_documents(test_texts)
    t_batch_end = time.time()
    print(f"   -> 一次性 embed_documents 20 条，耗时：{t_batch_end - t_batch_start:.4f} 秒。")
    
    print("\n【第三阶段：二次读取（触发缓存检查）】")
    t2 = time.time()
    res2 = embedder.embed_documents(test_texts)
    t3 = time.time()
    
    print(f"   -> 成功从本地 SQLite `.embedding_cache.db` 极速恢复 {len(res2)} 条向量，耗时：{t3 - t2:.4f} 秒。")
    
    print("\n【结论分析】")
    if t_single_end - t_single_start > 0:
        batch_speedup = (t_single_end - t_single_start) / max(t_batch_end - t_batch_start, 0.0001)
        print(f"⚡ 批量请求对比单条请求速度提升了 {batch_speedup:.1f} 倍！网络 IO 开销被大幅削减。")
        
    if t_batch_end - t_batch_start > 0:
        cache_speedup = (t_batch_end - t_batch_start) / max(t3 - t2, 0.0001)
        print(f"💥 缓存读取对比批量请求提速达 {cache_speedup:.1f} 倍！")

if __name__ == "__main__":
    import sqlite3
    test_embedding_cache_and_batch_speedup()
