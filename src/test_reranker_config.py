import os
import sys

# 手动让脚本找到上一级的 config 文件 (如果有需要的话)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# 加载你的 .env 配置
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 导入真正的路由里的被测试函数
from routers.db_routes import rerank_documents

def test_rerank_flow():
    print("="*60)
    print("🚀 开始进行 Reranker 双重机制测试")
    print("="*60)
    
    # 打印当前环境变量，确认 .env 已加载
    api_key = os.getenv("RERANKER_API_KEY")
    api_base = os.getenv("RERANKER_API_BASE")
    model = os.getenv("RERANKER_MODEL")
    print(f"[配置检查] RERANKER_API_KEY : {'已填入 (被遮蔽)' if api_key else '未填入或读取失败'}")
    print(f"[配置检查] RERANKER_API_BASE: {api_base}")
    print(f"[配置检查] RERANKER_MODEL   : {model}")
    print("-" * 60)

    # 1. 模拟数据：准备3条假文档（1条非常匹配，1条假匹配，1条毫不相关）
    query = "如何使用人工智能来提升在线教育的效果？"
    mock_docs = [
        {
            "id": 1,
            "title": "无关新闻",
            "content": "昨天晚上我们去吃了一顿火锅，虽然餐厅里有一个人工智能点餐机器人，但是饭菜味道一般没有吃饱。",
            "score": 5.0
        },
        {
            "id": 2,
            "title": "字面相关但意思相反",
            "content": "很多专家认为人工智能不能提升在线教育的效果。教育不应该被机器干预，如何使用机器都是不可取的。",
            "score": 10.0
        },
        {
            "id": 3,
            "title": "完美答案",
            "content": "在在线教育中，利用人工智能可以通过个性化推荐、自适应做题系统来极大提升学生的学习专注度和学习效果。",
            "score": 3.0
        }
    ]

    print("\n[待排序文档原始状态]: ")
    for doc in mock_docs:
        print(f" -> 文档ID:{doc['id']}, 原始分数: {doc['score']}, 预览: {doc['content'][:25]}...")

    # 2. 正式调用我们的业务函数（它会自动走: 专用 API -> 如果失败走现有 LLM -> 如果再失败按原样返回）
    print("\n[开始执行 Rerank 流程...]")
    ranked_results = rerank_documents(query, mock_docs, top_k=3)

    # 3. 结果验证
    print("\n" + "="*60)
    print("✅ Rerank 环节结束，最终排序结果如下：")
    print("="*60)
    for i, doc in enumerate(ranked_results):
        print(f"🥇 第 {i+1} 名")
        print(f"    - ID: {doc['id']}")
        print(f"    - 标题: {doc['title']}")
        print(f"    - 新打分 (rerank_score/score): {doc.get('score')}")
        print(f"    - 预览: {doc['content'][:45]}...")
        print("")

    if ranked_results and ranked_results[0]['id'] == 3:
        print("🎉 [测试通过] ID为3的完美答案成功被重排到了第一名！这证明您的机制生效了。")
    else:
        print("⚠️ [测试异常] 完美答案没有排到第一，请检查报错日志。")

if __name__ == "__main__":
    test_rerank_flow()