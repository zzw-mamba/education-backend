import json
import time
import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

try:
    from prompt import MATERIAL_PARSING_PROMPT
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prompt import MATERIAL_PARSING_PROMPT

load_dotenv()
LLM_API_BASE = os.getenv("LLM_API_BASE")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 60.0))
MODEL_NAME = os.getenv("MODEL_NAME")

def call_llm_api(task_name, content):
    """
    调用大模型 API 进行处理
    """
    if not LLM_API_BASE:
        print(f"Warning: LLM_API_BASE not set. returning mock data for {task_name}")
        return {
            "summary": f"Mock summary for {task_name}",
            "keywords": ["mock", "keyword"],
            "entities": [{"name": "MockEntity", "type": "MockType", "context": "MockContext"}],
            "events": [{"description": "MockEvent", "date": "2023", "location": "Internet"}]
        }

    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}" 
    }

    # Truncate content 
    prompt_content = MATERIAL_PARSING_PROMPT + "\n\n待分析内容：\n" + content[:3000]

    payload = {
        "model": MODEL_NAME or "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(f"{LLM_API_BASE}/chat/completions", json=payload, headers=headers, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        res_json = response.json()
        content_str = res_json['choices'][0]['message']['content']
        
        try:
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0].strip()
            elif "```" in content_str:
                content_str = content_str.split("```")[1].strip()
            return json.loads(content_str)
        except json.JSONDecodeError:
            return {"raw_text": content_str}

    except Exception as e:
        print(f"LLM API Call failed: {e}")
        return {"error": str(e)}


def process_material_workflow(material_item):
    """
    针对单个素材的流转处理函数
    :param material_item: 字典，包含 {"id": "xxx", "content": "..."}
    :return: 包含溯源ID的处理结果
    """
    m_id = material_item.get("id")
    content = material_item.get("content")
    
    # 初始化结果字典，确保 ID 永远在第一位以便溯源
    result = {
        "source_id": m_id,
        "status": "success",
        "data": {}
    }

    if not content:
        result['status'] = "failed"
        result['error'] = "Content is empty"
        return result

    try:
        # 调用综合解析
        analysis_result = call_llm_api("综合解析", content)
        result['data'] = analysis_result

    except Exception as e:
        result['status'] = "failed"
        result['error'] = str(e)
    
    return result

def batch_process_to_file(materials, output_file="final_results.json"):
    """
    批量处理并生成整体文件
    """
    all_results = []
    
    # 使用线程池并发调用 API（大模型API通常是I/O密集型）
    print(f"开始处理 {len(materials)} 个素材...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        all_results = list(executor.map(process_material_workflow, materials))

    # 存储为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成！结果已保存至: {output_file}")
    return all_results

# --- 使用示例 ---
if __name__ == "__main__":
    # 你的素材列表，每个带唯一ID
    my_materials = [
        {"id": "DOC_001", "content": "素材1的内容..."},
        {"id": "DOC_002", "content": "素材2的内容..."},
        {"id": "DOC_003", "content": "素材3的内容..."},
    ]

    # 执行并生成文件
    final_data = batch_process_to_file(my_materials)