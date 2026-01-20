import json
import time
import os
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# 配置参数
MAX_CHUNK_LENGTH = 8000  # 每个块的最大字符数
CHUNK_OVERLAP = 500      # 块之间的重叠字符数

def chunk_text(text, max_length=MAX_CHUNK_LENGTH, overlap=CHUNK_OVERLAP):
    """
    将长文本分割成多个重叠的块
    :param text: 原始文本
    :param max_length: 每块最大长度
    :param overlap: 块间重叠长度
    :return: 文本块列表
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = end - overlap
    
    return chunks

def merge_analysis_results(results_list):
    """
    合并多个分块的分析结果
    :param results_list: 分析结果列表
    :return: 合并后的结果
    """
    if not results_list:
        return {}
    
    if len(results_list) == 1:
        return results_list[0]
    
    # 合并策略
    merged = {
        "summary": "",
        "keywords": [],
        "entities": [],
        "events": []
    }
    
    # 合并摘要（拼接所有摘要）
    summaries = [r.get("summary", "") for r in results_list if r.get("summary")]
    merged["summary"] = " ".join(summaries)
    
    # 合并关键词（去重）
    keywords_set = set()
    for r in results_list:
        if "keywords" in r and isinstance(r["keywords"], list):
            keywords_set.update(r["keywords"])
    merged["keywords"] = list(keywords_set)[:10]  # 保留前10个
    
    # 合并实体（去重，基于name）
    entities_dict = {}
    for r in results_list:
        if "entities" in r and isinstance(r["entities"], list):
            for entity in r["entities"]:
                if isinstance(entity, dict) and "name" in entity:
                    name = entity["name"]
                    if name not in entities_dict:
                        entities_dict[name] = entity
    merged["entities"] = list(entities_dict.values())
    
    # 合并事件（去重，基于description）
    events_dict = {}
    for r in results_list:
        if "events" in r and isinstance(r["events"], list):
            for event in r["events"]:
                if isinstance(event, dict) and "description" in event:
                    desc = event["description"]
                    if desc not in events_dict:
                        events_dict[desc] = event
    merged["events"] = list(events_dict.values())
    
    return merged

def call_llm_api(prompt, content):
    """
    调用大模型 API 进行处理
    """
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}" 
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content[:3000]}
        ],
        "temperature": 0.3,
        "max_tokens": 16384,
    }

    response = requests.post(f"{LLM_API_BASE}/chat/completions", json=payload, headers=headers, timeout=LLM_TIMEOUT)
    response.raise_for_status()
    return response.json()


def process_material_workflow(material_item):
    """
    针对单个素材的流转处理函数
    :param material_item: 字典，包含 {"id": "xxx", "content": "..."}
    :return: 包含溯源ID的处理结果
    """
    m_id = material_item.get("id")
    content = material_item.get("content")
    
    result = {
        "source_id": m_id,
        "status": "success",
        "data": {},
        "chunks_processed": 0
    }

    if not content:
        result['status'] = "failed"
        result['error'] = "Content is empty"
        return result

    try:
        # 检查内容长度，决定是否分块
        if len(content) > MAX_CHUNK_LENGTH:
            print(f"素材 {m_id} 内容过长({len(content)} 字符)，进行分块处理...")
            chunks = chunk_text(content)
            result['chunks_processed'] = len(chunks)
            
            chunk_results = []
            for i, chunk in enumerate(chunks):
                print(f"  处理块 {i+1}/{len(chunks)}...")
                try:
                    analysis_result = call_llm_api(MATERIAL_PARSING_PROMPT, f"待分析内容如下 (第{i+1}/{len(chunks)}部分): " + chunk)
                    content_str = analysis_result['choices'][0]['message']['content']
                    
                    if "```json" in content_str:
                        content_str = content_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in content_str:
                        content_str = content_str.split("```")[1].strip()
                    
                    chunk_data = json.loads(content_str)
                    chunk_results.append(chunk_data)
                except Exception as e:
                    print(f"  块 {i+1} 处理失败: {e}")
                    continue
            
            # 合并所有块的结果
            result['data'] = merge_analysis_results(chunk_results)
        else:
            result['chunks_processed'] = 1
            analysis_result = call_llm_api(MATERIAL_PARSING_PROMPT, "待分析内容如下: " + content)
            content_str = analysis_result['choices'][0]['message']['content']
            
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0].strip()
            elif "```" in content_str:
                content_str = content_str.split("```")[1].strip()
            
            result['data'] = json.loads(content_str)

    except Exception as e:
        result['status'] = "failed"
        result['error'] = str(e)
    
    return result


def batch_process_to_file(materials, output_file=None):
    """
    批量处理并生成整体文件
    :param materials: 素材列表 [{"id": "...", "content": "..."}, ...]
    :param output_file: 输出文件路径，如果为 None 则自动生成
    :return: {"file_path": "...", "results": [...], "total": ..., "success": ..., "failed": ...}
    """
    all_results = []
    
    print(f"开始处理 {len(materials)} 个素材...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        all_results = list(executor.map(process_material_workflow, materials))

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = str(uuid.uuid4())[:8]
        output_file = os.path.join(ANALYSIS_OUTPUT_DIR, f"analysis_{timestamp}_{batch_id}.json")
    else:
        if not os.path.isabs(output_file):
            output_file = os.path.join(ANALYSIS_OUTPUT_DIR, output_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成！结果已保存至: {output_file}")
    
    total = len(all_results)
    success = sum(1 for r in all_results if r.get("status") == "success")
    failed = total - success
    
    return {
        "file_path": output_file,
        "results": all_results,
        "total": total,
        "success": success,
        "failed": failed
    }

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