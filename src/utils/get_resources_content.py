"""素材内容处理工具模块

提供论文素材的批量分析处理功能，包括文本分块、LLM 调用、结果合并等。

主要功能：
- 文本分块处理（支持重叠）
- 调用大模型 API 进行内容分析
- 合并多个分块的分析结果
- 批量处理并输出结果文件
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from prompt import MATERIAL_PARSING_PROMPT
from utils.model import ask_messages, LLMError

load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# 配置参数
MAX_CHUNK_LENGTH = 8000  # 每个块的最大字符数
CHUNK_OVERLAP = 500      # 块之间的重叠字符数


def parse_first_json_payload(text: str) -> Any:
    """从任意文本中提取并解析首个 JSON 对象/数组。

    支持处理 LLM 返回的多种格式：
    - 纯 JSON 字符串
    - 包含 markdown code fence 的 JSON
    - 前后带有解释性文本的 JSON

    Args:
        text: LLM 返回的原始文本

    Returns:
        解析后的 JSON 对象或数组

    Raises:
        ValueError: 当文本为空或无法解析时
    """
    if text is None:
        raise ValueError("LLM 返回为空")

    content = str(text).strip()
    if not content:
        raise ValueError("LLM 返回为空字符串")

    # 优先处理 markdown code fence
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    decoder = json.JSONDecoder()

    # 先尝试整段直接解析
    try:
        return json.loads(content)
    except Exception:
        pass

    # 再从首个 { 或 [ 开始做 raw_decode，容忍前后噪声文本
    first_obj = content.find("{")
    first_arr = content.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        raise ValueError(f"未找到 JSON 起始符，原始返回前200字符: {content[:200]}")

    start = min(starts)
    sub = content[start:]
    try:
        parsed, _ = decoder.raw_decode(sub)
        return parsed
    except Exception as exc:
        raise ValueError(f"无法解析 JSON，原始返回前200字符: {content[:200]}") from exc

def chunk_text(text: str, max_length: int = MAX_CHUNK_LENGTH, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """将长文本分割成多个重叠的块。

    通过滑动窗口方式处理长文本，确保相邻块之间有一定的重叠部分，
    以保持上下文的连续性。

    Args:
        text: 原始文本
        max_length: 每块最大字符数，默认为 8000
        overlap: 块间重叠字符数，默认为 500

    Returns:
        分割后的文本块列表
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

def merge_analysis_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并多个分块的分析结果。

    采用不同策略合并各类数据：
    - 摘要：直接拼接
    - 关键词：去重后保留前10个
    - 实体：按 name 字段去重
    - 事件：按 description 字段去重

    Args:
        results_list: 多个分块的分析结果列表

    Returns:
        合并后的统一分析结果字典
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

def call_llm_api(prompt: str, content: str) -> Dict[str, Any]:
    """调用大模型 API 进行内容分析。

    Args:
        prompt: 系统提示词，定义模型的角色和任务
        content: 用户输入内容，限制前 3000 字符

    Returns:
        模型返回的原始响应对象，包含 choices[0].message.content 字段

    Raises:
        RuntimeError: 当调用大模型服务失败时
    """
    try:
        result = ask_messages(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content[:3000]}
            ],
            temperature=0.3,
            max_tokens=16384,
        )
        # 与原结构保持兼容，调用方继续从 choices[0].message.content 取值
        return result.raw
    except LLMError as exc:
        raise RuntimeError(f"调用大模型服务失败: {exc}") from exc


def process_material_workflow(material_item: Dict[str, Any]) -> Dict[str, Any]:
    """针对单个素材的流转处理函数。

    根据内容长度决定是否分块处理：
    - 内容长度超过 MAX_CHUNK_LENGTH 时，进行分块分析并合并结果
    - 内容较短时，直接单次分析

    Args:
        material_item: 素材字典，包含 {"id": "xxx", "content": "..."}

    Returns:
        包含溯源ID、处理状态和分析结果的字典
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

                    chunk_data = parse_first_json_payload(content_str)
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

            result['data'] = parse_first_json_payload(content_str)

    except Exception as e:
        result['status'] = "failed"
        result['error'] = str(e)
    
    return result


def batch_process_to_file(materials: List[Dict[str, Any]], output_file: str | None = None) -> Dict[str, Any]:
    """批量处理素材并生成结果文件。

    使用线程池并发处理多个素材，提高处理效率。
    结果保存为 JSON 文件，包含处理统计信息。

    Args:
        materials: 素材列表，每个元素为 {"id": "...", "content": "..."}
        output_file: 输出文件路径，为 None 时自动生成带时间戳的文件名

    Returns:
        包含文件路径、处理结果列表和统计信息的字典
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