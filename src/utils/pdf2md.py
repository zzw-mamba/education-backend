import os
from marker.converters.pdf import PdfConverter
from marker.models import load_all_models
from marker.config.parser import ConfigParser

model_lst = load_all_models()

def extract_with_marker(pdf_path, source_id):
    # 1. 定义配置（可以自定义是否识别OCR、是否处理表格等）
    config_dict = {
        "output_format": "markdown",
        "parallel_factor": 2, # 并行度
    }
    config_parser = ConfigParser(config_dict)
    
    # 3. 初始化转换器
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_lst,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer()
    )
    
    result = converter(pdf_path)
    
    full_text = result.markdown
    
    return full_text