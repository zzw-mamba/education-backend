import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

model_lst = create_model_dict()

def extract_with_marker(pdf_path, source_id):
    config_dict = {
        "output_format": "markdown",
        "batch_multiplier": 1,
        "allow_ocr": True
    }
    
    config_parser = ConfigParser(config_dict)
    
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_lst,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    
    result = converter(pdf_path)
    

    if isinstance(result, dict):
        full_text = result.get("markdown", "")
    else:
        full_text = getattr(result, "markdown", "")
    
    return full_text