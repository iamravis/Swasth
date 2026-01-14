import unicodedata
import torch
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Normalize unicode and fix medical text artifacts."""
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    return text.replace('\u2011', '-')

def get_docling_converter():
    """Configure Docling with GPU support (CUDA for NVIDIA, MPS for Apple)."""
    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
    
    if torch.cuda.is_available():
        device = AcceleratorDevice.CUDA
    elif torch.backends.mps.is_available():
        device = AcceleratorDevice.MPS
    else:
        device = AcceleratorDevice.CPU
        
    logger.info(f"Using accelerator device: {device}")
    
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.accelerator_options = AcceleratorOptions(
        num_threads=8, 
        device=device
    )
    
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.HTML, InputFormat.DOCX],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
