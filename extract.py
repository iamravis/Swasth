import os
import logging
import json
import unicodedata
from pathlib import Path

def clean_text(text):
    """Normalize unicode and remove artifacts like non-breaking hyphens."""
    if not text:
        return ""
    # Normalize to NFKC to handle things like \u2011 (non-breaking hyphen)
    text = unicodedata.normalize('NFKC', text)
    # Explicitly replace common artifacts if normalization missed them
    text = text.replace('\u2011', '-')
    return text

# 1. Setup Logging & Paths
logging.basicConfig(level=logging.INFO)
INPUT_DIR = Path("./medical_guidelines_raw")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./guidelines_structured_md")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR = Path("./guidelines_chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def process_guidelines():
    # 2. Configure Docling for High-Quality Medical Parsing
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.chunking import HierarchicalChunker

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Handles scanned medical PDFs
    pipeline_options.do_table_structure = True  # Critical for dosage/symptom tables
    pipeline_options.images_scale = 2.0  # Better image resolution for VLM
    
    # Optional: If you had a VLM setup, you'd add it here
    # pipeline_options.enable_vlm = True 

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.HTML, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    chunker = HierarchicalChunker()

    # 3. Gather all files
    input_files = []
    for ext in ['*.pdf', '*.html', '*.docx']:
        input_files.extend(list(INPUT_DIR.glob(ext)))
    
    print(f"🚀 Found {len(input_files)} guidelines. Starting advanced conversion...")

    # 4. Batch Processing
    results = converter.convert_all(input_files, raises_on_error=False)

    for i, result in enumerate(results):
        source_name = input_files[i].stem
        try:
            # A. Export to structured Markdown (for reference)
            md_content = clean_text(result.document.export_to_markdown())
            md_file_path = OUTPUT_DIR / f"{source_name}.md"
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            # B. Advanced SFT-Optimized Extraction
            doc_metadata = {
                "title": getattr(result.document, "name", source_name),
                "num_pages": len(result.document.pages) if hasattr(result.document, "pages") else "N/A",
                "format": result.input.format.value if hasattr(result.input, "format") else "unknown"
            }
            
            chunks = list(chunker.chunk(result.document))
            sft_knowledge_units = []
            
            for chunk in chunks:
                headings = []
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    headings = chunk.meta.headings
                
                breadcrumb = clean_text(" > ".join(headings)) if headings else "General"
                chunk_text = clean_text(chunker.serialize(chunk))
                full_context_text = f"[Context: {breadcrumb}]\n\n{chunk_text}"
                
                # SFT-ready entry
                sft_knowledge_units.append({
                    "source": source_name,
                    "breadcrumb": breadcrumb,
                    "content": chunk_text,
                    "context_enriched_text": full_context_text,
                    "metadata": chunk.meta.export_json_dict() if hasattr(chunk.meta, "export_json_dict") else {}
                })

            chunk_file_path = CHUNKS_DIR / f"{source_name}_sft_units.json"
            with open(chunk_file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "document_metadata": doc_metadata,
                    "units": sft_knowledge_units
                }, f, indent=2)
                
            logging.info(f"✅ SFT READY: {source_name} | Units: {len(sft_knowledge_units)}")
        except Exception as e:
            logging.error(f"❌ Failed: {source_name} | Error: {e}")
            import traceback
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    process_guidelines()