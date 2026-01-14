import sys
import logging
import requests
import argparse
import shutil
from pathlib import Path
from src.pipeline import UnifiedPipeline
from src.generate_qa import QualityGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_ollama():
    """Check if Ollama is running and reachable."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama is running.")
            return True
    except Exception:
        pass
    
    logger.error("❌ Ollama is not detected at http://localhost:11434.")
    logger.error("Please start Ollama before running the generation stage.")
    return False

def ingest_file(file_path):
    """Copy a single file into the data/raw directory."""
    src = Path(file_path)
    if not src.exists():
        logger.error(f"❌ File not found: {file_path}")
        sys.exit(1)
        
    dest_dir = Path("data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    
    if dest.exists():
        logger.info(f"ℹ️ File {src.name} already exists in data/raw.")
    else:
        shutil.copy2(src, dest)
        logger.info(f"📥 Ingested {src.name} to data/raw/")

def run_all():
    parser = argparse.ArgumentParser(description="Unified Medical SFT Pipeline")
    parser.add_argument("--file", "-f", help="Path to a new PDF/HTML/DOCX to ingest and process")
    args = parser.parse_args()

    logger.info("🚀 Starting Unified Medical SFT Pipeline")
    
    # 0. Ingestion (Optional)
    if args.file:
        logger.info(f"--- Phase 0: Ingesting {args.file} ---")
        ingest_file(args.file)

    # 1. Extraction & Structuring
    logger.info("--- Phase 1: Extraction & Structuring ---")
    pipeline = UnifiedPipeline()
    pipeline.process()
    
    # 2. Service Health Check
    logger.info("--- Phase 2: Service Health Check ---")
    if not check_ollama():
        logger.warning("⚠️ Skipping Stage 3 (QA Generation) because Ollama is unreachable.")
        sys.exit(1)
        
    # 3. QA Generation
    logger.info("--- Phase 3: QA Generation ---")
    gen = QualityGenerator()
    gen.generate_all()
    
    logger.info("✨ Pipeline Execution Complete!")

if __name__ == "__main__":
    run_all()
