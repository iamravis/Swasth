import json
import logging
from pathlib import Path
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate():
    # Paths
    config_path = Path("config/sft_config.yaml")
    input_root = Path("data/processed/sdk_inputs")
    output_root = Path("data/processed/qa_results")
    output_root.mkdir(parents=True, exist_ok=True)

    # Initialize SDK Components
    client = LLMClient(config_path=config_path)
    generator = QAGenerator(client=client, config_path=config_path)
    
    # Process each guideline directory
    for doc_dir in input_root.iterdir():
        if not doc_dir.is_dir(): continue
        
        doc_name = doc_dir.name
        doc_output = output_root / doc_name
        doc_output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧬 Generating QA for: {doc_name}")
        
        # Process each topic file in order
        topic_files = sorted(list(doc_dir.glob("*.txt")))
        
        for topic_file in topic_files:
            try:
                text = topic_file.read_text(encoding="utf-8")
                # Generate 3 high-quality pairs per topic grouping
                # Use the SDK's process_document which handles summary and QA pairs
                results = generator.process_document(text, num_pairs=3, verbose=True)
                
                if results and "qa_pairs" in results:
                    output_file = doc_output / f"{topic_file.stem}_qa.json"
                    output_file.write_text(json.dumps(results, indent=2))
                    logger.info(f"  ✅ {topic_file.name} -> {len(results['qa_pairs'])} pairs")
                else:
                    logger.warning(f"  ⚠️ No pairs generated for {topic_file.name}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error processing {topic_file.name}: {e}")

if __name__ == "__main__":
    generate()
