import json
import logging
from pathlib import Path
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

    def __init__(self, config_path="config/sft_config.yaml"):
        self.config_path = Path(config_path)
        import yaml
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_pairs = self.config.get("generation", {}).get("num_pairs", 3)
        self.client = LLMClient(config_path=self.config_path)
        self.generator = QAGenerator(client=self.client, config_path=self.config_path)

    def generate_all(self, input_root="data/processed/sdk_inputs", output_root="data/processed/qa_results"):
        input_root = Path(input_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Process each guideline directory
        for doc_dir in input_root.iterdir():
            if not doc_dir.is_dir(): continue
            
            doc_name = doc_dir.name
            doc_output = output_root / doc_name
            doc_output.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🧬 Generating QA for: {doc_name}")
            
            topic_files = sorted(list(doc_dir.glob("*.txt")))
            
            for topic_file in topic_files:
                try:
                    output_file = doc_output / f"{topic_file.stem}_qa.json"
                    
                    # Resume logic: Skip if already exists
                    if output_file.exists():
                        logger.info(f"  ⏭ Skipping {topic_file.name} (already exists)")
                        continue

                    text = topic_file.read_text(encoding="utf-8")
                    # Use the SDK's process_document which handles summary and QA pairs
                    results = self.generator.process_document(text, num_pairs=self.num_pairs, verbose=True)
                    
                    if results and "qa_pairs" in results:
                        output_file.write_text(json.dumps(results, indent=2))
                        logger.info(f"  ✅ {topic_file.name} -> {len(results['qa_pairs'])} pairs")
                    else:
                        logger.warning(f"  ⚠️ No pairs generated for {topic_file.name}")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error processing {topic_file.name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

if __name__ == "__main__":
    QualityGenerator().generate_all()
