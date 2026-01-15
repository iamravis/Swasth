import json
import logging
import yaml
from pathlib import Path
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator

from synthetic_data_kit.generators.qa_generator import QAGenerator

logger = logging.getLogger("rich")

import os
import contextlib

class QualityGenerator:
    def __init__(self, config_path="config/sft_config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_pairs = self.config.get("generation", {}).get("num_pairs", 3)
        
        # Suppress SDK's internal print() statements
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
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
            
            topic_files = sorted(list(doc_dir.glob("*.txt")))
            
            # Count skips and identify new files
            to_process = []
            skipped_count = 0
            for tf in topic_files:
                if (doc_output / f"{tf.stem}_qa.json").exists():
                    skipped_count += 1
                else:
                    to_process.append(tf)
            
            if skipped_count > 0:
                logger.info(f"⏭ Skipped {skipped_count} sections in {doc_name} (already exist)")
            
            if not to_process:
                continue

            logger.info(f"🧬 Generating QA for {len(to_process)} new sections in: {doc_name}")
            
            for topic_file in to_process:
                try:
                    output_file = doc_output / f"{topic_file.stem}_qa.json"
                    text = topic_file.read_text(encoding="utf-8")
                    
                    # Generate QA pairs (Fast path)
                    qa_pairs = self.generator.generate_qa_pairs(text, summary="", num_pairs=self.num_pairs)
                    
                    if qa_pairs:
                        # Return the original text as 'summary' (context) for the SFT builder
                        results = {
                            "summary": text,
                            "qa_pairs": qa_pairs
                        }
                        output_file.write_text(json.dumps(results, indent=2))
                        logger.info(f"  ✅ {topic_file.name} -> {len(qa_pairs)} pairs")
                    else:
                        logger.warning(f"  ⚠️ No results generated for {topic_file.name}")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error processing {topic_file.name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

if __name__ == "__main__":
    QualityGenerator().generate_all()
