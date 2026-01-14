import json
import logging
from pathlib import Path
from docling.chunking import HierarchicalChunker
from .core import get_docling_converter, clean_text

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class UnifiedPipeline:
    def __init__(self, raw_dir="data/raw", out_root="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.out_root = Path(out_root)
        self.chunker = HierarchicalChunker()
        
    def process(self):
        converter = get_docling_converter()
        files = [f for ext in ['*.pdf', '*.html', '*.docx'] for f in self.raw_dir.glob(ext)]
        
        for file in files:
            source = file.stem
            doc_dir = self.out_root / f"sdk_inputs/{source}"
            
            # Skip if already processed
            if doc_dir.exists() and any(doc_dir.iterdir()):
                logger.info(f"  ⏭ Skipping extraction for {file.name} (already structured)")
                continue

            try:
                logger.info(f"📄 Processing {file.name}...")
                result = converter.convert(file)
                
                # 1. Markdown
                md = clean_text(result.document.export_to_markdown())
                (self.out_root / f"markdown/{source}.md").write_text(md)

                # 2. Units & SDK Inputs
                units = []
                for chunk in self.chunker.chunk(result.document):
                    headings = getattr(chunk.meta, "headings", []) or []
                    bc = clean_text(" > ".join(headings) or "General")
                    txt = clean_text(self.chunker.serialize(chunk))
                    units.append({"bc": bc, "txt": txt})

                self._export_units(source, units)
                logger.info(f"✅ Processed {source}")
            except Exception as e:
                logger.error(f"❌ Failed {file.name}: {e}")

    def _export_units(self, source, units):
        # Save JSON
        (self.out_root / f"json_units/{source}.json").write_text(json.dumps(units, indent=2))
        
        # Save Topic-Grouped TXT for SDK
        doc_dir = self.out_root / f"sdk_inputs/{source}"
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        groups = {}
        for u in units: groups.setdefault(u["bc"], []).append(u["txt"])
        
        for bc, contents in groups.items():
            safe_bc = bc.replace(" > ", "_").replace(" ", "_").replace("/", "_")[:50]
            (doc_dir / f"{safe_bc}.txt").write_text(f"### {bc}\n\n" + "\n\n".join(contents))

if __name__ == "__main__":
    UnifiedPipeline().process()
