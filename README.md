# Medical SFT Data Pipeline

A professional pipeline to transform clinical guidelines into context-enriched synthetic datasets for fine-tuning LLMs.

## 🚀 Features
- **Docling v2 Integration**: High-fidelity PDF-to-Markdown with structural awareness.
- **Contextual Breadcrumbs**: Injects heading hierarchy into every training unit.
- **NVIDIA GPU Optimized**: Built-in support for multiple 3090 GPUs.
- **Unicode Normalization**: Automatic cleanup of non-breaking hyphens and medical artifacts.

## 🛠 Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place raw PDFs in `data/raw/`.
3. Run the pipeline:
   ```bash
   python3 -m src.pipeline
   ```

## 🧠 GPU Resource Management (3x 3090)
To limit Ollama to 70% of your 3x 3090s while generating data:
```bash
# Limit Ollama parallel generation
export OLLAMA_NUM_PARALLEL=3
# Use SDK with local model
synthetic-data-kit create ./data/processed/sdk_inputs/ --api-base http://localhost:11434/v1 --model gpt-oss:20b
```

## 📁 Structure
- `src/`: Core logic and CLI.
- `config/`: Prompt templates and SDK configurations.
- `data/`: Raw inputs and multi-stage processed outputs.
