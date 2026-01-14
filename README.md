# Swasth: Professional Medical SFT Pipeline

A high-performance system for transforming clinical guidelines into context-enriched synthetic datasets for Supervised Fine-Tuning (SFT).

## 🚀 Key Features

- **Structural Preservation**: Leverages Docling v2 for high-fidelity conversion of complex medical PDFs and documents.
- **Hierarchical Context**: Automatically builds contextual breadcrumbs from document headers to enrich knowledge units.
- **Multi-Backend Acceleration**: Cross-platform support for hardware acceleration (CUDA/MPS) via a unified detection engine.
- **Universal Automation**: A single entry point for extraction, transformation, and generation with built-in service health checks.
- **Incremental Processing**: Intelligent "Resume" logic that skips previously structured or generated data to optimize compute cycles.

## 🛠 Usage

### 1. Unified Execution
Run the entire pipeline (Extraction -> Health Check -> Generation) with a single command:
```bash
python3 main.py
```

### 2. Deployment Guidelines
For large-scale dataset generation, ensure your local inference server (e.g., Ollama) is configured to handle the desired concurrency.
```bash
# Example: Configure parallel generation threads
export OLLAMA_NUM_PARALLEL=4
```

## 📁 Project Architecture

- `src/core.py`: Hardware detection and data sanitization modules.
- `src/pipeline.py`: Document processing and hierarchical structuring.
- `src/generate_qa.py`: High-fidelity synthetic data generation engine.
- `config/`: Enterprise-grade prompt templates and pipeline configurations.
- `data/`: Modular data lake for raw inputs and multi-stage outputs.

## ⚙️ Configuration

Custom SFT standards are defined in `config/sft_config.yaml`. The pipeline is pre-configured for clinical depth, expert clinical personas, and strict evidence grounding.
