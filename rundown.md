# Project Rundown: Medical SFT Data Pipeline

This document provides a microscopic breakdown of the pipeline's architecture, data flow, and the logic behind our "Scientist's Approach" to dataset generation.

## 1. High-Level Flow

The pipeline is a three-stage transformation process designed to turn unstructured medical PDFs into high-quality, clinical-grade training data.

```mermaid
graph TD
    subgraph "Stage 1: Extraction & Cleanup"
        A["Raw Clinical PDFs"] --> B["pipeline.py + core.py"]
        B --> C["Unicode Normalization"]
        C --> D["Structural Parsing (Docling)"]
    end

    subgraph "Stage 2: Context Engineering"
        D --> E["Hierarchical Chunking"]
        E --> F["Topic-Grouped TXT Files"]
    end

    subgraph "Stage 3: SFT Generation"
        F --> G["generate_qa.py"]
        G --> H["Llama Synthetic Data Kit"]
        H --> I["gpt-oss:20b (Local Ollama)"]
        I --> J["Final QA Dataset (JSON)"]
    end
```

---

## 2. Microscopic Code Breakdown

### `src/core.py`: The Hardware & Quality Guard
*   **Multi-Backend Acceleration**: It uses `torch` to detect the environment. It automatically initializes the appropriate accelerator (e.g., CUDA for NVIDIA GPUs or MPS for Apple Silicon). This ensures maximum performance across various workstation and server environments.
*   **Unicode Sanitization**: Medical PDFs often contain non-standard characters (like the `\u2011` non-breaking hyphen). `clean_text()` normalizes everything to NFKC standard to prevent training artifacts.

### `src/pipeline.py`: The Context Architect
*   **Hierarchical Clustering**: Instead of just cutting text at arbitrary lengths, this script uses `HierarchicalChunker`. It extracts the "Breadcrumb" (e.g., *Management > Surgery > Complications*) for every chunk.
*   **Topic Grouping**: This is the "Scientist's Choice." We group related chunks into a single Topic file. This provides the LLM with a coherent logical context, allowing it to generate "Real-Life" scenarios rather than trivia.

### `src/generate_qa.py`: The Generator Engine
*   **SDK Integration**: It interfaces directly with the `synthetic-data-kit` Python API.
*   **Two-Pass Generation**: For every topic, it first asks the model to generate a **Summary** (for internal world-building) and then generates the **QA pairs**.

### `config/sft_config.yaml`: The Brain
*   **Expert Personas**: We explicitly instruct the model to behave as a "Distinguished Medical Consultant."
*   **Strict Constraints**: It forbids triviality and mandates "Clinical Depth" and "Long-Form Answers."

---

---

## 3. High-Performance Compute Strategy
The system is architected for scalability across high-performance compute environments:
1.  **Extraction Tier (Docling)**: Leverages hardware acceleration to parallelize layout analysis and OCR.
2.  **Inference Tier**: Dynamically load-balances across available compute resources. By targeting high-parameter models (e.g., 20B+), the pipeline utilizes large memory buffers for the sophisticated reasoning required in clinical datasets.

---

## 4. Why This Works for SFT
Standard RAG often fails in medicine because context is fragmented. This pipeline ensures:
- **Zero Hallucination**: Every answer is strictly grounded in the grouped topic text.
- **Structural Integrity**: Tables and lists are kept whole within chunks.
- **Human-Like Reasoning**: The prompts force the model to explain *why* a clinical action is taken, which is critical for fine-tuning a model to "think" like a doctor.
